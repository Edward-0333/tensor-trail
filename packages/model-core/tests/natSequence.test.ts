import { describe, it, expect } from 'vitest';
import { registry } from '../src/modules/index';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { ports } from '../src/core/graph';
import type { Project, Params, TensorSpec } from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { sequenceCode } from '../src/modules/sequenceOps';
import { natSource } from '../src/modules/natSequence';

const t = (shape: (number | string)[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const infer = (type: string, inputs: Record<string, TensorSpec>, params: Params = {}) => {
  const node = registry.create(type, params);
  return registry.get(type)!.infer!(inputs, node.params);
};
function project(length = 20, params: Params = {}, dtype = 'float32'): Project {
  return {
    schemaVersion: 1,
    name: 'PLUTO NAT',
    variants: [],
    graph: {
      nodes: [
        registry.create(
          'Input',
          { shape: `3,9,${length}`, axes: 'agent,channel,time', dtype },
          'x',
        ),
        registry.create('NATSequenceEncoder', params, 'history_encoder'),
        registry.create('Output', {}, 'out'),
      ],
      edges: [
        {
          id: 'input',
          source: 'x',
          sourcePort: 'out',
          target: 'history_encoder',
          targetPort: 'in',
        },
        {
          id: 'output',
          source: 'history_encoder',
          sourcePort: 'out',
          target: 'out',
          targetPort: 'in',
        },
      ],
    },
  };
}
describe('行 81 的数据准备', () => {
  it('布尔索引保留数据类型和尾轴，数量保持符号而不假造真实掩码统计', () => {
    const r = infer('BooleanIndex', { in: t([98, 20, 9]), mask: t([98], 'bool') });
    expect(tensorValue(r.outputs.out).shape).toEqual(['N_valid', 20, 9]);
    expect(r.outputs.out.dtype).toBe('float32');
    expect(r.notes?.join()).toContain('count_nonzero');
    const permuted = infer('Permute', { in: r.outputs.out }, { dims: '0,2,1' }).outputs.out;
    expect(permuted.shape).toEqual(['N_valid', 9, 20]);
    expect(infer('Contiguous', { in: permuted }).outputs.out).toEqual(permuted);
  });
  it('支持空 batch 和符号 batch，不修改输入元数据', () => {
    const input = t(['B_A', 20, 9]),
      before = structuredClone(input);
    expect(
      infer('BooleanIndex', { in: input, mask: t(['B_A'], 'bool') }).outputs.out.shape[0],
    ).toBe('N_valid');
    expect(input).toEqual(before);
    expect(
      infer('BooleanIndex', { in: t([0, 20, 9]), mask: t([0], 'bool') }).outputs.out.shape,
    ).toEqual([0, 20, 9]);
  });
  it.each([t([98], 'int64'), t([98, 1], 'bool'), t([97], 'bool')])('拒绝错误掩码 %j', (mask) => {
    expect(() => infer('BooleanIndex', { in: t([98, 20, 9]), mask })).toThrow();
  });
});
describe('函数代码块接口', () => {
  it('邻域注意力保留时间轴；短于邻域窗口的序列仍可用', () => {
    const input = t(['N_valid', 2, 128]);
    expect(
      infer('CodeFunction', { in: input }, { dim: 128, num_heads: 8, kernel_size: 5 }).outputs.out,
    ).toEqual(input);
    expect(
      ports(registry.create('CodeFunction', { operation: 'Interpolate1D' }), registry).inputs.map(
        (p) => p.id,
      ),
    ).toEqual(['in', 'like']);
  });
  it.each<Params>([
    { dim: 33 },
    { num_heads: 3 },
    { kernel_size: 4 },
    { kernel_size: 1 },
    { attn_drop: 'bad' },
    { proj_drop: '1.2' },
  ])('邻域注意力拒绝无效参数 %j', (params) => {
    expect(() => infer('CodeFunction', { in: t([3, 20, 32]) }, params)).toThrow();
  });
  it('插值仅从 like 获取长度，不错误复制其 batch、通道或 dtype', () => {
    expect(
      infer(
        'CodeFunction',
        { in: t([3, 128, 5]), like: t([9, 64, 10], 'int64') },
        { operation: 'Interpolate1D' },
      ).outputs.out,
    ).toEqual(t([3, 128, 10]));
    expect(() =>
      infer(
        'CodeFunction',
        { in: t([3, 128, 0]), like: t([3, 128, 10]) },
        { operation: 'Interpolate1D' },
      ),
    ).toThrow();
  });
  it('DropPath 和 Dropout 概率校验，代码仅展示', () => {
    const input = t([3, 20, 32]);
    expect(
      infer('CodeFunction', { in: input }, { operation: 'DropPath', p: '0.2' }).outputs.out,
    ).toEqual(input);
    expect(infer('Dropout', { in: input }, { p: '0.1' }).outputs.out).toEqual(input);
    for (const p of ['-0.1', '1', 'NaN', ''])
      expect(() => infer('Dropout', { in: input }, { p })).toThrow();
    const code = sequenceCode(registry.create('CodeFunction', { operation: 'Interpolate1D' }));
    expect(code).toContain('mode="linear", align_corners=False');
    expect(sequenceCode(registry.create('CodeFunction'))).toContain('NeighborhoodAttention1D');
  });
});
describe('NATSequenceEncoder 分层数据流', () => {
  // Expected shapes were measured with local PLUTO + NATTEN 0.14.6, CPU eval,
  // random weights and batch=3. No training or cache generation is required.
  it.each([
    [20, 10, 5],
    [21, 11, 6],
    [7, 4, 2],
  ])('与 PyTorch 实测逐级形状一致：T=%i', (time, t1, t2) => {
    const p = project(time),
      before = structuredClone(p),
      r = inferProject(p, registry).nodes.history_encoder;
    expect(r.status).toBe('ok');
    expect(tensorValue(r.outputs.out).shape).toEqual([3, 128]);
    const child = r.child!.nodes;
    expect(tensorValue(child.embed.outputs.out).shape).toEqual([3, time, 32]);
    expect(tensorValue(child.level0.outputs.next).shape).toEqual([3, t1, 64]);
    expect(tensorValue(child.level0.outputs.xo).shape).toEqual([3, time, 32]);
    expect(tensorValue(child.level1.outputs.next).shape).toEqual([3, t2, 128]);
    expect(tensorValue(child.level1.outputs.xo).shape).toEqual([3, t1, 64]);
    expect(child.level2.outputs.next).toEqual(child.level2.outputs.xo);
    for (const [i, length] of [time, t1, t2].entries())
      expect(tensorValue(child[`lateral${i}`].outputs.out).shape).toEqual([3, 128, length]);
    expect(tensorValue(child.fpn_conv.outputs.out).shape).toEqual([3, 128, time]);
    expect(tensorValue(child.fuse1.outputs.out).shape).toEqual([3, 128, t1]);
    expect(tensorValue(child.fuse0.outputs.out).shape).toEqual([3, 128, time]);
    expect(p).toEqual(before);
  });
  it('FPN 先融合深层，再将融合结果传给浅层，而不是只传原始侧向输出', () => {
    const g = registry.create('NATSequenceEncoder').subgraph!;
    expect(g.edges).toContainEqual(
      expect.objectContaining({ source: 'fuse1', target: 'upsample1', targetPort: 'in' }),
    );
    expect(g.edges).toContainEqual(
      expect.objectContaining({ source: 'lateral0', target: 'upsample1', targetPort: 'like' }),
    );
    expect(g.nodes.find((n) => n.id === 'last_step')!.params.index).toBe('-1');
  });
  it('两次残差连接、三级深度、卷积 bias 和 DropPath 调度均保留', () => {
    const g = registry.create('NATSequenceEncoder').subgraph!,
      rates: number[] = [];
    for (let i = 0; i < 3; i++) {
      const level = g.nodes.find((n) => n.id === `level${i}`)!;
      expect(ports(level, registry).outputs.map((p) => p.id)).toEqual(['next', 'xo']);
      const layers = level.subgraph!.nodes.filter((n) => n.type === 'NATLayer');
      expect(layers).toHaveLength(2);
      expect(level.params.downsample).toBe(i < 2);
      for (const layer of layers) {
        rates.push(Number(layer.params.drop_path));
        const child = layer.subgraph!;
        expect(child.edges).toContainEqual(
          expect.objectContaining({ source: 'x', target: 'residual1', targetPort: 'a' }),
        );
        expect(child.edges).toContainEqual(
          expect.objectContaining({ source: 'residual1', target: 'residual2', targetPort: 'a' }),
        );
        expect(child.nodes.find((n) => n.id === 'drop_path1')!.params.p).toBe(
          child.nodes.find((n) => n.id === 'drop_path2')!.params.p,
        );
      }
      if (i < 2)
        expect(
          level
            .subgraph!.nodes.find((n) => n.id === 'downsample')!
            .subgraph!.nodes.find((n) => n.id === 'reduction')!.params.bias,
        ).toBe(false);
    }
    expect(rates).toEqual([0, 0.04, 0.08, 0.12, 0.16, 0.2]);
  });
  it('改参数后重新生成对应层数、宽度；支持符号有效车辆数量', () => {
    const p = project(20, { embed_dim: 16, depths: '1,2,1', num_heads: '1,2,4' });
    p.graph.nodes[0].params.shape = 'N_valid,9,20';
    const r = inferProject(p, registry).nodes.history_encoder;
    expect(r.status).toBe('ok');
    expect(tensorValue(r.outputs.out).shape).toEqual(['N_valid', 64]);
  });
  it('序列化重开保留层级与接口；没有伪装为普通 MHA', () => {
    const p = project();
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const code = natSource(p.graph.nodes[1]);
    expect(code).toContain('return out[:, :, -1]');
    const text = serializeProject(p);
    expect(text).toContain('NeighborhoodAttention1D');
    expect(text).not.toContain('MultiHeadAttention');
  });
  it.each<Params>([
    { depths: '0,2,2' },
    { depths: '2,2' },
    { depths: '9,2,2' },
    { num_heads: '3,4,8' },
    { kernel_size: '2,3,5' },
    { drop_path_rate: '1.1' },
  ])('拒绝无效生成参数 %j', (params) => {
    expect(() => registry.create('NATSequenceEncoder', params)).toThrow();
  });
  it('拒绝错误通道和非浮点输入', () => {
    const p = project();
    p.graph.nodes[0].params.shape = '3,8,20';
    expect(inferProject(p, registry).nodes.history_encoder.status).toBe('error');
    expect(inferProject(project(20, {}, 'int64'), registry).nodes.history_encoder.status).toBe(
      'error',
    );
  });
});
