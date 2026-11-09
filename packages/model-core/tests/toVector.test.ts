import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Dim, type Project, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules/index';
import { toVector } from '../src/modules/toVector';

const tensor = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  dtype,
  axes: shape.map(
    (_, index) => ['batch', 'agent', 'time', 'feature', 'channel'][index] ?? `axis${index}`,
  ),
});

function project(feat = tensor([2, 49, 21]), mask = tensor([2, 49, 21], 'bool')): Project {
  return {
    schemaVersion: 1,
    name: 'to_vector 相邻帧差分',
    variants: [],
    graph: {
      nodes: [
        registry.create(
          'ForwardInput',
          { schema: JSON.stringify({ kind: 'dict', fields: { feat, valid_mask: mask } }) },
          'data',
        ),
        registry.create('DictionaryGet', { path: 'feat' }, 'feat'),
        registry.create('DictionaryGet', { path: 'valid_mask' }, 'mask'),
        registry.create('ToVector', {}, 'vector'),
        registry.create('Output', {}, 'result'),
      ],
      edges: [
        { id: 'read_feat', source: 'data', sourcePort: 'out', target: 'feat', targetPort: 'in' },
        { id: 'read_mask', source: 'data', sourcePort: 'out', target: 'mask', targetPort: 'in' },
        { id: 'feat', source: 'feat', sourcePort: 'out', target: 'vector', targetPort: 'feat' },
        {
          id: 'mask',
          source: 'mask',
          sourcePort: 'out',
          target: 'vector',
          targetPort: 'valid_mask',
        },
        { id: 'result', source: 'vector', sourcePort: 'out', target: 'result', targetPort: 'in' },
      ],
    },
  };
}

describe('to_vector 函数组合', () => {
  it('注册可复用的 feat / valid_mask 两输入函数，内部保存真实差分与条件置零依赖', () => {
    const node = registry.create('ToVector');
    expect(registry.get('ToVector')).toBe(toVector);
    expect(toVector.category).toBe('组合计算');
    expect(toVector.fields).toEqual([]);
    expect(ports(node, registry).inputs.map((port) => port.id)).toEqual(['feat', 'valid_mask']);
    expect(ports(node, registry).outputs.map((port) => port.id)).toEqual(['out']);
    const graph = node.subgraph!;
    expect(graph.nodes.find((child) => child.id === 'mask_previous')!.params).toMatchObject({
      mode: 'slice',
      dim: '-1',
      start: '',
      stop: '-1',
      stopFromInput: false,
    });
    expect(graph.nodes.find((child) => child.id === 'feat_next')!.params).toMatchObject({
      mode: 'slice',
      dim: '2',
      start: '1',
      stop: '',
      stopFromInput: false,
    });
    expect(graph.nodes.find((child) => child.id === 'feat_previous')!.params).toMatchObject({
      mode: 'slice',
      dim: '2',
      start: '',
      stop: '-1',
      stopFromInput: false,
    });
    for (const [id, type] of [
      ['mask_pair', 'BitwiseAnd'],
      ['mask_aligned', 'AlignTrailingDims'],
      ['difference', 'Subtract'],
      ['zeros', 'ZerosLike'],
      ['where', 'Where'],
    ])
      expect(graph.nodes.find((child) => child.id === id)!.type).toBe(type);
    expect(graph.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: 'feat_next', target: 'difference', targetPort: 'a' }),
        expect.objectContaining({ source: 'feat_previous', target: 'difference', targetPort: 'b' }),
        expect.objectContaining({ source: 'feat', target: 'mask_aligned', targetPort: 'like' }),
        expect.objectContaining({
          source: 'mask_aligned',
          target: 'where',
          targetPort: 'condition',
        }),
        expect.objectContaining({ source: 'difference', target: 'where', targetPort: 'x' }),
        expect.objectContaining({ source: 'zeros', target: 'where', targetPort: 'y' }),
      ]),
    );
  });

  it.each<[string, number[]]>([
    ['heading', []],
    ['position / velocity', [2]],
    ['多轴特征', [2, 3]],
  ])('%s 在时间轴做差，掩码按需补齐尾部维度', (_, trailing) => {
    const feat = tensor([2, 49, 21, ...trailing]);
    const result = inferProject(project(feat), registry);
    expect(result.issues).toEqual([]);
    expect(result.nodes.vector.status, result.nodes.vector.messages.join()).toBe('ok');
    expect(result.nodes.result.outputs.out).toEqual({ ...feat, shape: [2, 49, 20, ...trailing] });
    const child = result.nodes.vector.child!;
    expect(tensorValue(child.nodes.mask_pair.outputs.out).shape).toEqual([2, 49, 20]);
    expect(tensorValue(child.nodes.mask_aligned.outputs.out).shape).toEqual([
      2,
      49,
      20,
      ...trailing.map(() => 1),
    ]);
    expect(child.nodes.zeros.outputs.out).toEqual(child.nodes.difference.outputs.out);
  });

  it.each([0, 1, 2])('时间长度 T=%s 时保持与 PyTorch 切片一致的空轴语义', (time) => {
    for (const trailing of [[], [2]]) {
      const feat = tensor([2, 3, time, ...trailing]);
      const result = inferProject(project(feat, tensor([2, 3, time], 'bool')), registry);
      expect(result.nodes.vector.status, result.nodes.vector.messages.join()).toBe('ok');
      expect(tensorValue(result.nodes.result.outputs.out).shape).toEqual([
        2,
        3,
        Math.max(time - 1, 0),
        ...trailing,
      ]);
    }
  });

  it.each(['float16', 'bfloat16', 'float32', 'float64'])('保留浮点特征类型 %s', (dtype) => {
    const result = inferProject(project(tensor([2, 49, 21, 2], dtype)), registry);
    expect(result.nodes.vector.status, result.nodes.vector.messages.join()).toBe('ok');
    expect(tensorValue(result.nodes.result.outputs.out).dtype).toBe(dtype);
  });

  it.each<[TensorSpec, TensorSpec, string]>([
    [tensor([2, 21]), tensor([2, 49, 21], 'bool'), '至少'],
    [tensor([2, 49, 21]), tensor([2, 21], 'bool'), '三个轴'],
    [tensor([2, 49, 21], 'int64'), tensor([2, 49, 21], 'bool'), '浮点'],
    [tensor([2, 49, 21]), tensor([2, 49, 21], 'float32'), 'bool'],
    [tensor([2, 49, 21]), tensor([1, 49, 21], 'bool'), '第 0 轴'],
    [tensor([2, 49, 21]), tensor([2, 48, 21], 'bool'), '第 1 轴'],
    [tensor([2, 49, 21]), tensor([2, 49, 20], 'bool'), '第 2 轴'],
  ])('在进入内部子图前拒绝不符合 helper 约定的输入 %#', (feat, mask, message) => {
    const result = inferProject(project(feat, mask), registry).nodes.vector;
    expect(result.status).toBe('error');
    expect(result.messages.join()).toContain(message);
    expect(result.child).toBeUndefined();
    expect(result.outputs).toEqual({});
  });

  it('符号时间维保留推导表达式，不虚构具体长度', () => {
    const result = inferProject(
      project(tensor(['B', 'A', 'T', 2]), tensor(['B', 'A', 'T'], 'bool')),
      registry,
    ).nodes.vector;
    expect(result.status).toBe('warning');
    const output = tensorValue(result.outputs.out);
    expect(output.shape.slice(0, 2)).toEqual(['B', 'A']);
    expect(String(output.shape[2])).toContain('T');
    expect(output.shape[3]).toBe(2);
    expect(output.dtype).toBe('float32');
  });

  it('保存重开保留完整函数结构，多个实例的内部节点独立', () => {
    const p = project(tensor([2, 49, 21, 2]));
    const restored = parseProject(serializeProject(p), registry);
    expect(restored).toEqual(p);
    expect(inferProject(restored, registry)).toEqual(inferProject(p, registry));
    const first = registry.create('ToVector'),
      second = registry.create('ToVector');
    first.subgraph!.nodes.find((node) => node.id === 'feat_next')!.params.start = '2';
    expect(second.subgraph!.nodes.find((node) => node.id === 'feat_next')!.params.start).toBe('1');
  });

  it('函数面包屑使用调用名称，进入视图不修改内部图', () => {
    const p = project(),
      before = structuredClone(p);
    expect(resolveForwardView(p.graph, ['vector']).current.title).toBe('to_vector()');
    expect(p).toEqual(before);
    p.graph.nodes.find((node) => node.id === 'vector')!.label = 'heading_vector()';
    expect(resolveForwardView(p.graph, ['vector']).current.title).toBe('heading_vector()');
  });
});
