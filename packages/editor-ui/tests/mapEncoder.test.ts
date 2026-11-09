import { useEditor } from '../src/editor/store';
import { describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules/index';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { Dim, Params, TensorSpec, ValueSpec, Project } from '@tensor-trail/model-core/core/types';
import { tensorValue } from '@tensor-trail/model-core/core/types';
const t = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const int = (value: Dim): ValueSpec => ({ kind: 'int', value });
const infer = (type: string, values: Record<string, ValueSpec>, params: Params = {}) => {
  const n = registry.create(type, params),
    d = registry.get(type)!;
  return d.inferValue
    ? d.inferValue(values, n.params)
    : d.infer!(
        Object.fromEntries(Object.entries(values).map(([k, v]) => [k, tensorValue(v)])),
        n.params,
      );
};
const failures = (r: ReturnType<typeof inferGraph>) =>
  Object.values(r.nodes).filter((v) => ['error', 'blocked'].includes(v.status));
const mapData = (): ValueSpec => ({
  kind: 'dict',
  fields: {
    map: {
      kind: 'dict',
      fields: {
        polygon_center: t([2, 58, 3]),
        point_position: t([2, 58, 3, 20, 2]),
        point_vector: t([2, 58, 3, 20, 2]),
        point_orientation: t([2, 58, 3, 20]),
        valid_mask: t([2, 58, 20], 'bool'),
        polygon_type: t([2, 58], 'int8'),
        polygon_on_route: t([2, 58], 'bool'),
        polygon_tl_status: t([2, 58], 'int8'),
        polygon_has_speed_limit: t([2, 58], 'bool'),
        polygon_speed_limit: t([2, 58]),
      },
    },
  },
});
const composite = (type: string, external: Record<string, ValueSpec>, params: Params = {}) => {
  const node = registry.create(type, params);
  return { node, r: inferGraph(node.subgraph!, registry, external) };
};

describe('多维 bool 筛选与赋值', () => {
  it('掩码消耗匹配的前缀轴，保留未索引尾轴', () => {
    expect(
      infer(
        'BooleanIndex',
        { in: t([2, 58, 128]), mask: t([2, 58], 'bool') },
        { count_symbol: 'N_speed' },
      ).outputs.out,
    ).toEqual({ shape: ['N_speed', 128], axes: ['selected', 'axis2'], dtype: 'float32' });
    expect(
      tensorValue(infer('BooleanIndex', { in: t([2, 58]), mask: t([2, 58], 'bool') }).outputs.out)
        .shape,
    ).toEqual(['N_valid']);
  });
  it('零元素掩码的 True 数量为 0，不留虚假的未知数量', () => {
    expect(
      tensorValue(
        infer('BooleanIndex', { in: t([2, 0, 128]), mask: t([2, 0], 'bool') }).outputs.out,
      ).shape,
    ).toEqual([0, 128]);
  });
  it.each([{ shape: [2, 1] }, { shape: [1, 58] }, { shape: [2, 58, 1] }])(
    '掩码本身不能广播 %j',
    ({ shape }) =>
      expect(() =>
        infer('BooleanIndex', { in: t([2, 58, 128]), mask: t(shape, 'bool') }),
      ).toThrow(),
  );
  it('多维回填保留原目标，不修改输入元数据', () => {
    const x = t([116, 20, 256]),
      before = structuredClone(x);
    const r = infer(
      'BooleanAssign',
      { in: x, mask: t([116, 20], 'bool'), values: t(['N_points', 256]) },
      { count_symbol: 'N_points' },
    );
    expect(r.outputs.out).toEqual(x);
    expect(r.outputs.out).not.toBe(x);
    expect(x).toEqual(before);
    expect(r.warnings).toEqual([]);
  });
  it('明确开启广播后，单行权重可写入任意个未知限速位置', () => {
    const r = infer(
      'BooleanAssign',
      { in: t([2, 58, 128]), mask: t([2, 58], 'bool'), values: t([1, 128]) },
      { count_symbol: 'N_unknown', allow_broadcast: true },
    );
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 58, 128]);
    expect(r.warnings).toEqual([]);
  });
  it('支持空结果上的单行广播，也支持标量广播', () => {
    expect(
      tensorValue(
        infer(
          'BooleanAssign',
          { in: t([0, 128]), mask: t([0], 'bool'), values: t([1, 128]) },
          { allow_broadcast: true },
        ).outputs.out,
      ).shape,
    ).toEqual([0, 128]);
    expect(
      tensorValue(
        infer(
          'BooleanAssign',
          { in: t([2, 3, 8]), mask: t([2, 3], 'bool'), values: t([]) },
          { allow_broadcast: true },
        ).outputs.out,
      ).shape,
    ).toEqual([2, 3, 8]);
  });
  it.each([t([1, 127]), t([2, 128, 1]), t([1, 128], 'float64')])('拒绝不兼容的广播 %j', (values) =>
    expect(() =>
      infer(
        'BooleanAssign',
        { in: t([2, 58, 128]), mask: t([2, 58], 'bool'), values },
        { allow_broadcast: true },
      ),
    ).toThrow(),
  );
  it('旧的未设置广播标志的图仍然可用', () => {
    const node = registry.create('BooleanAssign', {}, 'assign');
    delete node.params.allow_broadcast;
    const graph = {
      nodes: [
        registry.create('Inlet', { port: 'x' }, 'x'),
        registry.create('Inlet', { port: 'mask' }, 'mask'),
        registry.create('Inlet', { port: 'v' }, 'v'),
        node,
      ],
      edges: [
        { id: 'a', source: 'x', sourcePort: 'out', target: 'assign', targetPort: 'in' },
        { id: 'b', source: 'mask', sourcePort: 'out', target: 'assign', targetPort: 'mask' },
        { id: 'c', source: 'v', sourcePort: 'out', target: 'assign', targetPort: 'values' },
      ],
    };
    expect(
      inferGraph(graph, registry, {
        x: t([98, 128]),
        mask: t([98], 'bool'),
        v: t(['N_valid', 128]),
      }).nodes.assign.status,
    ).toBe('ok');
  });
});

describe('归约、重复和动态 Reshape', () => {
  it('Max 只输出 values，支持 keepdim，空轴不能 Max', () => {
    expect(
      tensorValue(infer('ReduceMax', { in: t([116, 20, 256]) }, { dim: 1 }).outputs.out).shape,
    ).toEqual([116, 256]);
    expect(
      tensorValue(infer('ReduceMax', { in: t([3, 4]) }, { dim: -1, keepdim: true }).outputs.out)
        .shape,
    ).toEqual([3, 1]);
    expect(() => infer('ReduceMax', { in: t([3, 0, 8]) }, { dim: 1 })).toThrow();
  });
  it('Sum 空轴仍有输出；整数和 bool 提升到 int64', () => {
    expect(infer('ReduceSum', { in: t([0, 2, 128]) }, { dim: 0 }).outputs.out).toEqual({
      shape: [2, 128],
      axes: ['axis1', 'axis2'],
      dtype: 'float32',
    });
    expect(
      tensorValue(infer('ReduceSum', { in: t([2, 3], 'bool') }, { dim: 1 }).outputs.out).dtype,
    ).toBe('int64');
  });
  it('repeat(1,n,1) 支持动态次数、新前导轴和零次数', () => {
    expect(
      tensorValue(infer('Repeat', { in: t([116, 1, 256]), d0: int(20) }).outputs.out).shape,
    ).toEqual([116, 20, 256]);
    expect(
      tensorValue(infer('Repeat', { in: t([2, 3]) }, { repeats: '4,1,2' }).outputs.out).shape,
    ).toEqual([4, 2, 6]);
    expect(
      tensorValue(infer('Repeat', { in: t([2, 3]) }, { repeats: '0,1' }).outputs.out).shape,
    ).toEqual([0, 3]);
    expect(() => infer('Repeat', { in: t([2, 3]) }, { repeats: '2' })).toThrow();
    expect(() => infer('Repeat', { in: t([2, 3]) }, { repeats: '1,-1' })).toThrow();
  });
  it('Reshape 接收局部参数，检查元素总数，保留旧的纯数值格式', () => {
    expect(
      tensorValue(
        infer(
          'Reshape',
          { in: t([2, 58, 20, 10]), d0: int(116), d1: int(20), d2: int(10) },
          { shape: 'd0,d1,d2' },
        ).outputs.out,
      ).shape,
    ).toEqual([116, 20, 10]);
    expect(
      ports(registry.create('Reshape', { shape: 'd0,d1,d2' }), registry).inputs.map((p) => p.id),
    ).toEqual(['in', 'd0', 'd1', 'd2']);
    expect(() => infer('Reshape', { in: t([2, 3]), d0: int(5) }, { shape: 'd0,2' })).toThrow();
  });
});

describe('地图复合模块', () => {
  it.each([true, false])('lane_boundary=%s，特征通道与真实构造一致', (boundary) => {
    const { node, r } = composite(
      'MapEncoder',
      { data: mapData() },
      { use_lane_boundary: boundary },
    );
    expect(failures(r)).toEqual([]);
    expect(tensorValue(r.nodes.polygon_feature.outputs.out).shape).toEqual([
      2,
      58,
      20,
      boundary ? 10 : 6,
    ]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 58, 128]);
    expect(node.subgraph!.nodes.some((n) => n.id === 'left_offset')).toBe(boundary);
  });
  it('PointsEncoder 保留两次先填零再 max、原地回填及独立 MLP', () => {
    const { node, r } = composite('PointsEncoder', {
      in: t([116, 20, 10]),
      mask: t([116, 20], 'bool'),
    });
    expect(failures(r)).toEqual([]);
    expect(tensorValue(r.outputs.out).shape).toEqual([116, 128]);
    expect(tensorValue(r.nodes.cat.outputs.out).shape).toEqual([116, 20, 512]);
    expect(node.subgraph!.edges.some((e) => e.source === 'fill2' && e.target === 'pool2')).toBe(
      true,
    );
    expect(
      node
        .subgraph!.nodes.find((n) => n.id === 'first_mlp')!
        .subgraph!.nodes.find((n) => n.id === 'linear0')!.params.bias,
    ).toBe(true);
  });
  it.each([0, 4])('Fourier 支持有效限速数 %s，空集合不伪造行', (count) => {
    const { r } = composite('FourierEmbedding', { in: t([count, 1]) });
    expect(failures(r)).toEqual([]);
    expect(tensorValue(r.nodes.cat.outputs.out).shape).toEqual([count, 1, 129]);
    expect(tensorValue(r.outputs.out).shape).toEqual([count, 128]);
  });
  it('Fourier 三维输入为三套独立 MLP，Stack 后 Sum', () => {
    const { node, r } = composite('FourierEmbedding', { in: t([2, 7, 3]) }, { input_dim: 3 });
    expect(failures(r)).toEqual([]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 7, 128]);
    expect(tensorValue(r.nodes.stack.outputs.out).shape).toEqual([3, 2, 7, 128]);
    expect(
      node.subgraph!.nodes.filter((n) => /^mlp\d$/.test(n.id)).every((n) => !n.valueSource),
    ).toBe(true);
  });
  it('未知限速参数为同一行权重的广播赋值', () => {
    const { node, r } = composite('MapEncoder', { data: mapData() });
    expect(
      node.subgraph!.nodes.find((n) => n.id === 'speed_unknown_fill')!.params.allow_broadcast,
    ).toBe(true);
    expect(tensorValue(r.nodes.unknown_speed_weight.outputs.out).shape).toEqual([1, 128]);
    expect(r.nodes.speed_unknown_fill.status).toBe('ok');
  });
  it('带动态端口、多维掩码的完整地图子图可保存重开', () => {
    const { node } = composite('MapEncoder', { data: mapData() });
    const project: Project = {
      schemaVersion: 1,
      name: 'map',
      variants: [],
      graph: { nodes: [node], edges: [] },
    };
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
  });
  it('BatchNorm 接受空集合但拒绝单个训练样本', () => {
    expect(tensorValue(infer('StateBatchNorm', { in: t([0, 128]) }).outputs.out).shape).toEqual([
      0, 128,
    ]);
    expect(() => infer('StateBatchNorm', { in: t([1, 128]) })).toThrow();
  });
});

describe('新操作的动态端口编辑', () => {
  it.each(['Reshape', 'Repeat', 'Zeros'])('%s 不允许带连线删除尺寸输入，断开后可以修改', (type) => {
    const key = type === 'Repeat' ? 'repeats' : 'shape';
    const node = registry.create(type, { [key]: type === 'Reshape' ? 'd0,d1,-1' : 'd0,d1' }, 'op');
    const project: Project = {
      schemaVersion: 1,
      name: 'edit',
      variants: [],
      graph: {
        nodes: [
          registry.create('Input', { shape: '6,4', axes: 'batch,feature' }, 'x'),
          registry.create('IntegerParameter', { expression: '2' }, 's0'),
          registry.create('IntegerParameter', { expression: '3' }, 's1'),
          node,
        ],
        edges: [
          {
            id: 'x',
            source: 'x',
            sourcePort: 'out',
            target: 'op',
            targetPort: type === 'Zeros' ? 'device_like' : 'in',
          },
          { id: 'd0', source: 's0', sourcePort: 'out', target: 'op', targetPort: 'd0' },
          { id: 'd1', source: 's1', sourcePort: 'out', target: 'op', targetPort: 'd1' },
        ],
      },
    };
    useEditor.setState({ project, path: [], past: [], future: [], hasProject: true, notice: '' });
    const changed = type === 'Reshape' ? 'd0,-1' : 'd0,3';
    expect(useEditor.getState().setParam('op', key, changed)).toBe(false);
    expect(useEditor.getState().project).toBe(project);
    expect(useEditor.getState().notice).toContain('先断开输入 d1');
    useEditor.getState().removeEdges(['d1']);
    expect(useEditor.getState().setParam('op', key, changed)).toBe(true);
    const saved = parseProject(serializeProject(useEditor.getState().project), registry);
    expect(inferGraph(saved.graph, registry).nodes.op.status).toBe('ok');
  });
});
