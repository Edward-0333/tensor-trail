import { describe, expect, it } from 'vitest';
import type { Graph, Params, Project, TensorSpec } from '../src/core/types';
import { inferGraph } from '../src/core/engine';
import { graphIssues } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';

const spec = (
  shape: (number | string)[],
  axes = shape.map((_, i) => `axis${i}`),
  dtype = 'float32',
): TensorSpec => ({ shape, axes, dtype });
function rule(type: string, inputs: Record<string, TensorSpec>, params: Params = {}) {
  const node = registry.create(type, params);
  return registry.get(type)!.infer!(inputs, node.params);
}
function single(type: string, params: Params = {}, inputParams: Params = {}): Graph {
  return {
    nodes: [
      registry.create('Input', inputParams, 'input'),
      registry.create(type, params, 'subject'),
    ],
    edges: [{ id: 'e', source: 'input', sourcePort: 'out', target: 'subject', targetPort: 'in' }],
  };
}
const check = (graph: Graph) => inferGraph(graph, registry);

describe('形状与轴语义', () => {
  it('Linear 仅变换最后一维', () =>
    expect(
      rule('Linear', { in: spec([2, 10, 64]) }, { out_features: 23 }).outputs.out.shape,
    ).toEqual([2, 10, 23]));
  it('Permute 同时更新轴名称', () =>
    expect(
      rule('Permute', { in: spec([8, 100, 64], ['batch', 'sequence', 'feature']) }).outputs.out,
    ).toEqual(spec([8, 64, 100], ['batch', 'feature', 'sequence'])));
  it('拒绝重复或缺失的 Permute 轴', () =>
    expect(() => rule('Permute', { in: spec([2, 3, 4]) }, { dims: '0, 0, 2' })).toThrow(
      '一次排列',
    ));
  it('Conv1d 正确处理步长、填充与膨胀', () =>
    expect(
      rule(
        'Conv1d',
        { in: spec([2, 4, 17]) },
        {
          in_channels: 4,
          out_channels: 8,
          kernel_size: 3,
          stride: 2,
          padding: 2,
          dilation: 2,
          groups: 2,
        },
      ).outputs.out.shape,
    ).toEqual([2, 8, 9]));
  it('Conv2d 使用两个空间轴', () =>
    expect(
      rule(
        'Conv2d',
        { in: spec([2, 3, 32, 40]) },
        { out_channels: 16, kernel_size: 5, padding: 0, stride: 2 },
      ).outputs.out.shape,
    ).toEqual([2, 16, 14, 18]));
  it('卷积不能产生非正尺寸', () =>
    expect(() => rule('Conv1d', { in: spec([2, 64, 2]) }, { kernel_size: 9, padding: 0 })).toThrow(
      '输出尺寸',
    ));
  it('卷积分组约束不会静默忽略', () =>
    expect(() => rule('Conv1d', { in: spec([2, 64, 10]) }, { groups: 3 })).toThrow('整除'));
  it('Reshape -1 依据总元素数推导', () =>
    expect(rule('Reshape', { in: spec([2, 3, 4]) }, { shape: '2, -1' }).outputs.out.shape).toEqual([
      2, 12,
    ]));
  it('Reshape 拒绝元素数不守恒和多个 -1', () => {
    expect(() => rule('Reshape', { in: spec([2, 3, 4]) }, { shape: '2, 13' })).toThrow('总数');
    expect(() => rule('Reshape', { in: spec([2, 3, 4]) }, { shape: '-1, -1' })).toThrow('最多');
  });
  it('Flatten 保留合并区间之外的轴', () =>
    expect(
      rule('Flatten', { in: spec([2, 3, 4, 5]) }, { start_dim: 1, end_dim: 2 }).outputs.out.shape,
    ).toEqual([2, 12, 5]));
  it('MatMul 正确广播批维并收缩矩阵维', () =>
    expect(
      rule('MatMul', { a: spec([2, 3, 5, 7]), b: spec([1, 3, 7, 11]) }).outputs.out.shape,
    ).toEqual([2, 3, 5, 11]));
  it('MatMul 拒绝不一致的收缩维', () =>
    expect(() => rule('MatMul', { a: spec([2, 5, 7]), b: spec([2, 8, 9]) })).toThrow('收缩维'));
  it('Add 支持单例广播并拒绝不相容尺寸', () => {
    expect(rule('Add', { a: spec([2, 3, 4]), b: spec([1, 3, 1]) }).outputs.out.shape).toEqual([
      2, 3, 4,
    ]);
    expect(() => rule('Add', { a: spec([2, 3, 4]), b: spec([2, 5, 4]) })).toThrow('对应维度');
  });
  it('Concat 只累加指定轴', () =>
    expect(rule('Concat', { a: spec([2, 3, 4]), b: spec([2, 3, 7]) }).outputs.out.shape).toEqual([
      2, 3, 11,
    ]));
  it('LSTM 双向输出与最终状态分离', () => {
    const result = rule(
      'LSTM',
      { in: spec([2, 5, 4]) },
      { input_size: 4, hidden_size: 7, num_layers: 3, bidirectional: true },
    );
    expect(result.outputs.out.shape).toEqual([2, 5, 14]);
    expect(result.outputs.hidden.shape).toEqual([6, 2, 7]);
    expect(result.outputs.cell.shape).toEqual([6, 2, 7]);
  });
  it('GRU 非 batch-first 时 hidden 仍以层数开头', () => {
    const result = rule(
      'GRU',
      { in: spec([5, 2, 4]) },
      { input_size: 4, hidden_size: 7, batch_first: false },
    );
    expect(result.outputs.hidden.shape).toEqual([1, 2, 7]);
    expect(result.outputs.out.shape).toEqual([5, 2, 7]);
  });
  it('符号尺寸保留且不可证明的约束标记警告', () => {
    const result = rule('Linear', { in: spec(['B', 'T', 'D']) });
    expect(result.outputs.out.shape).toEqual(['B', 'T', 64]);
    expect(result.warnings?.length).toBe(1);
    expect(rule('Reshape', { in: spec(['B', 3, 4]) }, { shape: '2, -1' }).warnings?.length).toBe(1);
  });
  it('拒绝整数 Tensor 进入浮点 Linear', () =>
    expect(() => rule('Linear', { in: spec([2, 64], undefined, 'int64') })).toThrow('浮点'));
});

describe('子图、多端口与错误传播', () => {
  it('标准 Transformer 的内外输出一致，并可观测分头与分数', () => {
    const result = check(example().graph),
      block = result.nodes.transformer;
    expect(block.status).toBe('ok');
    expect(block.outputs.out.shape).toEqual([8, 100, 64]);
    const attention = block.child!.nodes.attention;
    expect(attention.child!.nodes.q_heads.outputs.out.shape).toEqual([8, 8, 100, 8]);
    expect(attention.child!.nodes.scores.outputs.out.shape).toEqual([8, 8, 100, 100]);
    expect(attention.outputs.out).toEqual(attention.child!.outputs.out);
  });
  it.each(['Pre-Norm', 'Post-Norm'])('%s + RMSNorm + SwiGLU + causal 可完整推导', (order) => {
    const graph = single('Transformer', { order, norm: 'RMSNorm', ffn: 'SwiGLU', causal: true });
    const result = check(graph).nodes.subject;
    expect(result.status).toBe('ok');
    expect(result.outputs.out.shape).toEqual([8, 100, 64]);
    expect(result.child!.nodes.ffn.child!.nodes.gated.outputs.out.shape).toEqual([8, 100, 128]);
  });
  it('不整除头数时报错并阻断下游', () => {
    const project = example();
    const node = project.graph.nodes.find((n) => n.id === 'transformer')!;
    node.params.heads = 7;
    node.subgraph = registry.get('Transformer')!.build!(node.params);
    const result = check(project.graph);
    expect(result.nodes.transformer.status).toBe('error');
    expect(result.nodes.transformer.messages[0]).toContain('整除');
    expect(result.nodes.output.status).toBe('blocked');
    expect(result.nodes.output.outputs).toEqual({});
  });
  it('混合网络得到预期的卷积与 LSTM 输出', () => {
    const result = check(example('mixed').graph);
    expect(result.nodes.cnn.outputs.out.shape).toEqual([8, 128, 100]);
    expect(result.nodes.lstm.outputs.out.shape).toEqual([8, 100, 32]);
    expect(result.nodes.lstm.outputs.hidden.shape).toEqual([1, 8, 32]);
  });
  it('从 LSTM hidden 端口连接时不能误用序列输出', () => {
    const graph = single('LSTM');
    graph.nodes.push(registry.create('Output', {}, 'final'));
    graph.edges.push({
      id: 'hidden_edge',
      source: 'subject',
      sourcePort: 'hidden',
      target: 'final',
      targetPort: 'in',
    });
    expect(check(graph).nodes.final.inputs.in.shape).toEqual([1, 8, 32]);
  });
  it('不同变体实例的子图相互独立', () => {
    const original = registry.create('Transformer'),
      first = registry.fromVariant('甲', original.subgraph!),
      second = registry.fromVariant('乙', original.subgraph!);
    first.subgraph!.nodes.find((n) => n.id === 'norm1')!.type = 'RMSNorm';
    expect(second.subgraph!.nodes.find((n) => n.id === 'norm1')!.type).toBe('LayerNorm');
    expect(original.subgraph!.nodes.find((n) => n.id === 'norm1')!.type).toBe('LayerNorm');
  });
  it('自定义子图采用内部算子规则而非父层旧参数', () => {
    const graph = single('Transformer');
    const node = graph.nodes[1];
    node.customized = true;
    node.params.heads = 7;
    node.subgraph!.nodes.find((n) => n.id === 'norm1')!.type = 'RMSNorm';
    expect(check(graph).nodes.subject.status).toBe('ok');
    node.subgraph!.nodes.find((n) => n.id === 'norm1')!.params.features = 32;
    expect(check(graph).nodes.subject.status).toBe('error');
  });
  it('拒绝重复输入、错误端口和环路', () => {
    const graph = single('Linear');
    graph.edges.push({ ...graph.edges[0], id: 'other' });
    expect(graphIssues(graph, registry).join()).toContain('已有输入');
    graph.edges = [{ ...graph.edges[0], sourcePort: 'missing' }];
    expect(graphIssues(graph, registry).join()).toContain('没有输出端口');
    graph.nodes = [registry.create('Linear', {}, 'a'), registry.create('Linear', {}, 'b')];
    graph.edges = [
      { id: 'a_b', source: 'a', sourcePort: 'out', target: 'b', targetPort: 'in' },
      { id: 'b_a', source: 'b', sourcePort: 'out', target: 'a', targetPort: 'in' },
    ];
    expect(check(graph).issues.join()).toContain('环路');
  });
});

describe('项目保存协议', () => {
  it('JSON 往返完整保留子图和用户变体', () => {
    const project = example();
    project.variants.push({
      id: 'my_variant',
      name: '自己的模型',
      graph: registry.create('Transformer', { norm: 'RMSNorm' }).subgraph!,
    });
    const loaded = parseProject(serializeProject(project), registry);
    expect(loaded).toEqual(project);
    expect(check(loaded.graph).nodes.transformer.outputs.out.shape).toEqual([8, 100, 64]);
  });
  it('拒绝未知版本与未知模块', () => {
    expect(() =>
      parseProject(JSON.stringify({ ...example(), schemaVersion: 2 }), registry),
    ).toThrow('版本');
    const project = example();
    project.graph.nodes[0].type = 'Unregistered';
    expect(() => parseProject(serializeProject(project), registry)).toThrow('尚未注册');
  });
  it('拒绝重复节点与破坏对象键的标识', () => {
    const project = example();
    project.graph.nodes[1].id = 'input';
    expect(() => parseProject(serializeProject(project), registry)).toThrow('重复');
    project.graph.nodes[1].id = '__proto__';
    expect(() => parseProject(serializeProject(project), registry)).toThrow('ID');
  });
  it('缺失子图边界时明确拒绝', () => {
    const project: Project = example();
    project.graph.nodes[1].subgraph!.nodes = project.graph.nodes[1].subgraph!.nodes.filter(
      (n) => n.type !== 'Outlet',
    );
    expect(() => parseProject(serializeProject(project), registry)).toThrow('Outlet');
  });
  it('导入正常项目中的无效模型参数，仍能在界面诊断', () => {
    const project = example();
    project.graph.nodes[1].params.heads = 0;
    const loaded = parseProject(serializeProject(project), registry);
    expect(check(loaded.graph).nodes.transformer.status).toBe('error');
  });
});
