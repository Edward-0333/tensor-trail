import { tensorValue } from '../src/core/types';
import { describe, it, expect } from 'vitest';
import { registry } from '../src/modules/index';
import { inferGraph } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Dim, Params, TensorSpec, ValueSpec, Project } from '../src/core/types';
import { agentCompletionCode } from '../src/modules/agentCompletion';
const t = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const direct = (type: string, inputs: Record<string, TensorSpec>, params: Params = {}) => {
  const n = registry.create(type, params);
  return registry.get(type)!.infer!(inputs, n.params);
};
const state = (params: Params = {}, batch = 2) => {
  const n = registry.create('StateAttentionEncoder', params);
  return {
    n,
    r: inferGraph(n.subgraph!, registry, { in: t([batch, Number(n.params.state_channel)]) }),
  };
};
const branch = (
  params: Params = {},
  data: ValueSpec = { kind: 'dict', fields: { current_state: t([2, 7]) } },
) => {
  const n = registry.create('EgoHistoryBranch', params);
  return { n, r: inferGraph(n.subgraph!, registry, { data, x_agent: t([2, 49, 128]) }) };
};
const failures = (r: ReturnType<typeof inferGraph>) =>
  Object.values(r.nodes).filter((v) => ['error', 'blocked'].includes(v.status));

describe('AgentEncoder 最后一个分支和返回', () => {
  it('按当前配置替换 ego 并返回完整 [B,A,D]，分支有实际出口', () => {
    const { r } = branch();
    expect(failures(r)).toEqual([]);
    expect(r.outputs.out).toEqual(t([2, 49, 128]));
    expect(tensorValue(r.nodes.ego_state_emb.outputs.out).shape).toEqual([2, 128]);
    expect(tensorValue(r.nodes.state_slice.outputs.out).shape).toEqual([2, 6]);
  });
  it('use_ego_history=True 只直通，不构造或读取自车状态', () => {
    const { n, r } = branch({ use_ego_history: true }, { kind: 'dict', fields: {} });
    expect(n.subgraph!.nodes.map((n) => n.type)).toEqual(['Inlet', 'Inlet', 'Outlet']);
    expect(failures(r)).toEqual([]);
    expect(r.outputs.out).toEqual(t([2, 49, 128]));
  });
  it('状态注意力开关为 False 时选择正确的 BN MLP', () => {
    const { n, r } = branch({ state_attn_encoder: false });
    expect(failures(r)).toEqual([]);
    const mlp = n.subgraph!.nodes.find((n) => n.id === 'ego_state_emb')!;
    expect(mlp.type).toBe('EgoStateMLP');
    expect(mlp.subgraph!.nodes.find((n) => n.id === 'linear0')!.params.bias).toBe(false);
    expect(mlp.subgraph!.nodes.find((n) => n.id === 'linear1')!.params.bias).toBe(true);
  });
  it('Embedding 给类别索引增加 D 轴，不增加类别数轴', () => {
    const r = direct('Embedding', { in: t([2, 49], 'int64') });
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 49, 128]);
    expect(r.outputs.out.dtype).toBe('float32');
    expect(() => direct('Embedding', { in: t([2, 49]) })).toThrow();
  });
  it('整数赋值保持目标形状且不修改输入规格', () => {
    const x = t([2, 49, 128]),
      before = structuredClone(x);
    const r = direct('IndexAssign', { in: x, values: t([2, 128]) });
    expect(r.outputs.out).toEqual(x);
    expect(r.outputs.out).not.toBe(x);
    expect(x).toEqual(before);
  });
  it.each<Record<string, TensorSpec>>([
    { values: t([2, 127]) },
    { values: t([1, 128]) },
    { values: t([2, 128], 'float64') },
    { in: t([2, 0, 128]) },
  ])('拒绝错误替换输入 %j', (v) =>
    expect(() =>
      direct('IndexAssign', { in: t([2, 49, 128]), values: t([2, 128]), ...v }),
    ).toThrow(),
  );
  it('模块可保存重开，并保留各独立 Linear 的参数', () => {
    const { n } = branch();
    const p: Project = {
      name: 'test',
      schemaVersion: 1,
      variants: [],
      graph: { nodes: [n], edges: [] },
    };
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
  });
});

describe('StateAttentionEncoder 内部结构', () => {
  it('六个不同的 Linear 编码六个标量，再 Stack、加位置编码、注意力汇聚', () => {
    const { n, r } = state();
    expect(failures(r)).toEqual([]);
    const linear = n.subgraph!.nodes.filter((n) => n.type === 'StateLinear');
    expect(linear).toHaveLength(6);
    expect(new Set(linear.map((n) => n.id)).size).toBe(6);
    expect(linear.every((n) => !n.valueSource)).toBe(true);
    expect(tensorValue(r.nodes.stack.outputs.out).shape).toEqual([2, 6, 128]);
    expect(tensorValue(r.nodes.pos_repeat.outputs.out).shape).toEqual([2, 6, 128]);
    expect(tensorValue(r.nodes.query_repeat.outputs.out).shape).toEqual([2, 1, 128]);
    expect(tensorValue(r.nodes.attn.outputs.out).shape).toEqual([2, 1, 128]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 128]);
  });
  it.each([true, false])('training=%s 的遮挡语义有明确记录，输出形状不变', (training) => {
    const { r } = state({ training });
    expect(r.nodes.attn.notes.join()).toContain(training ? '前三个' : 'None');
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 128]);
  });
  it('概率 0 不建立遮挡，概率 1 仍保留前三项', () => {
    expect(state({ state_dropout: '0' }).r.nodes.attn.notes.join()).toContain('None');
    expect(state({ state_dropout: '1' }).r.nodes.attn.notes.join()).toContain('前三个');
  });
  it.each<Params>([
    { dim: 130 },
    { state_dropout: '1.1' },
    { state_dropout: '-0.1' },
    { state_dropout: '' },
    { state_channel: 2, training: true },
  ])('拒绝无效参数 %j', (p) => expect(() => state(p)).toThrow());
  it('eval 不执行训练 mask 分支，允许少于三个状态通道', () => {
    expect(tensorValue(state({ state_channel: 2, training: false }).r.outputs.out).shape).toEqual([
      2, 128,
    ]);
  });
  it('参数重复仅采用参考 batch，保留参数自己的 dtype 和尾轴', () => {
    expect(
      direct('RepeatBatch', { in: t([1, 6, 128], 'float64'), like: t([4, 7]) }).outputs.out,
    ).toEqual({ shape: [4, 6, 128], axes: ['batch', 'axis1', 'axis2'], dtype: 'float64' });
    expect(() => direct('RepeatBatch', { in: t([2, 6, 128]), like: t([4, 7]) })).toThrow();
  });
  it('注意力拒绝 token 宽度或 batch 不一致', () => {
    expect(() =>
      direct('StateAttentionReadout', { query: t([2, 1, 128]), tokens: t([2, 6, 64]) }),
    ).toThrow();
    expect(() =>
      direct('StateAttentionReadout', { query: t([3, 1, 128]), tokens: t([2, 6, 128]) }),
    ).toThrow();
  });
  it('源码保留 key_padding_mask=None、前三项可见，以及最终 x_state[:, 0]', () => {
    const source = agentCompletionCode(registry.create('StateAttentionReadout'))!;
    expect(source).toContain('key_padding_mask = None');
    expect(source).toContain('self.state_channel - 3');
    expect(agentCompletionCode(registry.create('StateAttentionEncoder'))).toContain(
      'return x_state[:, 0]',
    );
  });
});
