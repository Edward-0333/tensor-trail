import { validateModel } from '../src/agent';
import { describe, it, expect } from 'vitest';
import { registry } from '../src/modules';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { parseDictionarySchema } from '../src/core/dictionary';
import { tensorValue, shapeText } from '../src/core/types';
import type { Dim, TensorSpec, ValueSpec, Params, GraphResult, Project } from '../src/core/types';
import { plutoForwardCode } from '../src/modules/plutoForwardOps';
const t = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const dict = (fields: any): ValueSpec => ({ kind: 'dict', fields });
const data = (r = 3) =>
  dict({
    reference_line: dict({
      position: t([2, r, 120, 2]),
      vector: t([2, r, 120, 2]),
      orientation: t([2, r, 120]),
      valid_mask: t([2, r, 120], 'bool'),
    }),
  });
const memory = () => dict({ enc_emb: t([2, 109, 128]), enc_key_padding_mask: t([2, 109], 'bool') });
const errors = (r: GraphResult): string[] => [
  ...r.issues,
  ...Object.entries(r.nodes).flatMap(([id, v]) => [
    ...(['error', 'blocked'].includes(v.status) ? [`${id}: ${v.messages}`] : []),
    ...(v.child ? errors(v.child) : []),
  ]),
];
function run(type: string, inputs: Record<string, ValueSpec>, p: Params = {}) {
  const node = registry.create(type, p);
  const r = inferGraph(node.subgraph!, registry, inputs);
  return { node, r };
}
function op(type: string, inputs: Record<string, ValueSpec>, p: Params = {}) {
  const node = registry.create(type, p),
    def = registry.get(type)!;
  return def.inferValue
    ? def.inferValue(inputs, node.params)
    : def.infer!(
        Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, tensorValue(v)])),
        node.params,
      );
}

describe('字典组装及控制流值', () => {
  it('字段保留各自类型和尺寸，None 不伪装成零长度 Tensor', () => {
    const r = op('DictionaryPack', {
      trajectory: { kind: 'none' },
      probability: { kind: 'none' },
      prediction: t([2, 48, 80, 6]),
    });
    expect(r.outputs.out).toEqual(
      dict({
        trajectory: { kind: 'none' },
        probability: { kind: 'none' },
        prediction: t([2, 48, 80, 6]),
      }),
    );
    expect(parseDictionarySchema(JSON.stringify(r.outputs.out))).toEqual(r.outputs.out);
    expect(shapeText({ kind: 'none' })).toBe('None');
    expect(shapeText({ kind: 'bool', value: false })).toBe('bool · False');
  });
  it('动态端口字段顺序稳定，拒绝重复和保留名称', () => {
    expect(
      ports(
        registry.create('DictionaryPack', { keys: 'enc_emb,enc_key_padding_mask' }),
        registry,
      ).inputs.map((p) => p.id),
    ).toEqual(['enc_emb', 'enc_key_padding_mask']);
    expect(() => op('DictionaryPack', {}, { keys: 'x,x' })).toThrow();
    expect(() => op('DictionaryPack', {}, { keys: '__proto__' })).toThrow();
    expect(() =>
      op('DictionaryPack', { x: { kind: 'bool', value: true } }, { keys: 'x' }),
    ).toThrow();
    expect(() =>
      parseDictionarySchema('{"kind":"dict","fields":{"x":{"kind":"none","shape":[0]}}}'),
    ).toThrow();
  });
  it('条件来自 Python 整数尺寸，拒绝 Tensor 和未决符号', () => {
    expect(op('IntegerGreater', { in: { kind: 'int', value: 0 } }).outputs.out).toEqual({
      kind: 'bool',
      value: false,
    });
    expect(op('IntegerGreater', { in: { kind: 'int', value: 3 } }).outputs.out).toEqual({
      kind: 'bool',
      value: true,
    });
    expect(() => op('IntegerGreater', { in: t([], 'int64') })).toThrow();
    expect(() => op('IntegerGreater', { in: { kind: 'int', value: 'R' } })).toThrow();
  });
});
describe('场景编码与 agent 预测', () => {
  it('Encoder 保留 pre-norm、两个残差和 mask，输出 [B,L,D]', () => {
    const { node, r } = run(
      'PlutoEncoderLayer',
      { src: t([2, 109, 128]), key_padding_mask: t([2, 109], 'bool') },
      { drop_path: '0.2' },
    );
    expect(errors(r)).toEqual([]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 109, 128]);
    expect(node.subgraph!.nodes.find((n) => n.id === 'attn')!.params.dropout).toBe('0');
    expect(node.subgraph!.edges.some((e) => e.source === 'src' && e.target === 'residual1')).toBe(
      true,
    );
    expect(
      node.subgraph!.edges.some((e) => e.source === 'residual1' && e.target === 'residual2'),
    ).toBe(true);
  });
  it('多头注意力检查 query/key/mask 语义，允许不同 query/key 长度', () => {
    expect(
      tensorValue(
        op('MaskedMultiheadAttention', {
          query: t([2, 36, 128]),
          key: t([2, 109, 128]),
          value: t([2, 109, 128]),
          mask: t([2, 109], 'bool'),
        }).outputs.out,
      ).shape,
    ).toEqual([2, 36, 128]);
    expect(() =>
      op('MaskedMultiheadAttention', {
        query: t([2, 36, 128]),
        key: t([2, 109, 128]),
        value: t([2, 109, 128]),
        mask: t([2, 36], 'bool'),
      }),
    ).toThrow();
    expect(() =>
      op('MaskedMultiheadAttention', {
        query: t([2, 36, 128]),
        key: t([2, 109, 128]),
        value: t([2, 109, 128]),
        mask: t([2, 109]),
      }),
    ).toThrow();
  });
  it.each([0, 1, 48])('AgentPredictor N=%s 三个独立头输出 6 通道', (n) => {
    const { node, r } = run('PlutoAgentPredictor', { in: t([2, n, 128]) });
    expect(errors(r)).toEqual([]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, n, 80, 6]);
    const heads = node.subgraph!.nodes.filter((n) => n.type === 'MapMLP');
    expect(heads).toHaveLength(3);
    expect(heads.every((n) => !n.valueSource)).toBe(true);
  });
});
describe('参考线规划分支', () => {
  it.each([false, true])('PlanningDecoder cat_x=%s 保持模式、时间、状态轴', (cat_x) => {
    const { node, r } = run(
      'PlutoPlanningDecoder',
      { data: data(), enc_data: memory() },
      { cat_x },
    );
    expect(errors(r)).toEqual([]);
    expect(tensorValue(r.outputs.trajectory).shape).toEqual([2, 3, 12, 80, 6]);
    expect(tensorValue(r.outputs.probability).shape).toEqual([2, 3, 12]);
    expect(node.subgraph!.nodes.filter((n) => n.type === 'PlutoDecoderLayer')).toHaveLength(4);
    expect(node.subgraph!.nodes.some((n) => n.id === 'cat_x_proj')).toBe(cat_x);
    expect(node.subgraph!.nodes.some((n) => n.type === 'Softmax')).toBe(false);
    expect(tensorValue(r.nodes.r_feature.outputs.out).shape).toEqual([2, 3, 120, 6]);
  });
  it('当前 R>0 分支调用规划解码器，数据条件接线明确', () => {
    const { r } = run('ReferencePlanningBranch', {
      data: data(),
      enc_data: memory(),
      available: { kind: 'bool', value: true },
    });
    expect(errors(r)).toEqual([]);
    expect(tensorValue(r.outputs.trajectory).shape).toEqual([2, 3, 12, 80, 6]);
  });
  it('R=0 分支不创建/调用 decoder，返回两个 None', () => {
    const { node, r } = run(
      'ReferencePlanningBranch',
      { data: data(0), enc_data: memory(), available: { kind: 'bool', value: false } },
      { available: false },
    );
    expect(errors(r)).toEqual([]);
    expect(node.subgraph!.nodes.some((n) => n.type === 'PlutoPlanningDecoder')).toBe(false);
    expect(r.outputs).toEqual({ trajectory: { kind: 'none' }, probability: { kind: 'none' } });
  });
  it('输入条件变更时拒绝静默使用旧分支', () => {
    const { r } = run('ReferencePlanningBranch', {
      data: data(0),
      enc_data: memory(),
      available: { kind: 'bool', value: false },
    });
    expect(errors(r).join('\n')).toContain('参考线存在');
  });
  it('保留源码 R2R mask.repeat，M2M 保留有效筛选和零回填', () => {
    const { node, r } = run('PlutoDecoderLayer', {
      tgt: t([2, 3, 12, 128]),
      memory: t([2, 109, 128]),
      tgt_key_padding_mask: t([2, 3], 'bool'),
      memory_key_padding_mask: t([2, 109], 'bool'),
      m_pos: t([1, 12, 128]),
    });
    expect(errors(r)).toEqual([]);
    expect(tensorValue(r.outputs.out).shape).toEqual([2, 3, 12, 128]);
    expect(plutoForwardCode(node.subgraph!.nodes.find((n) => n.id === 'r2r')!)).toContain(
      'mask.repeat(M, 1)',
    );
    expect(plutoForwardCode(node.subgraph!.nodes.find((n) => n.id === 'm2m')!)).toContain(
      'tgt[tgt_valid_mask] = tgt_valid',
    );
  });
  it('完整分支图可序列化保存重开', () => {
    const { node } = run('ReferencePlanningBranch', { data: data(), enc_data: memory() });
    const project: Project = {
      schemaVersion: 1,
      name: 'forward',
      variants: [],
      graph: { nodes: [node], edges: [] },
    };
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
    expect(
      validateModel(project).diagnostics.filter((d) => d.code === 'UNKNOWN_PARAMETER'),
    ).toEqual([]);
  });
});
