import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { attentionMathCode } from '../src/modules/attentionMath';
import { opusCrossAttentionCode } from '../src/modules/opusCrossAttention';

const tensor = (shape: number[], dtype = 'float32'): TensorSpec => ({ shape, axes: shape.map((_, i) => `a${i}`), dtype });
function infer(type: string, values: Record<string, ValueSpec>, params: Params = {}) {
  const n = registry.create(type, params), d = registry.get(type)!;
  return d.inferValue ? d.inferValue(values, n.params) : d.infer!(values as Record<string, TensorSpec>, n.params);
}
function recipe(Q = 1, K = 720, P = 48): Recipe {
  return { name: 'expanded OPUS cross attention', nodes: [
    { id: 'q_points', type: 'Input', params: { shape: `2,${Q},3`, axes: '' } },
    { id: 'q_feat', type: 'Input', params: { shape: `2,${Q},256`, axes: '' } },
    { id: 'k_points', type: 'Input', params: { shape: `2,${K},${P},3`, axes: '' } },
    { id: 'k_feat', type: 'Input', params: { shape: `2,${K},256`, axes: '' } },
    { id: 'attn', type: 'OPUSCrossAttention' },
    { id: 'out', type: 'Output' }, { id: 'weights', type: 'Output' },
  ], connections: [
    ['q_points.out', 'attn.query_points'], ['q_feat.out', 'attn.query_feat'],
    ['k_points.out', 'attn.key_points'], ['k_feat.out', 'attn.key_feat'],
    ['attn.fused_ego_feat', 'out.in'], ['attn.weights', 'weights.in'],
  ] };
}

describe('expanded OPUSCrossAttention', () => {
  it.each([[1, 720, 48], [3, 1040, 4]])('derives Q=%i K=%i P=%i from actual inputs', (Q, K, P) => {
    const built = buildModel(recipe(Q, K, P));
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const result = inferProject(built.project!, registry), body = result.nodes.attn.child!;
    const shapes: Record<string, number[]> = {
      dist_negative: [2, Q, K], distance_weight: [2, Q, K], tau_raw: [2, K, 8],
      tau: [2, 8, K], dist_add_head_axis: [2, 1, Q, K], tau_add_query_axis: [2, 8, 1, K],
      bias_per_head: [2, 8, Q, K], attn_mask: [16, Q, K], query_zero: [2, Q, 3],
      query_pos: [2, Q, 256], key_points_metric: [2, K, P, 3], key_points_mean: [2, K, 3], key_pos: [2, K, 256],
    };
    for (const [id, shape] of Object.entries(shapes)) expect(body.nodes[id].outputs.out, id).toMatchObject({ shape });
    expect(body.nodes.dist_negative.outputs.out).toMatchObject({ autograd: { requiresGrad: false } });
    expect(body.nodes.distance_weight.outputs.out).toMatchObject({ autograd: { requiresGrad: false } });
    expect(body.nodes.tau.outputs.out).not.toHaveProperty('autograd');
    expect(body.outputs.fused_ego_feat).toMatchObject({ shape: [2, Q, 256] });
    expect(body.outputs.weights).toMatchObject({ shape: [2, Q, K] });
    const wrapper = body.nodes.attention.child!;
    expect(wrapper.nodes.query_sequence_first.outputs.out).toMatchObject({ shape: [Q, 2, 256] });
    expect(wrapper.nodes.key_sequence_first.outputs.out).toMatchObject({ shape: [K, 2, 256] });
    expect(wrapper.nodes.value_sequence_first.outputs.out).toMatchObject({ shape: [K, 2, 256] });
    expect(wrapper.nodes.out_sequence_first.outputs.out).toMatchObject({ shape: [Q, 2, 256] });
    expect(parseProject(serializeProject(built.project!), registry)).toEqual(built.project);
  });

  it('uses raw key for V, original query for identity, and both calls of the same position module', () => {
    const p = buildModel(recipe()).project!, graph = p.graph.nodes.find(n => n.id === 'attn')!.subgraph!;
    const wrapper = graph.nodes.find(n => n.id === 'attention')!.subgraph!;
    expect(wrapper.edges.find(e => e.target === 'transpose_value')!.source).toBe('key');
    expect(wrapper.edges.find(e => e.target === 'residual_add' && e.targetPort === 'b')!.source).toBe('query');
    expect(graph.nodes.find(n => n.id === 'query_pos_embed')!.label).toContain('共享同一实例');
    expect(graph.nodes.find(n => n.id === 'key_pos_embed')!.label).toContain('共享同一实例');
    expect(wrapper.nodes.find(n => n.id === 'proj_drop')!.params.p).toBe('0.0');
    expect(wrapper.nodes.find(n => n.id === 'dropout_layer')!.params.p).toBe('0.0');
    expect(wrapper.nodes.find(n => n.id === 'torch_attention')!.params.dropout).toBe('0.1');
    expect(opusCrossAttentionCode(p.graph.nodes.find(n => n.id === 'attn')!)).toContain('use_reentrant=False');
  });

  it('preserves sources through no_grad and native attention without inventing runtime gradient observations', () => {
    const built = buildModel(recipe()).project!;
    const enabled = patchModel(built, { operations: [{ op: 'set_lineage_enabled', enabled: true }] });
    expect(enabled.ok).toBe(true);
    const result = inferProject(enabled.project!, registry).nodes.attn;
    expect((result.outputs.fused_ego_feat as TensorSpec).lineage).toMatchObject({ unknown: false });
    expect((result.outputs.fused_ego_feat as TensorSpec).lineage!.sources.map(s => s.id)).toEqual(['q_points', 'q_feat', 'k_points', 'k_feat']);
    expect((result.child!.nodes.dist_negative.outputs.out as TensorSpec).lineage!.sources.map(s => s.id)).toEqual(['q_points', 'k_points']);
    expect(result.outputs.fused_ego_feat).not.toHaveProperty('autograd');
    expect(validateModel(enabled.project).ok).toBe(true);
  });
});

describe('attention math operators', () => {
  it('keeps Exp and Softplus elementwise while L2Norm reduces only the specified axis', () => {
    const x = tensor([2, 1, 720, 3]);
    for (const type of ['Exp', 'Softplus']) expect(infer(type, { in: x }).outputs.out).toMatchObject({ shape: x.shape, dtype: x.dtype });
    expect(infer('L2Norm', { in: x }).outputs.out).toMatchObject({ shape: [2, 1, 720] });
    expect(infer('L2Norm', { in: x }, { keepdim: true }).outputs.out).toMatchObject({ shape: [2, 1, 720, 1] });
    expect(() => infer('Softplus', { in: x }, { beta: 0 })).toThrow();
    expect(() => infer('L2Norm', { in: x }, { dim: 5 })).toThrow();
    expect(() => infer('Exp', { in: tensor([3], 'int64') })).toThrow();
  });
  it('marks a no_grad calculation result, not a new Parameter or runtime measurement', () => {
    const x = { ...tensor([2, 1, 720]), autograd: { requiresGrad: true, source: 'manual' as const } };
    expect(infer('NoGradResult', { in: x }).outputs.out).toMatchObject({ autograd: { requiresGrad: false, source: 'inferred' } });
    expect(x.autograd.requiresGrad).toBe(true);
    expect(attentionMathCode(registry.create('NoGradResult'))).toContain('不是源码中的额外 detach');
  });
  it.each([false, true])('models native MHA with batch_first=%s', batch_first => {
    const q = tensor(batch_first ? [2, 1, 256] : [1, 2, 256]), kv = tensor(batch_first ? [2, 720, 256] : [720, 2, 256]);
    const values = { query: q, key: kv, value: kv, attn_mask: tensor([16, 1, 720]) };
    const out = infer('TorchMultiheadAttention', values, { batch_first }).outputs;
    expect(out.out).toMatchObject({ shape: q.shape }); expect(out.weights).toMatchObject({ shape: [2, 1, 720] });
    expect(infer('TorchMultiheadAttention', values, { batch_first, average_attn_weights: false }).outputs.weights).toMatchObject({ shape: [2, 8, 1, 720] });
    expect(infer('TorchMultiheadAttention', { ...values, attn_mask: tensor([1, 720], 'bool') }, { batch_first }).notes!.join(' ')).toContain('True');
    expect(() => infer('TorchMultiheadAttention', { ...values, attn_mask: tensor([2, 1, 720]) }, { batch_first })).toThrow(/attn_mask/);
    expect(() => infer('TorchMultiheadAttention', values, { batch_first, dropout: '1.2' })).toThrow(/Dropout/);
    expect(() => infer('TorchMultiheadAttention', values, { batch_first, num_heads: 7 })).toThrow(/整除/);
  });
});
