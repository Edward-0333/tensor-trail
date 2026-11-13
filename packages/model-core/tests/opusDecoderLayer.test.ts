import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph, inferProject, resultAt } from '../src/core/engine';
import { annotateInformationSources } from '../src/core/informationSources';
import { annotateTensorOutputs } from '../src/core/tensorAppearance';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSequenceSpec, TensorSpec, ValueSpec } from '../src/core/types';
import { isTensorSequenceSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { decoderCounts, decoderLoopRecipe } from './fixtures/decoderLoop';
import { decoderLayerOpCode } from '../src/modules/decoderLayerOps';

const t = (shape: number[]): TensorSpec => ({ shape, axes: shape.map((_, i) => `d${i}`), dtype: 'float32' });
function source(v: ValueSpec, index: number): ValueSpec {
  if (isTensorSequenceSpec(v)) v = { ...v, items: v.items.map(item => source(item, index) as TensorSpec) };
  return { ...v, lineage: { sources: [{ id: `source${index}`, index, label: `input${index}` }],
    origins: [], initializations: [], unknown: false, evidence: 'root' } };
}
const inputs = (previous = 1, layout = 'channel_last'): Record<string, ValueSpec> => ({
  query_points: source({ ...t([2, 1040, previous, 3]), autograd: { requiresGrad: false, source: 'inferred' } }, 3),
  query_feat: { ...t([2, 1040, 256]), autograd: { requiresGrad: false, source: 'inferred' },
    lineage: { sources: [], origins: ['constant'], initializations: ['zeros'], unknown: false, evidence: 'inferred' } },
  mlvl_feats: source({ kind: 'list', items: [64, 32, 16, 8].map((h, i) =>
    t(layout === 'channel_last' ? [40, 6, h, 176 / 2 ** i, 64] : [40, 64, 6, h, 176 / 2 ** i])) }, 1),
  occ2img: source({ ...t([2, 30, 4, 4]), autograd: { requiresGrad: false, source: 'inferred' } }, 2),
  img_metas: source({ kind: 'forward', text: 'list[dict]' }, 2),
  index: { kind: 'int', value: 0 },
});
function run(type: string, values: Record<string, ValueSpec>, params: Params = {}) {
  const n = registry.create(type, params), d = registry.get(type)!;
  const raw = d.inferValue ? d.inferValue(values, n.params) : d.infer!(values as Record<string, TensorSpec>, n.params);
  return annotateInformationSources(n, values, annotateTensorOutputs(n, values, raw.outputs));
}
function expandedProject() {
  const built = buildModel(decoderLoopRecipe());
  expect(built.ok).toBe(true);
  const changed = patchModel(built.project!, { operations: [{ op: 'replace_node', path: ['loop'], id: 'layer',
    node: { id: 'layer', type: 'OPUSDecoderLayer' } }] }, { layout: 'preserve' });
  expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
  return changed.project!;
}

describe('Expanded OPUS decoder forward', () => {
  it('replaces the whole-layer contract with real dataflow for all six iterations', () => {
    const p = expandedProject(), r = inferProject(p, registry);
    expect(r.nodes.loop.status).toBe('ok');
    decoderCounts.forEach((P, index) => {
      const layer = resultAt(r, ['loop', 'layer'], { '["loop"]': index })!;
      const previous = index ? decoderCounts[index - 1] : 1;
      expect(layer.nodes.flatten.outputs.out).toMatchObject({ shape: [2, 1040, previous * 3] });
      expect(layer.nodes.position_dims.outputs.out).toMatchObject({ kind: 'int', value: previous * 3 });
      expect(layer.nodes.P.outputs.out).toMatchObject({ kind: 'int', value: P });
      for (const name of ['query_pos', 'positioned_feat', 'mixed_feat', 'norm1', 'norm2', 'norm3', 'ffn', 'updated_feat'])
        expect(layer.nodes[name].outputs.out).toMatchObject({ shape: [2, 1040, 256] });
      expect(layer.nodes.sampled_feat.outputs.out).toMatchObject({ shape: [2, 1040, 4, 20, 64] });
      expect(layer.nodes.cls_raw.outputs.out).toMatchObject({ shape: [2, 1040, P * 17] });
      expect(layer.nodes.reg_offset.outputs.out).toMatchObject({ shape: [2, 1040, P * 3] });
      expect(layer.outputs.cls_score).toMatchObject({ shape: [2, 1040, P, 17] });
      expect(layer.outputs.refine_pts).toMatchObject({ shape: [2, 1040, P, 3] });
      const refine = layer.nodes.refine_points.child!;
      expect(refine.nodes.center.outputs.out).toMatchObject({ shape: [2, 1040, 1, 3] });
      expect(refine.nodes.view_delta.outputs.out).toMatchObject({ shape: [2, 1040, P, 3] });
      expect(refine.nodes.add_delta.outputs.out).toMatchObject({ shape: [2, 1040, P, 3] });
      expect(layer.nodes.position_encoder.child!.nodes.linear0.inputs.in_features).toMatchObject({ value: previous * 3 });
      expect(layer.nodes.cls_branch.child!.nodes.linear2.inputs.out_features).toMatchObject({ value: P * 17 });
      expect(layer.nodes.reg_branch.child!.nodes.linear2.inputs.out_features).toMatchObject({ value: P * 3 });
    });
    expect((r.nodes.loop.outputs.refine_pts as TensorSequenceSpec).items.map(x => x.shape[2])).toEqual(decoderCounts);
    expect(validateModel(parseProject(serializeProject(p), registry)).ok).toBe(true);
  });

  it.each(['channel_last', 'channel_first'])('marks first-layer image fusion at sampling, not position encoding (%s)', layout => {
    const n = registry.create('OPUSDecoderLayer', { layout });
    const v = inputs(1, layout), before = structuredClone(v);
    const r = inferGraph(n.subgraph!, registry, v, 1, {}, undefined, [], { remaining: 20000 }, undefined, true);
    expect(r.issues).toEqual([]);
    const ids = (id: string) => r.nodes[id].outputs.out.lineage?.sources.map(s => s.index);
    expect(ids('query_pos')).toEqual([3]);
    expect(ids('positioned_feat')).toEqual([3]);
    for (const id of ['sampled_feat', 'mixed_feat', 'updated_feat', 'reg_offset']) expect(ids(id)).toEqual([1, 2, 3]);
    expect(r.nodes.flatten.outputs.out).toMatchObject({ autograd: { requiresGrad: false } });
    expect(r.nodes.query_pos.outputs.out).not.toHaveProperty('autograd');
    expect(r.outputs.refine_pts).not.toHaveProperty('autograd');
    expect(r.outputs.refine_pts.lineage?.unknown).toBe(false);
    expect(v).toEqual(before);
  });

  it('fails if iteration configuration does not match the incoming point count', () => {
    const n = registry.create('OPUSDecoderLayer'), v = { ...inputs(), index: { kind: 'int' as const, value: 2 } };
    const r = inferGraph(n.subgraph!, registry, v);
    expect(r.nodes.position_encoder.status).toBe('error');
    expect(r.outputs).not.toHaveProperty('query_feat');
  });

  it('keeps positional/classification/regression MLPs different and records residual FFN semantics', () => {
    const g = registry.create('OPUSDecoderLayer').subgraph!;
    const body = (id: string) => g.nodes.find(n => n.id === id)!.subgraph!;
    const count = (id: string, type: string) => body(id).nodes.filter(n => n.type === type).length;
    expect(count('position_encoder', 'LayerNorm')).toBe(2);
    expect(count('cls_branch', 'LayerNorm')).toBe(2);
    expect(count('reg_branch', 'LayerNorm')).toBe(0);
    expect(count('reg_branch', 'ReLU')).toBe(2);
    expect(g.nodes.find(n => n.id === 'scale_offset')!.params.expression).toBe('0.5 * x');
    expect(decoderLayerOpCode(g.nodes.find(n => n.id === 'ffn')!)).toContain('add_identity=True');
    expect(g.nodes.some(n => ['OPUSDecoderLayerContract', 'OPUSDecoderLayerAt'].includes(n.type))).toBe(false);
  });

  it('checks sampling/mixing/attention shape constraints rather than just declaring outputs', () => {
    expect(() => run('OPUSSamplingContract', inputs(), { num_groups: 3 })).toThrow(/整除/);
    expect(() => run('OPUSSamplingContract', inputs(), { num_frames: 4 })).toThrow();
    expect(() => run('OPUSSamplingContract', { ...inputs(), mlvl_feats: { kind: 'list', items: [] } })).toThrow();
    expect(() => run('AdaptiveMixingContract', { sampled_feat: t([2, 1040, 4, 16, 64]), query_feat: t([2, 1040, 256]) })).toThrow();
    expect(() => run('OPUSSelfAttentionContract', inputs(), { num_heads: 3 })).toThrow(/整除/);
    expect(() => run('MMCVFFNContract', { in: t([2, 1040, 128]) })).toThrow();
  });

  it('Linear dimensions are configuration, not data provenance', () => {
    const out = run('LinearByDimensions', { in: source(t([2, 3, 12]), 3),
      in_features: source({ kind: 'int', value: 12 }, 98), out_features: source({ kind: 'int', value: 256 }, 99) }).out;
    expect(out).toMatchObject({ shape: [2, 3, 256] });
    expect(out.lineage?.sources.map(s => s.index)).toEqual([3]);
    expect(() => run('LinearByDimensions', { in: t([2, 3, 12]), in_features: t([12]), out_features: { kind: 'int', value: 256 } })).toThrow();
  });

  it('refine transforms preserve shape, source IDs and no-grad when appropriate', () => {
    const x = source({ ...t([2, 1040, 1, 3]), autograd: { requiresGrad: false, source: 'inferred' } }, 3);
    const out = run('PointRangeTransform', { in: x }).out;
    expect(out).toMatchObject({ shape: [2, 1040, 1, 3], autograd: { requiresGrad: false } });
    expect(out.lineage?.sources.map(s => s.index)).toEqual([3]);
    expect(() => run('PointRangeTransform', { in: x }, { pc_range: '0,0,0,0,1,1' })).toThrow();
    expect(run('ReduceMean', { in: t([2, 1040, 32, 3]) }, { dim: 2, keepdim: true }).out).toMatchObject({ shape: [2, 1040, 1, 3] });
    expect(run('ReduceMean', { in: t([2, 1040, 32, 3]) }, { dim: -2, keepdim: false }).out).toMatchObject({ shape: [2, 1040, 3] });
  });
});
