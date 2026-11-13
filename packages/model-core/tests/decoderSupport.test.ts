import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { annotateInformationSources } from '../src/core/informationSources';
import { annotateTensorOutputs } from '../src/core/tensorAppearance';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { decoderSupportCode } from '../src/modules/decoderSupport';

const tensor = (shape: (number | string)[], dtype = 'float32'): TensorSpec => ({
  shape, dtype, axes: shape.map((_, i) => `d${i}`),
});
const metas: ValueSpec = { kind: 'forward', text: 'list[dict], actual contents not captured' };
function run(type: string, inputs: Record<string, ValueSpec>, params: Params = {}) {
  const node = registry.create(type, params);
  const def = registry.get(type)!;
  const raw = def.inferValue ? def.inferValue(inputs, node.params)
    : def.infer!(inputs as Record<string, TensorSpec>, node.params);
  return annotateInformationSources(node, inputs,
    annotateTensorOutputs(node, inputs, raw.outputs));
}
const source = (index: number, value: ValueSpec): ValueSpec => ({ ...value,
  lineage: { sources: [{ id: `source${index}`, index, label: `source${index}` }],
    origins: [], initializations: [], unknown: false, evidence: 'root' },
});
const numbers = (v: ValueSpec) => v.lineage?.sources.map(s => s.index);
function layerInputs(layout = 'channel_last', batch: number | string = 2, previous = 1): Record<string, ValueSpec> {
  return {
    query_points: source(3, tensor([batch, 1040, previous, 3])),
    query_feat: tensor([batch, 1040, 256]),
    mlvl_feats: source(1, { kind: 'list', items: [64, 32, 16, 8].map((h, i) =>
      tensor(layout === 'channel_last'
        ? [batch === 2 ? 40 : batch === 1 ? 20 : '(B*5*4)', 6, h, 176 / 2 ** i, 64]
        : [batch === 2 ? 40 : 20, 64, 6, h, 176 / 2 ** i])) }),
    occ2img: source(2, tensor([batch, 30, 4, 4])), img_metas: source(2, metas),
  };
}

describe('Decoder drawing operators', () => {
  it('detach clears gradient and parameter role but retains value source IDs, without mutation', () => {
    const x = source(3, { ...tensor([2, 1040, 1, 3]),
      autograd: { requiresGrad: true, source: 'runtime' },
      parameterRef: { module: 'self.points', name: 'weight' } });
    const before = structuredClone(x), out = run('Detach', { in: x }).out;
    expect(out).toMatchObject({ shape: [2, 1040, 1, 3],
      autograd: { requiresGrad: false, source: 'inferred' } });
    expect(out).not.toHaveProperty('parameterRef');
    expect(numbers(out)).toEqual([3]);
    expect(x).toEqual(before);
  });
  it('expand_as uses target shape only, not target dtype, gradients or provenance', () => {
    const out = run('ExpandAs', {
      in: source(2, { ...tensor([2, 1, 4, 4]), autograd: { requiresGrad: false, source: 'manual' } }),
      like: source(3, tensor([2, 30, 4, 4], 'float64')),
    }).out;
    expect(out).toMatchObject({ shape: [2, 30, 4, 4], dtype: 'float32',
      autograd: { requiresGrad: false } });
    expect(numbers(out)).toEqual([2]);
  });
  it.each([{ shape: [2, 2, 4, 4] }, { shape: [2, 1, 4, 4, 1] }])('rejects invalid expansion %j', ({ shape }) => {
    expect(() => run('ExpandAs', { in: tensor(shape), like: tensor([2, 30, 4, 4]) })).toThrow(/expand_as/);
  });
  it('expands a scalar or adds leading dimensions', () => {
    expect(run('ExpandAs', { in: tensor([]), like: tensor([2, 30]) }).out).toMatchObject({ shape: [2, 30] });
    expect(run('ExpandAs', { in: tensor([4, 4]), like: tensor([2, 30, 4, 4]) }).out)
      .toMatchObject({ shape: [2, 30, 4, 4] });
  });
  it('declares a metadata tensor contract with separate device and shape dependencies', () => {
    const out = run('BatchMetaTensor', {
      img_metas: source(2, metas), device: source(3, { kind: 'device', reference: 'query_points.device' }),
      d0: source(1, { kind: 'int', value: 2 }), d1: { kind: 'int', value: 30 },
    }).out;
    expect(out).toMatchObject({ shape: [2, 30, 4, 4], dtype: 'float32',
      autograd: { requiresGrad: false } });
    expect(numbers(out)).toEqual([2]);
    expect(decoderSupportCode(registry.create('BatchMetaTensor'))).toContain('显式契约');
  });
  it('does not accept a Tensor as a device or fabricate data from a non-list input', () => {
    const v = { img_metas: metas, device: { kind: 'device', reference: 'q.device' } as ValueSpec,
      d0: { kind: 'int', value: 2 } as ValueSpec, d1: { kind: 'int', value: 30 } as ValueSpec };
    expect(() => run('BatchMetaTensor', { ...v, device: tensor([2]) })).toThrow('GetDevice');
    expect(() => run('BatchMetaTensor', { ...v, img_metas: tensor([2]) })).toThrow();
  });
  it.each(['channel_first', 'channel_last'])('supports per-stage output counts with %s', layout => {
    let v = layerInputs(layout);
    const counts = [1, 4, 16, 24, 32, 48];
    counts.forEach((P, i) => {
      const out = run('OPUSDecoderLayerContract', v, { layout,
        num_refines: P, last_refines: i ? counts[i - 1] : 1 });
      expect(out.query_feat).toMatchObject({ shape: [2, 1040, 256] });
      expect(out.cls_score).toMatchObject({ shape: [2, 1040, P, 17] });
      expect(out.refine_pts).toMatchObject({ shape: [2, 1040, P, 3] });
      Object.values(out).forEach(value => {
        expect(numbers(value)).toEqual([1, 2, 3]);
        expect(value).not.toHaveProperty('autograd');
      });
      v = { ...v, query_feat: out.query_feat, query_points: run('Detach', { in: out.refine_pts }).out };
    });
  });
  it('derives batch from inputs rather than fixing it at two', () => {
    expect(run('OPUSDecoderLayerContract', layerInputs('channel_last', 1)).cls_score)
      .toMatchObject({ shape: [1, 1040, 1, 17] });
  });
  it('uses a layer index as configuration, not a source of fused features', () => {
    const v = layerInputs();
    const fixed = run('OPUSDecoderLayerContract', v);
    const selected = run('OPUSDecoderLayerAt', { ...v, index: source(99, { kind: 'int', value: 0 }) });
    expect(selected).toEqual(fixed);
  });
  it.each<Params>([{ num_groups: 3 }, { last_refines: 4 }, { num_levels: 3 },
    { embed_dims: 128 }, { num_refines: 0 }, { num_refines: 1.5 }, { layout: 'unknown' }])(
    'rejects incompatible layer contracts %j', params => {
      expect(() => run('OPUSDecoderLayerContract', layerInputs(), params)).toThrow();
    });
  it('rejects inconsistent camera and channel layouts', () => {
    expect(() => run('OPUSDecoderLayerContract', layerInputs(), { layout: 'channel_first' })).toThrow();
    expect(() => run('OPUSDecoderLayerContract', { ...layerInputs(), occ2img: tensor([2, 6, 4, 4]) })).toThrow();
  });
  it('supports draft/preserve API and save/reopen for generic operators', () => {
    const built = buildModel({ name: 'detach', nodes: [{ id: 'x', type: 'Input',
      params: { shape: '2,1,4,4', axes: '' }, position: { x: 99, y: 88 } }], connections: [] }, { mode: 'draft' });
    const patched = patchModel(built.project!, { operations: [
      { op: 'add_node', node: { id: 'detach', type: 'Detach' } },
      { op: 'connect', from: 'x.out', to: 'detach.in' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(patched.ok).toBe(true);
    expect(patched.project!.graph.nodes[0].position).toEqual({ x: 99, y: 88 });
    expect(validateModel(parseProject(serializeProject(patched.project!), registry), { mode: 'draft' }).ok).toBe(true);
    expect(patched.nodes.find(n => n.path[0] === 'detach')!.outputs.out)
      .toMatchObject({ autograd: { requiresGrad: false } });
  });
});
