import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import { parseDictionarySchema } from '../src/core/dictionary';
import { inferProject } from '../src/core/engine';
import { parseRuntimeSnapshot } from '../src/core/runtimeSnapshot';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { DictionarySpec, TensorSequenceSpec, TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const tensor = (shape: number[]): TensorSpec => ({
  shape, axes: shape.map((_, i) => `axis${i}`), dtype: 'float32',
  autograd: { requiresGrad: true, source: 'manual' },
});
const sequence = (channels = 17, kind: 'list' | 'tuple' = 'list'): TensorSequenceSpec => ({
  kind, items: [1, 4, 16, 24, 32, 48].map(p => tensor([2, 1040, p, channels])),
});
const contract = (): DictionarySpec => ({ kind: 'dict', fields: {
  all_cls_scores: sequence(), all_refine_pts: sequence(3), query_feat: tensor([2, 1040, 256]),
} });

describe('Dictionaries containing Tensor sequences', () => {
  it('packs heterogeneously shaped lists and tuples without stacking or mutating their metadata', () => {
    const fields: DictionarySpec['fields'] = { init_points: tensor([2, 1040, 1, 3]), ...contract().fields, tuple: sequence(3, 'tuple'), none: { kind: 'none' } };
    const before = structuredClone(fields);
    const output = registry.get('DictionaryPack')!.inferValue!(fields, { keys: Object.keys(fields).join(',') }).outputs.out as DictionarySpec;
    expect(output).toEqual({ kind: 'dict', fields });
    expect(fields).toEqual(before);
    expect(output.fields.all_cls_scores).not.toBe(fields.all_cls_scores);
    expect(parseDictionarySchema(output)).toEqual(output);
  });

  it('round-trips empty sequences, scalar tensors and nested dictionaries', () => {
    const schema: DictionarySpec = { kind: 'dict', keyType: 'int', fields: { 0: { kind: 'dict', fields: {
      empty: { kind: 'list', items: [] }, scalar: { kind: 'tuple', items: [tensor([])] },
    } } } };
    expect(parseDictionarySchema(JSON.stringify(schema))).toEqual(schema);
  });

  it.each([
    { kind: 'list' }, { kind: 'list', items: 'unknown' },
    { kind: 'list', items: [], shape: [0] }, { kind: 'list', items: [{ kind: 'none' }] },
    { kind: 'tuple', items: [{ kind: 'list', items: [] }] },
    { kind: 'list', items: [{ shape: [-1], dtype: 'float32' }] },
  ])('rejects malformed sequence descriptions %#', value => {
    expect(() => parseDictionarySchema({ kind: 'dict', fields: { value } })).toThrow();
  });

  it('keeps the global entry and nesting limits for sequence children', () => {
    expect(() => parseDictionarySchema({ kind: 'dict', fields: { x: { kind: 'list', items: Array(256).fill(tensor([])) } } })).toThrow('256');
    let nested: unknown = { kind: 'list', items: [tensor([])] };
    for (let i = 0; i < 8; i++) nested = { kind: 'dict', fields: { x: nested } };
    expect(() => parseDictionarySchema(nested)).toThrow('8');
  });

  it('preserves field-specific provenance and gradients through API serialization, packing and reading', () => {
    const built = buildModel({ name: 'dictionary return', nodes: [
      { id: 'init', type: 'Input', params: { shape: '2,1040,1,3', axes: 'B,Q,P,xyz' } },
      { id: 'predictions', type: 'ForwardInput', params: { schema: contract() } },
      ...['all_cls_scores', 'all_refine_pts', 'query_feat'].map(key => ({
        id: key, type: 'DictionaryGet', params: { path: JSON.stringify([key]) },
      })),
      { id: 'pack', type: 'DictionaryPack', params: { keys: 'init_points,all_cls_scores,all_refine_pts,query_feat' } },
      { id: 'get_scores', type: 'DictionaryGet', params: { path: '["all_cls_scores"]' } },
      { id: 'last', type: 'TensorSequenceGet', params: { index: '-1' } },
      { id: 'out', type: 'Output' },
    ], connections: [
      ['init.out', 'pack.init_points'],
      ...['all_cls_scores', 'all_refine_pts', 'query_feat'].flatMap(key => [
        ['predictions.out', `${key}.in`], [`${key}.out`, `pack.${key}`],
      ] as [string, string][]),
      ['pack.out', 'get_scores.in'], ['get_scores.out', 'last.in'], ['last.out', 'out.in'],
    ] });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    const reopened = parseProject(serializeProject(project), registry);
    expect(reopened).toEqual(project);
    expect(validateModel(reopened).ok).toBe(true);
    const result = inferProject(reopened, registry), packed = result.nodes.pack.outputs.out as DictionarySpec;
    expect(result.nodes.pack.outputs.out.lineage?.sources.map(s => s.index)).toEqual([1, 2]);
    expect((packed.fields.init_points as TensorSpec).lineage?.sources.map(s => s.index)).toEqual([1]);
    const scores = packed.fields.all_cls_scores as TensorSequenceSpec;
    expect(scores.items.map(t => t.shape)).toEqual(sequence().items.map(t => t.shape));
    expect(scores.items.every(t => t.autograd?.requiresGrad && t.lineage?.sources.map(s => s.index).join() === '2')).toBe(true);
    expect(result.nodes.last.outputs.out).toMatchObject({ shape: [2, 1040, 48, 17], autograd: { requiresGrad: true, source: 'manual' } });
    expect(result.nodes.last.outputs.out.lineage?.sources.map(s => s.index)).toEqual([2]);
  });

  it('marks Tensor gradients inside runtime dictionary lists without changing the source', () => {
    const value = contract(), original = structuredClone(value);
    const snapshot = parseRuntimeSnapshot({ format: 'tensor-trail.runtime-snapshot', version: 1,
      id: 'list-run', name: 'list run', capturedAt: '2026-11-12T00:00:00Z', context: {},
      bindings: [{ nodePath: ['input'], nodeType: 'ForwardInput', value }],
    });
    const fields = (snapshot.bindings[0].value as DictionarySpec).fields;
    expect((fields.all_cls_scores as TensorSequenceSpec).items.every(t => t.autograd?.source === 'runtime')).toBe(true);
    expect(value).toEqual(original);
  });
});
