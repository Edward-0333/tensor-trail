import { expect, it } from 'vitest';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { annotateInformationSources } from '../src/core/informationSources';
const t = (shape: number[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `a${i}`),
  dtype,
});
function run(type: string, inputs: Record<string, ValueSpec>, params: Params = {}) {
  const n = registry.create(type, params),
    d = registry.get(type)!;
  return d.inferValue
    ? d.inferValue(inputs, n.params)
    : d.infer!(inputs as Record<string, TensorSpec>, n.params);
}
it('uses an explicit missing-field declaration without overwriting a partial snapshot', () => {
  const data = { kind: 'dict' as const, fields: { unrelated: t([2, 1, 21]) } },
    original = structuredClone(data);
  const out = run(
    'TensorFieldContract',
    { in: data, d0: { kind: 'int', value: 2 } },
    { shape: 'd0,6,2' },
  ).outputs.out;
  expect(out).toMatchObject({
    shape: [2, 6, 2],
    dtype: 'float32',
    autograd: { requiresGrad: false, source: 'manual' },
  });
  expect(data).toEqual(original);
  const found = {
    kind: 'dict' as const,
    fields: {
      temporal_trajs: {
        ...t([2, 6, 2]),
        autograd: { requiresGrad: true, source: 'runtime' as const },
      },
    },
  };
  expect(run('TensorFieldContract', { in: found }).outputs.out).toEqual(
    found.fields.temporal_trajs,
  );
  expect(() => run('TensorFieldContract', { in: found }, { shape: '2,7,2' })).toThrow(/shape/);
});
it('stacks opaque metadata then explicitly creates a float32 CUDA tensor without guessing NumPy dtype', () => {
  const array = run('BatchMetaStack', {
    img_metas: { kind: 'forward', text: '' },
    d0: { kind: 'int', value: 2 },
  }).outputs.out;
  expect(array).toMatchObject({ kind: 'ndarray', dims: [2, 4, 4], dtype: 'unknown' });
  expect(run('ArrayToTensorExplicit', { in: array }).outputs.out).toMatchObject({
    shape: [2, 4, 4],
    dtype: 'float32',
    autograd: { requiresGrad: false },
  });
});
it('reshape-like reads only target dimensions, not target dtype, gradient or value origins', () => {
  const x = t([2, 37440, 3]),
    like = t([2, 780, 48, 3], 'float64');
  expect(run('ReshapeLike', { in: x, like }).outputs.out).toMatchObject({
    shape: like.shape,
    dtype: 'float32',
  });
  expect(() => run('ReshapeLike', { in: x, like: t([2, 781, 48, 3]) })).toThrow();
  const source = { id: 'points', index: 1, label: 'points' },
    reference = { id: 'shape_only', index: 2, label: 'shape_only' };
  const lineage = (s: typeof source) => ({
    sources: [s],
    origins: [],
    initializations: [],
    unknown: false,
    evidence: 'inferred' as const,
  });
  const inputs = {
    in: { ...x, lineage: lineage(source) },
    like: { ...like, lineage: lineage(reference) },
  };
  const out = annotateInformationSources(
    registry.create('ReshapeLike'),
    inputs,
    run('ReshapeLike', inputs).outputs,
  ).out as TensorSpec;
  expect(out.lineage!.sources).toEqual([source]);
});
it('multiplies a bool motion mask into float offsets without drawing a nonexistent cast', () => {
  expect(
    run('Mul', { a: t([2, 780, 48, 2]), b: t([2, 780, 48, 1], 'bool') }).outputs.out,
  ).toMatchObject({ shape: [2, 780, 48, 2], dtype: 'float32' });
  expect(
    run('Mul', { a: t([2, 780, 48, 1], 'bool'), b: t([2, 780, 48, 2]) }).outputs.out,
  ).toMatchObject({ shape: [2, 780, 48, 2], dtype: 'float32' });
  expect(
    run('ZerosLike', {
      in: { ...t([2, 1, 1]), autograd: { requiresGrad: true, source: 'manual' } },
    }).outputs.out,
  ).toMatchObject({ autograd: { requiresGrad: false } });
});
it('selects per-field provenance during dictionary update and does not merge inactive branch values', () => {
  const marked = (id: string, index: number) => ({
    ...t([2, 1, 2]),
    lineage: {
      sources: [{ id, index, label: id }],
      origins: [],
      initializations: [],
      unknown: false,
      evidence: 'inferred' as const,
    },
  });
  const a = marked('a', 1),
    b = marked('b', 2),
    base = { kind: 'dict' as const, fields: { x: a, y: a } },
    updates = { kind: 'dict' as const, fields: { y: b } };
  const n = registry.create('DictionaryUpdate'),
    out = annotateInformationSources(
      n,
      { base, updates },
      run(n.type, { base, updates }).outputs,
    ).out;
  expect(out).toMatchObject({
    fields: {
      x: { lineage: { sources: a.lineage.sources } },
      y: { lineage: { sources: b.lineage.sources } },
    },
  });
  const condition = { kind: 'bool' as const, value: false },
    selected = run('SelectValue', { condition, true_value: a, false_value: b }).outputs;
  expect(
    annotateInformationSources(
      registry.create('SelectValue'),
      { condition, true_value: a, false_value: b },
      selected,
    ).out,
  ).toMatchObject({ lineage: { sources: b.lineage.sources } });
});
