import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { parseProject, serializeProject } from '../src/core/serialization';
import { annotateTensorOutputs } from '../src/core/tensorAppearance';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { matchingPreparationCode } from '../src/modules/matchingPreparation';

const spec = (shape: (number | string)[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
  autograd: { requiresGrad: true, source: 'manual' },
  bufferRef: { reference: 'self.reference', persistent: true },
});
function infer(type: string, values: Record<string, ValueSpec>, params: Params = {}) {
  const node = registry.create(
    type,
    ['Arange', 'OnesScaled'].includes(type)
      ? { device_source: 'tensor_legacy', ...params }
      : params,
  );
  const definition = registry.get(type)!;
  const result = definition.inferValue
    ? definition.inferValue(values, node.params)
    : definition.infer!(values as Record<string, TensorSpec>, node.params);
  return { ...result, outputs: annotateTensorOutputs(node, values, result.outputs) };
}

describe('Matching preparation factories', () => {
  it('creates floating -1 sentinels without inheriting reference dtype, shape, grad or registration', () => {
    const result = infer('OnesScaled', {
      in: spec([2, 7], 'float64'),
      d0: { kind: 'int', value: 1040 },
    });
    expect(result.outputs.out).toEqual({
      shape: [1040],
      axes: ['axis0'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(
      infer(
        'OnesScaled',
        { in: spec([1]) },
        {
          shape: '2,0',
          axes: 'batch,query',
          default_dtype: 'float64',
          value: 2.5,
        },
      ).outputs.out,
    ).toMatchObject({ shape: [2, 0], dtype: 'float64' });
    expect(matchingPreparationCode(registry.create('OnesScaled'))).toContain('不是 torch.full');
  });

  it.each<Params>([
    { value: NaN },
    { value: Infinity },
    { value: '-1' },
    { default_dtype: 'int64' },
    { shape: '-1' },
    { shape: '1', axes: 'a,b' },
  ])('rejects malformed constant factories %j', (params) => {
    expect(() =>
      infer('OnesScaled', { in: spec([1]), d0: { kind: 'int', value: 4 } }, params),
    ).toThrow();
  });

  it.each([1040, 0, 'Q'])(
    'creates arange with end=%s and int64 independently of its reference',
    (end) => {
      const values: Record<string, ValueSpec> = {
        in: spec([3, 4]),
        end: { kind: 'int', value: end },
      };
      expect(infer('Arange', values).outputs.out).toEqual({
        shape: [end],
        axes: ['query'],
        dtype: 'int64',
        autograd: { requiresGrad: false, source: 'inferred' },
      });
    },
  );

  it.each([-1, 0.5, Number.MAX_SAFE_INTEGER + 1, ''])('rejects invalid arange end=%s', (value) => {
    expect(() => infer('Arange', { in: spec([1]), end: { kind: 'int', value } })).toThrow();
  });
  it('requires an integer end and Tensor reference, not a Tensor size or missing dependency', () => {
    expect(() => infer('Arange', { in: spec([1]), end: spec([1]) })).toThrow();
    expect(() => infer('Arange', { end: { kind: 'int', value: 2 } })).toThrow();
    expect(() => infer('OnesScaled', { in: spec([1]) })).toThrow();
  });
});

describe('Sort rows by maximum score', () => {
  it('preserves shape/dtype but not buffer identity, without mutating the input or inventing sort indices', () => {
    const x = spec([1040, 8]),
      before = structuredClone(x);
    x.autograd = { requiresGrad: false, source: 'manual' };
    const r = infer('SortRowsByMax', { in: x });
    expect(r.outputs.out).toEqual({
      shape: [1040, 8],
      axes: ['axis0', 'axis1'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(x).toEqual({ ...before, autograd: { requiresGrad: false, source: 'manual' } });
    expect(r.notes?.join(' ')).toContain('并列顺序不能从规格推断');
    const code = matchingPreparationCode(registry.create('SortRowsByMax'))!;
    expect(code).toContain('x[:, :-1]');
    expect(code).toContain('descending=True');
    expect(code).toContain('out = x[sort_index]');
    expect(infer('SortRowsByMax', { in: before }).outputs.out).not.toHaveProperty('autograd');
    expect(infer('SortRowsByMax', { in: spec([0, 8]) }).outputs.out).toMatchObject({
      shape: [0, 8],
    });
  });
  it('supports no metadata columns and reports symbolic-column preconditions', () => {
    expect(
      infer('SortRowsByMax', { in: spec([4, 1]) }, { trailing_columns: 0 }).outputs.out,
    ).toMatchObject({ shape: [4, 1] });
    expect(infer('SortRowsByMax', { in: spec(['Q', 'C']) }).warnings).toHaveLength(1);
    expect(
      matchingPreparationCode(
        registry.create('SortRowsByMax', { trailing_columns: 0, descending: false }),
      ),
    ).toContain('scores = x\n');
  });
  it.each<Params>([
    { trailing_columns: -1 },
    { trailing_columns: 1.5 },
    { trailing_columns: 8 },
    { descending: 1 },
  ])('rejects invalid sorting settings %j', (params) => {
    expect(() => infer('SortRowsByMax', { in: spec([4, 8]) }, params)).toThrow();
  });
  it('rejects non-matrix input, no scores and unsupported dtype', () => {
    for (const x of [
      spec([4]),
      spec([2, 3, 8]),
      spec([4, 1]),
      spec([4, 0]),
      spec([4, 8], 'complex64'),
    ])
      expect(() => infer('SortRowsByMax', { in: x })).toThrow();
  });
});

describe('Incremental matching submodule via draft API', () => {
  it('can annotate the joined score/id axis without changing dtype, shape or its inputs', () => {
    const a = spec([1040, 7]),
      b = spec([1040, 1], 'int64');
    const before = structuredClone({ a, b });
    expect(
      infer('Concat', { a, b }, { dim: -1, axis_name: 'score_or_id' }).outputs.out,
    ).toMatchObject({
      shape: [1040, 8],
      axes: ['axis0', 'score_or_id'],
      dtype: 'float32',
    });
    expect({ a, b }).toEqual(before);
    expect(infer('Concat', { a, b }, { dim: -1 }).outputs.out).toMatchObject({ axes: a.axes });
    expect(() => infer('Concat', { a, b }, { axis_name: 8 })).toThrow();
  });

  it('routes two branches, promotes appended ids, follows input size and round trips without a fabricated return', () => {
    const built = buildModel(
      {
        name: 'matching preparation',
        nodes: [
          { id: 'counts', type: 'BufferSource', params: { shape: '1040,7', axes: 'query,time' } },
          { id: 'maps', type: 'Cast', params: { dtype: 'float32' } },
          {
            id: 'matched',
            type: 'Submodule',
            subgraph: {
              nodes: [
                { id: 'maps', type: 'Inlet', params: { port: 'maps' } },
                { id: 'size', type: 'TensorSize', params: { axes: '0', names: 'num_points' } },
                { id: 'num_points', type: 'LocalParameter' },
                {
                  id: 'init',
                  type: 'OnesScaled',
                  params: { shape: 'd0', value: -1, device_source: 'tensor_legacy' },
                },
                { id: 'inds', type: 'TensorVariable' },
                { id: 'arange', type: 'Arange', params: { device_source: 'tensor_legacy' } },
                { id: 'column', type: 'Unsqueeze', params: { dim: -1 } },
                { id: 'concat', type: 'Concat', params: { dim: -1 } },
                { id: 'sort', type: 'SortRowsByMax' },
              ],
              connections: [
                ['maps.out', 'size.in'],
                ['size.dim0', 'num_points.in'],
                ['maps.out', 'init.in'],
                ['num_points.out', 'init.d0'],
                ['init.out', 'inds.in'],
                ['maps.out', 'arange.in'],
                ['num_points.out', 'arange.end'],
                ['arange.out', 'column.in'],
                ['maps.out', 'concat.a'],
                ['column.out', 'concat.b'],
                ['concat.out', 'sort.in'],
              ],
            },
          },
        ],
        connections: [
          ['counts.out', 'maps.in'],
          ['maps.out', 'matched.maps'],
        ],
      },
      { mode: 'draft' },
    );
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const trace = (id: string) => built.nodes.find((n) => n.path.join('/') === id)!;
    expect(trace('matched/inds').outputs.out).toMatchObject({
      shape: [1040],
      dtype: 'float32',
      autograd: { requiresGrad: false },
    });
    expect(trace('matched/column').outputs.out).toMatchObject({ shape: [1040, 1], dtype: 'int64' });
    expect(trace('matched/sort').outputs.out).toMatchObject({
      shape: [1040, 8],
      dtype: 'float32',
      autograd: { requiresGrad: false },
    });
    expect(trace('matched').outputs).toEqual({});
    expect(built.diagnostics.every((d) => d.severity === 'warning')).toBe(true);
    const changed = patchModel(
      built.project!,
      {
        operations: [
          { op: 'set_params', id: 'counts', params: { shape: '12,4' } },
          { op: 'duplicate_linked', path: ['matched'], source: 'num_points', id: 'near_arange' },
          { op: 'disconnect', path: ['matched'], from: 'num_points.out', to: 'arange.end' },
          { op: 'connect', path: ['matched'], from: 'near_arange.out', to: 'arange.end' },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
    expect(
      changed.nodes.find((n) => n.path.join('/') === 'matched/sort')!.outputs.out,
    ).toMatchObject({ shape: [12, 5] });
    const reopened = validateModel(parseProject(serializeProject(changed.project!), registry), {
      mode: 'draft',
    });
    expect(reopened.nodes).toEqual(changed.nodes);
  });
});
