import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import { annotateTensorOutputs } from '../src/core/tensorAppearance';
import type { Params, TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { reweightCode } from '../src/modules/reweight';
import { sortingCode } from '../src/modules/sorting';

const spec = (shape: (number | string)[], dtype = 'float32', grad = false): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
  autograd: { requiresGrad: grad, source: 'manual' },
  bufferRef: { reference: 'self.maps', persistent: true },
});
function infer(type: string, inputs: Record<string, TensorSpec>, params: Params = {}) {
  const node = registry.create(type, params);
  const result = registry.get(type)!.infer!(inputs, node.params);
  return annotateTensorOutputs(node, inputs, result.outputs) as Record<string, TensorSpec>;
}

describe('Max and sort with indices', () => {
  it('returns both max values and non-differentiable indices, not a single ReduceMax', () => {
    const x = spec([1040, 7], 'float32', true),
      before = structuredClone(x);
    const r = infer('MaxWithIndices', { in: x });
    expect(r.values).toEqual({ shape: [1040], axes: ['axis0'], dtype: 'float32' });
    expect(r.indices).toEqual({
      shape: [1040],
      axes: ['axis0'],
      dtype: 'int64',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(x).toEqual(before);
    expect(
      infer('MaxWithIndices', { in: spec([2, 7, 4]) }, { dim: 1, keepdim: true }).values,
    ).toMatchObject({ shape: [2, 1, 4], autograd: { requiresGrad: false } });
    expect(infer('MaxWithIndices', { in: spec([7]) }).values.shape).toEqual([]);
    expect(infer('MaxWithIndices', { in: spec([0, 7]) }).values.shape).toEqual([0]);
    expect(() => infer('MaxWithIndices', { in: spec([7, 0]) })).toThrow();
  });
  it('sorts along a valid axis, preserves dimensions, and permits an empty sort axis', () => {
    for (const shape of [[1040], [2, 4], [0], ['Q']]) {
      const r = infer('Sort', { in: spec(shape, 'float64', true) }, { descending: true });
      expect(r.values).toEqual({ shape, axes: shape.map((_, i) => `axis${i}`), dtype: 'float64' });
      expect(r.indices).toMatchObject({ shape, dtype: 'int64', autograd: { requiresGrad: false } });
    }
    expect(registry.create('Sort').params.stable).toBe(false);
    expect(sortingCode(registry.create('Sort', { descending: true }))).toContain(
      'descending=True, stable=False',
    );
  });
  it.each(['MaxWithIndices', 'Sort'])('%s rejects invalid rank, dtype, dim and flags', (type) => {
    for (const x of [spec([]), spec([2], 'complex64'), spec([2], 'bool')])
      expect(() => infer(type, { in: x })).toThrow();
    for (const dim of [2, -3, 0.5, '0'])
      expect(() => infer(type, { in: spec([2, 7]) }, { dim })).toThrow();
    expect(() =>
      infer(type, { in: spec([2, 7]) }, type === 'Sort' ? { stable: 1 } : { keepdim: 1 }),
    ).toThrow();
  });
});

describe('Row advanced indexing', () => {
  it('indexes full rows with arbitrary length and does not pretend to validate index values', () => {
    const r = infer('RowIndex', { in: spec([1040, 8]), indices: spec([9], 'int64') });
    expect(r.out).toEqual({
      shape: [9, 8],
      axes: ['axis0', 'axis1'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(
      infer('RowIndex', { in: spec([2, 8], 'float32', true), indices: spec([3], 'int32') }).out,
    ).not.toHaveProperty('autograd');
    expect(infer('RowIndex', { in: spec([0, 8]), indices: spec([0], 'int64') }).out.shape).toEqual([
      0, 8,
    ]);
    expect(
      infer('RowIndex', { in: spec([2, 3, 4]), indices: spec(['K'], 'int64') }).out.shape,
    ).toEqual(['K', 3, 4]);
  });
  it('rejects noninteger and nonvector indices or a provably invalid empty source', () => {
    for (const indices of [spec([2]), spec([2], 'bool'), spec([2, 1], 'int64')])
      expect(() => infer('RowIndex', { in: spec([5, 8]), indices })).toThrow();
    expect(() => infer('RowIndex', { in: spec([]), indices: spec([1], 'int64') })).toThrow();
    expect(() => infer('RowIndex', { in: spec([0, 8]), indices: spec([1], 'int64') })).toThrow();
  });
});

describe('Expandable reweight function', () => {
  const build = (shape = '1040,8', type = 'Reweight') =>
    buildModel({
      name: 'reweight',
      nodes: [
        {
          id: 'maps',
          type: 'BufferSource',
          params: { shape, axes: 'query,score_or_id', dtype: 'float32' },
        },
        { id: 'sort', type, label: 'reweight', position: { x: 400, y: 130 } },
        { id: 'result', type: 'Output' },
      ],
      connections: [
        ['maps.out', 'sort.in'],
        ['sort.out', 'result.in'],
      ],
    }, { mode: 'draft' });

  it('expands actual operations, unused outputs, linked original rows, and a real return', () => {
    const r = build();
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    const node = (id: string) => r.nodes.find((n) => n.path.join('/') === id)!;
    const sorted = node('sort').outputs.out;
    expect(sorted).toEqual({
      shape: [1040, 8],
      axes: ['query', 'score_or_id'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(node('sort/scores').outputs.out).toMatchObject({ shape: [1040, 7] });
    for (const id of ['max_index', 'sort_index'])
      expect(node(`sort/${id}`).outputs.out).toMatchObject({
        shape: [1040],
        dtype: 'int64',
        autograd: { requiresGrad: false },
      });
    expect(node('sort/max_map').outputs.out).toMatchObject({ shape: [1040], dtype: 'float32' });
    expect(node('sort/row_index').inputs.in).toEqual(node('maps').outputs.out);
    expect(node('sort/return_maps').outputs.out).toEqual(sorted);
    const fn = r.project!.graph.nodes.find((n) => n.id === 'sort')!;
    expect(ports(fn, registry).inputs.map((p) => p.id)).toEqual(['in']);
    expect(fn.subgraph!.nodes.find((n) => n.id === 'maps_near_index')!.valueSource).toBe('maps');
    expect(
      fn.subgraph!.edges.filter((e) => ['max_index', 'sorted_values'].includes(e.source)),
    ).toEqual([]);
    expect(resolveForwardView(r.project!.graph, ['sort']).current.title).toBe('reweight()');
    expect(reweightCode(fn)).toContain('maps = maps[sort_index]');
    expect(reweightCode(fn)).toContain('return maps');
  });
  it('responds to input dimensions instead of hardcoding 1040 or 7', () => {
    const r = build('12,5');
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    expect(r.nodes.find((n) => n.path.join('/') === 'sort/scores')!.outputs.out).toMatchObject({
      shape: [12, 4],
    });
    expect(r.nodes.find((n) => n.path.join('/') === 'sort')!.outputs.out).toMatchObject({
      shape: [12, 5],
    });
    const def = registry.get('Reweight')!;
    for (const x of [spec([4]), spec([4, 1]), spec([4, 0]), spec([4, 8], 'bool')])
      expect(() => def.validate!({ in: x }, {})).toThrow();
  });
  it('replaces an opaque node via API with the same external ports, geometry and output', () => {
    const old = build('1040,8', 'SortRowsByMax');
    expect(old.ok).toBe(true);
    const graph = old.project!.graph;
    graph.edges[1].labelPosition = 0.42;
    graph.edges[1].waypoints = [{ x: 710, y: 80 }];
    const changed = patchModel(
      old.project!,
      {
        operations: [
          {
            op: 'replace_node',
            id: 'sort',
            node: { id: 'sort', type: 'Reweight', label: 'reweight' },
          },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
    expect(changed.project!.graph.edges).toEqual(graph.edges);
    expect(changed.project!.graph.nodes.find((n) => n.id === 'sort')!.position).toEqual(
      graph.nodes.find((n) => n.id === 'sort')!.position,
    );
    expect(changed.nodes.find((n) => n.path.join('/') === 'sort')!.outputs).toEqual(
      old.nodes.find((n) => n.path.join('/') === 'sort')!.outputs,
    );
    const parsed = parseProject(serializeProject(changed.project!), registry);
    expect(validateModel(parsed, { mode: 'draft' }).nodes).toEqual(changed.nodes);
  });
});
