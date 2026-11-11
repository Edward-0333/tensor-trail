import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph, inferProject, resultAt } from '../src/core/engine';
import { checkForEachBody, inferForEach, loopIteration } from '../src/core/forEach';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { TensorSequenceSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { loopRecipe, reshapedFeatures } from './fixtures/forEach';

const build = () => {
  const r = buildModel(loopRecipe());
  expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
  return r.project!;
};

describe('ForEach Tensor shape inference', () => {
  it('maps different feature shapes with a shared B and ordered collection, without mutating inputs', () => {
    const p = build(),
      before = serializeProject(p),
      r = inferProject(p, registry),
      loop = r.nodes.loop;
    expect(loop.status).toBe('ok');
    expect((loop.outputs.out as TensorSequenceSpec).items.map((t) => t.shape)).toEqual(
      reshapedFeatures,
    );
    expect(loop.iterations).toHaveLength(4);
    expect(loop.outputs.out).not.toHaveProperty('autograd');
    expect(
      (loop.outputs.out as TensorSequenceSpec).items.every(
        (t) => t.dtype === 'float32' && t.autograd === undefined,
      ),
    ).toBe(true);
    loop.iterations!.forEach((child, i) => {
      expect(child.nodes.index.outputs.out).toEqual({ kind: 'int', value: i });
      expect(child.nodes.divide.outputs.out).toEqual({ kind: 'int', value: 30 });
      expect(child.nodes.B.outputs.out).toEqual({ kind: 'int', value: 2 });
    });
    expect(serializeProject(p)).toBe(before);
    expect(
      ports(p.graph.nodes.find((n) => n.id === 'loop')!, registry).inputs.map((p) => p.id),
    ).toEqual(['items', 'initial', 'B']);
    expect(validateModel(parseProject(before, registry)).ok).toBe(true);
  });

  it('switches iteration results without recomputation or changing the full output, and clamps stale selections', () => {
    const result = inferProject(build(), registry),
      before = structuredClone(result);
    for (let i = 0; i < 4; i++)
      expect(resultAt(result, ['loop'], { '["loop"]': i })!.outputs.out).toMatchObject({
        shape: reshapedFeatures[i],
      });
    expect(resultAt(result, ['loop'], { '["loop"]': 99 })).toBe(result.nodes.loop.iterations![3]);
    expect(loopIteration(result.nodes.loop, -1)).toBe(0);
    expect(loopIteration(result.nodes.loop, NaN)).toBe(0);
    expect(result).toEqual(before);
  });

  it('keeps the selected iteration when navigating further into a nested body module', () => {
    const p = build();
    const changed = patchModel(p, {
      operations: [
        {
          op: 'add_node',
          path: ['loop'],
          node: {
            id: 'nested',
            type: 'Submodule',
            subgraph: {
              nodes: [
                { id: 'input', type: 'Inlet', params: { port: 'in' } },
                { id: 'output', type: 'Outlet', params: { port: 'out' } },
              ],
              connections: [['input.out', 'output.in']],
            },
          },
        },
        { op: 'disconnect', path: ['loop'], from: 'view.out', to: 'collect.in' },
        { op: 'connect', path: ['loop'], from: 'view.out', to: 'nested.in' },
        { op: 'connect', path: ['loop'], from: 'nested.out', to: 'collect.in' },
      ],
    });
    expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
    const result = inferProject(changed.project!, registry);
    expect(resultAt(result, ['loop', 'nested'], { '["loop"]': 3 })!.outputs.out).toMatchObject({
      shape: reshapedFeatures[3],
    });
    expect(resultAt(result, ['loop', 'nested'])!.outputs.out).toMatchObject({
      shape: reshapedFeatures[0],
    });
  });

  it('validates EVERY iteration, not only the selected/first feature, and rejects bad shapes even in draft mode', () => {
    const p = build(),
      before = serializeProject(p);
    const result = patchModel(
      p,
      { operations: [{ op: 'set_params', id: 'input3', params: { shape: '61,256,8,22' } }] },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.severity === 'error' && d.message.includes('迭代 3')),
    ).toBe(true);
    expect(serializeProject(p)).toBe(before);
  });

  it('supports incremental body editing and recomputes batch dimensions through API', () => {
    const p = build();
    const pending = patchModel(
      p,
      { operations: [{ op: 'disconnect', path: ['loop'], from: 'size.dim3', to: 'view.d4' }] },
      { mode: 'draft' },
    );
    expect(pending.ok).toBe(true);
    expect(pending.nodes.find((n) => n.path.join('/') === 'loop')!.status).toBe('incomplete');
    const changed = patchModel(
      p,
      { operations: [{ op: 'set_params', id: 'B', params: { expression: '3' } }] },
      { layout: 'preserve' },
    );
    expect(changed.ok).toBe(true);
    const trace = changed.nodes.find((n) => n.path.join('/') === 'loop')!;
    expect(trace.iterations).toHaveLength(4);
    expect((trace.outputs.out as TensorSequenceSpec).items[0].shape).toEqual([3, 20, 256, 64, 176]);
    expect(changed.project!.graph.nodes.find((n) => n.id === 'loop')!.position).toEqual({
      x: 800,
      y: 100,
    });
  });

  it('skips empty sequences and returns the initial list; appends rather than replaces a nonempty prefix', () => {
    const body = registry.create('ForEachTensor').subgraph!;
    const item = {
      shape: [2, 3],
      axes: ['B', 'C'],
      dtype: 'float32',
      autograd: { requiresGrad: true, source: 'runtime' as const },
    };
    const initial: TensorSequenceSpec = { kind: 'list', items: [item] };
    let calls = 0;
    const run = (values: Record<string, ValueSpec>) => {
      calls++;
      return inferGraph(body, registry, values);
    };
    const empty = inferForEach(body, { items: { kind: 'tuple', items: [] }, initial }, run);
    expect(calls).toBe(0);
    expect(empty.iterations).toEqual([]);
    expect(empty.child).toBeUndefined();
    expect(empty.outputs.out).toEqual(initial);
    const r = inferForEach(body, { items: { kind: 'list', items: [item] }, initial }, run);
    expect(calls).toBe(1);
    expect(r.outputs.out).toEqual({ kind: 'list', items: [item, item] });
    expect(initial.items).toHaveLength(1);
  });

  it('rejects opaque inputs, non-list accumulators and non-Tensor body outputs', () => {
    const body = registry.create('ForEachTensor').subgraph!;
    const empty: TensorSequenceSpec = { kind: 'list', items: [] };
    const run = (values: Record<string, ValueSpec>) => inferGraph(body, registry, values);
    expect(() =>
      inferForEach(body, { items: { kind: 'list', text: 'unknown' }, initial: empty }, run),
    ).toThrow('items');
    expect(() =>
      inferForEach(body, { items: empty, initial: { kind: 'tuple', items: [] } }, run),
    ).toThrow('initial');
    const p = build();
    const bad = patchModel(
      p,
      {
        operations: [
          { op: 'disconnect', path: ['loop'], from: 'view.out', to: 'collect.in' },
          { op: 'connect', path: ['loop'], from: 'B.out', to: 'collect.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(bad.ok).toBe(false);
    expect(bad.diagnostics.some((d) => d.message.includes('必须为一个 Tensor'))).toBe(true);
  });

  it('checks reserved body boundaries and imposes static inference limits', () => {
    const body = registry.create('ForEachTensor').subgraph!;
    const bad = structuredClone(body);
    bad.nodes[0].params.port = 'items';
    expect(() => checkForEachBody(bad)).toThrow();
    const badOut = structuredClone(body);
    badOut.nodes[1].params.port = 'features';
    expect(() => checkForEachBody(badOut)).toThrow('out');
    const item = { shape: [1], axes: ['C'], dtype: 'float32' };
    const run = (values: Record<string, ValueSpec>) => inferGraph(body, registry, values);
    expect(() =>
      inferForEach(
        body,
        {
          items: { kind: 'tuple', items: Array(17).fill(item) },
          initial: { kind: 'list', items: [] },
        },
        run,
      ),
    ).toThrow('16');
    expect(() =>
      inferForEach(
        body,
        {
          items: { kind: 'tuple', items: [item] },
          initial: { kind: 'list', items: Array(256).fill(item) },
        },
        run,
      ),
    ).toThrow('256');
    const p = build();
    const limited = inferGraph(p.graph, registry, {}, 0, {}, undefined, [], {
      remaining: p.graph.nodes.length + 1,
    });
    expect(limited.nodes.loop.status).toBe('error');
    expect(limited.nodes.loop.messages.join()).toContain('20000');
  });
});

describe('Python-style int(a / b) for dimensions', () => {
  const divide = (a: number | string, b: number | string) =>
    registry.get('IntegerTruncDivide')!.inferValue!(
      { a: { kind: 'int', value: a }, b: { kind: 'int', value: b } },
      {},
    );
  it.each([
    [60, 2, 30],
    [61, 2, 30],
    [-5, 2, -2],
    [5, -2, -2],
    [0, 2, 0],
  ])('truncates %s / %s to %s', (a, b, value) =>
    expect(divide(a, b).outputs.out).toEqual({ kind: 'int', value }),
  );
  it('rejects zero denominator and retains symbolic uncertainty', () => {
    expect(() => divide(1, 0)).toThrow();
    expect(divide('BN', 'B').outputs.out).toEqual({ kind: 'int', value: 'int((BN)/(B))' });
    expect(divide('BN', 'B').warnings).not.toHaveLength(0);
  });
});
