import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import { inferWhile } from '../src/core/whileLoop';
import { isTensorSpec, type Params, type TensorSpec, type ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { matchingLoopBody } from './fixtures/matchingLoop';

const tensor = (shape: (number | string)[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
  autograd: { requiresGrad: false, source: 'manual' },
});
const integer = (value: number | string): ValueSpec => ({ kind: 'int', value });
const infer = (type: string, v: Record<string, ValueSpec>, params: Params = {}) => {
  const n = registry.create(type, params),
    d = registry.get(type)!;
  return d.inferValue
    ? d.inferValue(v, n.params)
    : d.infer!(v as Record<string, TensorSpec>, n.params);
};
function buildLoop() {
  return buildModel(
    {
      name: 'assignment',
      nodes: [
        {
          id: 'maps',
          type: 'BufferSource',
          params: { shape: '1040,8', axes: 'query,score_or_id', dtype: 'float32' },
        },
        {
          id: 'inds',
          type: 'BufferSource',
          params: { shape: '1040', axes: 'query', dtype: 'float32' },
        },
        { id: 'i', type: 'IntegerParameter', params: { expression: '0' } },
        { id: 'limit', type: 'IntegerParameter', params: { expression: '1040' } },
        { id: 'quotas', type: 'IntegerList' },
        {
          id: 'loop',
          type: 'WhileLoop',
          params: { states: 'maps,inds' },
          subgraph: matchingLoopBody(),
        },
        { id: 'result', type: 'Output' },
      ],
      connections: [
        ['maps.out', 'loop.maps'],
        ['inds.out', 'loop.inds'],
        ['i.out', 'loop.i'],
        ['limit.out', 'loop.num_points'],
        ['quotas.out', 'loop.num_inds'],
        ['loop.inds', 'result.in'],
      ],
    },
    { mode: 'draft' },
  );
}
describe('Stateful bounded while template', () => {
  it('infers one symbolic body, int scalars, conditional states and the real final inds contract', () => {
    const r = buildLoop();
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    const trace = (id: string) => r.nodes.find((n) => n.path.join('/') === id)!;
    expect(trace('loop').outputs.inds).toMatchObject({
      shape: [1040],
      dtype: 'float32',
      autograd: { requiresGrad: false },
    });
    expect(trace('loop').outputs.maps).toMatchObject({ shape: [1040, 8] });
    expect(trace('loop/i').outputs.out).toEqual(integer('i'));
    expect(trace('loop/select_query/scores').outputs.out).toMatchObject({ shape: [7] });
    expect(trace('loop/select_query/argmax').outputs.out).toMatchObject({
      shape: [],
      dtype: 'int64',
    });
    expect(trace('loop/ind').outputs.out).toEqual(integer('ind'));
    expect(trace('loop/original_id').outputs.out).toEqual(integer('original_id'));
    expect(trace('loop/assign').inputs.index).toEqual(integer('original_id'));
    expect(trace('loop/count_assigned/equal').outputs.out).toMatchObject({
      shape: [1040],
      dtype: 'bool',
    });
    expect(trace('loop/count_assigned/sum').outputs.out).toMatchObject({
      shape: [],
      dtype: 'int64',
    });
    expect(trace('loop/assigned_count').outputs.out).toEqual(integer('assigned_count'));
    expect(trace('loop/quota_full').outputs.true).toEqual({
      kind: 'control',
      expression: 'assigned_count == num_inds[ind]',
      branch: true,
    });
    expect(trace('loop/quota_full').outputs.false).toMatchObject({
      kind: 'control',
      branch: false,
    });
    expect(trace('loop/update_tail/tail').outputs.out).toMatchObject({
      shape: ['max(1040-(i + 1),0)', 8],
    });
    expect(trace('loop/update_tail/reweight').outputs.out).toMatchObject({
      shape: ['max(1040-(i + 1),0)', 8],
    });
    expect(trace('loop/update_tail/write_tail').outputs.out).toMatchObject({ shape: [1040, 8] });
    expect(trace('loop/false_maps').outputs.out).toEqual(trace('maps').outputs.out);
    expect(trace('loop/next_i').outputs.out).toEqual(integer('(i + 1)'));
    const node = r.project!.graph.nodes.find((n) => n.id === 'loop')!;
    expect(ports(node, registry).outputs.map((p) => p.id)).toEqual(['maps', 'inds']);
    expect(resolveForwardView(r.project!.graph, ['loop']).current.title).toContain('一轮符号模板');
    const engine = inferGraph(r.project!.graph, registry);
    expect(engine.nodes.loop.iterations).toBeUndefined();
    expect(engine.nodes.loop.notes.join(' ')).toContain('不生成分配结果');
    expect(Object.keys(engine.nodes.loop.child!.nodes).length).toBeLessThan(40);
    const reopened = validateModel(parseProject(serializeProject(r.project!), registry), {
      mode: 'draft',
    });
    expect(reopened.nodes).toEqual(r.nodes);
  });
  it('honors zero-iteration semantics without evaluating an invalid/empty argmax body', () => {
    const node = registry.create('WhileLoop');
    let visited = 0;
    const r = inferWhile(
      node,
      { i: integer(0), num_points: integer(0), maps: tensor([0, 8]), inds: tensor([0]) },
      () => {
        visited++;
        throw Error('must not run');
      },
    );
    expect(visited).toBe(0);
    expect(r.child).toBeUndefined();
    expect(r.outputs.maps).toEqual(tensor([0, 8]));
  });
  it('rejects state shape changes, missing feedback and a stalled counter', () => {
    const built = buildLoop();
    expect(built.ok).toBe(true);
    const stalled = patchModel(
      built.project!,
      {
        operations: [
          { op: 'disconnect', path: ['loop'], from: 'start.out', to: 'next_i.in' },
          { op: 'connect', path: ['loop'], from: 'i.out', to: 'next_i.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(stalled.ok).toBe(false);
    expect(JSON.stringify(stalled.diagnostics)).toContain('不能停滞');
    const resized = patchModel(
      built.project!,
      {
        operations: [
          { op: 'disconnect', path: ['loop'], from: 'assigned_inds.out', to: 'next_inds.in' },
          { op: 'connect', path: ['loop'], from: 'maps.out', to: 'next_inds.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(resized.ok).toBe(false);
    expect(JSON.stringify(resized.diagnostics)).toContain('shape/dtype');
    expect(
      patchModel(
        built.project!,
        { operations: [{ op: 'remove_node', path: ['loop'], id: 'next_inds' }] },
        { mode: 'draft' },
      ).ok,
    ).toBe(false);
  });
});

describe('Loop operations are honest static specs', () => {
  it('keeps Python lists, scalar Tensors and symbolic Python ints distinct', () => {
    expect(infer('IntegerList', {}).outputs.out).toEqual({
      kind: 'int-list',
      values: [720, 60, 60, 60, 60, 40, 40],
    });
    expect(
      infer('IntegerListGet', { in: { kind: 'int-list', values: [3, 2] }, index: integer(1) })
        .outputs.out,
    ).toEqual(integer(2));
    expect(() => infer('IntegerListGet', { in: tensor([7]), index: integer(0) })).toThrow();
    expect(() =>
      infer('IntegerListGet', { in: { kind: 'int-list', values: [3, 2] }, index: integer(2) }),
    ).toThrow();
    expect(() => infer('TensorItemInt', { in: tensor([]) })).toThrow();
    expect(
      infer('TensorItemInt', { in: tensor([]) }, { cast: true, symbol: 'original_id' }).outputs.out,
    ).toEqual(integer('original_id'));
    expect(() => infer('TensorItemInt', { in: tensor([1], 'int64') })).toThrow();
    expect(() => infer('Argmax', { in: tensor([0]) })).toThrow();
    expect(
      infer('TensorRowAt', { in: tensor([3, 8]), index: integer('i') }).outputs.out,
    ).toMatchObject({ shape: [8] });
    expect(() => infer('TensorRowAt', { in: tensor([3, 8]), index: integer(3) })).toThrow();
    expect(
      infer('ScalarIndexWrite', {
        in: tensor([3]),
        index: integer('original_id'),
        value: integer('ind'),
      }).outputs.out,
    ).toMatchObject({ shape: [3], dtype: 'float32' });
  });
  it('permits the empty last suffix and rejects invalid suffix writes', () => {
    const x = tensor([3, 8]),
      before = structuredClone(x);
    expect(infer('SliceRowsFrom', { in: x, start: integer(3) }).outputs.out).toMatchObject({
      shape: [0, 8],
    });
    expect(
      infer('ZeroTailColumn', { in: x, start: integer(3), column: integer(0) }).outputs.out,
    ).toMatchObject({ shape: [3, 8] });
    expect(
      infer('WriteTailRows', { in: x, start: integer(3), rows: tensor([0, 8]) }).outputs.out,
    ).toMatchObject({ shape: [3, 8] });
    expect(x).toEqual(before);
    expect(() => infer('SliceRowsFrom', { in: x, start: integer(-1) })).toThrow();
    expect(() =>
      infer('WriteTailRows', { in: x, start: integer(2), rows: tensor([2, 8]) }),
    ).toThrow();
    expect(() =>
      infer('ZeroTailColumn', { in: x, start: integer(1), column: integer(8) }),
    ).toThrow();
  });
  it('does not turn symbolic conditions into booleans or silently accept incompatible branches', () => {
    const r = infer('IfCondition', { count: integer('count'), quota: integer(60) });
    expect(r.outputs.out).toEqual({ kind: 'control', expression: 'count == 60' });
    expect(() =>
      infer('BranchValue', { in: tensor([2]), control: r.outputs.false }, { branch: true }),
    ).toThrow();
    expect(() =>
      infer('MergeBranches', { condition: r.outputs.out, true: tensor([2]), false: tensor([3]) }),
    ).toThrow();
    const value = infer('MergeBranches', {
      condition: r.outputs.out,
      true: tensor([2]),
      false: tensor([2]),
    }).outputs.out;
    expect(isTensorSpec(value) && value.autograd?.requiresGrad).toBe(false);
    expect(value).not.toHaveProperty('values');
  });
});
