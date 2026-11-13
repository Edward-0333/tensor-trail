import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import { inferGraph, inferProject } from '../src/core/engine';
import { inferWhile, MAX_WHILE_LINEAGE_PASSES } from '../src/core/whileLoop';
import { annotateInformationSources, mergeLineages, valueLineage } from '../src/core/informationSources';
import { clone, type GraphResult, type ModelNode, type Params, type ValueLineage, type ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { matchingLoopBody } from './fixtures/matchingLoop';

const lineage = (ids: number[] = [], origins: ValueLineage['origins'] = []): ValueLineage => ({
  sources: ids.map(index => ({ id: `input${index}`, index, label: `input ${index}` })),
  origins, initializations: [], unknown: false, evidence: 'inferred',
});
const tensor = (ids: number[] = [], origins: ValueLineage['origins'] = []): ValueSpec => ({
  shape: [4], axes: ['query'], dtype: 'float32',
  autograd: { requiresGrad: false, source: 'manual' }, lineage: lineage(ids, origins),
});
const integer = (value: number | string, ids: number[] = []): ValueSpec => ({
  kind: 'int', value, lineage: lineage(ids, ids.length ? [] : ['constant']),
});
const sources = (v: ValueSpec) => v.lineage?.sources.map(s => s.index);
const add = (loop: ModelNode, id: string, type: string, params: Params = {}) => {
  const node = registry.create(type, params, id);
  loop.subgraph!.nodes.push(node);
  return node;
};
const connect = (loop: ModelNode, from: string, to: string) => {
  const [source, sourcePort] = from.split('.'), [target, targetPort] = to.split('.');
  const g = loop.subgraph!;
  g.edges = g.edges.filter(e => e.target !== target || e.targetPort !== targetPort);
  g.edges.push({ id: `${source}_${sourcePort}_${target}_${targetPort}`, source, sourcePort, target, targetPort });
};
function run(loop: ModelNode, values: Record<string, ValueSpec>, enabled = true) {
  const inputs = { i: integer(0), num_points: integer(1040), ...values };
  const before = clone({ loop, inputs });
  let passes = 0;
  const budget = { remaining: 20000 };
  const result = inferWhile(loop, inputs, external => {
    passes++;
    return inferGraph(loop.subgraph!, registry, external, 1, {}, undefined, ['loop'], budget, undefined, enabled);
  }, enabled);
  expect({ loop, inputs }).toEqual(before);
  return { ...result, passes };
}
registry.register({
  type: 'TestOpaqueWhileOp', label: 'opaque', category: 'test', fields: [],
  description: 'Test-only operator with no provenance rule.',
  inputs: [{ id: 'in', label: 'in' }], outputs: [{ id: 'out', label: 'out' }],
  inferValue: values => ({ outputs: { out: clone(values.in) } }),
});

describe('While carried-state provenance', () => {
  it.each([0, 2])('returns original state with no unknown badge when i=%s already reaches the limit', i => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    const a = tensor([1]);
    const result = run(loop, { a, i: integer(i), num_points: integer(0) });
    expect(result.passes).toBe(0);
    expect(result.outputs.a).toEqual(a);
    expect(result.child).toBeUndefined();
  });

  it('keeps independent states separate and ignores an unused capture', () => {
    const loop = registry.create('WhileLoop', { states: 'a,b' });
    add(loop, 'unused', 'Inlet', { port: 'unused' });
    const result = run(loop, { a: tensor([1]), b: tensor([2]), unused: tensor([3]) });
    expect(result.status).toBe('ok');
    expect(sources(result.outputs.a)).toEqual([1]);
    expect(sources(result.outputs.b)).toEqual([2]);
    expect(result.outputs.a.lineage?.unknown).toBe(false);
    expect(result.passes).toBeLessThan(4);
    expect(result.iterations).toBeUndefined();
  });

  it('propagates dependencies arriving through another state on later iterations', () => {
    const loop = registry.create('WhileLoop', { states: 'a,b,c' });
    connect(loop, 'b.out', 'next_a.in');
    connect(loop, 'c.out', 'next_b.in');
    const values = { a: tensor([], ['constant']), b: tensor([], ['constant']), c: tensor([1]) };
    const once = run(loop, { ...values, num_points: integer(1) });
    expect(once.passes).toBe(1);
    expect(sources(once.outputs.a)).toEqual([]);
    expect(sources(once.outputs.b)).toEqual([1]);
    const result = run(loop, values);
    expect(sources(result.outputs.a)).toEqual([1]);
    expect(result.outputs.a.lineage?.unknown).toBe(false);
    expect(result.passes).toBeGreaterThan(1);
    expect(result.passes).toBeLessThan(5);
    expect(sources(result.child!.nodes.a.outputs.out)).toEqual([1]);
  });

  it('converges for mutually dependent states rather than oscillating forever', () => {
    const loop = registry.create('WhileLoop', { states: 'a,b' });
    connect(loop, 'b.out', 'next_a.in');
    connect(loop, 'a.out', 'next_b.in');
    const result = run(loop, { a: tensor([1]), b: tensor([2]) });
    expect(sources(result.outputs.a)).toEqual([1, 2]);
    expect(sources(result.outputs.b)).toEqual([1, 2]);
    expect(result.passes).toBeLessThan(4);
  });

  it('does not label a constant overwrite with the previous state or its dtype/device reference', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    add(loop, 'zeros', 'NewZeros', { shape: '4' });
    connect(loop, 'a.out', 'zeros.in');
    connect(loop, 'zeros.out', 'next_a.in');
    const positive = run(loop, { a: tensor([1]) });
    expect(sources(positive.outputs.a)).toEqual([]);
    expect(positive.outputs.a.lineage).toMatchObject({ unknown: false, initializations: ['zeros'] });
    const maybeZero = run(loop, { a: tensor([1]), num_points: integer('N') });
    expect(sources(maybeZero.outputs.a)).toEqual([1]);
    expect(maybeZero.outputs.a.lineage?.unknown).toBe(false);
  });

  it('includes data-dependent loop bounds but not unrelated captures', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    const result = run(loop, { a: tensor([1]), num_points: integer('N', [2]) });
    expect(sources(result.outputs.a)).toEqual([1, 2]);
  });

  it('retains unknown operators through recurrence without contaminating independent states', () => {
    const loop = registry.create('WhileLoop', { states: 'a,b,c' });
    add(loop, 'opaque', 'TestOpaqueWhileOp');
    connect(loop, 'b.out', 'opaque.in');
    connect(loop, 'opaque.out', 'next_b.in');
    connect(loop, 'b.out', 'next_a.in');
    const result = run(loop, { a: tensor([1]), b: tensor([2]), c: tensor([3]) });
    expect(result.outputs.a.lineage?.unknown).toBe(true);
    expect(result.outputs.b.lineage?.unknown).toBe(true);
    expect(result.outputs.c.lineage?.unknown).toBe(false);
    expect(sources(result.outputs.c)).toEqual([3]);
  });

  it('merges conditional alternatives and the condition, not just the first branch', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    add(loop, 'other', 'Inlet', { port: 'other' });
    add(loop, 'condition_value', 'Inlet', { port: 'condition_value' });
    add(loop, 'quota', 'IntegerParameter', { expression: '1' });
    add(loop, 'condition', 'IfCondition');
    add(loop, 'yes', 'BranchValue', { branch: true });
    add(loop, 'no', 'BranchValue', { branch: false });
    add(loop, 'merge', 'MergeBranches');
    for (const [from, to] of [
      ['condition_value.out', 'condition.count'], ['quota.out', 'condition.quota'],
      ['a.out', 'yes.in'], ['condition.true', 'yes.control'],
      ['other.out', 'no.in'], ['condition.false', 'no.control'],
      ['condition.out', 'merge.condition'], ['yes.out', 'merge.true'], ['no.out', 'merge.false'],
      ['merge.out', 'next_a.in'],
    ]) connect(loop, from, to);
    const result = run(loop, { a: tensor([1]), other: tensor([3]), condition_value: integer('flag', [2]) });
    expect(result.status).toBe('ok');
    expect(sources(result.outputs.a)).toEqual([1, 2, 3]);
    expect(result.outputs.a.lineage?.unknown).toBe(false);
    expect(result.child!.nodes.condition.outputs.out).toMatchObject({ kind: 'control', expression: 'flag == 1' });
  });

  it('handles nested loops without inventing root input IDs', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    add(loop, 'capture', 'Inlet', { port: 'capture' });
    const inner = add(loop, 'inner', 'WhileLoop', { states: 'a' });
    add(inner, 'capture', 'Inlet', { port: 'capture' });
    add(inner, 'merge', 'Add');
    connect(inner, 'a.out', 'merge.a');
    connect(inner, 'capture.out', 'merge.b');
    connect(inner, 'merge.out', 'next_a.in');
    add(loop, 'zero', 'IntegerParameter', { expression: '0' });
    add(loop, 'two', 'IntegerParameter', { expression: '2' });
    connect(loop, 'a.out', 'inner.a');
    connect(loop, 'capture.out', 'inner.capture');
    connect(loop, 'zero.out', 'inner.i');
    connect(loop, 'two.out', 'inner.num_points');
    connect(loop, 'inner.a', 'next_a.in');
    const result = run(loop, { a: tensor([1]), capture: tensor([2]) });
    expect(result.status).toBe('ok');
    expect(sources(result.outputs.a)).toEqual([1, 2]);
    expect(result.outputs.a.lineage?.unknown).toBe(false);
  });

  it('bounds analysis work and preserves uncertainty if a transfer keeps adding origins', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    let passes = 0;
    const result = inferWhile(loop, { a: tensor(), i: integer(0), num_points: integer('N') }, external => {
      const out = clone(external.a);
      out.lineage = mergeLineages([valueLineage(out), lineage([++passes])]);
      return { order: [], nodes: {}, issues: [], outputs: { a: out, i: integer('(i + 1)') } };
    }, true);
    expect(passes).toBe(MAX_WHILE_LINEAGE_PASSES);
    expect(result.status).toBe('warning');
    expect(result.outputs.a.lineage?.unknown).toBe(true);
    expect(sources(result.outputs.a)).toHaveLength(MAX_WHILE_LINEAGE_PASSES);
  });

  it('does not return success after a later analysis pass fails', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    let passes = 0;
    const result = inferWhile(loop, { a: tensor(), i: integer(0), num_points: integer(1040) }, (): GraphResult => {
      passes++;
      return passes === 1
        ? { order: [], nodes: {}, issues: [], outputs: { a: tensor([1]), i: integer('(i + 1)') } }
        : { order: [], nodes: {}, issues: ['budget exhausted'], outputs: {} };
    }, true);
    expect(result.status).toBe('error');
    expect(result.outputs).toEqual({});
    expect(result.messages.join(' ')).toContain('budget exhausted');
  });

  it('keeps single-template behavior when provenance is disabled', () => {
    const loop = registry.create('WhileLoop', { states: 'a' });
    expect(run(loop, { a: tensor([1]) }, false).passes).toBe(1);
  });
});

describe('Matching-loop provenance integration', () => {
  it.each([
    ['TensorRowAt', ['in', 'index']], ['TensorEqualInt', ['in', 'value']],
    ['IntegerListGet', ['in', 'index']], ['ScalarIndexWrite', ['in', 'index', 'value']],
    ['ZeroTailColumn', ['in', 'start', 'column']], ['SliceRowsFrom', ['in', 'start']],
    ['WriteTailRows', ['in', 'start', 'rows']], ['IfCondition', ['count', 'quota']],
    ['BranchValue', ['in', 'control']], ['MergeBranches', ['condition', 'true', 'false']],
  ] as const)('%s includes index/value/control dependencies, not just the target tensor', (type, ports) => {
    const inputs = Object.fromEntries(ports.map((port, index) => [port, tensor([index + 1])]));
    const result = annotateInformationSources(registry.create(type), inputs, { out: tensor() }).out;
    expect(sources(result)).toEqual(ports.map((_, i) => i + 1));
    expect(result.lineage?.unknown).toBe(false);
  });

  it('resolves buffer/constant origins through matching, serialization and the public API', () => {
    const built = buildModel({
      name: 'matching origins',
      nodes: [
        { id: 'maps', type: 'BufferSource', params: { shape: '1040,8', axes: 'query,column', dtype: 'float32', initial: 'ones' } },
        { id: 'inds', type: 'BufferSource', params: { shape: '1040', axes: 'query', dtype: 'float32', initial: 'ones' } },
        { id: 'start', type: 'IntegerParameter', params: { expression: '0' } },
        { id: 'count', type: 'IntegerParameter', params: { expression: '1040' } },
        { id: 'quotas', type: 'IntegerList' },
        { id: 'loop', type: 'WhileLoop', params: { states: 'maps,inds' }, subgraph: matchingLoopBody() },
        { id: 'out', type: 'Output' },
      ],
      connections: [['maps.out', 'loop.maps'], ['inds.out', 'loop.inds'], ['start.out', 'loop.i'],
        ['count.out', 'loop.num_points'], ['quotas.out', 'loop.num_inds'], ['loop.inds', 'out.in']],
    }, { mode: 'draft' });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    const before = JSON.stringify(project);
    const result = inferProject(project, registry);
    expect(result.nodes.loop.status).toBe('ok');
    expect(result.nodes.out.outputs.out).toMatchObject({
      shape: [1040], dtype: 'float32', autograd: { requiresGrad: false },
      lineage: { sources: [], origins: ['constant', 'state'], unknown: false },
    });
    const visit = (r: GraphResult) => {
      for (const n of Object.values(r.nodes)) {
        for (const v of Object.values(n.outputs)) expect(v.lineage?.unknown).toBe(false);
        if (n.child) visit(n.child);
      }
    };
    visit(result);
    expect(result.nodes.loop.iterations).toBeUndefined();
    expect(JSON.stringify(project)).toBe(before);
    const reopened = validateModel(JSON.parse(before), { mode: 'draft' });
    expect(reopened.ok, JSON.stringify(reopened.diagnostics)).toBe(true);
    expect(reopened.nodes.find(n => n.path.join('/') === 'loop')!.outputs.inds.lineage?.unknown).toBe(false);
  });
});
