import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferProject, resultAt } from '../src/core/engine';
import { checkForLoopBody, inferForLoop } from '../src/core/forLoop';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { TensorSequenceSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { decoderCounts, decoderLoopRecipe } from './fixtures/decoderLoop';

function build(iterations = 6) {
  const result = buildModel(decoderLoopRecipe(iterations));
  expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
  return result.project!;
}

describe('ForLoop state recurrence', () => {
  it('carries changing shapes and growing lists through all six independent layer contracts', () => {
    const p = build(), before = serializeProject(p), r = inferProject(p, registry), loop = r.nodes.loop;
    expect(loop.status).toBe('ok');
    expect(loop.iterations).toHaveLength(6);
    decoderCounts.forEach((P, i) => {
      const child = loop.iterations![i];
      expect(child.nodes.query_points.outputs.out).toMatchObject({ shape: [2, 1040, i ? decoderCounts[i - 1] : 1, 3] });
      expect(child.nodes.detach.outputs.out).toMatchObject({ autograd: { requiresGrad: false } });
      expect(child.nodes.layer.outputs.refine_pts).not.toHaveProperty('autograd');
      expect(child.outputs.query_points).toMatchObject({ shape: [2, 1040, P, 3] });
      expect(child.outputs.query_feat).toMatchObject({ shape: [2, 1040, 256] });
      expect((child.outputs.cls_scores as TensorSequenceSpec).items).toHaveLength(i + 1);
      expect(child.nodes.layer.notes.join(' ')).toContain(`decoder_layers[${i}]`);
      if (i) {
        expect(child.nodes.query_feat.outputs.out).toEqual(loop.iterations![i - 1].outputs.query_feat);
        expect(child.nodes.query_points.outputs.out).toEqual(loop.iterations![i - 1].outputs.query_points);
      }
      expect(child.nodes.mlvl_feats.outputs.out).toEqual(loop.inputs.mlvl_feats);
    });
    expect((loop.outputs.refine_pts as TensorSequenceSpec).items.map(t => t.shape[2])).toEqual(decoderCounts);
    expect((loop.outputs.refine_pts as TensorSequenceSpec).items.every(t => !t.autograd)).toBe(true);
    expect(serializeProject(p)).toBe(before);
    expect(validateModel(parseProject(before, registry)).ok).toBe(true);
    expect(ports(p.graph.nodes.find(n => n.id === 'loop')!, registry).inputs.some(p => p.id === 'index')).toBe(false);
  });

  it('exposes every iteration without mutating the graph or final state', () => {
    const p = build(), r = inferProject(p, registry), before = structuredClone(r);
    decoderCounts.forEach((P, index) => expect(resultAt(r, ['loop'], { '["loop"]': index })!.outputs.query_points)
      .toMatchObject({ shape: [2, 1040, P, 3] }));
    expect(r).toEqual(before);
  });

  it('skips a zero-length loop, preserving all initial states', () => {
    const loop = inferProject(build(0), registry).nodes.loop;
    expect(loop.iterations).toEqual([]);
    expect(loop.outputs.query_points).toEqual(loop.inputs.query_points);
    expect(loop.outputs.cls_scores).toEqual(loop.inputs.cls_scores);
    expect(loop.child).toBeUndefined();
  });

  it('rejects later-iteration errors even in draft mode; no partial final output', () => {
    const p = build(), before = serializeProject(p);
    const changed = patchModel(p, { operations: [{ op: 'set_params', path: ['loop'], id: 'layer', params: { refines: '1,4' } }] }, { mode: 'draft' });
    expect(changed.ok).toBe(false);
    expect(changed.diagnostics.some(d => d.message.includes('迭代 2'))).toBe(true);
    expect(serializeProject(p)).toBe(before);
  });

  it('uses generic integer states rather than requiring SparseWorld tensors', () => {
    const result = buildModel({ name: 'integer recurrence', nodes: [
      { id: 'start', type: 'IntegerParameter', params: { expression: '2' } },
      { id: 'loop', type: 'ForLoop', params: { iterations: 3, states: 'value' }, subgraph: {
        nodes: [{ id: 'index', type: 'Inlet', params: { port: 'index' } },
          { id: 'value', type: 'Inlet', params: { port: 'value' } }, { id: 'add', type: 'IntegerAddOne' },
          { id: 'next', type: 'Outlet', params: { port: 'value' } }],
        connections: [['value.out', 'add.in'], ['add.out', 'next.in']],
      } },
    ], connections: [['start.out', 'loop.value']] }, { mode: 'draft' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(inferProject(result.project!, registry).nodes.loop.outputs.value).toEqual({ kind: 'int', value: 5 });
  });

  it.each([-1, 1.5, 17, NaN])('rejects invalid iteration limit %s', iterations => {
    const node = registry.create('ForLoop', { iterations });
    expect(() => inferForLoop(node, { value: { kind: 'int', value: 0 } }, () => { throw Error('must not run'); }))
      .toThrow(/0-16/);
  });

  it('validates state boundaries and reserved names', () => {
    const n = registry.create('ForLoop');
    n.subgraph!.nodes = n.subgraph!.nodes.filter(n => n.type !== 'Outlet');
    expect(() => checkForLoopBody(n)).toThrow(/Outlet/);
    n.params.states = 'index';
    expect(() => checkForLoopBody(n)).toThrow(/保留/);
  });

  it('allows incomplete drafts while preserving the loop layout', () => {
    const p = build();
    const change = patchModel(p, { operations: [{ op: 'disconnect', path: ['loop'], from: 'detach.out', to: 'layer.query_points' }] }, { mode: 'draft', layout: 'preserve' });
    expect(change.ok, JSON.stringify(change.diagnostics)).toBe(true);
    expect(change.nodes.find(n => n.path.join('/') === 'loop')!.status).toBe('incomplete');
    expect(change.project!.graph.nodes.find(n => n.id === 'loop')!.position).toEqual(p.graph.nodes.find(n => n.id === 'loop')!.position);
  });

  it('uses a connected integer count instead of the fallback parameter and skips zero iterations', () => {
    const recipe = decoderLoopRecipe(1);
    recipe.nodes.find(n => n.id === 'loop')!.params!.iterations_input = true;
    recipe.nodes.push({ id: 'count', type: 'IntegerParameter', params: { expression: '3' } });
    recipe.connections.push(['count.out', 'loop.iterations']);
    const changed = buildModel(recipe);
    expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
    expect(inferProject(changed.project!, registry).nodes.loop.iterations).toHaveLength(3);
    expect(inferProject(changed.project!, registry).nodes.loop.child!.nodes).not.toHaveProperty('iterations');
    const zero = patchModel(changed.project!, { operations: [
      { op: 'set_params', id: 'count', params: { expression: '0' } },
    ] });
    expect(zero.ok).toBe(true);
    const loop = inferProject(zero.project!, registry).nodes.loop;
    expect(loop.iterations).toEqual([]);
    expect(loop.outputs.query_feat).toEqual(loop.inputs.query_feat);
  });

  it('requires a connected known integer for input-controlled counts, including in drafts', () => {
    const node = registry.create('ForLoop', { iterations_input: true });
    const run = (iterations: ValueSpec | undefined) => inferForLoop(node,
      { value: { kind: 'int', value: 0 }, ...(iterations ? { iterations } : {}) }, () => { throw Error('must not run'); });
    const invalid: (ValueSpec | undefined)[] = [undefined, { kind: 'int', value: 'N' }, { kind: 'int', value: 17 },
      { kind: 'int', value: -1 }, { kind: 'int', value: 1.5 }, { shape: [], axes: [], dtype: 'int64' }];
    for (const value of invalid)
      expect(() => run(value)).toThrow(/0-16/);
    const recipe = decoderLoopRecipe();
    recipe.nodes.find(n => n.id === 'loop')!.params!.iterations_input = true;
    const pending = buildModel(recipe, { mode: 'draft' });
    expect(pending.ok).toBe(true);
    expect(pending.nodes.find(n => n.path.join('/') === 'loop')!.status).toBe('incomplete');
  });

  it('reserves the count port only when explicitly enabled', () => {
    const node = registry.create('ForLoop', { states: 'iterations' });
    expect(() => checkForLoopBody(node)).not.toThrow();
    node.params.iterations_input = true;
    expect(() => checkForLoopBody(node)).toThrow(/控制端口/);
  });

  it('keeps an unfinished recurrence as a draft without inventing final states', () => {
    const changed = patchModel(build(1), { operations: [
      { op: 'disconnect', path: ['loop'], from: 'layer.query_feat', to: 'next_query_feat.in' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(changed.ok, JSON.stringify(changed.diagnostics)).toBe(true);
    const loop = changed.nodes.find(n => n.path.join('/') === 'loop')!;
    expect(loop.status).toBe('incomplete');
    expect(loop.outputs).toEqual({});
  });
});
