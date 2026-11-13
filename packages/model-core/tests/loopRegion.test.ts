import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { ifRegionMembers } from '../src/core/ifRegions';
import { canMergeCalculations, mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';
import { parseProject, serializeProject } from '../src/core/serialization';
import { copySelection, duplicateSelection } from '../src/editing/clipboard';
import { registry } from '../src/modules';
import type { Recipe } from '../src/agent/types';

function recipe(): Recipe {
  return { name: 'visible loop arguments', nodes: [
    { id: 'x', type: 'Input', params: { shape: '2,1,3', axes: '' } },
    { id: 'cast', type: 'Cast', params: { dtype: 'float32' }, position: { x: 100, y: 200 } },
    { id: 'sum', type: 'ReduceSum', params: { dim: -1, keepdim: true }, position: { x: 400, y: 200 } },
    { id: 'out', type: 'Output' },
    { id: 'scope', type: 'LoopRegion', params: { statement: 'for interval in range(num_fu_frames)', members: 'cast,sum' } },
  ], connections: [['x.out', 'cast.in'], ['cast.out', 'sum.in'], ['sum.out', 'out.in']] };
}

describe('inline loop scope', () => {
  it('is a visual scope, not an iterative executor or another forward input', () => {
    const built = buildModel(recipe());
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const p = built.project!, result = inferProject(p, registry);
    expect(result.nodes.scope.outputs).toEqual({});
    expect(result.nodes.scope.iterations).toBeUndefined();
    expect(result.nodes.scope.child).toBeUndefined();
    expect(result.nodes.sum.outputs.out).toMatchObject({ shape: [2, 1, 1] });
    expect(registry.get('LoopRegion')!.inputs).toEqual([]);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const changed = patchModel(p, { operations: [{ op: 'set_params', id: 'scope', params: { statement: 'while False' } }] });
    expect(changed.ok).toBe(true);
    expect(changed.outputs).toEqual(built.outputs);
    expect(changed.project!.graph.nodes.find(n => n.id === 'cast')!.position).toEqual({ x: 100, y: 200 });
  });

  it.each(['', 'if ready', 'for', 'while', 'print(x)'])(
    'rejects an invalid loop heading %s', statement => {
      const r = recipe(); r.nodes[4].params!.statement = statement;
      expect(buildModel(r).ok).toBe(false);
    },
  );

  it.each(['missing', 'scope', 'cast,cast'])(
    'rejects invalid members %s', members => {
      const r = recipe(); r.nodes[4].params!.members = members;
      expect(buildModel(r).ok).toBe(false);
    },
  );

  it.each(['LoopRegion', 'IfRegion'])('prevents duplicate membership but allows nesting across %s', type => {
    const p = buildModel(recipe()).project!;
    expect(patchModel(p, { operations: [{ op: 'add_node', node: { id: 'other', type, params: { members: 'cast' } } }] }).ok).toBe(false);
    expect(patchModel(p, { operations: [{ op: 'add_node', node: { id: 'other', type, params: { members: 'scope' } } }] }).ok).toBe(true);
  });

  it('removes frame membership when code is deleted, but keeps code when the frame is deleted', () => {
    const p = buildModel(recipe()).project!;
    const result = patchModel(p, { operations: [{ op: 'remove_node', id: 'cast' }] }, { mode: 'draft' });
    expect(result.ok).toBe(true);
    expect(ifRegionMembers(result.project!.graph.nodes.find(n => n.id === 'scope')!)).toEqual(['sum']);
    const unframed = patchModel(p, { operations: [{ op: 'remove_node', id: 'scope' }] });
    expect(unframed.ok).toBe(true);
    expect(unframed.project!.graph.edges).toEqual(p.graph.edges);
    expect(unframed.project!.graph.nodes.map(n => n.id)).toEqual(['x', 'cast', 'sum', 'out']);
  });

  it('copies code with its frame and remaps local IDs', () => {
    const p = buildModel(recipe()).project!, fragment = copySelection(p.graph, ['scope']);
    expect(fragment.nodes.map(n => n.id)).toEqual(['cast', 'sum', 'scope']);
    const copy = duplicateSelection(fragment, 50);
    expect(ifRegionMembers(copy.nodes.find(n => n.type === 'LoopRegion')!))
      .toEqual(copy.nodes.filter(n => n.type !== 'LoopRegion').map(n => n.id));
  });

  it('preserves scope through calculation merge/split and rejects cross-scope merging', () => {
    const p = buildModel(recipe()).project!, g = p.graph;
    const id = mergeCalculations(g, 'cast', 'sum', registry);
    expect(ifRegionMembers(g.nodes.find(n => n.id === 'scope')!)).toEqual([id]);
    expect(validateModel(p).ok).toBe(true);
    const ids = splitCalculationGroup(g, id);
    expect(ifRegionMembers(g.nodes.find(n => n.id === 'scope')!)).toEqual(ids);
    expect(validateModel(p).ok).toBe(true);
    g.nodes.find(n => n.id === 'scope')!.params.members = ids[0];
    expect(canMergeCalculations(g, ids[0], ids[1], registry)).toBe(false);
  });
});
