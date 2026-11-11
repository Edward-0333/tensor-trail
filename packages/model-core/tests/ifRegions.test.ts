import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { ifRegionMembers } from '../src/core/ifRegions';
import {
  copySelection,
  duplicateSelection,
  readClipboard,
  clipboardText,
} from '../src/editing/clipboard';
import {
  canMergeCalculations,
  mergeCalculations,
  splitCalculationGroup,
} from '../src/core/calculationGroups';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';

const recipe = () => ({
  name: 'conditional scope',
  nodes: [
    {
      id: 'x',
      type: 'Input',
      params: { shape: '1040,7', axes: 'query,time' },
      position: { x: 20, y: 50 },
    },
    { id: 'cast', type: 'Cast', params: { dtype: 'float32' }, position: { x: 400, y: 50 } },
    {
      id: 'sum',
      type: 'ReduceSum',
      params: { dim: -1, keepdim: true },
      position: { x: 800, y: 50 },
    },
    { id: 'out', type: 'Output' },
    { id: 'scope', type: 'IfRegion', params: { condition: 'self.pretrain', members: 'cast,sum' } },
  ],
  connections: [
    ['x.out', 'cast.in'],
    ['cast.out', 'sum.in'],
    ['sum.out', 'out.in'],
  ],
});

describe('if visual scopes', () => {
  it('preserves output shapes, layout and serialization without evaluating the condition', () => {
    const built = buildModel(recipe());
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const p = built.project!;
    const result = inferProject(p, registry);
    expect(result.nodes.scope.outputs).toEqual({});
    expect(result.nodes.sum.outputs.out).toMatchObject({ shape: [1040, 1] });
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const changed = patchModel(p, {
      operations: [{ op: 'set_params', id: 'scope', params: { condition: 'False' } }],
    });
    expect(changed.ok).toBe(true);
    expect(changed.outputs).toEqual(built.outputs);
    expect(changed.project!.graph.nodes.find((n) => n.id === 'cast')!.position).toEqual({
      x: 400,
      y: 50,
    });
  });
  it.each(['missing', 'scope', 'cast,cast', '__proto__'])(
    'rejects invalid members %s',
    (members) => {
      const r = recipe();
      r.nodes[4].params = { condition: 'self.pretrain', members } as (typeof r.nodes)[4]['params'];
      expect(buildModel(r).ok).toBe(false);
    },
  );
  it('rejects overlapping same-level regions', () => {
    const p = buildModel(recipe()).project!;
    const result = patchModel(p, {
      operations: [
        { op: 'add_node', node: { id: 'other', type: 'IfRegion', params: { members: 'cast' } } },
      ],
    });
    expect(result.ok).toBe(false);
  });
  it('prunes membership on node deletion, but deleting a frame retains code', () => {
    const p = buildModel(recipe()).project!;
    const deleted = patchModel(
      p,
      { operations: [{ op: 'remove_node', id: 'cast' }] },
      { mode: 'draft' },
    );
    expect(deleted.ok, JSON.stringify(deleted.diagnostics)).toBe(true);
    expect(ifRegionMembers(deleted.project!.graph.nodes.find((n) => n.id === 'scope')!)).toEqual([
      'sum',
    ]);
    const unframed = patchModel(p, { operations: [{ op: 'remove_node', id: 'scope' }] });
    expect(unframed.ok).toBe(true);
    expect(unframed.project!.graph.nodes.map((n) => n.id)).toEqual(
      p.graph.nodes.filter((n) => n.id !== 'scope').map((n) => n.id),
    );
    expect(unframed.project!.graph.edges).toEqual(p.graph.edges);
  });
  it('copies framed code and remaps only local membership on paste', () => {
    const p = buildModel(recipe()).project!;
    const fragment = copySelection(p.graph, ['scope']);
    expect(fragment.nodes.map((n) => n.id)).toEqual(['cast', 'sum', 'scope']);
    const pasted = duplicateSelection(readClipboard(clipboardText(fragment))!, 50);
    const frame = pasted.nodes.find((n) => n.type === 'IfRegion')!;
    expect(ifRegionMembers(frame)).toEqual(
      pasted.nodes.filter((n) => n.type !== 'IfRegion').map((n) => n.id),
    );
    expect(pasted.nodes[0].position).toEqual({ x: 450, y: 100 });
  });
  it('retains membership through calculation merge and split, disallows cross-scope merging', () => {
    const p = buildModel(recipe()).project!,
      g = p.graph;
    const id = mergeCalculations(g, 'cast', 'sum', registry);
    expect(ifRegionMembers(g.nodes.find((n) => n.id === 'scope')!)).toEqual([id]);
    expect(validateModel(p).ok).toBe(true);
    const members = splitCalculationGroup(g, id);
    expect(ifRegionMembers(g.nodes.find((n) => n.id === 'scope')!)).toEqual(members);
    expect(validateModel(p).ok).toBe(true);
    g.nodes.find((n) => n.id === 'scope')!.params.members = members[0];
    expect(canMergeCalculations(g, members[0], members[1], registry)).toBe(false);
  });
});

describe('explicit versions of an attribute assignment', () => {
  it('loads legacy SET/GET nodes without inserting new serialized fields', () => {
    const built = buildModel({
      name: 'legacy',
      nodes: [
        { id: 'x', type: 'Input' },
        { id: 'set', type: 'SetAttribute', params: { owner: '[]', name: 'x' } },
        { id: 'get', type: 'GetAttribute', params: { owner: '[]', name: 'x' } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['x.out', 'set.in'],
        ['get.out', 'out.in'],
      ],
    });
    for (const n of built.project!.graph.nodes)
      if (['SetAttribute', 'GetAttribute'].includes(n.type)) delete n.params.version;
    const serialized = serializeProject(built.project!);
    expect(validateModel(built.project!).ok).toBe(true);
    expect(serializeProject(built.project!)).toBe(serialized);
    expect(inferProject(built.project!, registry).nodes.get.outputs.out).toEqual(
      inferProject(built.project!, registry).nodes.x.outputs.out,
    );
  });
  const p = () =>
    buildModel({
      name: 'versions',
      nodes: [
        { id: 'a', type: 'Input', params: { shape: '1040', axes: 'query' } },
        { id: 'b', type: 'Input', params: { shape: '20', axes: 'query' } },
        {
          id: 'initial',
          type: 'SetAttribute',
          params: { owner: '[]', name: 'ind_stamps_all', version: 'initial' },
        },
        {
          id: 'refresh',
          type: 'SetAttribute',
          params: { owner: '[]', name: 'ind_stamps_all', version: 'pretrain' },
        },
        {
          id: 'get',
          type: 'GetAttribute',
          params: { owner: '[]', name: 'ind_stamps_all', version: 'pretrain' },
        },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['a.out', 'initial.in'],
        ['b.out', 'refresh.in'],
        ['get.out', 'out.in'],
      ],
    });
  it('selects the explicit write, not graph order or last-writer guesses', () => {
    const built = p();
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(built.outputs.out).toMatchObject({ shape: [20] });
    const changed = patchModel(built.project!, {
      operations: [{ op: 'set_params', id: 'get', params: { version: 'initial' } }],
    });
    expect(changed.ok).toBe(true);
    expect(changed.outputs.out).toMatchObject({ shape: [1040] });
  });
  it('does not guess the version for an unqualified reader', () => {
    expect(
      patchModel(p().project!, {
        operations: [{ op: 'set_params', id: 'get', params: { version: '' } }],
      }).ok,
    ).toBe(false);
  });
  it('rejects two writes of the same version', () => {
    expect(
      patchModel(p().project!, {
        operations: [{ op: 'set_params', id: 'refresh', params: { version: 'initial' } }],
      }).ok,
    ).toBe(false);
  });
  it('requires explicit versions for all writes when an attribute is overwritten', () => {
    expect(
      patchModel(p().project!, {
        operations: [{ op: 'set_params', id: 'initial', params: { version: '' } }],
      }).ok,
    ).toBe(false);
  });
  it('rejects an unsafe version', () => {
    expect(
      patchModel(p().project!, {
        operations: [{ op: 'set_params', id: 'refresh', params: { version: 'a.b()' } }],
      }).ok,
    ).toBe(false);
  });
});
