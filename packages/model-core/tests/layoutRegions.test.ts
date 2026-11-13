import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { copySelection, duplicateSelection } from '../src/editing/clipboard';
import { removeValueNodes } from '../src/editing/valueLinkEditing';
import { canMergeCalculations, mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';
import { patchModel, validateModel } from '../src/agent';
import type { Graph, Project } from '../src/core/types';
import {
  assignLayoutRegion, constrainLayoutMoves, fitLayoutRegion, layoutRegionIssues,
  layoutRegionMembers, layoutRegionOf, layoutRegionRect, moveLayoutNodes, resizeLayoutRegion,
} from '../src/core/layoutRegions';

function fixture() {
  const graph: Graph = { nodes: [
    registry.create('Input', { shape: '2,3', axes: 'B,C' }, 'input'),
    registry.create('ReLU', {}, 'a'), registry.create('ReLU', {}, 'b'),
    registry.create('Output', {}, 'output'), registry.create('LayoutRegion', {}, 'region'),
  ], edges: [
    { id: 'in', source: 'input', sourcePort: 'out', target: 'a', targetPort: 'in' },
    { id: 'inside', source: 'a', sourcePort: 'out', target: 'b', targetPort: 'in' },
    { id: 'out', source: 'b', sourcePort: 'out', target: 'output', targetPort: 'in' },
  ] };
  graph.nodes.forEach((n, i) => { n.position = { x: i * 400, y: 800 }; });
  const region = graph.nodes[4]; region.position = { x: 2000, y: 100 };
  const project: Project = { schemaVersion: 1, name: 'layout test', graph, variants: [] };
  const sizes = { a: { width: 210, height: 100 }, b: { width: 270, height: 140 } };
  return { graph, project, region, sizes };
}

describe('Layout regions', () => {
  it('moves new members into a region without changing connections or inferred data', () => {
    const f = fixture(), before = inferProject(f.project, registry), edges = structuredClone(f.graph.edges);
    assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    const [a, b] = ['a', 'b'].map(id => f.graph.nodes.find(n => n.id === id)!);
    expect(a.position).toEqual({ x: 2024, y: 160 });
    expect(b.position).toEqual({ x: 2274, y: 160 });
    expect(layoutRegionMembers(f.region)).toEqual(['a', 'b']);
    expect(f.graph.edges).toEqual(edges);
    expect(inferProject(f.project, registry).nodes).toEqual(before.nodes);
    expect(layoutRegionIssues(f.graph)).toEqual([]);
  });

  it.each(['auto', 'manual'])('clamps individual and multi-selection movement in %s mode until explicitly released', sizing => {
    const f = fixture(); f.region.params.sizing = sizing;
    assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    const rect = layoutRegionRect(f.graph, f.region, f.sizes);
    const moves = constrainLayoutMoves(f.graph, [
      { id: 'a', position: { x: -9999, y: -9999 } }, { id: 'b', position: { x: 9999, y: 9999 } },
      { id: 'input', position: { x: -20, y: -30 } },
    ], f.sizes);
    expect(moves[0].position).toEqual({ x: rect.x + 24, y: rect.y + 60 });
    expect(moves[1].position).toEqual({ x: rect.x + rect.width - 24 - 270, y: rect.y + rect.height - 24 - 140 });
    expect(moves[2].position).toEqual({ x: -20, y: -30 });
    moveLayoutNodes(f.graph, moves, f.sizes);
    assignLayoutRegion(f.graph, ['a'], null, f.sizes);
    moveLayoutNodes(f.graph, [{ id: 'a', position: { x: -999, y: -999 } }], f.sizes);
    expect(f.graph.nodes.find(n => n.id === 'a')!.position).toEqual({ x: -999, y: -999 });
    expect(layoutRegionOf(f.graph, 'a')).toBeUndefined();
  });

  it('moves the frame and members once, preserving internal routes and unrelated geometry', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    for (const edge of f.graph.edges) { edge.waypoints = [{ x: 2120, y: 200 }]; edge.labelPosition = .5; }
    const before = structuredClone(f.graph);
    moveLayoutNodes(f.graph, [{ id: 'region', position: { x: 2100, y: 150 } },
      { id: 'a', position: { x: 0, y: 0 } }], f.sizes);
    for (const id of ['a', 'b']) {
      const old = before.nodes.find(n => n.id === id)!.position;
      expect(f.graph.nodes.find(n => n.id === id)!.position).toEqual({ x: old.x + 100, y: old.y + 50 });
    }
    expect(f.graph.nodes[0]).toEqual(before.nodes[0]);
    expect(f.graph.edges[1].waypoints).toEqual([{ x: 2220, y: 250 }]);
    expect(f.graph.edges[1].labelPosition).toBe(.5);
    expect(f.graph.edges[0].waypoints).toBeUndefined();
    expect(f.graph.edges[2].waypoints).toBeUndefined();
  });

  it('keeps cards inside when resizing any edge and supports automatic fit and changing measurements', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    const positions = f.graph.nodes.map(n => [n.id, { ...n.position }]);
    resizeLayoutRegion(f.graph, 'region', { x: 2400, y: 300, width: 100, height: 50 }, f.sizes);
    const rect = layoutRegionRect(f.graph, f.region, f.sizes);
    expect(rect.x).toBe(2000); expect(rect.y).toBe(100);
    expect(rect.width).toBeGreaterThanOrEqual(568); expect(rect.height).toBeGreaterThanOrEqual(224);
    expect(f.region.params.sizing).toBe('manual');
    fitLayoutRegion(f.graph, 'region', f.sizes);
    expect(f.region.params.sizing).toBe('auto');
    expect(f.graph.nodes.filter(n => n.id !== 'region').map(n => [n.id, n.position]))
      .toEqual(positions.filter(([id]) => id !== 'region'));
    const before = structuredClone(f.graph);
    const grown = layoutRegionRect(f.graph, f.region, { ...f.sizes, b: { width: 500, height: 500 } });
    expect(grown.width).toBeGreaterThan(rect.width); expect(grown.height).toBeGreaterThan(rect.height);
    expect(f.graph).toEqual(before);
  });

  it('packs later additions without overlapping existing members and supports explicit transfer', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a'], 'region', f.sizes);
    const a = { ...f.graph.nodes[1].position };
    assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    expect(f.graph.nodes[1].position).toEqual(a);
    expect(f.graph.nodes[2].position.y).toBeGreaterThanOrEqual(a.y + f.sizes.a.height);
    const other = registry.create('LayoutRegion', {}, 'other'); other.position = { x: -1000, y: -1000 };
    f.graph.nodes.push(other); assignLayoutRegion(f.graph, ['b'], 'other', f.sizes);
    expect(layoutRegionMembers(f.region)).toEqual(['a']); expect(layoutRegionOf(f.graph, 'b')?.id).toBe('other');
  });

  it('validates same-level, unique membership and safe geometry and colors', () => {
    const f = fixture();
    for (const members of ['missing', 'region', 'a,a']) {
      f.region.params.members = members; expect(layoutRegionIssues(f.graph).length).toBeGreaterThan(0);
    }
    f.region.params.members = 'a';
    const second = registry.create('LayoutRegion', { members: 'a' }, 'second'); f.graph.nodes.push(second);
    expect(layoutRegionIssues(f.graph).join()).toContain('只能属于一个');
    f.graph.nodes.pop(); f.region.params.members = '';
    f.region.params.color = 'url(bad)'; expect(layoutRegionIssues(f.graph).join()).toContain('颜色');
    f.region.params.color = '#38b2ac'; f.region.params.width = Infinity;
    expect(layoutRegionIssues(f.graph).join()).toContain('宽度');
  });

  it('copies a frame with its members, remaps membership and preserves deletion semantics', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    const fragment = copySelection(f.graph, ['region']);
    expect(fragment.nodes.map(n => n.id).sort()).toEqual(['a', 'b', 'region']);
    const pasted = duplicateSelection(fragment, 100), region = pasted.nodes.find(n => n.type === 'LayoutRegion')!;
    expect(layoutRegionMembers(region).sort()).toEqual(pasted.nodes.filter(n => n !== region).map(n => n.id).sort());
    expect(layoutRegionIssues(pasted)).toEqual([]);
    expect(copySelection(f.graph, ['a']).nodes).toHaveLength(1);
    removeValueNodes(f.graph, [], ['b']); expect(layoutRegionMembers(f.region)).toEqual(['a']);
    removeValueNodes(f.graph, [], ['region']); expect(f.graph.nodes.some(n => n.id === 'a')).toBe(true);
  });

  it('preserves membership through calculation merge/split and rejects cross-region merging', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a'], 'region', f.sizes);
    expect(canMergeCalculations(f.graph, 'a', 'b', registry)).toBe(false);
    assignLayoutRegion(f.graph, ['b'], 'region', f.sizes);
    // Use actual composable arithmetic operators.
    f.graph.nodes[1].type = 'Detach'; f.graph.nodes[2].type = 'Detach';
    expect(canMergeCalculations(f.graph, 'a', 'b', registry)).toBe(true);
    const merged = mergeCalculations(f.graph, 'a', 'b', registry);
    expect(layoutRegionMembers(f.region)).toEqual([merged]);
    const split = splitCalculationGroup(f.graph, merged);
    expect(layoutRegionMembers(f.region)).toEqual(split); expect(layoutRegionIssues(f.graph)).toEqual([]);
  });

  it('round-trips regions without persisting UI measurements', () => {
    const f = fixture(); assignLayoutRegion(f.graph, ['a', 'b'], 'region', f.sizes);
    expect(parseProject(serializeProject(f.project), registry)).toEqual(f.project);
    expect(serializeProject(f.project)).not.toContain('nodeSizes');
  });
});

describe('Region editing API', () => {
  it('does not mark enclosing neural modules customized for purely visual edits', () => {
    const parent = registry.create('Transformer', {}, 'parent');
    const project: Project = { schemaVersion: 1, name: 'nested', variants: [], graph: { nodes: [parent], edges: [] } };
    const path = ['parent'], member = parent.subgraph!.nodes.find(n => n.type === 'Inlet')!.id;
    const patched = patchModel(project, { operations: [
      { op: 'add_node', path, node: { id: 'region', type: 'LayoutRegion' } },
      { op: 'set_params', path, id: 'region', params: { color: '#6699aa' } },
      { op: 'assign_region', path, ids: [member], region: 'region' },
      { op: 'fit_region', path, id: 'region' },
      { op: 'remove_node', path, id: 'region' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(patched.ok, JSON.stringify(patched.diagnostics)).toBe(true);
    expect(patched.project!.graph.nodes[0].customized).toBeFalsy();
    expect(patched.project!.graph.nodes[0].subgraph!.nodes.some(n => n.id === member)).toBe(true);
  });

  it('assigns, constrains moves, resizes, fits and releases without modifying the input project', () => {
    const f = fixture(), original = structuredClone(f.project);
    const assigned = patchModel(f.project, { operations: [
      { op: 'assign_region', ids: ['a', 'b'], region: 'region' },
      { op: 'set_position', id: 'a', position: { x: -1000, y: -1000 } },
      { op: 'resize_region', id: 'region', width: 900, height: 700 },
      { op: 'fit_region', id: 'region' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(assigned.ok, JSON.stringify(assigned.diagnostics)).toBe(true);
    expect(assigned.project!.graph.nodes[1].position).toEqual({ x: 2024, y: 160 });
    expect(f.project).toEqual(original); expect(validateModel(assigned.project!).ok).toBe(true);
    const released = patchModel(assigned.project!, { operations: [
      { op: 'assign_region', ids: ['a'], region: null },
      { op: 'set_position', id: 'a', position: { x: -1000, y: -1000 } },
    ] });
    expect(released.ok).toBe(true); expect(released.project!.graph.nodes[1].position.x).toBe(-1000);
  });

  it('does not break containment when applying explicit automatic layout or moving an unplaced new region', () => {
    const f = fixture();
    const patched = patchModel(f.project, { operations: [
      { op: 'add_node', node: { id: 'new_region', type: 'LayoutRegion' } },
      { op: 'assign_region', ids: ['a', 'b'], region: 'new_region' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(patched.ok, JSON.stringify(patched.diagnostics)).toBe(true);
    const graph = patched.project!.graph, region = graph.nodes.find(n => n.id === 'new_region')!;
    expect(graph.nodes[1].position.x).toBe(region.position.x + 24);
    const positions = graph.nodes.filter(n => ['a', 'b', 'new_region'].includes(n.id)).map(n => n.position);
    const arranged = patchModel(patched.project!, { operations: [{ op: 'layout' }] });
    expect(arranged.ok).toBe(true);
    expect(arranged.project!.graph.nodes.filter(n => ['a', 'b', 'new_region'].includes(n.id)).map(n => n.position)).toEqual(positions);
  });

  it('rejects bad assignments and resize values atomically', () => {
    const f = fixture(), before = structuredClone(f.project);
    expect(patchModel(f.project, { operations: [{ op: 'assign_region', ids: ['a'], region: 'absent' }] }).ok).toBe(false);
    expect(patchModel(f.project, { operations: [{ op: 'assign_region', ids: ['region'], region: 'region' }] }).ok).toBe(false);
    expect(patchModel(f.project, { operations: [{ op: 'resize_region', id: 'region', width: Infinity, height: 700 }] }).ok).toBe(false);
    expect(f.project).toEqual(before);
  });
});
