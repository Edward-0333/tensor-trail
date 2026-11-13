import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { ifRegionIssues, ifRegionMembers, isCodeRegion } from '../src/core/ifRegions';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Graph } from '../src/core/types';
import {
  clipboardText,
  copySelection,
  duplicateSelection,
  readClipboard,
} from '../src/editing/clipboard';
import { removeValueNodes } from '../src/editing/valueLinkEditing';
import { registry } from '../src/modules';

function recipe(): Recipe {
  return {
    name: 'nested forecast scopes',
    nodes: [
      { id: 'x', type: 'Input', params: { shape: '2,1,3', axes: '' } },
      { id: 'cast', type: 'Cast', params: { dtype: 'float32' }, position: { x: 100, y: 200 } },
      { id: 'sum', type: 'ReduceSum', params: { dim: -1, keepdim: true } },
      { id: 'out', type: 'Output' },
      { id: 'training', type: 'IfRegion', params: { condition: 'self.training', members: 'sum' } },
      {
        id: 'interval',
        type: 'IfRegion',
        params: { condition: 'interval < 6', members: 'cast,training' },
      },
      {
        id: 'loop',
        type: 'LoopRegion',
        params: { statement: 'for interval in range(num_fu_frames)', members: 'interval' },
      },
    ],
    connections: [
      ['x.out', 'cast.in'],
      ['cast.out', 'sum.in'],
      ['sum.out', 'out.in'],
    ],
  };
}

function project() {
  const built = buildModel(recipe());
  expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
  return built.project!;
}

const members = (graph: Graph, id: string) =>
  ifRegionMembers(graph.nodes.find((n) => n.id === id)!);

describe('nested visual code scopes', () => {
  it.each([false, true])(
    'round-trips a strict tree without changing dataflow (reverse order: %s)',
    (reverse) => {
      const r = recipe();
      if (reverse) r.nodes.reverse();
      const built = buildModel(r);
      expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
      const p = built.project!;
      expect(built.outputs.out).toMatchObject({ shape: [2, 1, 1] });
      expect(ifRegionIssues(p.graph)).toEqual([]);
      expect(parseProject(serializeProject(p), registry)).toEqual(p);
      const flat = buildModel({ ...r, nodes: r.nodes.filter((n) => !isCodeRegion(n)) });
      expect(flat.outputs).toEqual(built.outputs);
      expect(members(p.graph, 'loop')).toEqual(['interval']);
      expect(members(p.graph, 'interval')).toEqual(['cast', 'training']);
    },
  );

  it('patches sibling conditions into a loop without moving code or manual wire routes', () => {
    const p = project();
    p.graph.edges[0].waypoints = [
      { x: 70, y: 80 },
      { x: 90, y: 80 },
    ];
    p.graph.edges[0].labelPosition = 0.3;
    const before = structuredClone(p);
    const result = patchModel(
      p,
      {
        operations: [
          { op: 'set_params', id: 'interval', params: { members: 'cast' } },
          { op: 'set_params', id: 'loop', params: { members: 'interval,training' } },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.edges).toEqual(p.graph.edges);
    expect(result.project!.graph.nodes.map((n) => n.position)).toEqual(
      p.graph.nodes.map((n) => n.position),
    );
    expect(p).toEqual(before);
  });

  it.each([
    ['loop', 'loop', '自身'],
    ['interval', 'cast,loop', '循环'],
    ['training', 'sum,loop', '循环'],
    ['training', 'sum,cast', '直接父'],
    ['loop', 'interval,sum', '直接父'],
    ['loop', 'interval,training', '直接父'],
    ['loop', 'interval,interval', '唯一'],
    ['loop', 'missing', '不存在'],
  ])('rejects invalid %s membership %s atomically', (id, value, message) => {
    const p = project();
    const before = structuredClone(p);
    const bad = structuredClone(p);
    bad.graph.nodes.find((n) => n.id === id)!.params.members = value;
    expect(ifRegionIssues(bad.graph).join(' ')).toContain(message);
    expect(validateModel(bad).ok).toBe(false);
    expect(() => readClipboard(clipboardText(bad.graph))).toThrow();
    expect(
      patchModel(
        p,
        { operations: [{ op: 'set_params', id, params: { members: value } }] },
        { mode: 'draft', layout: 'preserve' },
      ).ok,
    ).toBe(false);
    expect(p).toEqual(before);
  });

  it('does not resolve membership into a callable subgraph', () => {
    const graph = project().graph;
    const child = registry.create('Submodule', {}, 'call');
    child.subgraph = { nodes: [registry.create('Input', {}, 'hidden')], edges: [] };
    graph.nodes.push(child);
    graph.nodes.find((n) => n.id === 'training')!.params.members = 'hidden';
    expect(ifRegionIssues(graph).join(' ')).toContain('不存在');
  });

  it.each([false, true])(
    'copies all descendants once and remaps only direct parents (reverse: %s)',
    (reverse) => {
      const p = project();
      if (reverse) p.graph.nodes.reverse();
      const before = structuredClone(p);
      const fragment = copySelection(p.graph, ['loop', 'interval', 'sum']);
      expect(new Set(fragment.nodes.map((n) => n.id))).toEqual(
        new Set(['loop', 'interval', 'training', 'cast', 'sum']),
      );
      expect(fragment.nodes).toHaveLength(5);
      expect(fragment.edges).toHaveLength(1);
      const pasted = duplicateSelection(readClipboard(clipboardText(fragment))!, 32);
      const ids = new Map(fragment.nodes.map((n, i) => [n.id, pasted.nodes[i].id]));
      for (const node of fragment.nodes) {
        const copy = pasted.nodes.find((n) => n.id === ids.get(node.id))!;
        expect(copy.position).toEqual({ x: node.position.x + 32, y: node.position.y + 32 });
        if (isCodeRegion(node))
          expect(ifRegionMembers(copy)).toEqual(ifRegionMembers(node).map((id) => ids.get(id)));
      }
      expect(ifRegionIssues({ nodes: [...p.graph.nodes, ...pasted.nodes], edges: [] })).toEqual([]);
      expect(p).toEqual(before);
    },
  );

  it('copies a child subtree or a leaf without copying its ancestors', () => {
    const graph = project().graph;
    expect(copySelection(graph, ['training']).nodes.map((n) => n.id)).toEqual(['sum', 'training']);
    expect(copySelection(graph, ['sum']).nodes.map((n) => n.id)).toEqual(['sum']);
    expect(readClipboard(clipboardText(copySelection(graph, ['training'])))!.nodes).toHaveLength(2);
  });

  it('terminates descendant copying for a cyclic invalid draft', () => {
    const graph = project().graph;
    graph.nodes.find((n) => n.id === 'training')!.params.members = 'sum,loop';
    const fragment = copySelection(graph, ['loop']);
    expect(fragment.nodes).toHaveLength(5);
    expect(ifRegionIssues(fragment).join(' ')).toContain('循环');
  });

  it('deleting an inner frame promotes its direct children, not its grandchildren', () => {
    const p = project();
    const before = structuredClone(p);
    const removed = patchModel(p, { operations: [{ op: 'remove_node', id: 'interval' }] });
    expect(removed.ok, JSON.stringify(removed.diagnostics)).toBe(true);
    const graph = removed.project!.graph;
    expect(members(graph, 'loop')).toEqual(['cast', 'training']);
    expect(members(graph, 'training')).toEqual(['sum']);
    expect(graph.nodes).toHaveLength(p.graph.nodes.length - 1);
    expect(graph.edges).toEqual(p.graph.edges);
    expect(p).toEqual(before);
  });

  it('deleting the outer frame keeps the child scopes and their code', () => {
    const p = project();
    const removed = patchModel(p, { operations: [{ op: 'remove_node', id: 'loop' }] });
    expect(removed.ok).toBe(true);
    expect(removed.project!.graph.nodes).toEqual(p.graph.nodes.filter((n) => n.id !== 'loop'));
    expect(removed.project!.graph.edges).toEqual(p.graph.edges);
  });

  it.each([
    ['interval', 'training'],
    ['training', 'interval'],
    ['interval', 'training', 'cast'],
  ])('promotes only surviving code through simultaneous frame deletion: %j', (...ids) => {
    const graph = project().graph;
    removeValueNodes(graph, [], ids);
    expect(members(graph, 'loop')).toEqual(ids.includes('cast') ? ['sum'] : ['cast', 'sum']);
    expect(ifRegionIssues(graph)).toEqual([]);
  });

  it('prunes a deleted leaf only from its direct frame', () => {
    const graph = project().graph;
    removeValueNodes(graph, [], ['sum']);
    expect(members(graph, 'training')).toEqual([]);
    expect(members(graph, 'interval')).toEqual(['cast', 'training']);
    expect(members(graph, 'loop')).toEqual(['interval']);
    expect(ifRegionIssues(graph)).toEqual([]);
  });
});
