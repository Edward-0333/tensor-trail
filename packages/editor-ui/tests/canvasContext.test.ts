import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { layoutRegionOf } from '@tensor-trail/model-core/core/layoutRegions';
import { contextNodeSelection, contextTargets, menuPosition } from '../src/editor/canvasContext';
import { useEditor } from '../src/editor/store';
import { createModuleClipboard } from '../src/editor/moduleClipboard';

const state = () => useEditor.getState();
beforeEach(() => {
  const a = registry.create('Input', { shape: '2,3', axes: 'B,C' }, 'a');
  const b = registry.create('ReLU', {}, 'b'); b.position = { x: 500, y: 100 };
  const c = registry.create('Output', {}, 'c'); c.position = { x: 900, y: 100 };
  const region = registry.create('LayoutRegion', {}, 'region'); region.position = { x: 1400, y: 500 };
  state().load({ schemaVersion: 1, name: 'menu', variants: [], graph: { nodes: [a, b, c, region], edges: [
    { id: 'ab', source: 'a', sourcePort: 'out', target: 'b', targetPort: 'in', waypoints: [{ x: 300, y: 80 }] },
    { id: 'bc', source: 'b', sourcePort: 'out', target: 'c', targetPort: 'in' },
  ] } });
  useEditor.setState({ selected: null, selectedIds: [], past: [], future: [], clipboard: null, clipboardOrigin: null });
});

describe('Canvas context actions', () => {
  it('keeps selected groups, replaces selection on an unselected node, and preserves selection on the pane', () => {
    const graph = state().project.graph;
    expect(contextNodeSelection(graph, ['a', 'b'], 'b')).toEqual(['a', 'b']);
    expect(contextNodeSelection(graph, ['a', 'b'], 'c')).toEqual(['c']);
    expect(contextNodeSelection(graph, ['a', 'b'])).toEqual(['a', 'b']);
    expect(contextNodeSelection(graph, ['a', 'a', 'missing'])).toEqual(['a']);
    expect(contextNodeSelection(graph, [], 'a')).toEqual(['a']);
  });

  it('excludes layout/code frames from region membership and protects real function boundaries from deletion', () => {
    const graph = state().project.graph;
    graph.nodes.push(registry.create('IfRegion', {}, 'if'), registry.create('Inlet', {}, 'in'));
    graph.nodes.push({ ...structuredClone(graph.nodes.at(-1)!), id: 'in_copy', valueSource: 'in' });
    const targets = contextTargets(graph, ['a', 'region', 'if', 'in', 'in_copy']);
    expect(targets.regionMembers).toEqual(['a', 'in', 'in_copy']);
    expect(targets.deletable).toEqual(['a', 'region', 'if', 'in_copy']);
    expect(targets.assigned).toEqual([]);
  });

  it('adds a selection to a region as one undoable edit without changing values or input count', () => {
    const before = structuredClone(state().project), outputs = inferProject(before, registry).outputs;
    const ids = contextTargets(before.graph, ['a', 'b']).regionMembers;
    state().assignLayoutRegion(ids, 'region');
    expect(state().project.graph.nodes.find(n => n.id === 'region')!.params.members).toBe('a,b');
    expect(inferProject(state().project, registry).outputs).toEqual(outputs);
    expect(state().past).toHaveLength(1);
    state().undo(); expect(state().project).toEqual(before);
    state().redo(); expect(layoutRegionOf(state().project.graph, 'b')!.id).toBe('region');
  });

  it('creates a region with the selected modules, transfers membership, and undoes in one step', () => {
    state().assignLayoutRegion(['a'], 'region');
    const before = structuredClone(state().project), count = state().past.length;
    state().createLayoutRegion(contextTargets(before.graph, ['a', 'b', 'region']).regionMembers);
    const added = state().project.graph.nodes.find(n => n.id === state().selected)!;
    expect(added.params.members).toBe('a,b');
    expect(state().project.graph.nodes.find(n => n.id === 'region')!.params.members).toBe('');
    expect(state().past).toHaveLength(count + 1);
    state().undo(); expect(state().project).toEqual(before);
  });

  it('pastes at the context position with relative positions and manual routes intact, in one edit', () => {
    state().selectMany(['a', 'b']); const clipboard = createModuleClipboard();
    const fragment = clipboard.copy()!, before = structuredClone(state().project);
    expect(clipboard.paste({ x: 2000, y: 600 })).toBe(true);
    const copied = state().project.graph.nodes.filter(n => state().selectedIds.includes(n.id));
    expect(copied[0].position).toEqual({ x: 2000, y: 600 });
    expect(copied[1].position).toEqual({ x: 2500, y: 700 });
    const edge = state().project.graph.edges.find(e => e.source === copied[0].id)!;
    expect(edge.waypoints).toEqual([{ x: 2300, y: 680 }]);
    expect(state().clipboard).toEqual(fragment);
    expect(state().past).toHaveLength(1);
    state().undo(); expect(state().project).toEqual(before);
  });

  it('context pasting uses the usual linked-local-parameter policy, rejects invalid coordinates', () => {
    const source = registry.create('IntegerParameter', { value: 2 }, 'source');
    const node = registry.create('LocalParameter', {}, 'size');
    state().load({ schemaVersion: 1, name: 'parameters', variants: [], graph: { nodes: [source, node], edges: [
      { id: 'value', source: 'source', sourcePort: 'out', target: 'size', targetPort: 'in' },
    ] } });
    state().select(node.id); const clipboard = createModuleClipboard(); clipboard.copy();
    expect(clipboard.paste({ x: 100, y: 300 })).toBe(true);
    expect(state().project.graph.nodes.at(-1)!.valueSource).toBe('size');
    const before = state().project;
    expect(clipboard.paste({ x: NaN, y: 100 })).toBe(false);
    expect(state().project).toBe(before);
  });

  it('clamps menus to the viewport, including narrow screens and oversized submenus', () => {
    expect(menuPosition({ x: 990, y: 790 }, { width: 268, height: 400 }, { width: 1000, height: 800 }))
      .toEqual({ x: 724, y: 392 });
    expect(menuPosition({ x: -10, y: 10 }, { width: 268, height: 900 }, { width: 300, height: 500 }))
      .toEqual({ x: 8, y: 8 });
  });
});
