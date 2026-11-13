import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ifRegionIssues, ifRegionMembers } from '@tensor-trail/model-core/core/ifRegions';
import type { Graph } from '@tensor-trail/model-core/core/types';
import { clipboardText } from '@tensor-trail/model-core/editing/clipboard';
import { registry } from '@tensor-trail/model-core/modules';
import { IfRegionPanel } from '../src/editor/IfRegionPanel';
import { createModuleClipboard } from '../src/editor/moduleClipboard';
import { useEditor } from '../src/editor/store';
import { IfRegionNode } from '../src/views/IfRegionNode';
import { layoutIfRegions } from '../src/views/ifRegionLayout';
import type { TrailNode } from '../src/views/ModuleNode';

function flow(id: string, type: string, members = ''): TrailNode {
  const model = registry.create(type, type === 'Input' ? {} : { members }, id);
  model.label = id;
  return {
    id,
    type: 'module',
    position: { x: 100, y: 200 },
    measured: { width: 200, height: 80 },
    data: {
      model,
      inputs: [],
      outputs: [],
      category: '',
      reverse: false,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
}

function nodes() {
  const result = [
    flow('a', 'Input'),
    flow('training', 'IfRegion', 'a'),
    flow('interval', 'IfRegion', 'training'),
    flow('loop', 'LoopRegion', 'interval'),
  ];
  result[1].data.model.params.condition = 'self.training';
  result[2].data.model.params.condition = 'interval < 6';
  result[3].data.model.params.statement = 'for interval in range(num_fu_frames)';
  return result;
}

const graph = (): Graph => ({ nodes: nodes().map((n) => n.data.model), edges: [] });

function load() {
  useEditor
    .getState()
    .load({ schemaVersion: 1, name: 'nested scopes', graph: graph(), variants: [] });
  useEditor.setState({ clipboard: null, clipboardOrigin: null, past: [], future: [] });
  return structuredClone(useEditor.getState().project);
}

describe('nested frame layout', () => {
  it.each([false, true])(
    'includes every child heading and minimum width (reverse order: %s)',
    (reverse) => {
      const input = nodes();
      if (reverse) input.reverse();
      const before = JSON.stringify(input);
      const result = layoutIfRegions(input);
      expect(result.map((n) => n.id)).toEqual(input.map((n) => n.id));
      const byId = new Map(result.map((n) => [n.id, n]));
      expect(byId.get('training')).toMatchObject({
        position: { x: 60, y: 106 },
        style: { width: 440, height: 214 },
        zIndex: -1,
      });
      expect(byId.get('interval')).toMatchObject({
        position: { x: 20, y: 12 },
        style: { width: 520, height: 348 },
        zIndex: -2,
      });
      expect(byId.get('loop')).toMatchObject({
        position: { x: -20, y: -82 },
        style: { width: 600, height: 482 },
        zIndex: -3,
        draggable: false,
      });
      expect(byId.get('a')).toBe(input.find((n) => n.id === 'a'));
      expect(JSON.stringify(input)).toBe(before);
      expect(layoutIfRegions(result)).toEqual(result);
    },
  );

  it('updates all ancestor bounds after a leaf moves or is measured, ignoring stale frame sizes', () => {
    const input = nodes();
    const first = layoutIfRegions(input);
    input[0].position.x += 50;
    input[0].position.y += 70;
    input[0].measured = { width: 800, height: 400 };
    for (const node of input.slice(1)) node.measured = { width: 9999, height: 9999 };
    const updated = layoutIfRegions(input);
    expect(updated[3].position).toEqual({
      x: first[3].position.x + 50,
      y: first[3].position.y + 70,
    });
    expect(updated[3].style).toMatchObject({ width: 1040, height: 802 });
    expect(updated[0]).toBe(input[0]);
  });

  it('includes empty child frames at their saved positions without measurement feedback', () => {
    const input = nodes();
    input[1].data.model.params.members = '';
    const result = layoutIfRegions(input);
    expect(result[1]).toMatchObject({
      position: { x: 100, y: 200 },
      style: { width: 440, height: 220 },
      draggable: true,
    });
    expect(result[2]).toMatchObject({
      position: { x: 60, y: 106 },
      style: { width: 520, height: 354 },
      draggable: false,
    });
    expect(result[3]).toMatchObject({
      position: { x: 20, y: 12 },
      style: { width: 600, height: 488 },
      draggable: false,
    });
  });

  it('wraps sibling conditions without changing their flat layout', () => {
    const a = flow('a', 'Input');
    const b = flow('b', 'Input');
    b.position = { x: 700, y: 400 };
    const training = flow('training', 'IfRegion', 'a');
    const interval = flow('interval', 'IfRegion', 'b');
    const loop = flow('loop', 'LoopRegion', 'training,interval');
    const flat = [a, b, training, interval];
    const result = layoutIfRegions([...flat, loop]);
    expect(result.slice(0, 4)).toEqual(layoutIfRegions(flat));
    expect(result[4]).toMatchObject({
      position: { x: 20, y: 12 },
      style: { width: 1120, height: 548 },
      zIndex: -2,
    });
  });

  it.each(['training', 'loop', 'missing'])(
    'terminates on an invalid draft reference to %s',
    (id) => {
      const input = nodes();
      input[1].data.model.params.members = id;
      const result = layoutIfRegions(input);
      expect(result).toHaveLength(input.length);
      for (const node of result.slice(1)) {
        expect(Number.isFinite(node.position.x)).toBe(true);
        expect(Number.isFinite(node.position.y)).toBe(true);
        expect(Number.isFinite(node.style!.width)).toBe(true);
        expect(Number.isFinite(node.style!.height)).toBe(true);
      }
    },
  );

  it('handles a deep scope tree without recursive call-stack growth', () => {
    const input = [flow('leaf', 'Input')];
    for (let i = 0; i < 1999; i++) input.push(flow(`r${i}`, 'IfRegion', i ? `r${i - 1}` : 'leaf'));
    expect(ifRegionIssues({ nodes: input.map((n) => n.data.model), edges: [] })).toEqual([]);
    const result = layoutIfRegions(input);
    expect(result.at(-1)!.zIndex).toBe(-1999);
    expect(result.at(-1)!.position).toEqual({ x: 100 - 40 * 1999, y: 200 - 94 * 1999 });
  });

  it('renders the loop and both condition headings as scope frames', () => {
    const html = nodes()
      .slice(1)
      .map((n) =>
        renderToStaticMarkup(createElement(IfRegionNode, { model: n.data.model, selected: false })),
      )
      .join('');
    expect(html).toContain('for interval in range(num_fu_frames):');
    expect(html).toContain('if interval &lt; 6:');
    expect(html).toContain('if self.training:');
    expect(html).not.toContain('react-flow__handle');
  });
});

describe('nested scope editing', () => {
  it('offers unowned child frames but disables ancestors and already-owned descendants', () => {
    const g = graph();
    const free = registry.create('LoopRegion', {}, 'free');
    free.label = 'free';
    g.nodes.push(free);
    const panel = (id: string) =>
      renderToStaticMarkup(
        createElement(IfRegionPanel, { graph: g, node: g.nodes.find((n) => n.id === id)! }),
      );
    const checkbox = (html: string, id: string) =>
      html
        .split('<label>')
        .find((part) => part.includes(`<span>${id}</span>`))!
        .split('</label>')[0];
    const outer = panel('loop');
    expect(checkbox(outer, 'interval')).toContain('checked');
    expect(checkbox(outer, 'interval')).not.toContain('disabled');
    expect(checkbox(outer, 'training')).toContain('disabled');
    expect(checkbox(outer, 'a')).toContain('disabled');
    expect(checkbox(outer, 'free')).not.toContain('disabled');
    const inner = panel('training');
    expect(checkbox(inner, 'loop')).toContain('disabled');
    expect(checkbox(inner, 'interval')).toContain('disabled');
    expect(checkbox(inner, 'a')).not.toContain('disabled');
    expect(inner).not.toContain('<span>training</span>');
  });

  it('rejects an invalid membership edit without history and undoes a valid one', () => {
    const before = load();
    expect(useEditor.getState().setParam('training', 'members', 'a,loop')).toBe(false);
    expect(useEditor.getState().setParam('loop', 'members', 'interval,a')).toBe(false);
    expect(useEditor.getState().project).toEqual(before);
    expect(useEditor.getState().past).toHaveLength(0);
    expect(useEditor.getState().setParam('interval', 'members', '')).toBe(true);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(
      useEditor.getState().project.graph.nodes.find((n) => n.id === 'interval')!.params.members,
    ).toBe('');
  });

  it('copies and pastes complete nested scopes locally and through native clipboard text', () => {
    const before = load();
    useEditor.getState().select('loop');
    const clipboard = createModuleClipboard();
    const copied = clipboard.copy()!;
    expect(copied.nodes).toHaveLength(4);
    expect(clipboard.paste()).toBe(true);
    const pasted = useEditor.getState().project;
    expect(pasted.graph.nodes).toHaveLength(8);
    expect(ifRegionIssues(pasted.graph)).toEqual([]);
    const newIds = new Set(pasted.graph.nodes.slice(4).map((n) => n.id));
    for (const frame of pasted.graph.nodes.slice(5))
      expect(ifRegionMembers(frame).every((id) => newIds.has(id))).toBe(true);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(pasted);
    load();
    expect(createModuleClipboard().pasteText(clipboardText(copied))).toBe(true);
    expect(useEditor.getState().project.graph.nodes).toHaveLength(8);
    expect(ifRegionIssues(useEditor.getState().project.graph)).toEqual([]);
  });

  it('deletes multiple inner frames without losing the parent scope and supports undo/redo', () => {
    const before = load();
    useEditor.getState().removeNodes(['interval', 'training']);
    const after = useEditor.getState().project;
    expect(after.graph.nodes.map((n) => n.id)).toEqual(['a', 'loop']);
    expect(after.graph.nodes[1].params.members).toBe('a');
    expect(ifRegionIssues(after.graph)).toEqual([]);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
});
