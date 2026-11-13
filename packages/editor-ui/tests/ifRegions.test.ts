import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, it, expect } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules';
import { buildModel } from '@tensor-trail/model-core/agent';
import type { TrailNode } from '../src/views/ModuleNode';
import { IfRegionNode } from '../src/views/IfRegionNode';
import { layoutIfRegions } from '../src/views/ifRegionLayout';
import { useEditor } from '../src/editor/store';

function flow(id: string, type: string, x: number, y: number, width = 200, height = 80): TrailNode {
  const model = registry.create(type, type === 'IfRegion' ? { members: 'a,b' } : {}, id);
  return {
    id,
    type: 'module',
    position: { x, y },
    measured: { width, height },
    data: {
      model,
      inputs: [],
      outputs: [],
      category: '分支运算',
      reverse: false,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
}
describe('if range-frame UI', () => {
  it('renders a loop scope as a dashed frame with a for heading, not a nested callable', () => {
    const model = registry.create('LoopRegion', { statement: 'for interval in range(num_fu_frames)', members: 'a,b' });
    const html = renderToStaticMarkup(createElement(IfRegionNode, { model, selected: true }));
    expect(html).toContain('for interval in range(num_fu_frames):');
    expect(html).not.toContain('if for');
    expect(html).toContain('if-region-node selected');
    expect(html).toContain('每轮执行');
    expect(html).toContain('不模拟迭代');
    expect(html).not.toContain('react-flow__handle');
    expect(html).not.toContain('进入 forward');
  });
  it('fits a loop frame to the caller-level members without relocating them', () => {
    const frame = flow('scope', 'LoopRegion', 0, 0);
    frame.data.model.params.members = 'a,b';
    const nodes = [flow('a', 'Input', 100, 200), flow('b', 'Cast', 500, 260, 300, 120), frame];
    const shown = layoutIfRegions(nodes);
    expect(shown[2].position).toEqual({ x: 60, y: 106 });
    expect(shown[2].style).toMatchObject({ width: 780, height: 314 });
    expect(shown.slice(0, 2)).toEqual(nodes.slice(0, 2));
    expect(frame.position).toEqual({ x: 0, y: 0 });
  });
  it('shows a condition heading, static scope note and no Tensor or diamond', () => {
    const model = registry.create('IfRegion', { condition: 'self.pretrain', members: 'a,b' });
    const html = renderToStaticMarkup(createElement(IfRegionNode, { model, selected: true }));
    expect(html).toContain('if self.pretrain:');
    expect(html).toContain('if-region-node selected');
    expect(html).toContain('未执行条件判断');
    expect(html).not.toContain('react-flow__handle');
    expect(html).not.toContain('tensor-grad-badge');
    expect(html).not.toContain('condition-node');
  });
  it('auto-wraps actual node bounds without moving members or their saved positions', () => {
    const nodes = [
      flow('a', 'Input', 100, 200),
      flow('b', 'Cast', 500, 260, 300, 120),
      flow('scope', 'IfRegion', 0, 0),
    ];
    const before = structuredClone(nodes.map((n) => ({ id: n.id, position: n.position })));
    const displayed = layoutIfRegions(nodes),
      frame = displayed[2];
    expect(frame.position).toEqual({ x: 60, y: 106 });
    expect(frame.style).toMatchObject({ width: 780, height: 314 });
    expect(frame.zIndex).toBe(-1);
    expect(frame.draggable).toBe(false);
    expect(displayed.slice(0, 2)).toEqual(nodes.slice(0, 2));
    expect(nodes.map((n) => ({ id: n.id, position: n.position }))).toEqual(before);
    nodes[1].position.x += 100;
    expect(layoutIfRegions(nodes)[2].style?.width).toBe(880);
  });
  it('keeps an empty frame selectable and placeable', () => {
    const node = flow('scope', 'IfRegion', 100, 200);
    node.data.model.params.members = '';
    const frame = layoutIfRegions([node])[0];
    expect(frame.draggable).toBe(true);
    expect(frame.position).toEqual({ x: 100, y: 200 });
    expect(frame.style).toMatchObject({ width: 440, height: 220 });
  });
  it('undoes member edits without changing the surrounding graph', () => {
    const built = buildModel(
      {
        name: 'frame edit',
        nodes: [
          { id: 'x', type: 'Input' },
          { id: 'scope', type: 'IfRegion' },
        ],
        connections: [],
      },
      { mode: 'draft' },
    );
    useEditor.getState().load(built.project!);
    const before = structuredClone(useEditor.getState().project);
    expect(useEditor.getState().setParam('scope', 'members', 'x')).toBe(true);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(
      useEditor.getState().project.graph.nodes.find((n) => n.id === 'scope')!.params.members,
    ).toBe('x');
  });
});
