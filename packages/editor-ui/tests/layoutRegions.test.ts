import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { layoutRegionMembers, layoutRegionOf, layoutRegionOutputs, layoutRegionOutputRegions, layoutRegionRect } from '@tensor-trail/model-core/core/layoutRegions';
import { useEditor } from '../src/editor/store';
import { LayoutRegionMembership, LayoutRegionPanel } from '../src/editor/LayoutRegionPanel';
import { LayoutRegionNode } from '../src/views/LayoutRegionNode';
import { layoutRegions } from '../src/views/layoutRegions';
import { layoutIfRegions } from '../src/views/ifRegionLayout';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

const state = () => useEditor.getState();
beforeEach(() => {
  const a = registry.create('Input', { shape: '2,3', axes: 'B,C' }, 'a');
  const b = registry.create('Output', {}, 'b'); b.position = { x: 1600, y: 100 };
  state().load({ schemaVersion: 1, name: 'region UI', variants: [], graph: {
    nodes: [a, b], edges: [{ id: 'edge', source: 'a', sourcePort: 'out', target: 'b', targetPort: 'in' }],
  } });
  useEditor.setState({ past: [], future: [] });
  state().setNodeSizes({ a: { width: 200, height: 80 }, b: { width: 200, height: 80 } });
});

describe('Layout region editor', () => {
  it('allows layout-region drag handles below nested loop/if frames without reparenting members', () => {
    state().createLayoutRegion(['a', 'b']); const id = state().selected!;
    const graph = state().project.graph;
    const condition = registry.create('IfRegion', { members: `${id},a,b` }, 'condition');
    const loop = registry.create('LoopRegion', { members: 'condition' }, 'loop');
    const nodes: TrailNode[] = [...graph.nodes, condition, loop].map(model => ({
      id: model.id, type: 'module', position: { ...model.position }, measured: { width: 200, height: 80 },
      data: { model, inputs: [], outputs: [], category: '', reverse: false, active: false, dimmed: false, open: () => {} },
    }));
    const before = nodes.map(n => ({ id: n.id, position: n.position, model: n.data.model }));
    const displayed = layoutRegions(layoutIfRegions(nodes));
    const region = displayed.find(n => n.id === id)!;
    expect(region.draggable).toBe(true);
    expect(region.dragHandle).toBe('.layout-region-heading');
    for (const scope of displayed.filter(n => ['condition', 'loop'].includes(n.id))) {
      expect(scope.zIndex).toBeGreaterThan(region.zIndex!);
      expect(scope.style!.pointerEvents).toBe('none');
      expect(scope.draggable).toBe(false);
      expect(scope.dragHandle).toBe('.if-region-heading');
    }
    expect(nodes.map(n => ({ id: n.id, position: n.position, model: n.data.model }))).toEqual(before);
  });
  it('saves notes and output roles, displays edit controls, and restores them with undo/redo', () => {
    state().createLayoutRegion(['a', 'b']); const id = state().selected!;
    state().setParam(id, 'note', '提取多视角视觉特征');
    state().setParam(id, 'outputs', 'b');
    const graph = state().project.graph, region = graph.nodes.find(n => n.id === id)!;
    const panel = renderToStaticMarkup(createElement(LayoutRegionPanel, { graph, node: region }));
    expect(panel).toContain('分区注释'); expect(panel).toContain('分区输出');
    const card = renderToStaticMarkup(createElement(ReactFlowProvider, null,
      createElement(LayoutRegionNode, { model: region, selected: false })));
    expect(card).toContain('layout-region-note'); expect(card).toContain('提取多视角视觉特征');
    state().select('b');
    const member = renderToStaticMarkup(createElement(LayoutRegionMembership, { graph, node: graph.nodes[1] }));
    expect(member).toContain('标记为分区输出');
    state().undo(); expect(layoutRegionOutputs(state().project.graph.nodes.find(n => n.id === id)!)).toEqual([]);
    state().redo(); expect(layoutRegionOutputs(state().project.graph.nodes.find(n => n.id === id)!)).toEqual(['b']);
    state().assignLayoutRegion(['b'], null);
    expect(layoutRegionOutputs(state().project.graph.nodes.find(n => n.id === id)!)).toEqual([]);
  });

  it.each(['ValueVariable', 'ReLU'])('shows a distinct region output badge for %s without changing final-return identity', type => {
    const model = registry.create(type, {}, 'value');
    const props: NodeProps<TrailNode> = { id: 'value', type: 'module', selected: false, dragging: false, zIndex: 0,
      selectable: true, deletable: true, draggable: true, isConnectable: true, positionAbsoluteX: 0, positionAbsoluteY: 0,
      data: { model, inputs: [], outputs: [], category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {},
        regionOutputs: [{ id: 'vision', label: '视觉感知', color: '#38b2ac' }],
        result: { status: 'ok', inputs: {}, outputs: { out: { shape: [2, 3], axes: ['B', 'C'], dtype: 'float32' } }, notes: [], messages: [] } },
    };
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
    expect(html).toContain('data-region-output="vision"'); expect(html).toContain('分区输出 · 视觉感知');
    expect(html).not.toContain('data-final-output');
    if (type === 'ValueVariable') {
      props.data.finalOutput = true;
      const both = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
      expect(both).toContain('data-region-output="vision"'); expect(both).toContain('data-final-output="true"');
    }
  });

  it('renders inherited and local region roles together, while keeping alias membership unchanged', () => {
    const value = registry.create('TensorVariable', {}, 'value');
    const alias = { ...structuredClone(value), id: 'alias', valueSource: 'value' };
    const vision = registry.create('LayoutRegion', { members: 'value', outputs: 'value', color: '#38b2ac' }, 'vision');
    const target = registry.create('LayoutRegion', { members: 'alias', outputs: 'alias', color: '#60a5fa' }, 'target');
    vision.label = '视觉感知'; target.label = '预测';
    const graph = { nodes: [value, alias, vision, target], edges: [] };
    const props: NodeProps<TrailNode> = { id: 'alias', type: 'module', selected: false, dragging: false, zIndex: 0,
      selectable: true, deletable: true, draggable: true, isConnectable: true, positionAbsoluteX: 0, positionAbsoluteY: 0,
      data: { model: alias, inputs: [], outputs: [], category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {},
        regionOutputs: layoutRegionOutputRegions(graph).get(alias.id)!.map(r => ({ id: r.id, label: r.label, color: String(r.params.color) })),
      },
    };
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
    expect(html).toContain('data-region-output="vision"'); expect(html).toContain('data-region-output="target"');
    expect(html).toContain('分区输出 · 视觉感知'); expect(html).toContain('分区输出 · 预测');
    expect(html).toContain('--region-color:#38b2ac'); expect(html).toContain('--region-color:#60a5fa');
    expect(html.match(/class="region-output-badges /g)).toHaveLength(1);
    expect(layoutRegionOf(graph, alias.id)!.id).toBe('target');
    const panel = renderToStaticMarkup(createElement(LayoutRegionMembership, { graph, node: alias }));
    expect(panel).toContain('关联来源的分区输出：'); expect(panel).toContain('视觉感知');
    expect(panel).toContain('随主节点同步');
  });

  it('recomputes inherited roles after toggles, undo/redo and unlinking', () => {
    const graph = structuredClone(state().project.graph);
    graph.nodes.push({ ...structuredClone(graph.nodes[0]), id: 'alias', valueSource: 'a' });
    state().load({ ...state().project, graph });
    state().createLayoutRegion(['a']); const id = state().selected!;
    state().setParam(id, 'outputs', 'a');
    const markers = () => layoutRegionOutputRegions(state().project.graph).get('alias')?.map(r => r.id) ?? [];
    expect(markers()).toEqual([id]);
    state().setParam(id, 'outputs', ''); expect(markers()).toEqual([]);
    state().undo(); expect(markers()).toEqual([id]);
    state().redo(); expect(markers()).toEqual([]);
    state().undo();
    const panel = renderToStaticMarkup(createElement(LayoutRegionMembership, {
      graph: state().project.graph, node: state().project.graph.nodes.find(n => n.id === 'alias')!,
    }));
    expect(panel).toContain('关联分区输出');
    expect(panel).not.toContain('aria-label="标记为分区输出"');
    expect(layoutRegionOf(state().project.graph, 'alias')).toBeUndefined();
    state().unlinkValue('alias'); expect(markers()).toEqual([]);
    state().undo(); expect(markers()).toEqual([id]);
  });

  it('keeps nested layout creation, color edits and removal separate from computation customization', () => {
    const parent = registry.create('Transformer', {}, 'parent');
    state().load({ schemaVersion: 1, name: 'nested', variants: [], graph: { nodes: [parent], edges: [] } });
    state().navigate(['parent']);
    const inlet = parent.subgraph!.nodes.find(n => n.type === 'Inlet')!;
    state().createLayoutRegion([inlet.id]); const id = state().selected!;
    state().setParam(id, 'color', '#6699aa');
    expect(state().project.graph.nodes[0].customized).toBeFalsy();
    state().removeNodes([id]);
    expect(state().project.graph.nodes[0].customized).toBeFalsy();
    state().add('LayoutRegion', undefined, { x: 300, y: 400 });
    expect(state().project.graph.nodes[0].customized).toBeFalsy();
  });

  it('creates, moves, undoes and restores a group as one edit without changing outputs', () => {
    const before = structuredClone(state().project), result = inferProject(before, registry).nodes.b.outputs;
    state().createLayoutRegion(['a', 'b']);
    const grouped = structuredClone(state().project), id = state().selected!;
    expect(layoutRegionMembers(grouped.graph.nodes.find(n => n.id === id)!)).toEqual(['a', 'b']);
    expect(inferProject(grouped, registry).nodes.b.outputs).toEqual(result);
    expect(state().past).toHaveLength(1);
    state().undo(); expect(state().project).toEqual(before);
    state().redo(); expect(state().project).toEqual(grouped);
    const start = state().project.graph.nodes[0].position;
    const region = state().project.graph.nodes.find(n => n.id === id)!;
    state().finishNodeDrag([{ id, position: { x: region.position.x + 300, y: region.position.y + 100 } }]);
    expect(state().project.graph.nodes[0].position).toEqual({ x: start.x + 300, y: start.y + 100 });
    state().undo(); expect(state().project).toEqual(grouped);
  });

  it('clamps drag commits, permits removal from the node property and retains nodes when deleting a frame', () => {
    state().createLayoutRegion(['a']); const id = state().selected!;
    const original = state().project.graph.nodes.find(n => n.id === id)!;
    state().resizeLayoutRegion(id, layoutRegionRect(state().project.graph, original, state().nodeSizes));
    state().finishNodeDrag([{ id: 'a', position: { x: -10000, y: -10000 } }]);
    const region = state().project.graph.nodes.find(n => n.id === id)!;
    expect(state().project.graph.nodes[0].position).toEqual({ x: region.position.x + 24, y: region.position.y + 60 });
    state().assignLayoutRegion(['a'], null); state().move('a', { x: -10000, y: -10000 });
    expect(state().project.graph.nodes[0].position.x).toBe(-10000);
    expect(layoutRegionOf(state().project.graph, 'a')).toBeUndefined();
    state().assignLayoutRegion(['a', 'b'], id); state().removeNodes([id]);
    expect(state().project.graph.nodes.map(n => n.id)).toEqual(['a', 'b']);
    expect(state().project.graph.edges).toHaveLength(1);
  });

  it('commits an auto member drag and frame growth as one undoable edit', () => {
    state().createLayoutRegion(['a', 'b']); const id = state().selected!;
    const before = structuredClone(state().project), history = state().past.length;
    state().finishNodeDrag([{ id: 'a', position: { x: -300, y: -400 } }]);
    const after = structuredClone(state().project), frame = after.graph.nodes.find(n => n.id === id)!;
    expect(after.graph.nodes[0].position).toEqual({ x: -300, y: -400 });
    expect(frame.position).toEqual({ x: -324, y: -460 });
    expect(frame.params.sizing).toBe('auto'); expect(layoutRegionMembers(frame)).toEqual(['a', 'b']);
    expect(after.graph.nodes[1]).toEqual(before.graph.nodes[1]);
    expect(state().past).toHaveLength(history + 1);
    state().undo(); expect(state().project).toEqual(before);
    state().redo(); expect(state().project).toEqual(after);
  });

  it('previews auto growth during a drag without mutating the project or moving other members', () => {
    state().createLayoutRegion(['a', 'b']); const graph = state().project.graph;
    const before = structuredClone(graph), history = state().past.length;
    const nodes: TrailNode[] = graph.nodes.map(model => ({
      id: model.id, type: 'module', position: model.id === 'a' ? { x: -300, y: -400 } : { ...model.position },
      measured: { width: 200, height: 80 },
      data: { model, inputs: [], outputs: [], category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {} },
    }));
    const rendered = layoutRegions(nodes), frame = rendered.find(n => n.data.model.type === 'LayoutRegion')!;
    expect(frame.position).toEqual({ x: -324, y: -460 });
    expect(Number(frame.style!.width)).toBeGreaterThan(Number(frame.data.model.params.width));
    expect(Number(frame.style!.height)).toBeGreaterThan(Number(frame.data.model.params.height));
    expect(rendered.find(n => n.id === 'b')!.position).toEqual(graph.nodes[1].position);
    expect(graph).toEqual(before); expect(state().past).toHaveLength(history);
  });

  it('supports manual sizing, safe shrinking and automatic fitting in undo history', () => {
    state().createLayoutRegion(['a', 'b']); const id = state().selected!;
    const graph = state().project.graph, region = graph.nodes.find(n => n.id === id)!;
    const before = structuredClone(state().project);
    state().resizeLayoutRegion(id, { ...region.position, width: 1200, height: 800 });
    expect(state().project.graph.nodes.find(n => n.id === id)!.params.sizing).toBe('manual');
    state().fitLayoutRegion(id);
    const fitted = state().project.graph.nodes.find(n => n.id === id)!;
    expect(fitted.params.sizing).toBe('auto');
    expect(layoutRegionRect(state().project.graph, fitted, state().nodeSizes).width).toBeLessThan(1200);
    state().undo(); state().undo(); expect(state().project).toEqual(before);
  });

  it('provides region membership in node properties and a dedicated region panel', () => {
    state().createLayoutRegion(['a']); const graph = state().project.graph, node = graph.nodes[0];
    state().select('a');
    const html = renderToStaticMarkup(createElement(LayoutRegionMembership, { graph, node }));
    expect(html).toContain('所属分区'); expect(html).toContain('解除分区归属');
    const frame = graph.nodes.find(n => n.type === 'LayoutRegion')!;
    const panel = renderToStaticMarkup(createElement(LayoutRegionPanel, { graph, node: frame }));
    expect(panel).toContain('自动适配大小'); expect(panel).toContain('分区颜色');
    expect(panel).toContain('搜索分区成员'); expect(panel).toContain('分区宽度');
    const card = renderToStaticMarkup(createElement(ReactFlowProvider, null,
      createElement(LayoutRegionNode, { model: frame, selected: false })));
    expect(card).toContain('layout-region-heading'); expect(card).toContain('--region-color:#38b2ac');
    expect(card).not.toContain('react-flow__handle');
  });

  it('uses actual measurements and respects live resize dimensions without modifying model positions', () => {
    state().createLayoutRegion(['a', 'b']); const graph = state().project.graph;
    const before = structuredClone(graph);
    const nodes: TrailNode[] = graph.nodes.map(model => ({
      id: model.id, type: 'module', position: { ...model.position }, measured: { width: 200, height: 80 },
      data: { model, inputs: [], outputs: [], category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {} },
    }));
    const id = graph.nodes.find(n => n.type === 'LayoutRegion')!.id;
    const frame = layoutRegions(nodes).find(n => n.id === id)!;
    expect(frame.dragHandle).toBe('.layout-region-heading');
    expect(frame.zIndex! + 1000).toBeLessThan(0); // Selection elevation must not cover members.
    expect(frame.style!.pointerEvents).toBe('none');
    expect(frame.style!.height).toBe(200);
    nodes.find(n => n.id === id)!.resizing = true;
    nodes.find(n => n.id === id)!.measured = { width: 1000, height: 900 };
    expect(layoutRegions(nodes).find(n => n.id === id)!.style).toMatchObject({ width: 1000, height: 900 });
    expect(graph).toEqual(before);
  });
});
