import { readFileSync } from 'node:fs';
import { parseHTML } from 'linkedom';
import { act, createElement as h, useMemo, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Position, ReactFlowProvider } from '@xyflow/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { projectCanvasGraph } from '@tensor-trail/model-core/core/canvasProjection';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import type { PortSide } from '@tensor-trail/model-core/core/types';
import { resolveGraphTensorTones } from '@tensor-trail/model-core/core/tensorGeometry';
import { fitLayoutRegion, moveLayoutNodes } from '@tensor-trail/model-core/core/layoutRegions';
import { layoutRegions, regionDragPositions } from '../src/views/layoutRegions';
import { layoutIfRegions } from '../src/views/ifRegionLayout';
import { registry } from '@tensor-trail/model-core/modules';
import { Canvas } from '../src/editor/Canvas';
import { Inspector } from '../src/editor/Inspector';
import { createModuleClipboard } from '../src/editor/moduleClipboard';
import { useEditor } from '../src/editor/store';
import { createViewportMemory } from '../src/editor/viewportMemory';
import { EdgeJunctions } from '../src/views/EdgeJunctions';
import { EditableEdge, type TrailEdge } from '../src/views/EditableEdge';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

// The DOM fixture has no layout engine. Supply measured endpoints while retaining Canvas's
// production node/edge props, route stores, portal events and Inspector selection.
const remeasure = vi.hoisted(() => vi.fn());
vi.mock('@xyflow/react', async (importOriginal) => {
  const flow = await importOriginal<typeof import('@xyflow/react')>();
  const { createPortal } = await import('react-dom');
  return {
    ...flow,
    useUpdateNodeInternals: () => remeasure,
    EdgeLabelRenderer: ({ children }: { children: ReactNode }) =>
      createPortal(children, document.querySelector('#labels')!),
    ReactFlow: ({ nodes, edges }: { nodes: TrailNode[]; edges: TrailEdge[] }) =>
      h(
        'div',
        null,
        nodes.map(renderModule),
        h(
          'svg',
          null,
          edges.map((edge, i) =>
            h(EditableEdge, {
              key: edge.id,
              ...edge,
              type: 'editable',
              sourceX: 0,
              sourceY: 0,
              targetX: 300,
              targetY: i * 40,
              sourcePosition: Position.Right,
              targetPosition: Position.Left,
              sourceHandleId: edge.sourceHandle,
              targetHandleId: edge.targetHandle,
              markerStart: undefined,
              markerEnd: 'url(#arrow)',
            }),
          ),
        ),
        h(EdgeJunctions),
      ),
  };
});

function renderModule(node: TrailNode) {
  return h(ModuleNode, {
    key: node.id, ...node, selected: !!node.selected,
    zIndex: 0, dragging: false, draggable: true, deletable: true, selectable: true, isConnectable: true,
    positionAbsoluteX: node.position.x, positionAbsoluteY: node.position.y,
  });
}

const project = buildModel({
  name: 'Compact flow',
  nodes: [
    { id: 'query', type: 'Input', label: 'query · features', params: { shape: '2,2,4' } },
    {
      id: 'linear',
      type: 'Linear',
      presentation: 'edge',
      params: { in_features: 4, out_features: 4 },
    },
    { id: 'reshape', type: 'Reshape', presentation: 'edge', params: { shape: '4,4' } },
    { id: 'values', type: 'Output' },
    { id: 'other', type: 'Input', params: { shape: '2,2,4' } },
    { id: 'cat', type: 'Concat', presentation: 'compact', params: { dim: -1 } },
    { id: 'sum', type: 'Add', presentation: 'compact' },
    { id: 'joined', type: 'Output' },
  ],
  connections: [
    ['query.out', 'linear.in'],
    ['linear.out', 'reshape.in'],
    ['reshape.out', 'values.in'],
    ['other.out', 'cat.a'],
    ['query.out', 'cat.b'],
    ['cat.out', 'sum.a'],
    ['cat.out', 'sum.b'],
    ['sum.out', 'joined.in'],
  ],
}).project!;
const anchor = project.graph.edges.find((e) => e.source === 'query' && e.target === 'cat')!;
anchor.presentation = 'anchor';
const clipboard = createModuleClipboard(),
  viewports = createViewportMemory();
const state = useEditor.getState;
function Workbench() {
  const p = useEditor((s) => s.project);
  const result = useMemo(() => inferProject(p, registry), [p]);
  const graphTones = useMemo(() => resolveGraphTensorTones(p.graph, result), [p, result]);
  const projection = useMemo(
    () => projectCanvasGraph(p.graph, result, graphTones),
    [p, result, graphTones],
  );
  return h(
    ReactFlowProvider,
    null,
    h(Canvas, {
      graph: p.graph,
      graphTones,
      projection,
      referenceMaxDim: 8,
      result,
      step: -1,
      theme: 'dark',
      clipboard,
      viewports,
    }),
    h(Inspector, { graph: p.graph, graphTones, result, hiddenNodeIds: projection.hiddenNodeIds }),
  );
}

describe('opt-in compact canvas', () => {
  let root: Root;
  beforeEach(async () => {
    const { window } = parseHTML('<html><body><main></main><div id="labels"></div></body></html>');
    vi.stubGlobal('window', window);
    vi.stubGlobal('document', window.document);
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.stubGlobal('requestAnimationFrame', (fn: FrameRequestCallback) =>
      setTimeout(() => fn(0), 0),
    );
    vi.stubGlobal('cancelAnimationFrame', clearTimeout);
    state().load(project);
    root = createRoot(document.querySelector('main')!);
    await act(() => root.render(h(Workbench)));
  });
  afterEach(async () => {
    await act(() => root.unmount());
    vi.unstubAllGlobals();
  });
  const card = (id: string) => document.querySelector<HTMLElement>(`[data-testid="node-${id}"]`);
  const click = async (el: Element) => act(() => (el as HTMLButtonElement).click());
  const toggle = (index: number) =>
    click(document.querySelectorAll('.presentation-options button')[index]);
  async function chooseSelect(label: string, value: string) {
    const select = document.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;
    select.querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.selected = true;
    await act(() => { select.dispatchEvent(new window.Event('change', { bubbles: true })); });
  }
  const chooseStyle = async (style: string) => {
    const select = document.querySelector<HTMLSelectElement>('.edge-settings select')!;
    select.querySelector<HTMLOptionElement>(`option[value="${style}"]`)!.selected = true;
    await act(() => {
      select.dispatchEvent(new window.Event('change', { bubbles: true }));
    });
  };

  it.each(['Identity', 'ForwardInput'])('preserves a Unicode %s name on blur without adding an undo entry', async type => {
    await act(() => state().add(type));
    const id = state().selected!;
    const name = 'A' + '𠮷'.repeat(100);
    const field = () => document.querySelector<HTMLInputElement | HTMLTextAreaElement>('[aria-label="模块名称"]')!;
    expect(field().hasAttribute('maxlength')).toBe(false);
    field().value = name;
    await act(() => { field().dispatchEvent(new window.Event('focusout', { bubbles: true })); });
    expect(state().project.graph.nodes.find(n => n.id === id)!.label).toBe(name);
    const { project, past } = state();
    await act(() => { field().dispatchEvent(new window.Event('focusout', { bubbles: true })); });
    expect(state().project).toBe(project);
    expect(state().past).toBe(past);
  });

  it('renders compact junction symbols, dimension and accessible handles without the card chrome', async () => {
    for (const [id, symbol] of [
      ['sum', '+'],
      ['cat', '‖'],
    ]) {
      const node = card(id)!;
      expect(node.classList.contains('compact-junction-node')).toBe(true);
      expect(node.textContent).toContain(symbol);
      expect(node.querySelectorAll('.react-flow__handle[aria-label]')).toHaveLength(3);
      expect(node.querySelector('.node-axis-labels,.node-port-label,.module-node')).toBeNull();
    }
    expect(card('cat')!.querySelector('.compact-junction-sub')?.textContent).toBe('dim=-1');
    await act(() => state().select('sum'));
    await toggle(0);
    expect(card('sum')!.classList.contains('compact-junction-node')).toBe(false);
    await toggle(1);
    expect(card('sum')!.classList.contains('compact-junction-node')).toBe(true);
    await act(() => state().undo());
    expect(card('sum')!.classList.contains('compact-junction-node')).toBe(false);
    await act(() => state().redo());
    expect(card('sum')!.classList.contains('compact-junction-node')).toBe(true);
  });

  it('selects and edits real pipeline steps, and restores their cards with Inspector and undo/redo', async () => {
    const steps = () => [...document.querySelectorAll('.edge-op-step')];
    expect(steps().map((s) => s.textContent)).toEqual(['Linear 4→4', 'Reshape(4,4)']);
    expect(document.querySelector('.edge-op-pipeline small')?.textContent).toContain('[4, 4]');
    expect(card('linear')).toBeNull();
    expect(card('reshape')).toBeNull();
    await click(steps()[0]);
    expect(state().selected).toBe('linear');
    expect(steps()[0].getAttribute('aria-pressed')).toBe('true');
    expect(document.querySelector('.inspector-title')?.textContent).toContain('Linear');
    expect(document.querySelector('.node-presentation-settings small')?.textContent).toContain(
      '当前内嵌',
    );
    expect(document.querySelector('input[aria-label="输出特征 (out_features)"]')).toBeTruthy();
    await toggle(1);
    expect(card('linear')!.classList.contains('compact-layer-node')).toBe(true);
    expect(card('linear')!.querySelector('.module-glyph')).toBeTruthy();
    expect(card('linear')!.querySelector('.compact-layer-spec')?.textContent).toBe('Linear 4→4');
    await act(() => state().undo());
    expect(card('linear')).toBeNull();
    await act(() => {
      state().setParam('linear', 'out_features', 8);
    });
    expect(steps()[0].textContent).toBe('Linear 4→8');
    expect(steps()[1].getAttribute('data-status')).toBe('error');
    await act(() => state().undo());
    await click(steps()[1]);
    expect(state().selected).toBe('reshape');
    await toggle(0);
    expect(card('reshape')).toBeTruthy();
    expect(steps().map((s) => s.textContent)).toEqual(['Linear 4→4']);
    await act(() => state().undo());
    expect(card('reshape')).toBeNull();
    await act(() => state().redo());
    expect(card('reshape')).toBeTruthy();
  });

  it('renders projection factories, source code and the maximum junction with live diagnostics', async () => {
    const built = buildModel({
      name: 'Depth comparison',
      nodes: [
        { id: 'z', type: 'Linspace', params: { start: '-2', end: '6', steps: 4 } },
        { id: 'other', type: 'Input', params: { axes: 'depth', shape: 'D' } },
        { id: 'ones', type: 'OnesLike' }, { id: 'zeros', type: 'ZerosLike' },
        { id: 'max', type: 'Maximum' }, { id: 'out', type: 'Output' },
      ],
      connections: [['z.out', 'max.a'], ['other.out', 'ones.in'], ['ones.out', 'zeros.in'],
        ['zeros.out', 'max.b'], ['max.out', 'out.in']],
    });
    expect(built.ok).toBe(true);
    await act(() => { state().load(built.project!); });
    for (const [id, label, glyph, spec] of [
      ['z', 'Linspace', '等距采样', 'linspace(-2, 6, 4)'],
      ['ones', 'OnesLike', '填充 1', 'ones_like'],
      ['zeros', 'ZerosLike', '填充 0', 'zeros_like'],
      ['max', 'Maximum', '取最大值', 'max'],
    ]) {
      expect(card(id)!.querySelector('.module-glyph title')?.textContent).toContain(glyph);
      await act(() => state().select(id));
      expect(document.querySelector('.source-code-help pre')?.textContent).toContain('out = torch.');
      await toggle(1);
      const node = card(id)!;
      if (label === 'Maximum') {
        expect(node.classList.contains('compact-junction-node')).toBe(true);
        expect(node.textContent).toContain('max');
        expect(node.querySelectorAll('.react-flow__handle')).toHaveLength(3);
        expect(node.querySelector('[aria-label="警告"]')).toBeTruthy();
        expect(node.title).toContain('D 为 1 或 4');
        expect(node.title).toContain('输出 [4]');
      } else {
        expect(node.querySelector('.module-glyph title')?.textContent).toContain(glyph);
        expect(node.querySelector('.compact-layer-spec')?.textContent).toBe(spec);
      }
    }
    await act(() => state().select('z'));
    const axisField = () => document.querySelector<HTMLInputElement>('input[aria-label="轴名 (axis)"]')!;
    const axis = '𠮷'.repeat(64);
    expect(axisField().hasAttribute('maxlength')).toBe(false);
    await act(() => { expect(state().setParam('z', 'axis', axis)).toBe(true); });
    expect(axisField().value).toBe(axis);
    await act(() => { expect(state().setParam('z', 'axis', axis + '𠮷')).toBe(false); });
    expect(axisField().value).toBe(axis);
    await act(() => { state().setParam('z', 'steps', 8); });
    expect(card('z')!.textContent).toContain('linspace(-2, 6, 8)');
    await act(() => { state().setParam('z', 'end', 'NaN'); });
    expect(card('z')!.textContent).toContain('linspace(?)');
    expect(card('z')!.title).toContain('end');
    expect(card('max')!.title).toContain('暂无有效输出');
    expect(card('max')!.title).not.toContain('[D]');
    const source = state().project.graph.edges.find(e => e.target === 'ones')!;
    await act(() => { state().removeEdges([source.id]); });
    for (const id of ['ones', 'zeros']) {
      expect(card(id)!.getAttribute('data-status')).toBe('blocked');
      expect(card(id)!.title).toContain('暂无有效输出');
    }
  });

  it('keeps the projection steps editable when a comparison threshold is temporarily invalid', async () => {
    const operators = ['Cos', 'Sin', 'Function', 'TensorCompareScalar', 'OnesLike', 'ZerosLike'];
    const ids = ['x', ...operators, 'out'];
    const built = buildModel({
      name: 'Projection pipeline', nodes: [
        { id: 'x', type: 'Input' },
        ...operators.map(type => ({ id: type, type, presentation: 'edge' as const })),
        { id: 'out', type: 'Output' },
      ], connections: ids.slice(1).map((id, i) => [`${ids[i]}.out`, `${id}.in`]),
    });
    expect(built.ok).toBe(true);
    await act(() => state().load(built.project!));
    const steps = () => [...document.querySelectorAll('.edge-op-step')];
    expect(steps().map(el => el.textContent)).toEqual(['cos()', 'sin()', 'x', '>= 0', 'ones_like', 'zeros_like']);
    const formula = 'x' + ' + 1'.repeat(9);
    await act(() => { state().setParam('Function', 'expression', formula); });
    expect(steps()[2].textContent).toBe(formula.slice(0, 23) + '…');
    expect(steps()[2].getAttribute('title')).toContain(formula);
    await click(steps()[3]);
    expect(state().selected).toBe('TensorCompareScalar');
    await act(() => { state().setParam('TensorCompareScalar', 'value', '1e'); });
    expect(steps()[3].getAttribute('data-status')).toBe('error');
    expect(document.querySelector('.source-code-help pre')?.textContent).toContain('finite decimal');
    await act(() => { state().setParam('TensorCompareScalar', 'value', '1e-5'); });
    expect(steps()[3].textContent).toBe('>= 1e-5');
    expect(steps()[3].getAttribute('data-status')).toBe('ok');
  });

  it('renders Normalize in both card styles and explains channel constraints and invalid outputs', async () => {
    const built = buildModel({
      name: 'Channel normalization',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '3,8,16', axes: 'C,H,W' } },
        { id: 'norm', type: 'Normalize' },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'norm.in'], ['norm.out', 'out.in']],
    });
    expect(built.ok).toBe(true);
    await act(() => {
      state().load(built.project!);
      state().select('norm');
    });
    const node = () => card('norm')!;
    const spec = () => node().querySelector('.compact-layer-spec')?.textContent;
    const glyph = () => node().querySelector('.module-glyph')!;
    expect(glyph().querySelector('title')?.textContent).toContain('逐通道减去均值');
    expect(glyph().querySelector('.normalize-glyph [fill="currentColor"]')).toBeTruthy();
    const standardGlyph = glyph().innerHTML;
    expect(document.querySelector('.operation-port-help')?.textContent).toContain('(x - mean) / std');
    const help = document.querySelector('.operation-port-help pre')!.textContent!;
    expect(help).toMatch(/dim=-3[^\n]*\n.*torchvision\.transforms\.Normalize\(mean=mean, std=std\)\(x\)/);
    expect(help).toContain('HWC');
    expect(help).toContain('dim=2');
    expect(help).toContain('shape[dim] = -1');
    expect(help).toContain('(x - x.new_tensor(mean).view(shape)) / x.new_tensor(std).view(shape)');
    await toggle(1);
    expect(node().classList.contains('compact-layer-node')).toBe(true);
    expect(glyph().innerHTML).toBe(standardGlyph);
    expect(spec()).toBe('Normalize · C=3');
    await act(() => { state().setParam('x', 'shape', 'C,8,16'); });
    expect(node().getAttribute('data-status')).toBe('warning');
    expect(node().textContent).toContain('警告');
    expect(node().title).toContain('C = 3');
    expect(node().title).toContain('输出 [C, 8, 16]');
    await act(() => {
      state().setParam('norm', 'mean', '0');
      state().setParam('norm', 'std', '1');
    });
    expect(node().getAttribute('data-status')).toBe('ok');
    expect(spec()).toBe('Normalize · 广播');
    await act(() => { state().setParam('norm', 'std', '0'); });
    expect(node().getAttribute('data-status')).toBe('error');
    expect(spec()).toBe('Normalize');
    expect(node().title).toContain('暂无有效输出');
    expect(node().title).toContain('std');
    expect(node().title).not.toContain('[C, 8, 16]');
    const inputEdge = state().project.graph.edges.find(e => e.target === 'norm')!;
    await act(() => { state().removeEdges([inputEdge.id]); });
    expect(node().getAttribute('data-status')).toBe('blocked');
    expect(node().title).toContain('请连接输入：in');
  });

  it('uses the two-line title and full tooltip only on real Submodule cards', async () => {
    const titles = ['TemporalSelfAttention + LN', 'SpatialCrossAttention + LN'];
    const graph = structuredClone(project);
    for (const [index, label] of titles.entries())
      graph.graph.nodes.push({ ...registry.create('Submodule', {}, `sub${index}`), label });
    graph.graph.nodes.push(registry.create('EgoStateMLP', {}, 'mlp'));
    await act(() => { state().load(graph); });
    for (const [index, label] of titles.entries()) {
      const title = card(`sub${index}`)!.querySelector('strong.node-kind.submodule-title')!;
      expect(title.textContent).toBe(label);
      expect(title.getAttribute('title')).toBe(label);
    }
    expect(card('mlp')!.querySelector('.node-kind')?.hasAttribute('title')).toBe(false);
    expect(document.querySelectorAll('.submodule-title')).toHaveLength(2);
  });

  const portNames = ['ref_cam_uv', 'lidar2img', 'bev_pos', 'img_mem'];
  function moduleProject(count: number) {
    const inlets = portNames.slice(0, count).map(id => ({ id, type: 'Inlet', params: { port: id } }));
    return buildModel({
      name: '封装端口', nodes: [
        { id: 'x', type: 'Input', params: { shape: '1,256,256', axes: 'B,Q,C' } },
        { id: 'unit', type: 'Submodule', label: 'SpatialCrossAttention + LN', subgraph: {
          nodes: [...inlets, { id: 'step', type: 'Identity' }, { id: 'ret', type: 'Outlet' }],
          connections: [['ref_cam_uv.out', 'step.in'], ['step.out', 'ret.in']],
        } }, { id: 'y', type: 'Output' },
      ], connections: [...inlets.map(p => ['x.out', 'unit.' + p.id]), ['unit.out', 'y.in']],
    }).project!;
  }

  it.each([1, 2, 3, 4])('keeps %i inputs in independent slots when sides change', async count => {
    const p = moduleProject(count);
    await act(() => { state().load(p); });
    const layouts: [PortSide, PortSide][] = [['left', 'right'], ['right', 'left'], ['top', 'bottom'],
      ['left', 'left'], ['bottom', 'bottom'], ['right', 'top']];
    for (const [inputSide, outputSide] of layouts) {
      await act(() => {
        state().setPortSide('unit', 'input', inputSide);
        state().setPortSide('unit', 'output', outputSide);
      });
      const unit = card('unit')!;
      expect(unit.querySelector('.submodule-content .submodule-port-label')).toBeNull();
      expect(unit.querySelectorAll('.react-flow__handle')).toHaveLength(count + 1);
      for (const [i, id] of [...portNames.slice(0, count), 'out'].entries()) {
        const side = i === count ? outputSide : inputSide;
        const group = side === 'left' || side === 'right' ? 'sides' : side;
        const label = unit.querySelector<HTMLElement>(`[data-band="${group}"] [data-port-id="${id}"]`)!;
        expect(label.title).toBe(id);
        expect(label.dataset.side).toBe(side);
        const handle = unit.querySelector<HTMLElement>(`.react-flow__handle-${side}[aria-label="${i === count ? '输出' : '输入'} ${id}"]`)!;
        expect(handle.parentElement).toBe(unit);
        expect(handle.style.gridArea).toBe(group);
        if (group === 'sides') expect(Number(label.style.gridRow)).toBe(i === count && inputSide !== outputSide ? 1 : i + 1);
      }
    }
  });

  it('remeasures only the changed module when the same four slots move to the opposite side', async () => {
    await act(() => { state().load(moduleProject(4)); state().setPortSide('unit', 'output', 'top'); });
    remeasure.mockClear();
    await act(() => { state().setPortSide('unit', 'input', 'right'); });
    expect(remeasure.mock.calls).toEqual([['unit']]);
    await act(() => { state().setSubmoduleVisual('unit', { tone: 'rose' }); });
    expect(remeasure.mock.calls).toEqual([['unit']]);
  });

  it('commits Concat output side and count edits through the controlled selects', async () => {
    await act(() => state().select('cat'));
    await toggle(0);
    expect(card('cat')!.classList.contains('cat-node')).toBe(true);
    for (const [side, count] of [['bottom', 2], ['bottom', 3], ['bottom', 2], ['right', 2]] as const) {
      remeasure.mockClear();
      const cat = () => state().project.graph.nodes.find(node => node.id === 'cat')!;
      if (cat().portLayout?.output !== side) await chooseSelect('输出端口位置', side);
      else await chooseSelect('输入数量 (input_count)', String(count));
      expect(cat().portLayout?.output).toBe(side);
      expect(Number(cat().params.input_count ?? 2)).toBe(count);
      expect(document.querySelector<HTMLSelectElement>('[aria-label="输出端口位置"]')!.value).toBe(side);
      expect(card('cat')!.querySelector('.source')?.getAttribute('data-handlepos')).toBe(side);
      expect(card('cat')!.querySelectorAll('.target')).toHaveLength(count);
      expect(remeasure.mock.calls).toEqual([['cat']]);
    }
  });

  it.each([
    ['error', '推导错误'], ['blocked', '已阻断'], ['incomplete', '待补齐'], ['warning', '警告'],
  ] as const)('shows the %s diagnostic on the Submodule badge', async (status, label) => {
    const p = moduleProject(1), model = p.graph.nodes[1];
    const messages = ['内部采样步骤', '检查输入尺寸'];
    const result = { ...inferProject(p, registry).nodes.unit, status, messages };
    const node: TrailNode = {
      id: model.id, type: 'module', position: model.position,
      data: { model, result, inputs: [], outputs: [{ id: 'out', label: 'out' }], category: '网络结构',
        active: false, reverse: false, dimmed: false, open: () => {} },
    };
    await act(() => root.render(h(ReactFlowProvider, null, renderModule(node))));
    const badge = card('unit')!.querySelector<HTMLElement>('.submodule-status')!;
    expect(badge.dataset.status).toBe(status);
    expect(badge.textContent).toBe(label);
    expect(badge.title).toBe(messages.join('\n'));
    expect(card('unit')!.querySelector('.submodule-spec code')?.textContent)
      .toBe(status === 'warning' ? '[1, 256, 256]' : undefined);
  });

  function expectInferredBadge(status: string, label: string, message: string, shape?: string) {
    const unit = card('unit')!;
    const badge = unit.querySelector<HTMLElement>('.submodule-status')!;
    expect(badge.dataset.status).toBe(status);
    expect(badge.textContent).toBe(label);
    expect(badge.title).toContain(message);
    expect(unit.querySelector('.submodule-spec code')?.textContent).toBe(shape);
  }

  it('shows a blocked badge when the Submodule input is disconnected', async () => {
    const p = moduleProject(1);
    p.graph.edges = p.graph.edges.filter(edge => edge.target !== 'unit');
    await act(() => state().load(p));
    expectInferredBadge('blocked', '已阻断', '请连接输入：ref_cam_uv');
  });

  it('shows an error badge when the internal Outlet is disconnected', async () => {
    const p = moduleProject(1);
    const child = p.graph.nodes[1].subgraph!;
    child.edges = child.edges.filter(edge => edge.target !== 'ret');
    await act(() => state().load(p));
    expectInferredBadge('error', '推导错误', '模块输出：请连接输入：in');
  });

  it('shows the symbolic dimension warning and the inferred Linear output shape', async () => {
    const p = moduleProject(1);
    p.graph.nodes[0].params.shape = '1,256,C';
    p.graph.nodes[1].subgraph!.nodes[1] = registry.create(
      'Linear', { in_features: 256, out_features: 32 }, 'step',
    );
    await act(() => state().load(p));
    expectInferredBadge('warning', '警告', 'C = 256', '[1, 256, 32]');
  });

  it('renders the actual contract summary and distinguishes a single semantic step', async () => {
    const p = moduleProject(1), graph = p.graph.nodes[1].subgraph!;
    const step = registry.create('SemanticOp', { summary: '待展开的图像采样契约' }, 'step');
    graph.nodes[1] = step;
    graph.edges[0].targetPort = 'x';
    await act(() => { state().load(p); });
    expect(card('unit')!.querySelector('[data-status="contract"]')?.textContent).toBe('待展开 · 契约');
    expect(card('unit')!.querySelector('.submodule-summary')?.textContent).toBe('待展开的图像采样契约');
    const revised = structuredClone(p);
    revised.graph.nodes[1].subgraph!.nodes[1].params.summary = '按固定系数缩放';
    await act(() => { state().load(revised); });
    expect(card('unit')!.querySelector('[data-status="semantic"]')?.textContent).toBe('1 个语义步骤');
  });

  it('edits the customized card from the inspector, restores automatic style, and opens forward()', async () => {
    await act(() => { state().load(moduleProject(2)); state().select('unit'); });
    const unit = () => card('unit')!;
    expect(unit().querySelector('[data-submodule-icon="cross_attention"]')).toBeTruthy();
    expect(unit().classList.contains('submodule-tone-teal')).toBe(true);
    expect(unit().querySelector('.submodule-status')?.textContent).toBe('已展开 · 1 步');
    expect(unit().querySelector('.submodule-spec code')?.textContent).toBe('[1, 256, 256]');
    expect(unit().querySelector('.node-footer')).toBeNull();
    const controls = () => document.querySelector('[aria-label="封装模块样式"]')!;
    await chooseSelect('图标原型', 'mlp');
    await chooseSelect('主题色标', 'rose');
    const summary = controls().querySelector('input')!;
    summary.value = '  图像特征融合  ';
    await act(() => { summary.dispatchEvent(new window.Event('focusout', { bubbles: true })); });
    expect(unit().querySelector('.submodule-summary')?.getAttribute('title')).toBe('图像特征融合');
    expect(unit().querySelector('[data-submodule-icon="mlp"]')).toBeTruthy();
    expect(unit().classList.contains('submodule-tone-rose')).toBe(true);
    await click(controls().querySelector('button')!);
    expect(unit().querySelector('.submodule-summary')?.textContent).toBe('Identity');
    await act(() => state().undo());
    expect(unit().querySelector('.submodule-summary')?.textContent).toBe('图像特征融合');
    await act(() => state().redo());
    expect(unit().querySelector('[data-submodule-icon="cross_attention"]')).toBeTruthy();
    const enter = unit().querySelector('button[aria-label="进入 forward()"]')!;
    expect(enter.textContent).toBe('↗');
    await click(enter);
    expect(state().path).toEqual(['unit']);
  });

  it('validates the raw inspector summary before trimming and restores it after an overlong edit', async () => {
    await act(() => { state().load(moduleProject(1)); state().select('unit'); });
    const text = '🧊'.repeat(120);
    const input = () => document.querySelector<HTMLInputElement>('input[aria-label="功能摘要"]')!;
    async function commit(value: string) {
      input().value = value;
      await act(() => { input().dispatchEvent(new window.Event('focusout', { bubbles: true })); });
    }
    for (const value of [' ' + '🧊'.repeat(119), text]) {
      await commit(value);
      expect(input().value).toBe(value.trim());
      expect(state().project.graph.nodes[1].submoduleVisual?.summary).toBe(value.trim());
    }
    const saved = state().project;
    for (const value of [' ' + text, text + '🧊']) {
      await commit(value);
      expect(state().notice).toContain('120');
      expect(input().value).toBe(text);
      expect(state().project).toBe(saved);
    }
  });

  it('keeps one exterior handle per port with both media and tensor geometry', async () => {
    const p = moduleProject(2), unit = p.graph.nodes[1];
    unit.visualMedia = { items: [{ id: 'image', src: 'data:image/png;base64,YQ==' }] };
    unit.tensorVisual = { mode: 'block' };
    await act(() => { state().load(p); });
    const element = card('unit')!;
    expect(element.querySelector('.node-visual-media img')).toBeTruthy();
    expect(element.querySelector('.tensor-geometry-card')).toBeTruthy();
    expect(element.querySelectorAll('.react-flow__handle')).toHaveLength(3);
    expect(element.querySelector('.submodule-content .react-flow__handle')).toBeNull();
  });

  it('selects the real TensorIndex and Permute nodes by clicking successive edge steps', async () => {
    const built = buildModel({
      name: 'Index and reorder',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,512,1,256', axes: 'B,Q,T,C' } },
        { id: 'pick', type: 'TensorIndex', presentation: 'edge' },
        { id: 'perm', type: 'Permute', params: { dims: '1, 0, 2' }, presentation: 'edge' },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'pick.in'], ['pick.out', 'perm.in'], ['perm.out', 'out.in']],
    });
    expect(built.ok).toBe(true);
    await act(() => { state().load(built.project!); });
    const steps = () => [...document.querySelectorAll('.edge-op-step')];
    expect(steps().map(s => s.textContent)).toEqual(['[0]', 'permute(1,0,2)']);
    expect(document.querySelector('.edge-op-pipeline small')?.textContent).toContain('[1, 512, 256]');
    for (const [i, id] of ['pick', 'perm'].entries()) {
      expect(card(id)).toBeNull();
      await click(steps()[i]);
      expect(state().selected).toBe(id);
      expect(steps()[i].getAttribute('aria-pressed')).toBe('true');
      expect(document.querySelector('.inspector-title')?.textContent)
        .toContain(built.project!.graph.nodes.find(n => n.id === id)!.label);
    }
  });

  it('ellipsizes long edge steps and exposes the complete index expression on hover', async () => {
    const index = 'selected_temporal_feature_index + temporal_window_offset';
    const built = buildModel({
      name: 'Long index expression',
      hyperparameters: {
        selected_temporal_feature_index: { type: 'int', value: 1 },
        temporal_window_offset: { type: 'int', value: 2 },
      },
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,512', axes: 'B,Q' } },
        { id: 'pick', type: 'TensorIndex', params: { dim: '1', index }, presentation: 'edge' },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'pick.in'], ['pick.out', 'out.in']],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    await act(() => { state().load(built.project!); });
    const step = document.querySelector<HTMLButtonElement>('.edge-op-step')!;
    expect(step.textContent).toBe(`[:, ${index}]`);
    expect(step.title).toContain(`[:, ${index}]`);

    const style = document.createElement('style');
    style.textContent = readFileSync(new URL('../src/editor/workbench.css', import.meta.url), 'utf8');
    document.head.append(style);
    const rule = Array.from(style.sheet!.cssRules).find((r): r is CSSStyleRule =>
      'selectorText' in r && r.selectorText === '.edge-op-step')!;
    expect(rule.style.getPropertyValue('max-width')).toBe('240px');
    expect(rule.style.getPropertyValue('overflow')).toBe('hidden');
    expect(rule.style.getPropertyValue('text-overflow')).toBe('ellipsis');
    expect(rule.style.getPropertyValue('white-space')).toBe('nowrap');
  });

  async function loadGridSample() {
    const sampled = buildModel({
      name: 'Grid sampling',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '1,8,6,4', axes: '' } },
        { id: 'coords', type: 'Input', params: { shape: '1,5,7,2', axes: '' } },
        { id: 'sample', type: 'GridSample', label: '采样特征' },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'sample.in'], ['coords.out', 'sample.grid'], ['sample.out', 'out.in']],
    });
    expect(sampled.ok).toBe(true);
    await act(() => {
      state().load(sampled.project!);
      state().select('sample');
    });
  }

  it('switches GridSample cards and keeps each port label beside its handle as the layout changes', async () => {
    await loadGridSample();
    const node = () => card('sample')!;
    const checkGlyph = () => {
      const svg = node().querySelector('.module-glyph')!;
      expect(svg.querySelector('title')?.textContent).toContain('四个邻域角点');
      expect(svg.querySelectorAll('.grid-sample-glyph line')).toHaveLength(4);
      expect(svg.querySelectorAll('.grid-sample-glyph circle')).toHaveLength(5);
    };
    expect(node().classList.contains('module-node')).toBe(true);
    checkGlyph();
    expect([...document.querySelectorAll('.presentation-options button')].map(b => b.textContent))
      .toEqual(['标准卡片', '精简图标']);
    expect(document.querySelector('.operation-port-help pre')?.textContent).toContain('F.grid_sample(');
    await toggle(1);
    checkGlyph();
    expect(node().classList.contains('compact-grid-sample')).toBe(true);
    expect(node().querySelector('.compact-layer-spec')?.textContent).toBe('GridSample · bilinear');
    await act(() => { state().setParam('sample', 'mode', 'nearest'); });
    expect(node().querySelector('.compact-layer-spec')?.textContent).toBe('GridSample · nearest');
    expect(document.querySelector('.operation-port-help pre')?.textContent)
      .toContain('mode=mode, padding_mode=padding_mode, align_corners=align_corners');

    function checkPorts(input: PortSide, output: PortSide) {
      const sides = { in: input, grid: input, out: output };
      expect(node().querySelectorAll('.react-flow__handle')).toHaveLength(3);
      expect(node().querySelectorAll('.node-port-label')).toHaveLength(3);
      for (const [port, side] of Object.entries(sides)) {
        const handle = node().querySelector<HTMLElement>(`[data-handleid="${port}"][data-handlepos="${side}"]`)!;
        const label = node().querySelector<HTMLElement>(`.port-label-${side}[data-port-id="${port}"]`)!;
        expect(handle.getAttribute('aria-label')).toBe(`${port === 'out' ? '输出' : '输入'} ${port}`);
        expect(label.textContent).toBe(port);
        const lateral = side === 'left' || side === 'right';
        const offset = lateral ? 'top' : 'left';
        const peers = Object.keys(sides).filter(id => sides[id as keyof typeof sides] === side);
        expect(label.style[offset]).toBe(handle.style[offset]);
        expect(parseFloat(label.style[offset])).toBeCloseTo(100 * (peers.indexOf(port) + 1) / (peers.length + 1));
        const padding = node().style.getPropertyValue(`padding-${side}`);
        expect(parseFloat(padding)).toBeGreaterThanOrEqual(lateral ? 34 : 20);
        const minSize = lateral ? node().style.minHeight : node().style.minWidth;
        expect(parseFloat(minSize) / (peers.length + 1)).toBeGreaterThanOrEqual(lateral ? 18 : 32);
      }
    }
    checkPorts('left', 'right');
    await act(() => state().move('out', { x: -1000, y: 0 }));
    checkPorts('right', 'left');
    for (const [input, output] of [
      ['top', 'bottom'], ['bottom', 'top'], ['top', 'top'], ['bottom', 'bottom'],
      ['left', 'left'], ['right', 'right'],
    ] as const) {
      await act(() => {
        state().setPortSide('sample', 'input', input);
        state().setPortSide('sample', 'output', output);
      });
      checkPorts(input, output);
    }
    await toggle(0);
    expect(node().classList.contains('module-node')).toBe(true);
    checkGlyph();
  });

  it('shows GridSample warnings and replaces the output shape with diagnostics when sampling is blocked', async () => {
    await loadGridSample();
    await toggle(1);
    const node = () => card('sample')!;
    await act(() => { state().setParam('coords', 'shape', '1,5,7,K'); });
    expect(node().getAttribute('data-status')).toBe('warning');
    expect(node().textContent).toContain('警告');
    expect(node().title).toContain('K = 2');
    expect(node().title).toContain('输出 [1, 8, 5, 7]');

    await act(() => { state().setParam('coords', 'shape', '1,5,7,3'); });
    expect(node().getAttribute('data-status')).toBe('error');
    expect(node().title).toContain('暂无有效输出');
    expect(node().title).toContain('grid 末维应为 2');
    expect(node().title).not.toContain('[1, 8, 6, 4]');

    const gridEdge = state().project.graph.edges.find(edge => edge.target === 'sample' && edge.targetPort === 'grid')!;
    await act(() => { state().removeEdges([gridEdge.id]); });
    expect(node().getAttribute('data-status')).toBe('blocked');
    expect(node().title).toContain('请连接输入：grid');
    expect(node().title).toContain('暂无有效输出');
    expect(node().title).not.toContain('[1, 8, 6, 4]');
  });

  it('drags the pipeline using the terminal canonical edge as the label anchor', async () => {
    const step = document.querySelector<HTMLButtonElement>('.edge-op-step')!;
    step.setPointerCapture = vi.fn();
    step.hasPointerCapture = () => false;
    for (const [name, x] of [
      ['pointerdown', 150],
      ['pointermove', 225],
      ['pointerup', 225],
    ] as const)
      await act(() => {
        step.dispatchEvent(
          Object.assign(new window.Event(name, { bubbles: true }), {
            button: 0,
            pointerId: 1,
            clientX: x,
            clientY: 0,
          }),
        );
      });
    expect(state().project.graph.edges[2].labelPosition).toBeCloseTo(0.75);
    expect(
      state()
        .project.graph.edges.slice(0, 2)
        .every((e) => e.labelPosition === undefined),
    ).toBe(true);
    await act(() => state().undo());
    expect(state().project.graph.edges[2].labelPosition).toBeUndefined();
  });

  it('reveals an anchor on hover, edge selection and source selection, with no residual bus', async () => {
    const chip = () => document.querySelector<HTMLButtonElement>('.edge-anchor-chip')!;
    const wire = () => document.querySelector('.edge-anchor-hover-wire')!;
    const junction = () => document.querySelector('.edge-junction[data-source="query"]');
    expect(chip().textContent).toBe('● query');
    expect(wire().getAttribute('data-revealed')).toBe('false');
    expect(junction()).toBeNull();
    const pointer = async (name: string) =>
      act(() => {
        chip().dispatchEvent(new window.Event(name, { bubbles: true }));
      });
    await pointer('pointerover');
    expect(wire().getAttribute('data-revealed')).toBe('true');
    await pointer('pointerout');
    expect(wire().getAttribute('data-revealed')).toBe('false');
    await act(() => state().select('query'));
    expect(wire().getAttribute('data-revealed')).toBe('true');
    await click(chip());
    expect(state().selectedEdgeId).toBe(anchor.id);
    expect(state().selected).toBeNull();
    expect(wire().getAttribute('data-revealed')).toBe('true');
    expect(junction()).toBeNull();
    await chooseStyle('slide');
    expect(chip()).toBeNull();
    expect(junction()).toBeTruthy();
    await chooseStyle('anchor');
    expect(chip()).toBeTruthy();
    expect(junction()).toBeNull();
    await act(() => state().undo());
    expect(junction()).toBeTruthy();
    await act(() => state().redo());
    expect(chip()).toBeTruthy();
    expect(junction()).toBeNull();
  });
});

it('fits projected frames to visible members and translates hidden members by the heading drag delta', () => {
  const graph = structuredClone(project.graph);
  graph.nodes.find((n) => n.id === 'linear')!.position.x = -8500;
  const members = 'query,linear,reshape,values';
  const region = registry.create('LayoutRegion', { members }, 'layout');
  const scope = registry.create('IfRegion', { members }, 'scope');
  graph.nodes.push(region, scope);
  fitLayoutRegion(graph, region.id);
  const before = structuredClone(graph);
  const view = projectCanvasGraph(graph, undefined, new Map());
  const nodes: TrailNode[] = view.visibleNodes.map((model) => ({
    id: model.id,
    type: 'module',
    position: model.position,
    measured: { width: 80, height: 40 },
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
  }));
  const frames = layoutRegions(layoutIfRegions(nodes), view.hiddenNodeIds);
  const frame = frames.find((n) => n.id === region.id)!;
  expect(frame.position.x).toBeGreaterThan(-1000);
  expect(Number(frame.style!.width)).toBeLessThan(Number(region.params.width));
  expect(frames.find((n) => n.id === scope.id)!.position.x).toBeGreaterThan(-1000);
  expect(graph).toEqual(before);
  const desired = { x: frame.position.x + 50, y: frame.position.y + 25 };
  const moves = regionDragPositions(
    graph,
    [{ id: frame.id, position: desired }],
    frames,
    view.hiddenNodeIds,
  );
  moveLayoutNodes(graph, moves);
  for (const id of members.split(',')) {
    const old = before.nodes.find((n) => n.id === id)!.position;
    expect(graph.nodes.find((n) => n.id === id)!.position).toEqual({
      x: old.x + 50,
      y: old.y + 25,
    });
  }
});
