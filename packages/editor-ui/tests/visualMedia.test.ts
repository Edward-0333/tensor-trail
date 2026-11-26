import { parseHTML } from 'linkedom';
import { act, createElement, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { Position, ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { ports } from '@tensor-trail/model-core/core/graph';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { example } from '@tensor-trail/model-core/modules/examples';
import { registry } from '@tensor-trail/model-core/modules';
import type { NodeVisualMedia } from '@tensor-trail/model-core/core/types';
import { Inspector } from '../src/editor/Inspector';
import { useEditor } from '../src/editor/store';
import { uploadNodeImages } from '../src/editor/NodeVisualMediaPanel';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { NodeVisualMedia as Gallery } from '../src/views/NodeVisualMedia';
import { EditableEdge } from '../src/views/EditableEdge';
import { EdgeRoutesProvider } from '../src/views/EdgeJunctions';
import { createRoutes } from '../src/views/routeStore';
import { valueEdgeStyle } from '../src/views/valueEdgeStyle';

// React Flow normally mounts this portal in its measured canvas. Supply the same HTML
// destination when exercising a single edge in a DOM without a layout engine.
vi.mock('@xyflow/react', async (importOriginal) => {
  const flow = await importOriginal<typeof import('@xyflow/react')>();
  const { createPortal } = await import('react-dom');
  return {
    ...flow,
    EdgeLabelRenderer: ({ children }: { children: ReactNode }) =>
      createPortal(children, document.querySelector('#labels')!),
  };
});

const state = useEditor.getState;
const sources = ['data:image/png;base64,YQ==', 'data:image/jpeg;base64,Yg=='];
const media: NodeVisualMedia = {
  items: [
    { id: 'first', src: sources[0], caption: '保留的区域', badge: '有效', emphasis: 'highlight' },
    { id: 'second', src: sources[1], caption: '过滤的区域', emphasis: 'dim' },
  ],
};
const project = buildModel({
  name: '图集',
  nodes: [
    { id: 'x', type: 'Input', params: { shape: '2,3', axes: '' } },
    { id: 'y', type: 'Output' },
  ],
  connections: [['x.out', 'y.in']],
}).project!;

describe('图集排版', () => {
  it.each([
    ['auto', 1, 1],
    ['auto', 2, 2],
    ['auto', 3, 3],
    ['auto', 4, 2],
    ['auto', 12, 3],
    ['row', 6, 6],
    ['grid', 1, 2],
    ['grid', 6, 2],
    ['column', 4, 1],
  ] as const)('%s 的 %i 张图使用 %i 列', (layout, count, columns) => {
    const items = Array.from({ length: count }, (_, i) => ({ ...media.items[0], id: String(i) }));
    expect(
      renderToStaticMarkup(createElement(Gallery, { media: { layout, tileWidth: 160, items } })),
    ).toContain(`repeat(${columns}, 160px)`);
  });

  it.each([
    'Input',
    'Output',
    'Inlet',
    'Outlet',
    'TensorVariable',
    'ValueVariable',
    'BufferSource',
    'SemanticOp',
    'CalculationGroup',
    'Submodule',
    'Sigmoid',
  ])('%s 的图片和端口均位于外层卡片', (type) => {
    const model = {
      ...registry.create(type),
      visualMedia: media,
      portLayout: { output: 'bottom' as const },
    };
    const nodePorts = ports(model, registry);
    const props: NodeProps<TrailNode> = {
      id: model.id,
      type: 'module',
      selected: false,
      dragging: false,
      draggable: true,
      selectable: true,
      deletable: true,
      isConnectable: true,
      zIndex: 0,
      positionAbsoluteX: 0,
      positionAbsoluteY: 0,
      data: {
        model,
        ...nodePorts,
        category: '',
        reverse: false,
        active: false,
        dimmed: false,
        open: vi.fn(),
      },
    };
    const html = renderToStaticMarkup(
      createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
    );
    const { document } = parseHTML(html);
    const card = document.querySelector('.illustrated-node')!;
    expect(
      Array.from(card.querySelectorAll('figure img'), (img) => img.getAttribute('src')),
    ).toEqual(sources);
    expect(card.querySelector('[data-emphasis="highlight"] figcaption')?.textContent).toBe(
      '保留的区域',
    );
    expect(card.querySelector('.media-badge')?.textContent).toBe('有效');
    expect(card.querySelector('[data-emphasis="dim"] figcaption')?.textContent).toBe('过滤的区域');
    const handles = card.querySelectorAll('.react-flow__handle');
    expect(handles).toHaveLength(nodePorts.inputs.length + nodePorts.outputs.length);
    for (const handle of handles) expect(handle.parentElement).toBe(card);
  });
});

describe('属性面板展示编辑', () => {
  let root: Root;
  beforeEach(() => {
    const { window } = parseHTML('<html><body><main></main><div id="labels"></div></body></html>');
    vi.stubGlobal('window', window);
    vi.stubGlobal('document', window.document);
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.stubGlobal('requestAnimationFrame', (fn: FrameRequestCallback) =>
      setTimeout(() => fn(0), 0),
    );
    vi.stubGlobal('cancelAnimationFrame', clearTimeout);
    root = createRoot(document.querySelector('main')!);
    state().load(project);
    state().select('x');
  });
  afterEach(async () => {
    await act(() => root.unmount());
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });
  function Properties() {
    const graph = useEditor((s) => s.project.graph);
    return createElement(Inspector, { graph, result: inferProject(state().project, registry) });
  }
  async function change(
    selector: string,
    type: string,
    prepare: (input: HTMLInputElement) => void,
  ) {
    const input = document.querySelector<HTMLInputElement>(selector)!;
    expect(input).toBeTruthy();
    prepare(input);
    await act(async () => {
      input.dispatchEvent(new window.Event(type, { bubbles: true }));
    });
  }
  const images = () => state().project.graph.nodes[0].visualMedia?.items;
  const file = (body: string, type = 'image/png') => new File([body], 'sample', { type });
  const pick = (selector: string, files: File[]) =>
    change(selector, 'change', (input) =>
      Object.defineProperty(input, 'files', { value: files, configurable: true }),
    );

  it('多选上传、编辑说明与角标、单张换图、调整顺序和删除，支持撤销重做', async () => {
    await act(() => root.render(createElement(Properties)));
    await pick('input[aria-label="上传图片"]', [file('a'), file('b', 'image/jpeg')]);
    expect(images()!.map((item) => item.src)).toEqual(sources);
    await change('fieldset input:not([type=file])', 'focusout', (el) => {
      el.value = '筛选结果';
    });
    const badge = document.querySelectorAll<HTMLInputElement>('fieldset input:not([type=file])')[1];
    badge.value = '保留';
    await act(() => {
      badge.dispatchEvent(new window.Event('focusout', { bubbles: true }));
    });
    await change('fieldset select', 'change', (select) => {
      select.querySelector<HTMLOptionElement>('option[value=highlight]')!.selected = true;
    });
    const before = state().project;
    await pick('input[aria-label="单张换图"]', [file('c', 'image/webp')]);
    expect(images()![0]).toEqual({
      ...before.graph.nodes[0].visualMedia!.items[0],
      src: 'data:image/webp;base64,Yw==',
      caption: '筛选结果',
      badge: '保留',
      emphasis: 'highlight',
    });
    expect(images()![1].src).toBe(sources[1]);
    const after = state().project;
    await act(() => state().undo());
    expect(state().project).toEqual(before);
    await act(() => {
      state().redo();
      state().select('x');
    });
    expect(state().project).toEqual(after);
    await act(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="下移"]')!.click(),
    );
    expect(images()![0].src).toBe(sources[1]);
    await act(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="删除插图"]')!.click(),
    );
    await act(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="删除插图"]')!.click(),
    );
    expect(images()).toBeUndefined();
  });

  it('解码期间切换项目时取消写入，即使新项目存在同名节点', async () => {
    const image = file('old image');
    let release!: (buffer: ArrayBuffer) => void;
    vi.spyOn(image, 'arrayBuffer').mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    const pending = uploadNodeImages('x', [image]);
    state().load(project);
    const next = state().project;
    release(new Uint8Array([42]).buffer);
    await pending;
    expect(state().project).toBe(next);
    expect(images()).toBeUndefined();
  });

  it('Canvas 压缩长边并指定 JPEG 质量，同批错误不会部分写入', async () => {
    const draw = vi.fn(),
      encode = vi.fn(() => 'data:image/jpeg;base64,YQ==');
    const prototype = Object.getPrototypeOf(document.createElement('canvas')) as HTMLCanvasElement;
    vi.spyOn(prototype, 'getContext').mockReturnValue({
      drawImage: draw,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(prototype, 'toDataURL').mockImplementation(encode);
    vi.stubGlobal(
      'Image',
      class {
        naturalWidth = 1200;
        naturalHeight = 500;
        async decode() {}
      },
    );
    await uploadNodeImages('x', [file('a', 'image/jpeg')]);
    expect(draw).toHaveBeenCalledWith(expect.anything(), 0, 0, 480, 200);
    expect(encode).toHaveBeenCalledWith('image/jpeg', 0.85);
    const before = state().project;
    await expect(uploadNodeImages('x', [file('b'), file('bad', 'image/svg+xml')])).rejects.toThrow(
      /PNG/,
    );
    expect(state().project).toBe(before);
  });

  it('选边、编辑箭头、绘制双行标签并恢复位置', async () => {
    const id = project.graph.edges[0].id;
    const routes = createRoutes();
    function EdgeAndInspector() {
      const edge = useEditor((s) => s.project.graph.edges[0]);
      return createElement(
        ReactFlowProvider,
        null,
        createElement(EdgeRoutesProvider, {
          routes,
          children: createElement(
            'svg',
            null,
            createElement(EditableEdge, {
              ...edge,
              type: 'editable',
              sourceX: 0,
              sourceY: 0,
              targetX: 300,
              targetY: 0,
              sourcePosition: Position.Right,
              targetPosition: Position.Left,
              label: '[2, 3]',
              markerEnd: 'url(#arrow)',
              style: valueEdgeStyle(undefined, false, false, edge.presentation),
              data: {
                ...edge,
                changeRoute: vi.fn(),
                changeSharedRoutes: vi.fn(),
                changeLabelPosition: (position) => state().setEdgeLabelPosition(id, position),
                select: () => state().selectEdge(id),
              },
            }),
          ),
        }),
        createElement(Properties),
      );
    }
    await act(() => root.render(createElement(EdgeAndInspector)));
    await act(() => {
      document
        .querySelector('.react-flow__edge-interaction')!
        .dispatchEvent(new window.Event('click', { bubbles: true }));
    });
    expect(state().selectedEdgeId).toBe(id);
    expect(document.querySelector('.edge-settings')!.textContent).toContain('x.out → y.in');
    expect(document.querySelector('.edge-settings')!.textContent).toContain('[2, 3]');
    await change('.edge-settings select', 'change', (el) => {
      el.querySelector<HTMLOptionElement>('option[value=slide]')!.selected = true;
    });
    await change('.edge-settings textarea', 'focusout', (el) => {
      el.value = '编码后的特征';
    });
    expect(document.querySelector('.edge-slide-label strong')?.textContent).toBe('编码后的特征');
    expect(document.querySelector('.edge-slide-label span')?.textContent).toBe('[2, 3]');
    expect(document.querySelector<SVGElement>('.wire-normal')?.style.strokeWidth).toBe('2.5');
    expect(document.querySelector('marker')?.getAttribute('markerWidth')).toBe('20');
    const label = document.querySelector<HTMLDivElement>('.edge-slide-label')!;
    label.setPointerCapture = vi.fn();
    label.hasPointerCapture = () => false;
    for (const [event, x] of [
      ['pointerdown', 150],
      ['pointermove', 210],
      ['pointerup', 210],
    ] as const) {
      await act(() => {
        label.dispatchEvent(
          Object.assign(new window.Event(event, { bubbles: true }), {
            clientX: x,
            clientY: 0,
            button: 0,
            pointerId: 1,
          }),
        );
      });
    }
    expect(state().project.graph.edges[0].labelPosition).toBeCloseTo(0.7);
    await act(() => document.querySelector<HTMLButtonElement>('.edge-settings button')!.click());
    expect(state().project.graph.edges[0].labelPosition).toBeUndefined();
    await act(() => {
      state().undo();
      state().undo();
      state().undo();
    });
    expect(state().project.graph.edges[0].annotation).toBeUndefined();
    await act(() => state().redo());
    expect(state().project.graph.edges[0]).toMatchObject({
      presentation: 'slide',
      annotation: '编码后的特征',
    });
    await act(() => {
      state().setEdgePresentation(id, { presentation: null });
    });
    expect(document.querySelector('.edge-slide-label strong')?.textContent).toBe('编码后的特征');
    expect(document.querySelector<SVGElement>('.wire-normal')?.style.strokeWidth).toBe('1.5');
  });

  it('元数据与控制流的粗箭头保留语义颜色、错误红和虚线', () => {
    const device = { kind: 'device' as const, reference: 'cpu' };
    const control = { kind: 'control' as const, expression: 'valid' };
    for (const value of [device, control]) {
      const regular = valueEdgeStyle(value);
      expect(regular.strokeDasharray).toBeTruthy();
      expect(valueEdgeStyle(value, false, false, 'slide')).toEqual({
        ...regular,
        strokeWidth: 2.5,
      });
      expect(valueEdgeStyle(value, true, false, 'slide')).toEqual({
        ...regular,
        strokeWidth: 2.5,
        stroke: 'var(--danger)',
      });
    }
  });

  it('在生成模块内编辑和撤销展示属性，仍可修改父模块参数', () => {
    const original = example();
    state().load(original);
    state().navigate(['transformer']);
    const child = () => state().project.graph.nodes.find((n) => n.id === 'transformer')!.subgraph!;
    const edge = child().edges[0];
    state().setNodeVisualMedia(edge.source, media);
    state().setEdgePresentation(edge.id, { presentation: 'slide', annotation: '连接说明' });
    expect(
      state().project.graph.nodes.find((n) => n.id === 'transformer')!.customized,
    ).toBeUndefined();
    expect(inferProject(state().project, registry)).toEqual(inferProject(original, registry));
    state().undo();
    expect(child().edges[0].annotation).toBeUndefined();
    state().redo();
    expect(child().edges[0].annotation).toBe('连接说明');
    expect(state().setParam('transformer', 'heads', 4)).toBe(true);
    expect(child().nodes.find((n) => n.id === edge.source)!.visualMedia).toEqual(media);
    expect(child().edges[0].presentation).toBe('slide');
  });

  it('清除选边，关联副本不携带插图', () => {
    const edge = project.graph.edges[0].id;
    state().setNodeVisualMedia('x', media);
    state().selectEdge(edge);
    state().duplicateLinked('x');
    expect(state().selectedEdgeId).toBeNull();
    expect(state().project.graph.nodes.at(-1)!.visualMedia).toBeUndefined();
    for (const action of [
      () => state().select('x'),
      () => state().navigate([]),
      () => state().load(project),
      () => state().removeEdges([edge]),
    ]) {
      state().selectEdge(edge);
      action();
      expect(state().selectedEdgeId).toBeNull();
    }
  });
});
