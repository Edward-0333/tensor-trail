import { act, createElement, Fragment } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { parseHTML } from 'linkedom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import {
  buildTensorGeometryDiagram,
  buildTensorBlockDiagram,
  resolveGraphTensorTones,
  resolveGraphReferenceMaxDim,
} from '@tensor-trail/model-core/core/tensorGeometry';
import { registry } from '@tensor-trail/model-core/modules';
import App from '../src/App';
import { canvasApi } from '../src/editor/agentBridge';
import { configureEditor } from '../src/editor/platform';
import { useEditor } from '../src/editor/store';
import { TensorGeometryView } from '../src/views/TensorGeometryView';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

// LinkeDOM has no browser layout engine. Keep the real canvas derivation, provider,
// node/port components, toolbar and inspector; replace just React Flow's measurement surface.
const remeasure = vi.hoisted(() => vi.fn());
vi.mock('@xyflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@xyflow/react')>()),
  useUpdateNodeInternals: () => remeasure,
  ReactFlow: ({ nodes }: { nodes: TrailNode[] }) =>
    createElement(
      Fragment,
      null,
      nodes.map((n) =>
        createElement(ModuleNode, {
          key: n.id,
          id: n.id,
          data: n.data,
          type: 'module',
          selected: !!n.selected,
          isConnectable: true,
          dragging: false,
          draggable: true,
          selectable: true,
          deletable: true,
          zIndex: 0,
          positionAbsoluteX: n.position.x,
          positionAbsoluteY: n.position.y,
        }),
      ),
    ),
}));

const recipe = {
  name: 'Tensor geometry',
  nodes: [
    { id: 'query', type: 'Input', params: { shape: '1,256,256', axes: 'B,N,D' } },
    {
      id: 'pos',
      type: 'Input',
      params: { shape: '1,1,256', axes: 'B,N,D' },
      visualMedia: { items: [{ id: 'photo', src: 'data:image/png;base64,YQ==' }] },
    },
    { id: 'add', type: 'Add' },
    { id: 'features', type: 'Concat', params: { dim: -1 } },
    { id: 'history', type: 'Concat', params: { dim: 0 } },
    { id: 'named', type: 'TensorVariable' },
    { id: 'layers', type: 'TensorVariable' },
    { id: 'linear', type: 'Linear', params: { in_features: 512, out_features: 32 } },
    { id: 'reshape', type: 'Reshape', params: { shape: '1,256,32' } },
    { id: 'narrow', type: 'TensorVariable' },
    { id: 'chunk', type: 'Chunk', params: { dim: -1, chunks: 3 } },
    { id: 'out', type: 'Output', tensorVisual: { mode: 'off' } },
  ],
  connections: [
    ['query.out', 'add.a'],
    ['pos.out', 'add.b'],
    ['query.out', 'features.a'],
    ['add.out', 'features.b'],
    ['query.out', 'history.a'],
    ['query.out', 'history.b'],
    ['features.out', 'named.in'],
    ['history.out', 'layers.in'],
    ['layers.out', 'out.in'],
    ['named.out', 'linear.in'],
    ['linear.out', 'reshape.in'],
    ['reshape.out', 'narrow.in'],
    ['narrow.out', 'chunk.in'],
  ],
};
const project = buildModel(recipe).project!;
const result = inferProject(project, registry);
const tones = resolveGraphTensorTones(project.graph, result);

describe('SVG geometry', () => {
  const render = (diagram: ReturnType<typeof buildTensorBlockDiagram>) =>
    parseHTML(renderToStaticMarkup(createElement(TensorGeometryView, { diagram }))).document;
  const tensor = (shape: (number | string)[]) => ({ shape, axes: [], dtype: 'float32' as const });
  it('draws two side-by-side continuous deep prisms with four axis guides and custom color', () => {
    const diagram = buildTensorBlockDiagram(tensor([2, 256, 8, 32]), 'out', undefined, {
      axisLabels: ['queue', 'query', 'head', 'dim'],
      color: '#3b82f6',
    });
    expect(diagram.output.depthPx).toBeGreaterThan(12);
    const doc = render(diagram);
    const volumes = [...doc.querySelectorAll('[data-volume]')];
    expect(volumes).toHaveLength(2);
    const offsets = volumes.map((v) =>
      v
        .getAttribute('transform')!
        .match(/[-.\d]+/g)!
        .map(Number),
    );
    expect(offsets[0][0]).toBeLessThan(offsets[1][0]);
    expect(offsets[0][1]).toBe(offsets[1][1]);
    for (const volume of volumes) {
      expect(volume.querySelectorAll('[data-slice]')).toHaveLength(1);
      expect(volume.querySelectorAll('[data-face]')).toHaveLength(3);
      expect(volume.querySelector('[data-face="front"]')?.getAttribute('fill')).toBe(
        'var(--tv-custom-front)',
      );
      expect(volume.querySelector('[data-face="side"]')?.getAttribute('d')).toContain(
        'l' + diagram.output.depthPx,
      );
    }
    expect(doc.querySelector('figure')?.getAttribute('style')).toContain('#3b82f6');
    expect(doc.querySelector('[data-guide="outer"]')).toBeTruthy();
    expect([...doc.querySelectorAll('.tv-dim text')].map((t) => t.textContent)).toEqual([
      'dim 32',
      'head 8',
      'query 256',
      'queue 2',
    ]);
    expect(doc.querySelectorAll('.tv-chip')).toHaveLength(0);
  });

  it.each([
    [1, 1, false],
    [3, 3, false],
    [8, 2, true],
    ['B', 2, true],
  ] as const)(
    'fits the row and bracket for outer size %s at minimum card width',
    (n, count, elided) => {
      const d = buildTensorBlockDiagram(tensor([n, 16, 8, 8]), 'out', undefined, {
        width: 140,
        boxWidth: 160,
        boxHeight: 120,
        boxDepth: 72,
        depthStyle: 'slices',
      });
      const doc = render(d);
      const volumes = [...doc.querySelectorAll('[data-volume]')];
      expect(volumes).toHaveLength(count);
      expect(doc.querySelector('.tv-ellipsis')?.textContent ?? '').toBe(elided ? '···' : '');
      const bracket = doc
        .querySelector('[data-guide="outer"]')!
        .getAttribute('d')!
        .match(/[-.\d]+/g)!
        .map(Number);
      const svgWidth = Number(doc.querySelector('svg')!.getAttribute('viewBox')!.split(' ')[2]);
      expect(bracket[0]).toBeGreaterThan(0);
      expect(bracket[0] + bracket[3]).toBeLessThan(svgWidth);
      let end = 0;
      for (const volume of volumes) {
        const x = Number(volume.getAttribute('transform')!.match(/[.\d]+/)![0]);
        expect(x).toBeGreaterThan(end);
        const top = volume.querySelector('[data-slice="2"] [data-face="top"]')!.getAttribute('d')!;
        const backOffset = Number(
          volume
            .querySelector('[data-slice="2"]')!
            .getAttribute('transform')!
            .match(/[.\d]+/)![0],
        );
        const [, z, w] = top.match(/l([.\d]+),[-.\d]+h([.\d]+)/)!;
        end = x + backOffset + Number(z) + Number(w);
      }
      expect(end).toBeCloseTo(bracket[0] + bracket[3]);
    },
  );

  it('paints outer joins and deep joins in source order, including explicit solid mode', () => {
    const segments = [
      { port: 'a', size: 4, start: 0, end: 0.5, tones: ['teal' as const] },
      { port: 'b', size: 4, start: 0.5, end: 1, tones: ['#ff8844' as const] },
    ];
    const outer = render(
      buildTensorBlockDiagram(tensor([8, 16, 8, 8]), 'out', undefined, {
        join: { axis: 0, direction: 'outer', segments },
      }),
    );
    const fills = [...outer.querySelectorAll('[data-volume]')].map((v) => [
      v.getAttribute('data-segment'),
      v.querySelector('[data-face="front"]')?.getAttribute('fill'),
    ]);
    expect(fills).toEqual([
      ['a', 'var(--tv-teal-front)'],
      ['b', '#ff8844'],
    ]);

    const depthJoin = { axis: 0, direction: 'depth' as const, segments };
    const joined = (depthStyle: 'auto' | 'solid') =>
      render(
        buildTensorBlockDiagram(tensor([8, 8, 8]), 'out', undefined, {
          depthStyle,
          join: depthJoin,
        }),
      );
    const slices = joined('auto');
    expect(
      [...slices.querySelectorAll('[data-slice]')].map((s) => s.getAttribute('data-segment')),
    ).toEqual(['b', 'b', 'a']);
    const solid = joined('solid');
    expect(solid.querySelectorAll('[data-slice]')).toHaveLength(1);
    expect(solid.querySelectorAll('[data-face="front"]')).toHaveLength(1);
    expect(
      solid.querySelector('[data-segment="b"] [data-face="side"]')?.getAttribute('fill'),
    ).toContain('#ff8844');
  });

  it.each(['query', 'add', 'named', 'layers'])(
    'renders the inferred faces, layers and callouts for %s',
    (id) => {
      const node = project.graph.nodes.find((n) => n.id === id)!;
      const diagram = buildTensorGeometryDiagram(node, result.nodes[id], tones, {
        mode: 'block',
        referenceMaxDim: resolveGraphReferenceMaxDim(project.graph, result),
      })!;
      const html = renderToStaticMarkup(createElement(TensorGeometryView, { diagram }));
      const svg = parseHTML(html).document.querySelector('svg')!;
      expect(svg.getAttribute('role')).toBe('img');
      expect(svg.querySelector('title')?.textContent).toContain('float32');
      for (const face of ['top', 'front', 'side'])
        expect(svg.querySelector(`[data-face="${face}"]`)).toBeTruthy();
      const output = svg.querySelector('[data-box="output"]')!;
      expect(svg.querySelectorAll('[data-box]')).toHaveLength(1);
      expect(svg.querySelector('.tv-symbol')).toBeNull();
      expect(svg.textContent).not.toMatch(/[→↓+‖]/);
      if (id === 'add') {
        expect(output.querySelector('[data-face="front"]')?.getAttribute('fill')).toMatch(
          /^url\(#.+blue-violet-front\)$/,
        );
      } else if (id === 'named') {
        expect(output.querySelector('[data-segments="width"] .tv-seam')).toBeTruthy();
        expect(output.querySelector('[data-segment="a"] path')?.getAttribute('fill')).toBe(
          'var(--tv-blue-front)',
        );
        expect(output.querySelector('[data-segment="b"] path')?.getAttribute('fill')).toContain(
          'blue-violet-front',
        );
        expect(
          Array.from(svg.querySelectorAll('[data-direction="width"] text'), (t) => t.textContent),
        ).toEqual(['256', '|', '256']);
      } else if (id === 'layers') {
        expect(output.querySelectorAll('[data-slice]')).toHaveLength(2);
        expect(output.querySelector('[data-slice="1"]')?.getAttribute('data-segment')).toBe('b');
        expect(output.textContent).toContain('B 2');
      }
      for (const path of svg.querySelectorAll('path'))
        expect(path.getAttribute('d')).not.toMatch(/NaN|Infinity|undefined/);
    },
  );
});

describe('canvas and inspector', () => {
  let root: Root, dispose: () => void;
  const state = useEditor.getState;
  const nodeCard = (id: string) => document.querySelector(`[data-testid="node-${id}"]`)!;
  const button = (text: string) =>
    [...document.querySelectorAll<HTMLButtonElement>('.tensor-visual-panel button')].find(
      (b) => b.textContent === text,
    )!;
  beforeEach(async () => {
    const { window } = parseHTML('<html><body><main></main></body></html>');
    vi.stubGlobal('window', window);
    vi.stubGlobal('document', window.document);
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.stubGlobal('requestAnimationFrame', (fn: FrameRequestCallback) =>
      setTimeout(() => fn(0), 0),
    );
    vi.stubGlobal('cancelAnimationFrame', clearTimeout);
    dispose = configureEditor({
      kind: 'web',
      files: { download: vi.fn() },
      preferences: { getTheme: () => 'light', setTheme: vi.fn() },
    });
    state().load(project);
    useEditor.setState({ globalTensorBlocks: false });
    state().select('pos');
    root = createRoot(document.querySelector('main')!);
    await act(() => root.render(createElement(App)));
    remeasure.mockClear();
  });
  afterEach(async () => {
    await act(() => root.unmount());
    dispose();
    vi.unstubAllGlobals();
  });
  it('toolbar previews inferred tensors, respects media and off, restores the compact node and its handles', async () => {
    const previous = state().project,
      past = state().past;
    const compact = nodeCard('query').outerHTML;
    const media = nodeCard('pos').outerHTML;
    const toggle = document.querySelector<HTMLButtonElement>('[aria-label="全画布 3D 张量积木"]')!;
    expect(nodeCard('query').querySelector('figure')).toBeNull();
    await act(() => toggle.click());
    expect(toggle.getAttribute('aria-pressed')).toBe('true');
    for (const id of ['add', 'features', 'history', 'linear', 'reshape']) {
      expect(nodeCard(id).querySelector('.tensor-geometry-card')).toBeNull();
      expect(remeasure).not.toHaveBeenCalledWith(id);
    }
    for (const id of ['query', 'named', 'layers', 'narrow']) {
      expect(nodeCard(id).classList.contains('illustrated-tensor-node')).toBe(true);
      expect(nodeCard(id).querySelector('.tensor-node')).toBeNull();
      expect(nodeCard(id).querySelector('.tensor-card-header')).toBeTruthy();
      expect(nodeCard(id).querySelector('.tv-heading')).toBeNull();
      expect(nodeCard(id).querySelectorAll('.tensor-geometry-card [data-box]')).toHaveLength(1);
    }
    const faceWidth = (card: Element) =>
      Number(
        card
          .querySelector('[data-face="front"]')!
          .getAttribute('d')!
          .match(/h([\d.]+)/)![1],
      );
    expect(Math.round(faceWidth(nodeCard('query')))).toBe(69);
    expect(Math.round(faceWidth(nodeCard('narrow')))).toBe(23);
    await act(() => state().select('narrow'));
    expect(faceWidth(document.querySelector('.tensor-inspector')!)).toBe(
      faceWidth(nodeCard('narrow')),
    );
    expect(nodeCard('out').querySelector('figure')).toBeNull();
    await act(() => state().select('pos'));
    expect(nodeCard('pos').outerHTML).toBe(media);
    expect(nodeCard('add').querySelectorAll('.react-flow__handle')).toHaveLength(3);
    expect(remeasure).toHaveBeenCalledWith('query');
    await act(() => toggle.click());
    expect(nodeCard('query').outerHTML).toBe(compact);
    expect(state().project).toBe(previous);
    expect(state().past).toBe(past);
  });
  it('edits local geometry alongside media, remeasures, undoes, redoes and resets it', async () => {
    const config = () => state().project.graph.nodes.find((n) => n.id === 'pos')!.tensorVisual;
    await act(() => button('3D/4D 积木').click());
    await act(() =>
      document.querySelector<HTMLButtonElement>('.tv-swatches [aria-label="玫红"]')!.click(),
    );
    expect(config()).toEqual({ mode: 'block', tone: 'rose' });
    const card = nodeCard('pos');
    expect(card.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,YQ==');
    expect(
      card.querySelector('.tensor-geometry-svg [data-face="front"]')?.getAttribute('fill'),
    ).toBe('var(--tv-rose-front)');
    expect(card.querySelectorAll('.react-flow__handle')).toHaveLength(1);
    expect(remeasure).toHaveBeenCalledWith('pos');
    await act(() => state().undo());
    expect(config()).toEqual({ mode: 'block' });
    await act(() => {
      state().redo();
      state().select('pos');
    });
    expect(config()?.tone).toBe('rose');
    await act(() => button('恢复默认').click());
    expect(config()).toBeUndefined();
    expect(nodeCard('pos').querySelector('.tensor-geometry-card')).toBeNull();
  });
  it('edits visual parameters live and restores each change through undo and redo', async () => {
    await act(() => state().select('query'));
    await act(() => button('3D/4D 积木').click());
    const config = () => state().project.graph.nodes.find((n) => n.id === 'query')!.tensorVisual!;
    const input = (label: string) =>
      document.querySelector<HTMLInputElement>(`.tensor-visual-panel [aria-label="${label}"]`)!;
    const edit = async (label: string, value: string, event = 'input') => {
      const field = input(label);
      field.value = value;
      await act(() => field.dispatchEvent(new window.Event(event, { bubbles: true })));
    };
    const start = state().past.length;
    expect(input('自定义轴名').getAttribute('placeholder')).toBe('B, N, D');
    await edit('自定义颜色', '#2563eb');
    expect(nodeCard('query').getAttribute('style')).toContain('--tensor-card-accent:#2563eb');
    expect(nodeCard('query').querySelector('figure')?.getAttribute('style')).toContain('#2563eb');
    await act(() => button('实心体积').click());
    await edit('积木宽度', '400');
    await edit('积木块宽', '90');
    await edit('积木块高', '40');
    await edit('积木块深', '55');
    await edit('自定义轴名', ' query，head, dim ', 'focusout');
    await edit('积木说明', '参数预览', 'focusout');
    const final = {
      mode: 'block',
      color: '#2563eb',
      depthStyle: 'solid',
      width: 400,
      boxWidth: 90,
      boxHeight: 40,
      boxDepth: 55,
      axisLabels: ['query', 'head', 'dim'],
      caption: '参数预览',
    };
    expect(config()).toEqual(final);
    expect(nodeCard('query').querySelector('[data-face="top"]')?.getAttribute('d')).toContain(
      'l55,-33h90',
    );
    expect(nodeCard('query').querySelector('[data-face="front"]')?.getAttribute('d')).toContain(
      'v40',
    );
    expect(nodeCard('query').querySelector('.tv-dim')?.textContent).toContain('query 1');
    expect(nodeCard('query').querySelector('figcaption')?.textContent).toBe('参数预览');
    const steps = state().past.length - start;
    expect(steps).toBe(8);
    for (let i = 0; i < steps; i++) await act(() => state().undo());
    expect(config()).toEqual({ mode: 'block' });
    for (let i = 0; i < steps; i++) await act(() => state().redo());
    expect(config()).toEqual(final);
    await act(() => state().select('query'));
    await act(() => input('块深 Z恢复自动').click());
    await act(() =>
      document.querySelector<HTMLButtonElement>('.tv-swatches [aria-label="紫"]')!.click(),
    );
    await edit('自定义轴名', '', 'focusout');
    await edit('积木说明', '', 'focusout');
    expect(config()).toEqual({
      mode: 'block',
      tone: 'violet',
      depthStyle: 'solid',
      width: 400,
      boxWidth: 90,
      boxHeight: 40,
    });
    expect(nodeCard('query').querySelector('[data-face="front"]')?.getAttribute('fill')).toBe(
      'var(--tv-violet-front)',
    );
  });

  it('applies a parametric runtime API patch to the canvas as one undoable edit', async () => {
    const snapshot = canvasApi.getModel();
    const visual = {
      mode: 'block',
      color: '#fd024e',
      depthStyle: 'slices',
      width: 390,
      boxDepth: 16,
      boxHeight: 60,
      boxWidth: 84,
      axisLabels: ['Z', 'H', 'W'],
    };
    const prepared = canvasApi.patchModel(snapshot.project, {
      operations: [{ op: 'set_node_tensor_visual', id: 'query', visual }],
    });
    expect(prepared.ok).toBe(true);
    await act(() => {
      expect(
        canvasApi.applyModel({
          project: prepared.project,
          expectedRevision: snapshot.revision,
          allowWarnings: true,
        }).ok,
      ).toBe(true);
    });
    expect(nodeCard('query').querySelector('[data-face="front"]')?.getAttribute('d')).toContain(
      'h84v60',
    );
    expect(
      canvasApi.getModel().project.graph.nodes.find((n) => n.id === 'query')?.tensorVisual,
    ).toEqual(visual);
    await act(() => state().undo());
    expect(nodeCard('query').querySelector('figure')).toBeNull();
  });

  it('inspects a segmented result while the canvas is off and explicitly enables an operator block', async () => {
    await act(() => state().select('named'));
    const card = document.querySelector('.tensor-inspector .tensor-geometry-card')!;
    expect(card.querySelector('.tv-seam')).toBeTruthy();
    const output = card.querySelector('[data-box="output"]')!;
    await act(() => output.dispatchEvent(new window.Event('click', { bubbles: true })));
    expect(card.querySelector('.tv-detail')?.textContent).toBe('B=1 · N=256 · D=512 · float32');
    expect(nodeCard('named').querySelector('figure')).toBeNull();

    await act(() => state().select('features'));
    expect(document.querySelectorAll('.tv-options button')).toHaveLength(2);
    await act(() => button('3D/4D 积木').click());
    expect(nodeCard('features').querySelectorAll('[data-box]')).toHaveLength(1);
    expect(nodeCard('features').querySelector('.tv-seam')).toBeTruthy();
    const width = document.querySelector('[aria-label="积木宽度"]')!;
    expect([width.getAttribute('min'), width.getAttribute('max')]).toEqual(['140', '420']);
  });
  it('switches the single inspector block to the selected output port', async () => {
    await act(() => state().select('chunk'));
    const inspector = document.querySelector('.tensor-inspector')!;
    const tabs = inspector.querySelectorAll<HTMLButtonElement>('.port-tabs button');
    expect(tabs).toHaveLength(3);
    const widthLabel = () => inspector.querySelector('[data-box] .tv-dim text')!.textContent;
    expect(widthLabel()).toContain('11');
    await act(() => tabs[2].click());
    expect(widthLabel()).toContain('10');
    expect(inspector.querySelectorAll('[data-box]')).toHaveLength(1);
  });
});
