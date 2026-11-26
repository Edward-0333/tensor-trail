import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { beforeEach, describe, expect, it } from 'vitest';
import type { Graph, ModelNode, TensorSpec } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { DEFAULT_TENSOR_PALETTE } from '@tensor-trail/model-core/core/tensorAppearance';
import { levelInputs } from '../src/views/levelInputs';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { LevelInputsBar } from '../src/editor/LevelInputsBar';
import { useEditor } from '../src/editor/store';

const inlet = (id: string) => registry.create('Inlet', { port: id }, id);
const graphOf = (...nodes: ModelNode[]): Graph => ({ nodes, edges: [] });
const alias = (node: ModelNode): ModelNode => ({ ...node, id: `${node.id}_copy`, valueSource: node.id });

function renderNode(graph: Graph, id: string, root: boolean, spec?: TensorSpec) {
  const model = graph.nodes.find(node => node.id === id)!;
  const props: NodeProps<TrailNode> = {
    id, type: 'module', selected: false, dragging: false, zIndex: 0,
    selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, ...ports(model, registry),
      inputBoundary: levelInputs(graph, root).find(input => input.id === id),
      ...(spec ? { result: { status: 'ok' as const, inputs: {}, outputs: { out: spec }, notes: [], messages: [] } } : {}),
      category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {},
    },
  };
  return renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
}

beforeEach(() => useEditor.getState().load({
  schemaVersion: 1, name: 'Inputs', variants: [], graph: graphOf(),
}));

describe('Current-level input identity', () => {
  it('counts only root input types, not parameters, attributes, variables, operators or child inputs', () => {
    const graph = graphOf(
      registry.create('Input', {}, 'image'),
      registry.create('ForwardInput', {}, 'metadata'),
      registry.create('DictionaryInput', {}, 'batch'),
      registry.create('TensorVariable', {}, 'features'),
      registry.create('LocalParameter', {}, 'B'),
      registry.create('GetAttribute', {}, 'attribute'),
      registry.create('GetParameter', {}, 'weight'),
      registry.create('BufferSource', {}, 'buffer'),
      registry.create('Linear', {}, 'linear'),
      registry.create('Submodule', {}, 'child'),
      registry.create('Output', {}, 'result'),
    );
    const before = structuredClone(graph);
    expect(levelInputs(graph, true).map(({ id, index, total }) => ({ id, index, total }))).toEqual([
      { id: 'image', index: 1, total: 3 },
      { id: 'metadata', index: 2, total: 3 },
      { id: 'batch', index: 3, total: 3 },
    ]);
    expect(graph).toEqual(before);
  });

  it('uses only Inlets in a subgraph, in the same order as its parent ports', () => {
    const module = registry.create('Submodule', {}, 'transformer');
    module.subgraph = graphOf(
      inlet('query_points'),
      registry.create('Input', {}, 'internal_constant'),
      inlet('query_feat'), inlet('mlvl_feats'), inlet('img_metas'),
      registry.create('Outlet', { port: 'out' }, 'out'),
      registry.create('Submodule', {}, 'nested'),
    );
    const inputs = levelInputs(module.subgraph, false);
    expect(inputs.map(input => input.port)).toEqual(ports(module, registry).inputs.map(port => port.id));
    expect(inputs.map(input => input.index)).toEqual([1, 2, 3, 4]);
    expect(inputs.every(input => input.total === 4)).toBe(true);
    expect(renderNode(module.subgraph, 'internal_constant', false)).not.toContain('level-input-badge');
  });

  it.each(['Input', 'ForwardInput', 'DictionaryInput', 'Inlet'])(
    'does not double-count or badge an associated %s copy', type => {
      const node = registry.create(type, {}, 'source');
      const graph = graphOf(alias(node), node);
      const inputs = levelInputs(graph, type !== 'Inlet');
      expect(inputs.map(input => input.id)).toEqual(['source']);
      expect(renderNode(graph, 'source_copy', type !== 'Inlet')).not.toContain('level-input-badge');
      const html = renderNode(graph, 'source', type !== 'Inlet');
      expect(html).toContain('class="level-input-badge"');
      expect(html).toContain('IN · 1/1');
      expect(html).toContain('aria-label="本层输入 1/1"');
    },
  );

  it('keeps ordinals stable when nodes move and updates names and totals without writing metadata', () => {
    const graph = graphOf(inlet('a'), inlet('b'));
    graph.nodes[0].position = { x: 1200, y: 700 };
    graph.nodes[1].position = { x: -300, y: -600 };
    expect(levelInputs(graph, false).map(input => input.id)).toEqual(['a', 'b']);
    graph.nodes[0].label = 'renamed';
    graph.nodes.push(inlet('c'));
    const before = JSON.stringify(graph);
    expect(levelInputs(graph, false)[0]).toEqual({ id: 'a', label: 'renamed', port: 'a', index: 1, total: 3 });
    renderNode(graph, 'a', false);
    expect(JSON.stringify(graph)).toBe(before);
  });

  it.each([true, false])('preserves requiresGrad=%s, custom borders and handles alongside input identity', requiresGrad => {
    const node = inlet('query_feat');
    node.tensorBorderColor = '#123456';
    const spec: TensorSpec = {
      shape: [2, 1040, 256], axes: ['batch', 'query', 'embedding'], dtype: 'float32',
      autograd: { requiresGrad, source: 'manual' },
    };
    const html = renderNode(graphOf(node), node.id, false, spec);
    const state = requiresGrad ? 'on' : 'off';
    expect(html).toContain(`--tensor-bg:${DEFAULT_TENSOR_PALETTE[state]}`);
    expect(html).toContain('--tensor-border:#123456');
    expect(html).toContain(`data-grad-state="${state}"`);
    expect(html).toContain('IN · 1/1');
    expect(html).toContain('aria-label="输出 out"');
    expect(html).not.toContain('aria-label="输入 in"');
  });

  it('renders one navigation entry per container with incomplete-count wording', () => {
    const graph = graphOf(inlet('mlvl_feats'), inlet('img_metas'));
    graph.nodes.push(alias(graph.nodes[0]));
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null,
      createElement(LevelInputsBar, { graph, root: false })));
    expect(html).toContain('已绘制输入');
    expect(html).toContain('class="level-inputs-count">2</span>');
    expect(html).toContain('不代表代码参数已完整录入');
    expect(html).toContain('aria-label="定位输入 1/2：模块输入"');
    expect(html).toContain('data-input-id="mlvl_feats"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).not.toContain('data-input-id="mlvl_feats_copy"');
    expect(html.match(/<button /g)).toHaveLength(2);
  });

  it('shows a neutral zero count for a layer with no drawn inputs', () => {
    const graph = graphOf();
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null,
      createElement(LevelInputsBar, { graph, root: false })));
    expect(html).toContain('class="level-inputs-count">0</span>');
    expect(html).toContain('尚未绘制输入');
    expect(html).not.toContain('<button');
  });

  it('badges split input aliases with IN without double-counting and splits fan-out 1-to-1 with undo/redo', () => {
    const source = inlet('query');
    source.tensorVisual = { mode: 'block', tone: 'blue', width: 200 };
    const copy = alias(source);
    const graph = graphOf(source, copy);
    const boundary = levelInputs(graph, false)[0];
    const props: NodeProps<TrailNode> = {
      id: copy.id, type: 'module', selected: false, dragging: false, zIndex: 0,
      selectable: true, deletable: true, draggable: true, isConnectable: true,
      positionAbsoluteX: 0, positionAbsoluteY: 0,
      data: {
        model: copy, ...ports(copy, registry),
        inputAlias: boundary,
        category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {},
      },
    };
    const aliasHtml = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
    expect(aliasHtml).toContain('level-input-alias-badge');
    expect(aliasHtml).toContain('IN · 1/1');

    const add1 = registry.create('Add', {}, 'add1');
    const add2 = registry.create('Add', {}, 'add2');
    const add3 = registry.create('Add', {}, 'add3');
    const sub = registry.create('Submodule', {}, 'sub');
    sub.subgraph = {
      nodes: [source, add1, add2, add3, registry.create('Outlet', { port: 'out' }, 'out')],
      edges: [
        { id: 'e1', source: 'query', sourcePort: 'out', target: 'add1', targetPort: 'a' },
        { id: 'e2', source: 'query', sourcePort: 'out', target: 'add2', targetPort: 'a', presentation: 'anchor' },
        { id: 'e3', source: 'query', sourcePort: 'out', target: 'add3', targetPort: 'a' },
      ],
    };
    useEditor.getState().load({ schemaVersion: 1, name: 'Split', variants: [], graph: { nodes: [sub], edges: [] } });
    useEditor.getState().navigate(['sub']);
    expect(useEditor.getState().splitInputFanout('query')).toBe(true);
    const splitGraph = useEditor.getState().project.graph.nodes[0].subgraph!;
    const inlets = splitGraph.nodes.filter(n => n.type === 'Inlet');
    expect(inlets).toHaveLength(3);
    expect(levelInputs(splitGraph, false)).toHaveLength(1);
    for (const node of inlets) {
      expect(splitGraph.edges.filter(e => e.source === node.id)).toHaveLength(1);
      expect(node.tensorVisual).toEqual(source.tensorVisual);
    }
    expect(splitGraph.edges.some(e => e.presentation === 'anchor')).toBe(false);
    useEditor.getState().undo();
    const undone = useEditor.getState().project.graph.nodes[0].subgraph!;
    expect(undone.nodes.filter(n => n.type === 'Inlet')).toHaveLength(1);
    expect(undone.edges.filter(e => e.source === 'query')).toHaveLength(3);
    useEditor.getState().redo();
    expect(useEditor.getState().project.graph.nodes[0].subgraph!.nodes.filter(n => n.type === 'Inlet')).toHaveLength(3);
  });
});
