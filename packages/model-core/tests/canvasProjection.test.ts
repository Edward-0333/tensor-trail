import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel } from '../src/agent';
import { projectCanvasGraph } from '../src/core/canvasProjection';
import { inferProject } from '../src/core/engine';
import { setNodePresentation, validateNodePresentation } from '../src/core/nodePresentation';
import { parseProject, serializeProject } from '../src/core/serialization';
import { resolveGraphTensorTones } from '../src/core/tensorGeometry';
import type { Graph, ModelNode, Params } from '../src/core/types';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';

const chain = {
  name: 'Attention values',
  nodes: [
    { id: 'val_queue', type: 'Input', params: { shape: '2,256,256', axes: 'B,N,C' } },
    { id: 'linear', type: 'Linear', params: { in_features: 256, out_features: 256 } },
    { id: 'reshape', type: 'Reshape', params: { shape: '2, 256, 8, 32' } },
    { id: 'value_8heads', type: 'Output' },
  ],
  connections: [
    ['val_queue.out', 'linear.in'],
    ['linear.out', 'reshape.in'],
    ['reshape.out', 'value_8heads.in'],
  ],
};
function linearGraph(): Graph {
  return buildModel({
    name: 'Offsets',
    nodes: [
      {
        id: 'query_cat_512',
        type: 'Input',
        label: 'query · 轨迹查询 (兼 identity)',
        params: { shape: '512', axes: '' },
      },
      {
        id: 'offset',
        type: 'Linear',
        presentation: 'edge',
        params: { in_features: 512, out_features: 128 },
      },
      { id: 'offsets_128', type: 'Output' },
    ],
    connections: [
      ['query_cat_512.out', 'offset.in'],
      ['offset.out', 'offsets_128.in'],
    ],
  }).project!.graph;
}
const project = (graph: Graph) => projectCanvasGraph(graph, undefined, new Map());

describe('presentation contract', () => {
  it.each([
    ['Input', 'compact'],
    ['Reshape', 'compact'],
    ['Add', 'edge'],
    ['IfRegion', 'compact'],
    ['IfRegion', 'edge'],
    ['LoopRegion', 'default'],
    ['LayoutRegion', 'compact'],
    ['Linear', 'unknown'],
  ])('rejects %s with %s in recipes and project files', (type, presentation) => {
    const model = registry.create(type);
    expect(() => validateNodePresentation(model, presentation)).toThrow();
    const invalid = { id: model.id, type, presentation };
    expect(buildModel({ ...chain, nodes: [...chain.nodes, invalid] }, { mode: 'draft' }).ok).toBe(
      false,
    );
    const p = buildModel(chain).project!;
    p.graph.nodes.push(Object.assign(model, { presentation }));
    expect(() => parseProject(serializeProject(p), registry)).toThrow();
  });

  it('accepts explicit defaults, compact junctions and unary operators, and clears the field on reset', () => {
    const schemas = getSchemas();
    const ajv = new Ajv2020({ strict: false, inlineRefs: false });
    const validateRecipe = ajv.compile(schemas.recipe);
    const validatePatch = ajv.compile(schemas.patch);
    expect(
      validateRecipe({ ...chain, nodes: [{ id: 'n', type: 'Add', presentation: 'edge' }] }),
    ).toBe(false);
    expect(
      validatePatch({
        operations: [{ op: 'set_node_presentation', id: 'node', presentation: null }],
      }),
    ).toBe(true);
    expect(
      validatePatch({
        operations: [
          { op: 'set_edge_presentation', from: 'a.out', to: 'b.in', presentation: 'anchor' },
        ],
      }),
    ).toBe(true);
    for (const [type, style] of [
      ['Input', 'default'],
      ['Concat', 'compact'],
      ['Linear', 'compact'],
      ['LayerNorm', 'compact'],
      ['Dropout', 'edge'],
      ['SemanticOp', 'edge'],
      ['TensorIndex', 'edge'],
    ]) {
      const node = registry.create(type);
      validateNodePresentation(node, style);
      expect(validateRecipe({ ...chain, nodes: [{ id: 'n', type, presentation: style }] })).toBe(
        true,
      );
      setNodePresentation(node, style);
      for (const reset of ['default', null]) {
        setNodePresentation(node, reset);
        expect(node).not.toHaveProperty('presentation');
      }
    }
  });

  it('round-trips style patches, including anchor, with identical inference and CLI traces', () => {
    const before = buildModel(chain);
    const edited = patchModel(
      before.project,
      {
        operations: [
          { op: 'set_node_presentation', id: 'linear', presentation: 'edge' },
          {
            op: 'set_edge_presentation',
            from: 'reshape.out',
            to: 'value_8heads.in',
            presentation: 'anchor',
          },
        ],
      },
      { layout: 'auto' },
    );
    expect(edited.ok).toBe(true);
    expect(edited.nodes).toEqual(before.nodes);
    expect(edited.outputs).toEqual(before.outputs);
    expect(edited.project!.graph.nodes[1].presentation).toBe('edge');
    expect(edited.project!.graph.edges[2].presentation).toBe('anchor');
    expect(parseProject(serializeProject(edited.project!), registry)).toEqual(edited.project);
    expect(
      patchModel(edited.project, {
        operations: [
          { op: 'set_node_presentation', id: 'linear', presentation: null },
          {
            op: 'set_edge_presentation',
            from: 'reshape.out',
            to: 'value_8heads.in',
            presentation: null,
          },
        ],
      }).project,
    ).toEqual(before.project);
  });

  it('preserves styling on surviving generated nodes after changing a parent parameter', () => {
    const p = example();
    const parent = p.graph.nodes.find((n) => n.id === 'transformer')!;
    const node = parent.subgraph!.nodes.find((n) => n.type === 'LayerNorm')!;
    const styled = patchModel(p, {
      operations: [
        { op: 'set_node_presentation', path: [parent.id], id: node.id, presentation: 'edge' },
      ],
    });
    expect(styled.project!.graph.nodes.find((n) => n.id === parent.id)!.customized).toBe(
      parent.customized,
    );
    const rebuilt = patchModel(styled.project, {
      operations: [{ op: 'set_params', id: parent.id, params: { heads: 4 } }],
    });
    expect(
      rebuilt
        .project!.graph.nodes.find((n) => n.id === parent.id)!
        .subgraph!.nodes.find((n) => n.id === node.id)?.presentation,
    ).toBe('edge');
  });
});

describe('canvas projection', () => {
  function selectionGraph(params: Params = {}) {
    const built = buildModel({
      name: 'Indexed features',
      hyperparameters: { end: { type: 'int', value: 5 }, axis: { type: 'int', value: 1 } },
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,512,1,256', axes: 'B,Q,T,C' } },
        { id: 'pick', type: 'TensorIndex', params, presentation: 'edge' },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'pick.in'], ['pick.out', 'out.in']],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    return built.project!;
  }

  it.each([
    [{ dim: '1', index: '-1' }, '[:, -1]', '[2, 1, 256]'],
    [{ mode: 'slice', dim: '1', start: '1', stop: '2' }, '[:, 1:2]', '[2, 1, 1, 256]'],
    [{ mode: 'slice', dim: '-1', stop: 'end - 1', step: 2 }, '[..., :end - 1:2]', '[2, 512, 1, 2]'],
    [{ dim: 'axis', index: 'end - 1' }, '[轴 axis: end - 1]', '[2, 1, 256]'],
  ] as const)('collapses a single selection %o into %s', (params, badge, shapeText) => {
    const p = selectionGraph(params);
    const view = projectCanvasGraph(p.graph, inferProject(p, registry), new Map());
    expect(view.hiddenNodeIds).toEqual(new Set(['pick']));
    expect(view.edges).toHaveLength(1);
    expect(view.edges[0]).toMatchObject({
      source: 'x', target: 'out', edgeOps: [{ id: 'pick', badge, shapeText, status: 'ok' }],
    });
  });

  it.each([
    ['Cos', {}, 'cos()', 'orange'], ['Sin', {}, 'sin()', 'orange'],
    ['Function', { expression: '  x * 2 - 1  ' }, 'x * 2 - 1', 'orange'],
    ['Function', { expression: ' '.repeat(3) }, '映射', 'orange'],
    ['Function', { expression: 'x' + ' + 1'.repeat(9) }, ('x' + ' + 1'.repeat(9)).slice(0, 23) + '…', 'orange'],
    ['TensorCompareScalar', { operator: 'gt', value: ' 1e-5 ' }, '> 1e-5', 'pink'],
    ['TensorCompareScalar', { operator: 'gt', value: '0.0' }, '> 0.0', 'pink'],
    ['TensorCompareScalar', { operator: 'lt', value: '1.0' }, '< 1.0', 'pink'],
    ['TensorCompareScalar', { operator: 'ge', value: '0' }, '>= 0', 'pink'],
    ['TensorCompareScalar', { operator: 'le', value: '1' }, '<= 1', 'pink'],
    ['OnesLike', {}, 'ones_like', 'teal'], ['ZerosLike', {}, 'zeros_like', 'teal'],
  ] as const)('folds %s %o only on explicit request', (type, params, badge, tone) => {
    const graph = linearGraph();
    const op = registry.create(type, params, 'offset');
    op.label = '映射';
    graph.nodes[1] = op;
    expect(project(graph).hiddenNodeIds.size).toBe(0);
    setNodePresentation(op, 'edge');
    const original = structuredClone(graph);
    expect(project(graph).edges).toMatchObject([{ edgeOps: [{ id: 'offset', type, badge, tone }] }]);
    expect(graph).toEqual(original);
  });

  it.each([
    ['   ', '  映射  ', '映射'],
    [' \t ', ' \t ', 'Function'],
    ['  x' + ' + 1'.repeat(9) + '  ', '映射', 'x' + ' + 1'.repeat(9)],
  ])('retains the full Function text and trims both fallback values', (expression, label, detail) => {
    const graph = linearGraph();
    const op = registry.create('Function', { expression }, 'offset');
    Object.assign(op, { label, presentation: 'edge' });
    graph.nodes[1] = op;
    expect(project(graph).edges[0].edgeOps[0]).toMatchObject({
      detail, badge: detail.length <= 24 ? detail : detail.slice(0, 23) + '…',
    });
  });

  it('collapses Unflatten onto the edge and preserves the symbolic dimension', () => {
    const built = buildModel({
      name: 'Unflatten features',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '1,M,256', axes: 'B,M,C' } },
        {
          id: 'unflatten',
          type: 'Unflatten',
          params: { dim: -1, sizes: '8, 1, 16, 2' },
          presentation: 'edge',
        },
        { id: 'out', type: 'Output' },
      ],
      connections: [['x.out', 'unflatten.in'], ['unflatten.out', 'out.in']],
    });
    expect(built.diagnostics).toEqual([]);
    const p = built.project!;
    const result = inferProject(p, registry);
    expect(result.nodes.unflatten.outputs.out).toMatchObject({ shape: [1, 'M', 8, 1, 16, 2] });
    const view = projectCanvasGraph(p.graph, result, new Map());
    expect(view.hiddenNodeIds).toEqual(new Set(['unflatten']));
    expect(view.edges).toMatchObject([{
      source: 'x',
      target: 'out',
      edgeOps: [{
        id: 'unflatten',
        badge: 'unflatten(-1,(8,1,16,2))',
        shapeText: '[1, M, 8, 1, 16, 2]',
        status: 'ok',
      }],
    }]);
  });

  it('stacks [0] above permute and keeps their intermediate and final shapes', () => {
    const p = selectionGraph();
    const permute = registry.create('Permute', { dims: '1, 0, 2' }, 'permute');
    permute.presentation = 'edge';
    p.graph.nodes.push(permute);
    const last = p.graph.edges[1];
    p.graph.edges.push({ ...last, id: 'to-permute', target: permute.id });
    last.source = permute.id;
    const view = projectCanvasGraph(p.graph, inferProject(p, registry), new Map());
    expect(view.hiddenNodeIds).toEqual(new Set(['pick', 'permute']));
    expect(view.edges).toHaveLength(1);
    expect(view.edges[0].edgeOps).toMatchObject([
      { id: 'pick', badge: '[0]', shapeText: '[512, 1, 256]' },
      { id: 'permute', badge: 'permute(1,0,2)', shapeText: '[1, 512, 256]' },
    ]);
  });

  it('normalizes legacy Slice defaults for the badge without editing the saved node', () => {
    const p = selectionGraph();
    const old = p.graph.nodes[1];
    old.type = 'Slice';
    old.params = {};
    expect(project(p.graph).edges[0].edgeOps).toMatchObject([{ id: 'pick', type: 'Slice', badge: '[..., :2]' }]);
    old.params = { dim: 1, start: '1', stop: 'end', step: 2 };
    expect(project(p.graph).edges[0].edgeOps[0].badge).toBe('[:, 1:end:2]');
    expect(old).toMatchObject({ type: 'Slice', params: { dim: 1, start: '1', stop: 'end', step: 2 } });
  });

  it.each(['TensorIndex', 'Slice'])('retains a dynamic %s with its stop port both disconnected and connected', type => {
    const { graph } = selectionGraph({ mode: 'slice' });
    graph.nodes[1].type = type;
    graph.nodes[1].params.stopFromInput = true;
    expect(project(graph).visibleNodes).toEqual(graph.nodes);
    graph.nodes.push(registry.create('IntegerParameter', { expression: '1' }, 'stop'));
    graph.edges.push({ id: 'bound', source: 'stop', sourcePort: 'out', target: 'pick', targetPort: 'stop' });
    expect(project(graph).visibleNodes).toEqual(graph.nodes);
    expect(project(graph).edges.every(edge => !edge.edgeOps.length)).toBe(true);
  });

  it('keeps branched and cross-boundary selections visible', () => {
    const { graph } = selectionGraph();
    graph.nodes.push(registry.create('Output', {}, 'second'));
    graph.edges.push({ ...graph.edges[1], id: 'fork', target: 'second' });
    expect(project(graph).visibleNodes).toEqual(graph.nodes);
    graph.edges.pop();
    graph.nodes.push(registry.create('LayoutRegion', { members: 'x,pick' }, 'region'));
    expect(project(graph).visibleNodes).toEqual(graph.nodes);
  });

  it('keeps an explicitly marked layout output visible within a single region', () => {
    const { graph } = selectionGraph();
    const region = registry.create('LayoutRegion', { members: 'x,pick,out', outputs: 'pick' }, 'region');
    graph.nodes.push(region);
    const view = project(graph);
    expect(view.visibleNodes).toEqual(graph.nodes);
    expect(view.edges.map(edge => [edge.source, edge.target])).toEqual([['x', 'pick'], ['pick', 'out']]);
    region.params.outputs = '';
    expect(project(graph).hiddenNodeIds.has('pick')).toBe(true);
  });

  it('collapses a single linear layer and retains its real node identity', () => {
    const projection = project(linearGraph());
    expect(projection.visibleNodes.map((n) => n.id)).toEqual(['query_cat_512', 'offsets_128']);
    expect(projection.edges).toHaveLength(1);
    expect(projection.edges[0]).toMatchObject({
      id: 'edge_1',
      source: 'query_cat_512',
      target: 'offsets_128',
      canonicalEdgeIds: ['edge_0', 'edge_1'],
      edgeOps: [{ id: 'offset', badge: 'Linear 512→128', tone: 'blue' }],
    });
  });

  it('orders a pipeline by dataflow, carrying terminal geometry, output shape and slide styling', () => {
    const p = buildModel(chain).project!;
    const original = inferProject(p, registry);
    p.graph.nodes[1].presentation = p.graph.nodes[2].presentation = 'edge';
    Object.assign(p.graph.edges[0], { presentation: 'slide', annotation: 'start' });
    Object.assign(p.graph.edges[2], {
      annotation: 'end',
      waypoints: [{ x: 413, y: 51 }],
      labelPosition: 0.73,
    });
    p.graph.edges.reverse();
    const before = serializeProject(p);
    const result = inferProject(p, registry);
    const view = projectCanvasGraph(p.graph, result, resolveGraphTensorTones(p.graph, result));
    expect(result).toEqual(original);
    expect(view.hiddenNodeIds).toEqual(new Set(['linear', 'reshape']));
    expect(view.edges).toHaveLength(1);
    expect(view.edges[0]).toMatchObject({
      id: 'edge_2',
      source: 'val_queue',
      target: 'value_8heads',
      sourcePort: 'out',
      targetPort: 'in',
      canonicalEdgeIds: ['edge_0', 'edge_1', 'edge_2'],
      presentation: 'slide',
      annotation: 'end',
      labelPosition: 0.73,
      waypoints: [{ x: 413, y: 51 }],
      edgeOps: [
        { badge: 'Linear 256→256', shapeText: '[2, 256, 256]' },
        { badge: 'Reshape(2,256,8,32)', shapeText: '[2, 256, 8, 32]', status: 'ok' },
      ],
    });
    expect(serializeProject(p)).toBe(before);
    delete p.graph.edges[0].annotation;
    expect(project(p.graph).edges[0].annotation).toBe('start');
  });

  it.each([
    'fan-out',
    'missing input',
    'visualMedia',
    'block',
    'valueSource',
    'alias owner',
    'subgraph',
    'incoming anchor',
    'outgoing anchor',
  ])('preserves the card and its original edges for %s', (reason) => {
    const graph = linearGraph(),
      op = graph.nodes[1];
    switch (reason) {
      case 'fan-out':
        graph.nodes.push(registry.create('Output', {}, 'other'));
        graph.edges.push({ ...graph.edges[1], id: 'branch', target: 'other' });
        break;
      case 'missing input':
        graph.edges.shift();
        break;
      case 'visualMedia':
        op.visualMedia = { items: [{ id: 'image', src: 'data:image/png;base64,YQ==' }] };
        break;
      case 'block':
        op.tensorVisual = { mode: 'block' };
        break;
      case 'valueSource':
        op.valueSource = graph.nodes[0].id;
        break;
      case 'alias owner':
        graph.nodes[2].valueSource = op.id;
        break;
      case 'subgraph':
        op.subgraph = { nodes: [], edges: [] };
        break;
      case 'incoming anchor':
        graph.edges[0].presentation = 'anchor';
        break;
      case 'outgoing anchor':
        graph.edges[1].presentation = 'anchor';
        break;
    }
    const view = project(graph);
    expect(view.visibleNodes).toEqual(graph.nodes);
    expect(view.edges.map((e) => e.id)).toEqual(graph.edges.map((e) => e.id));
    expect(view.edges.every((e) => e.edgeOps.length === 0)).toBe(true);
  });

  it.each(['IfRegion', 'LoopRegion', 'LayoutRegion'])('preserves both borders of a %s', (type) => {
    const g = linearGraph();
    const region = registry.create(type, {}, 'region');
    g.nodes.push(region);
    for (const members of [
      'offset',
      'query_cat_512,offset',
      'offset,offsets_128',
      'query_cat_512,offsets_128',
    ]) {
      region.params.members = members;
      expect(project(g).hiddenNodeIds.size).toBe(0);
    }
    region.params.members = 'query_cat_512,offset,offsets_128';
    expect(project(g).hiddenNodeIds.has('offset')).toBe(true);
  });

  it('leaves closed cycles inspectable, including a cycle with one visible endpoint', () => {
    const nodes: ModelNode[] = ['a', 'b', 'c'].map((id) => ({
      ...registry.create('Identity', {}, id),
      presentation: 'edge',
    }));
    const g = {
      nodes,
      edges: nodes.map((n, i) => ({
        id: n.id,
        source: n.id,
        target: nodes[(i + 1) % 3].id,
        sourcePort: 'out',
        targetPort: 'in',
      })),
    };
    for (const mode of ['edge', 'default'] as const) {
      nodes[0].presentation = mode;
      expect(project(g).hiddenNodeIds.size).toBe(0);
      expect(project(g).edges).toHaveLength(3);
    }
  });

  it('preserves unwired declared SemanticOp ports and collapses the true unary form', () => {
    const g = linearGraph();
    const op = registry.create('SemanticOp', { inputs: 'x', outputs: 'y,other' }, 'offset');
    op.presentation = 'edge';
    op.label = '编码步骤'.repeat(10);
    g.nodes[1] = op;
    g.edges[0].targetPort = 'x';
    g.edges[1].sourcePort = 'y';
    expect(project(g).hiddenNodeIds.size).toBe(0);
    op.params.outputs = 'y';
    expect(project(g).edges[0].edgeOps[0].badge).toHaveLength(24);
  });

  it('resolves source colors and ordered concat segments on the canonical graph', () => {
    const p = buildModel({
      name: 'Joined features',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2', axes: '' } },
        { id: 'y', type: 'Input', params: { shape: '3', axes: '' } },
        { id: 'join', type: 'Concat', presentation: 'compact', params: { dim: 0 } },
        { id: 'detach', type: 'Detach', presentation: 'edge' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['y.out', 'join.b'],
        ['x.out', 'join.a'],
        ['join.out', 'detach.in'],
        ['detach.out', 'out.in'],
      ],
    }).project!;
    const result = inferProject(p, registry);
    const tones = resolveGraphTensorTones(p.graph, result);
    expect(projectCanvasGraph(p.graph, result, tones).hiddenNodeIds.has('detach')).toBe(true);
    expect(tones.get('out')).toMatchObject({
      tones: ['blue', 'violet'],
      join: {
        segments: [
          { port: 'a', size: 2, tones: ['blue'] },
          { port: 'b', size: 3, tones: ['violet'] },
        ],
      },
    });
  });

  it('uses a short annotation or source name and the first canonical source tone for anchors', () => {
    const g = linearGraph(),
      edge = g.edges[0];
    g.nodes[0].tensorVisual = { mode: 'off', tone: 'teal' };
    edge.presentation = 'anchor';
    for (const [annotation, expected] of [
      ['', 'query'],
      ['residual', 'residual'],
      ['x'.repeat(21), 'query'],
    ]) {
      edge.annotation = annotation;
      expect(projectCanvasGraph(g, undefined, resolveGraphTensorTones(g)).edges[0]).toMatchObject({
        sourceTone: 'teal',
        sourceLabel: expected,
        edgeOps: [],
        canonicalEdgeIds: [edge.id],
      });
    }
  });
});
