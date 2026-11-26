import { describe, expect, it } from 'vitest';
import { buildModel, patchModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import {
  buildTensorGeometryDiagram,
  buildTensorBlockDiagram,
  projectOuterVolumes,
  computeScaledDims,
  canvasTensorVisual,
  resolveGraphReferenceMaxDim,
  mapShapeToBox3D,
  projectTensorSlices,
  resolveGraphTensorTones,
} from '../src/core/tensorGeometry';
import {
  type Dim,
  type Graph,
  type GraphResult,
  type NodeResult,
  type TensorSpec,
} from '../src/core/types';
import { registry } from '../src/modules';

const spec = (shape: Dim[], axes = shape.map((_, i) => 'd' + i)): TensorSpec => ({
  shape,
  axes,
  dtype: 'float32',
});
const box = (shape: Dim[]) => mapShapeToBox3D(spec(shape));

describe('shape projection', () => {
  it.each(
    [[], [64], [3, 5], [2, 4, 8], [1, 3, 16, 32], [2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]].map(
      (s) => [s],
    ),
  )('preserves every axis of %j', (shape) => {
    const b = box(shape);
    expect(b.ndim).toBe(shape.length);
    expect(b.leadingChips).toEqual(shape.slice(0, -4).map((size, i) => ({ label: 'd' + i, size })));
    [b.width, b.height, b.depth].forEach((a, i) => {
      expect(a.size).toBe(shape.at(-1 - i) ?? 1);
      expect(a.dimIndex).toBe(i < shape.length ? shape.length - 1 - i : undefined);
    });
  });
  it.each([
    [1, 1, false],
    [2, 2, false],
    [3, 3, false],
    [4, 2, true],
    ['B', 2, true],
  ] as const)('maps outer size %s to %i volumes', (n, count, elided) => {
    const d = buildTensorBlockDiagram(spec([n, 256, 8, 32]));
    expect(d.output.outer).toEqual({ axis: 'd0', dimIndex: 0, size: n, count, elided });
    expect(d.output.leadingChips).toEqual([]);
    expect(d.width).toBe(count > 1 ? 220 : 180);
    expect(d.output.depth).toMatchObject({ solid: true, sliceCount: 1 });
    expect(d.output.depthPx).toBeGreaterThan(12);
  });

  it('overrides inferred names, face dimensions and continuous or stacked depth independently', () => {
    const tensor = spec(['T', 2, 256, 8, 32]);
    const options = {
      boxWidth: 75,
      boxHeight: 30,
      boxDepth: 50,
      color: '#abcdef',
      axisLabels: ['step', 'queue', 'query', 'head', 'dim'],
    };
    const d = buildTensorBlockDiagram(tensor, 'out', undefined, options);
    expect(d).toMatchObject({
      customColor: options.color,
      output: {
        leadingChips: [{ label: 'step', size: 'T' }],
        outer: { axis: 'queue' },
        width: { axis: 'dim' },
        height: { axis: 'head' },
        depth: { axis: 'query' },
        wPx: 75,
        hPx: 30,
        depthPx: 50,
      },
    });
    for (const z of [256, 'Z']) {
      const slices = buildTensorBlockDiagram(spec([z, 2, 3]), 'out', undefined, {
        ...options,
        depthStyle: 'slices',
      });
      expect(slices.output).toMatchObject({
        depth: { solid: false, elided: true, sliceCount: 3 },
        depthPx: 6,
        sliceGapPx: 22,
      });
    }
    expect(mapShapeToBox3D(spec([2, 8, 8]), { depthStyle: 'solid' }).depth).toMatchObject({
      solid: true,
      sliceCount: 1,
      elided: false,
    });
    const depths = [8, 16, 32, 256, 8192].map(
      (z) => buildTensorBlockDiagram(spec([z, 8, 8])).output.depthPx,
    );
    expect(depths[1] - depths[0]).toBeCloseTo(depths[2] - depths[1]);
    expect(depths[3]).toBeGreaterThan(depths[2]);
    expect(depths[4]).toBe(38);
  });

  it('elides deep stacks, bounds symbolic geometry, keeps scalars and thin vectors distinct', () => {
    expect([1, 2, 3, 4, 5, 8192, 'B'].map((n) => box([n, 8, 8]).depth.sliceCount)).toEqual([
      1, 2, 3, 4, 1, 1, 1,
    ]);
    expect(box(['B', 8, 'T']).depth.solid).toBe(true);
    const [scalar, vector, symbolic, small, huge] = computeScaledDims([
      box([]),
      box([50]),
      box(['B', 'T']),
      box([1]),
      box([1e14]),
    ]);
    expect(scalar.hPx).toBe(scalar.wPx);
    expect(vector.hPx).toBeLessThan(scalar.hPx);
    expect(symbolic.wPx).toBeGreaterThan(small.wPx);
    expect(symbolic.wPx).toBeLessThan(huge.wPx);
    expect(huge.wPx).toBeLessThan(100);
  });
});

const block = { mode: 'block' } as const;
function diagram(graph: Graph, result: GraphResult, id: string) {
  return buildTensorGeometryDiagram(
    graph.nodes.find((n) => n.id === id)!,
    result.nodes[id],
    resolveGraphTensorTones(graph, result),
    { ...block, referenceMaxDim: resolveGraphReferenceMaxDim(graph, result) },
  )!;
}

describe('graph reference scale', () => {
  it('uses the same reference across separate tensor cards, ignoring leading axes', () => {
    const sizes = [512, 256, 128, 64, 32, 1];
    const graph = buildModel(
      {
        name: 'Scale',
        connections: [],
        nodes: sizes.map((n) => ({
          id: 'x' + n,
          type: 'Input',
          params: { shape: '9999,1,8,' + n, axes: 'B,Z,H,W' },
        })),
      },
      { mode: 'draft' },
    ).project!;
    const result = inferProject(graph, registry);
    expect(resolveGraphReferenceMaxDim(graph.graph, result)).toBe(512);
    const widths = sizes.map((n) => diagram(graph.graph, result, 'x' + n).output.wPx);
    expect(widths.map(Math.round)).toEqual([98, 69, 47, 32, 23, 12]);
    expect(widths.every((width, i) => !i || widths[i - 1] > width)).toBe(true);
    expect(resolveGraphReferenceMaxDim({ ...graph.graph, nodes: [] }, result)).toBe(4);
    expect(resolveGraphReferenceMaxDim()).toBeUndefined();
    const first = result.nodes.x512;
    first.inputs = { in: spec([8192, 'H', 'W']) };
    expect(resolveGraphReferenceMaxDim(graph.graph, result)).toBe(2048);
    expect(computeScaledDims([box([512])], 128)[0].wPx).toBe(98);
  });
});

describe('single output block and color propagation', () => {
  function attention() {
    const project = buildModel({
      name: 'TemporalSelfAttention',
      nodes: [
        { id: 'query', type: 'Input', params: { shape: '1,256,256', axes: 'B,N,D' } },
        { id: 'pos', type: 'Input', params: { shape: '1,1,256', axes: 'B,N,D' } },
        { id: 'fused', type: 'Add' },
        { id: 'features', type: 'Concat', params: { dim: -1 } },
        { id: 'history', type: 'Concat', params: { dim: 0 } },
        { id: 'named', type: 'TensorVariable' },
        { id: 'layers', type: 'TensorVariable' },
        { id: 'held', type: 'ValueVariable' },
        { id: 'contiguous', type: 'Contiguous' },
        { id: 'detached', type: 'Detach' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['pos.out', 'fused.b'],
        ['query.out', 'fused.a'],
        ['fused.out', 'features.b'],
        ['query.out', 'features.a'],
        ['query.out', 'history.a'],
        ['fused.out', 'history.b'],
        ['features.out', 'named.in'],
        ['history.out', 'layers.in'],
        ['named.out', 'held.in'],
        ['held.out', 'contiguous.in'],
        ['contiguous.out', 'detached.in'],
        ['detached.out', 'out.in'],
      ],
    }).project!;
    return patchModel(project, {
      operations: [{ op: 'duplicate_linked', source: 'named', id: 'copy' }],
    }).project!;
  }

  it('carries the two ordered feature colors and sizes through same-shape values and aliases', () => {
    const p = attention();
    const result = inferProject(p, registry);
    const colors = resolveGraphTensorTones(p.graph, result);
    expect(colors.get('fused')?.tones).toEqual(['blue', 'violet']);
    const featureJoin = colors.get('features')?.join;
    expect(featureJoin).toMatchObject({
      axis: 2,
      direction: 'width',
      segments: [
        { port: 'a', size: 256, tones: ['blue'], start: 0, end: 0.5 },
        { port: 'b', size: 256, tones: ['blue', 'violet'], start: 0.5, end: 1 },
      ],
    });
    for (const id of ['named', 'held', 'contiguous', 'detached', 'out', 'copy']) {
      const d = diagram(p.graph, result, id);
      expect(d).toMatchObject({ mode: 'block', badge: '3D', inputs: [], join: featureJoin });
      expect(d).not.toHaveProperty('symbol');
      expect(d.output.width.size).toBe(512);
    }
    const depth = diagram(p.graph, result, 'layers');
    expect(depth.inputs).toEqual([]);
    expect(depth.join?.direction).toBe('depth');
    expect(depth.output.depth.sliceCount).toBe(2);
    expect(projectTensorSlices(depth.output, depth.join)).toEqual([
      { index: 1, segment: 'b', tones: ['blue', 'violet'] },
      { index: 0, segment: 'a', tones: ['blue'] },
    ]);

    // An Outlet can also be inspected in a partially inferred child with only its input.
    const boundary = p.graph.nodes.find((n) => n.id === 'out')!;
    boundary.type = 'Outlet';
    result.nodes.out.outputs = {};
    expect(diagram(p.graph, result, 'out')).toMatchObject({
      join: featureJoin,
      observeInput: true,
    });
  });

  it('drops inherited segments at an explicit color or a change in output shape', () => {
    const p = attention();
    const named = p.graph.nodes.find((n) => n.id === 'named')!;
    const result = inferProject(p, registry);
    named.tensorVisual = { ...block, tone: 'rose' };
    for (const id of ['named', 'out', 'copy']) {
      const d = diagram(p.graph, result, id);
      expect(d.join).toBeUndefined();
      expect(d.output.tones).toEqual(['rose']);
    }
    named.tensorVisual.color = '#1a2b3c';
    for (const id of ['named', 'out', 'copy']) {
      const d = diagram(p.graph, result, id);
      expect(d.customColor).toBe('#1a2b3c');
      expect(d.join).toBeUndefined();
    }
    delete named.tensorVisual.color;
    named.tensorVisual.tone = 'auto';
    expect(diagram(p.graph, result, 'named').join).toBeDefined();
    result.nodes.named.outputs.out = spec([512]);
    expect(diagram(p.graph, result, 'named').join).toBeUndefined();
    expect(diagram(p.graph, result, 'copy').join).toBeUndefined();
  });

  it.each([
    ['Concat', -2, [1, 3, 4], 'height', 3],
    ['Concat', 0, [4, 3, 8], 'depth', 4],
    ['Stack', -1, [3, 4], 'width', 1],
    ['Stack', 0, [3, 4], 'depth', 1],
    ['Stack', 0, [2, 3, 4], 'outer', 1],
    ['Stack', 0, [2, 2, 3, 4], 'leading', 1],
  ] as const)('%s dim=%i propagates %j as %s segments', (type, dim, shape, direction, size) => {
    const p = buildModel(
      {
        name: 'Join',
        nodes: [
          {
            id: 'x',
            type: 'Input',
            params: { shape: shape.join(','), axes: shape.map((_, i) => 'd' + i).join(',') },
          },
          { id: 'join', type, params: { dim, input_count: 3 } },
          { id: 'value', type: 'TensorVariable' },
        ],
        connections: [
          ['x.out', 'join.c'],
          ['x.out', 'join.a'],
          ['x.out', 'join.b'],
          ['join.out', 'value.in'],
        ],
      },
      { mode: 'draft' },
    ).project!;
    const d = diagram(p.graph, inferProject(p, registry), 'value');
    expect(d.join).toMatchObject({
      direction,
      segments: ['a', 'b', 'c'].map((port) => ({ port, size })),
    });
    expect(d.inputs).toEqual([]);
    if (direction === 'depth')
      expect(projectTensorSlices(d.output, d.join).map((s) => s.segment)).toEqual(['c', 'b', 'a']);
  });

  it.each([
    ['Stack', [256, 8, 32], [256, 8, 32], ['a', 'b']],
    ['Concat', [2, 256, 8, 32], [1, 256, 8, 32], ['a', 'a', 'b']],
    ['Concat', [4, 256, 8, 32], [1, 256, 8, 32], ['a', 'b']],
  ] as const)('samples %s outer volumes from their actual input segments', (type, a, b, ports) => {
    const p = buildModel({
      name: type,
      nodes: [
        ...[a, b].map((shape, i) => ({
          id: 'x' + i,
          type: 'Input',
          params: { shape: shape.join(','), axes: shape.map((_, j) => 'd' + j).join(',') },
          tensorVisual: i ? { mode: 'block', color: '#ef6789' } : { mode: 'block', tone: 'teal' },
        })),
        { id: 'join', type, params: { dim: 0 } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['x0.out', 'join.a'],
        ['x1.out', 'join.b'],
        ['join.out', 'out.in'],
      ],
    }).project!;
    const d = diagram(p.graph, inferProject(p, registry), 'out');
    expect(d.join?.direction).toBe('outer');
    expect(projectOuterVolumes(d.output, d.join)).toEqual(
      ports.map((port, index) => ({
        index,
        segment: port,
        tones: [port === 'a' ? 'teal' : '#ef6789'],
      })),
    );
  });

  it('selects the requested tensor port and skips off, unknown and non-tensor outputs', () => {
    const node = registry.create('SemanticOp');
    const result: NodeResult = {
      inputs: { in: spec([128]) },
      outputs: { count: { kind: 'int', value: 1 }, small: spec([1]), wide: spec([512]) },
      notes: [],
      messages: [],
      status: 'ok',
    };
    const preview = (port?: string) =>
      buildTensorGeometryDiagram(node, result, new Map(), { ...block, outputPort: port });
    expect(preview()?.output.port).toBe('small');
    expect(preview('wide')).toMatchObject({
      inputs: [],
      output: { port: 'wide', width: { size: 512 } },
    });
    expect(preview('count')).toBeUndefined();
    expect(preview('missing')).toBeUndefined();
    expect(buildTensorGeometryDiagram(node, result, new Map(), { mode: 'off' })).toBeUndefined();
  });
});

describe('canvas defaults', () => {
  it.each([
    'Input',
    'Inlet',
    'TensorVariable',
    'ValueVariable',
    'LearnedParameter',
    'BufferSource',
    'Outlet',
    'Output',
  ])('previews %s only when enabled', (type) => {
    const node = registry.create(type);
    expect(canvasTensorVisual(node, false)).toBeUndefined();
    expect(canvasTensorVisual(node, true)).toEqual({ ...block, tone: 'auto' });
    node.tensorVisual = { mode: 'off' };
    expect(canvasTensorVisual(node, true)).toBe(node.tensorVisual);
  });
  it.each(['Add', 'Concat', 'Stack', 'Linear', 'Reshape', 'Permute', 'LayerNorm', 'SemanticOp'])(
    'keeps %s compact until explicitly configured',
    (type) => {
      const node = registry.create(type);
      expect(canvasTensorVisual(node, true)).toBeUndefined();
      node.tensorVisual = block;
      expect(canvasTensorVisual(node, false)).toBe(node.tensorVisual);
    },
  );
});
