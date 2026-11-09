import { tensorValue } from '../src/core/types';
import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, getSchemas, patchModel, validateModel } from '../src/agent/index';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { clone, type Graph, type Params } from '../src/core/types';
import { registry } from '../src/modules/index';
import crossAttention from '../../../examples/cross-attention.recipe.json';

function graph(shapes = ['2, 3, 64', '2, 5, 32', '2, 5, 48'], params: Params = {}): Graph {
  return {
    nodes: [
      ...shapes.map((shape, i) => registry.create('Input', { shape, axes: '' }, `input${i}`)),
      registry.create('MultiheadAttention', { kdim: 32, vdim: 48, ...params }, 'attention'),
      registry.create('Output', {}, 'output'),
    ],
    edges: [
      ...['q', 'k', 'v'].map((port, i) => ({
        id: port,
        source: `input${i}`,
        sourcePort: 'out',
        target: 'attention',
        targetPort: port,
      })),
      { id: 'out', source: 'attention', sourcePort: 'out', target: 'output', targetPort: 'in' },
    ],
  };
}

describe('Q/K/V multi-head attention', () => {
  it('projects distinct feature widths and preserves query length through rectangular attention', () => {
    const result = inferGraph(graph(), registry);
    const attention = result.nodes.attention;
    expect(result.issues).toEqual([]);
    expect(attention.status, attention.messages.join()).toBe('ok');
    expect(tensorValue(attention.outputs.out).shape).toEqual([2, 3, 64]);
    expect(tensorValue(attention.child!.nodes.q_heads.outputs.out).shape).toEqual([2, 8, 3, 8]);
    expect(tensorValue(attention.child!.nodes.k_heads.outputs.out).shape).toEqual([2, 8, 5, 8]);
    expect(tensorValue(attention.child!.nodes.v_heads.outputs.out).shape).toEqual([2, 8, 5, 8]);
    expect(tensorValue(attention.child!.nodes.scores.outputs.out).shape).toEqual([2, 8, 3, 5]);
    expect(tensorValue(attention.child!.nodes.context.outputs.out).shape).toEqual([2, 8, 3, 8]);
    expect(result.nodes.output.inputs.in).toEqual(attention.outputs.out);
  });

  it('uses d_model for default K/V widths and supports one tensor connected to all three ports', () => {
    const g = graph(['2, 7, 128'], { d_model: 128, heads: 4, kdim: 0, vdim: 0 });
    g.edges.slice(0, 3).forEach((edge) => {
      edge.source = 'input0';
    });
    const result = inferGraph(g, registry).nodes.attention;
    expect(result.status, result.messages.join()).toBe('ok');
    expect(tensorValue(result.outputs.out).shape).toEqual([2, 7, 128]);
    expect(tensorValue(result.child!.nodes.scores.outputs.out).shape).toEqual([2, 4, 7, 7]);
  });

  it('keeps rectangular score shapes when causal masking is enabled', () => {
    const result = inferGraph(graph(undefined, { causal: true }), registry).nodes.attention;
    expect(result.status, result.messages.join()).toBe('ok');
    expect(tensorValue(result.child!.nodes.mask.outputs.out).shape).toEqual([2, 8, 3, 5]);
    expect(tensorValue(result.outputs.out).shape).toEqual([2, 3, 64]);
  });

  it.each([
    ['Q/K batch', ['2, 3, 64', '1, 5, 32', '2, 5, 48'], {}, 'batch'],
    ['Q/V batch', ['2, 3, 64', '2, 5, 32', '1, 5, 48'], {}, 'batch'],
    ['K/V length', ['2, 3, 64', '2, 5, 32', '2, 6, 48'], {}, '序列长度'],
    ['Q features', ['2, 3, 63', '2, 5, 32', '2, 5, 48'], {}, 'Q 输入特征'],
    ['K features', ['2, 3, 64', '2, 5, 31', '2, 5, 48'], {}, 'K 输入特征'],
    ['V features', ['2, 3, 64', '2, 5, 32', '2, 5, 47'], {}, 'V 输入特征'],
    ['input rank', ['2, 64', '2, 5, 32', '2, 5, 48'], {}, '3 维'],
    ['head division', ['2, 3, 64', '2, 5, 32', '2, 5, 48'], { heads: 7 }, '整除'],
  ] satisfies [string, string[], Params, string][])(
    'rejects incompatible %s and blocks downstream',
    (_, shapes, params, message) => {
      const result = inferGraph(graph(shapes, params), registry);
      expect(result.nodes.attention.status).toBe('error');
      expect(result.nodes.attention.messages.join()).toContain(message);
      expect(result.nodes.output.status).toBe('blocked');
      expect(result.nodes.attention.outputs).toEqual({});
    },
  );

  it.each(['int64', 'float16'])('rejects incompatible input dtype %s', (dtype) => {
    const g = graph();
    g.nodes[1].params.dtype = dtype;
    const attention = inferGraph(g, registry).nodes.attention;
    expect(attention.status).toBe('error');
    expect(attention.messages.join()).toContain(dtype === 'int64' ? '浮点' : 'dtype');
  });

  it('reports a missing V input instead of producing a partial output', () => {
    const g = graph();
    g.edges = g.edges.filter((edge) => edge.targetPort !== 'v');
    const result = inferGraph(g, registry).nodes.attention;
    expect(result.status).toBe('blocked');
    expect(result.messages.join()).toContain('v');
    expect(result.outputs).toEqual({});
  });

  it('preserves symbolic lengths and warns about unproven cross-input constraints', () => {
    const result = inferGraph(graph(['B, Tq, 64', 'C, S, 32', 'B, U, 48']), registry).nodes
      .attention;
    expect(result.status).toBe('warning');
    expect(tensorValue(result.outputs.out).shape).toEqual(['B', 'Tq', 64]);
    expect(result.messages.join()).toContain('batch');
    expect(result.messages.join()).toContain('序列长度');
  });
});

describe('Q/K/V attention AI and persistence', () => {
  it('exposes the three named ports and input width parameters in the catalog and schemas', () => {
    const entry = describeModules().modules.find((m) => m.type === 'MultiheadAttention')!;
    expect(entry.inputs.map((p) => p.id)).toEqual(['q', 'k', 'v']);
    expect(entry.paramsSchema.properties.kdim).toMatchObject({ default: 0, minimum: 0 });
    expect(getSchemas().recipe.$defs.node.oneOf).toContainEqual(
      expect.objectContaining({
        properties: expect.objectContaining({ type: { const: entry.type } }),
      }),
    );
    expect(ports(registry.create(entry.type), registry).inputs.map((p) => p.id)).toEqual([
      'q',
      'k',
      'v',
    ]);
  });

  it('builds and round-trips the documented cross-attention recipe including port layout and routes', () => {
    const result = buildModel(crossAttention);
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(tensorValue(result.outputs.output).shape).toEqual([2, 3, 64]);
    const project = result.project!;
    project.graph.nodes.find((n) => n.id === 'attention')!.portLayout = {
      input: 'top',
      output: 'bottom',
    };
    project.graph.edges[0].waypoints = [{ x: 50, y: 20 }];
    const loaded = parseProject(serializeProject(project), registry);
    expect(loaded).toEqual(project);
    expect(validateModel(loaded).ok).toBe(true);
  });

  it('rebuilds heads through Patch while keeping source data and external connections intact', () => {
    const original = buildModel(crossAttention).project!;
    const before = clone(original);
    const result = patchModel(original, {
      operations: [{ op: 'set_params', id: 'attention', params: { heads: 4 } }],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(tensorValue(result.outputs.output).shape).toEqual([2, 3, 64]);
    expect(
      tensorValue(result.nodes.find((n) => n.path.join('/') === 'attention/scores')!.outputs.out)
        .shape,
    ).toEqual([2, 4, 3, 5]);
    expect(result.project!.graph.edges).toEqual(before.graph.edges);
    expect(original).toEqual(before);
  });
});
