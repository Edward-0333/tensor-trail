import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Params, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { embeddingInitializationCode } from '../src/modules/embeddingInitialization';

const init = (params: Params = {}) => {
  const node = registry.create('EmbeddingXYZInit', params, 'init');
  return { node, result: inferGraph({ nodes: [node], edges: [] }, registry) };
};

describe('Embedding parameter construction and initialization', () => {
  it.each([true, false])(
    'declares requires_grad=%s and preserves it through in-place initialization',
    (requires_grad) => {
      const { node, result } = init({ requires_grad });
      expect(result.nodes.init.status).toBe('ok');
      expect(result.nodes.init.outputs.out).toEqual({
        shape: [1040, 3],
        axes: ['query', 'coordinate'],
        dtype: 'float32',
        autograd: { requiresGrad: requires_grad, source: 'inferred' },
      });
      for (const r of Object.values(result.nodes.init.child!.nodes)) {
        expect(r.status).toBe('ok');
        expect(r.outputs.out).toEqual(result.nodes.init.outputs.out);
      }
      expect(node.subgraph!.nodes.find((n) => n.id === 'init_x')!.params).toMatchObject({
        start: 0,
        stop: 1,
        low: '0',
        high: '1.1',
      });
      expect(node.subgraph!.nodes.find((n) => n.id === 'init_yz')!.params).toMatchObject({
        start: 1,
        stop: 3,
        low: '0',
        high: '1',
      });
      expect(
        node.subgraph!.nodes.find((n) => n.id === 'registered_weight')!.tensorBorderColor,
      ).toBe('#60a5fa');
    },
  );

  it('has no lookup input and exposes weight rather than a module-object Tensor', () => {
    const { node, result } = init({ num_embeddings: 7, dtype: 'float64' });
    expect(node.subgraph!.nodes.some((n) => n.type === 'Inlet')).toBe(false);
    expect(tensorValue(result.nodes.init.outputs.out).shape).toEqual([7, 3]);
    expect(tensorValue(result.nodes.init.outputs.out).dtype).toBe('float64');
    const code = embeddingInitializationCode(node)!;
    expect(code).toContain('nn.Embedding(7, 3');
    expect(code).toContain('weight[:, 0], 0, 1.1');
    expect(code).toContain('weight[:, 1:], 0, 1');
    expect(code).toContain('_init_layers');
    expect(resolveForwardView({ nodes: [node], edges: [] }, ['init']).current.title).not.toMatch(
      /forward\(\)$/,
    );
  });

  it.each<Params>([
    { num_embeddings: 0 },
    { embedding_dim: -1 },
    { requires_grad: 1 },
    { dtype: 'int64' },
  ])('rejects invalid construction %j', (params) => {
    const node = registry.create('EmbeddingWeight', params);
    expect(() => registry.get(node.type)!.infer!({}, node.params)).toThrow();
  });

  it('reports the full parameter after in-place column initialization', () => {
    const source: TensorSpec = {
      shape: [8, 3],
      axes: ['query', 'coordinate'],
      dtype: 'float32',
      autograd: { requiresGrad: true, source: 'runtime' },
    };
    const before = structuredClone(source);
    const node = registry.create('UniformInitColumns', { start: 0, stop: 1, high: '1.1' });
    const result = registry.get(node.type)!.infer!({ in: source }, node.params);
    expect(result.outputs.out).toEqual(source);
    result.outputs.out.shape[0] = 3;
    expect(source).toEqual(before);
    expect(embeddingInitializationCode(node)).toContain('weight[:, 0], 0, 1.1');
  });

  it.each<Params>([
    { start: -1 },
    { stop: 0 },
    { stop: 4 },
    { low: '' },
    { high: 'NaN' },
    { high: 'Infinity' },
    { high: '0' },
    { low: '2', high: '1' },
  ])('rejects invalid initialization %j', (params) => {
    const node = registry.create('UniformInitColumns', params);
    expect(() =>
      registry.get(node.type)!.infer!(
        { in: { shape: [8, 3], axes: ['Q', 'D'], dtype: 'float32' } },
        node.params,
      ),
    ).toThrow();
  });

  it('does not invent grad metadata for an unannotated Tensor', () => {
    const node = registry.create('UniformInitColumns');
    const result = registry.get(node.type)!.infer!(
      { in: { shape: [8, 3], axes: ['Q', 'D'], dtype: 'float32' } },
      node.params,
    );
    expect(result.outputs.out).not.toHaveProperty('autograd');
  });

  it('requires a weight outlet and does not relax ordinary Submodule boundaries', () => {
    const built = buildModel({ name: 'init', nodes: [{ id: 'init', type: 'EmbeddingXYZInit' }], connections: [] }, { mode: 'draft' });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const missing = structuredClone(built.project!);
    missing.graph.nodes[0].subgraph!.nodes = missing.graph.nodes[0].subgraph!.nodes.filter(n => n.type !== 'Outlet');
    missing.graph.nodes[0].subgraph!.edges = [];
    expect(validateModel(missing, { mode: 'draft' }).ok).toBe(false);
    const regular = structuredClone(built.project!);
    regular.graph.nodes[0].type = 'Submodule';
    regular.graph.nodes[0].params = {};
    expect(validateModel(regular, { mode: 'draft' }).ok).toBe(false);
  });

  it('round trips through the API and preserves geometry when changing constructor settings', () => {
    const before = buildModel(
      {
        name: 'provenance',
        nodes: [{ id: 'init', type: 'EmbeddingXYZInit', position: { x: 90, y: 60 } }],
        connections: [],
      },
      { mode: 'draft' },
    );
    expect(before.ok).toBe(true);
    const result = patchModel(
      before.project!,
      { operations: [{ op: 'set_params', id: 'init', params: { num_embeddings: 17 } }] },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.nodes[0].position).toEqual({ x: 90, y: 60 });
    expect(
      tensorValue(result.nodes.find((n) => n.path.join('/') === 'init')!.outputs.out).shape,
    ).toEqual([17, 3]);
    expect(
      validateModel(parseProject(serializeProject(result.project!), registry), { mode: 'draft' })
        .ok,
    ).toBe(true);
  });
});
