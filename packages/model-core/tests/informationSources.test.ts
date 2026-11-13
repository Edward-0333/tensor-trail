import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel, validateModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import {
  annotateInformationSources,
  valueLineage,
  validateInformationSource,
  mergeLineages,
} from '../src/core/informationSources';
import { rootInputSources } from '../src/core/rootInputs';
import type {
  ModelNode,
  Params,
  Project,
  RootInputSource,
  TensorSpec,
  ValueSpec,
} from '../src/core/types';
import { isDictionarySpec } from '../src/core/types';
import { registry } from '../src/modules';
import { loopRecipe } from './fixtures/forEach';

const tensor: TensorSpec = { shape: [2, 3], axes: ['batch', 'feature'], dtype: 'float32' };
const node = (type: string, params: Params = {}): ModelNode => ({
  id: 'test',
  type,
  label: type,
  params,
  position: { x: 0, y: 0 },
  definitionVersion: 1,
});
const root = (id: string, index: number, label = id): RootInputSource => ({ id, index, label });
const a = root('signal_a', 1),
  b = root('signal_b', 2),
  c = root('payload', 3);
const sourced = (source: RootInputSource, value: ValueSpec = tensor) =>
  annotateInformationSources(node('Input'), {}, { out: value }, source).out;
const run = (
  type: string,
  inputs: Record<string, ValueSpec>,
  value: ValueSpec = tensor,
  params: Params = {},
) => annotateInformationSources(node(type, params), inputs, { out: value }).out;
const numbers = (value: ValueSpec) => value.lineage?.sources.map((s) => s.index);

function fixture(): Project {
  const built = buildModel({
    name: 'Any model',
    nodes: [
      { id: 'signal_a', type: 'Input', params: { shape: '2,3', axes: 'batch,feature' } },
      { id: 'signal_b', type: 'Input', params: { shape: '2,3', axes: 'batch,feature' } },
      { id: 'merge', type: 'Add' },
      { id: 'out', type: 'Output' },
    ],
    connections: [
      ['signal_a.out', 'merge.a'],
      ['signal_b.out', 'merge.b'],
      ['merge.out', 'out.in'],
    ],
  });
  expect(built.ok).toBe(true);
  return built.project!;
}

describe('Root input provenance', () => {
  it('automatically numbers only root arguments, without semantic declarations or mutating the graph', () => {
    const project = fixture();
    expect(inferProject(project, registry).nodes.merge.outputs.out.lineage).toBeUndefined();
    project.lineageEnabled = true;
    const before = serializeProject(project),
      result = inferProject(project, registry);
    expect(numbers(result.nodes.signal_a.outputs.out)).toEqual([1]);
    expect(numbers(result.nodes.signal_b.outputs.out)).toEqual([2]);
    expect(numbers(result.nodes.merge.outputs.out)).toEqual([1, 2]);
    expect(result.nodes.merge.outputs.out.lineage?.unknown).toBe(false);
    expect(serializeProject(project)).toBe(before);
    expect(inferProject(parseProject(before, registry), registry)).toEqual(result);
    project.lineageEnabled = false;
    expect(inferProject(project, registry).nodes.merge.outputs.out.lineage).toBeUndefined();
  });

  it('ignores coordinates and labels when assigning identity, and renumbers from root interface order', () => {
    const p = fixture();
    p.lineageEnabled = true;
    p.graph.nodes[0].label = 'any renamed input';
    p.graph.nodes[0].position.x = 99999;
    expect(rootInputSources(p.graph)).toEqual([
      root('signal_a', 1, 'any renamed input'),
      root('signal_b', 2, p.graph.nodes[1].label),
    ]);
    [p.graph.nodes[0], p.graph.nodes[1]] = [p.graph.nodes[1], p.graph.nodes[0]];
    expect(rootInputSources(p.graph).map((s) => [s.id, s.index])).toEqual([
      ['signal_b', 1],
      ['signal_a', 2],
    ]);
    expect(numbers(inferProject(p, registry).nodes.merge.outputs.out)).toEqual([1, 2]);
  });

  it('does not count aliases, model state, parameter factories or nested Input nodes', () => {
    const p = fixture();
    p.graph.nodes.push(
      { ...p.graph.nodes[0], id: 'alias', valueSource: 'signal_a' },
      registry.create('EmbeddingModule', {}, 'parameter'),
      registry.create('BufferSource', {}, 'buffer'),
      {
        ...node('Submodule'),
        id: 'child',
        subgraph: { nodes: [registry.create('Input', {}, 'hidden')], edges: [] },
      },
    );
    expect(rootInputSources(p.graph).map((s) => s.id)).toEqual(['signal_a', 'signal_b']);
    p.lineageEnabled = true;
    const result = inferProject(p, registry);
    expect(result.nodes.child.child!.nodes.hidden.outputs.out.lineage).toMatchObject({
      sources: [],
      unknown: true,
    });
    expect(result.nodes.alias.outputs.out.lineage).toEqual(
      result.nodes.signal_a.outputs.out.lineage,
    );
  });

  it('uses generic forward/dictionary inputs, including legacy input aliases', () => {
    const graph = {
      nodes: ['Input', 'ForwardInput', 'DictionaryInput', 'DictInput', 'ListInput'].map(
        (type, i) => ({ ...node(type), id: `arg${i}` }),
      ),
      edges: [],
    };
    expect(rootInputSources(graph).map((s) => s.index)).toEqual([1, 2, 3, 4, 5]);
  });

  it('legacy semantic annotations cannot override root IDs or introduce new origins in nested calculations', () => {
    const p = fixture();
    p.lineageEnabled = true;
    p.graph.nodes[0].informationSource = { kind: 'parameter', initialization: 'checkpoint' };
    p.graph.nodes[2].informationSource = { kind: 'ego' };
    const r = inferProject(parseProject(serializeProject(p), registry), registry);
    expect(numbers(r.nodes.signal_a.outputs.out)).toEqual([1]);
    expect(numbers(r.nodes.merge.outputs.out)).toEqual([1, 2]);
    expect(r.nodes.signal_a.outputs.out.lineage?.origins).toEqual([]);
  });

  it.each(['View', 'Repeat', 'Cast', 'Detach', 'Contiguous', 'ReLU', 'Tanh'])(
    '%s propagates value IDs, never shape arguments',
    (type) => {
      const value = sourced(a, { ...tensor, autograd: { requiresGrad: false, source: 'manual' } });
      expect(run(type, { in: value, d0: sourced(b) }).lineage).toMatchObject({
        sources: [a],
        unknown: false,
      });
    },
  );

  it('keeps priors unnumbered until input values arrive and deduplicates multi-input fusion', () => {
    const weight = run('EmbeddingWeight', {});
    expect(weight.lineage).toMatchObject({
      sources: [],
      origins: ['parameter'],
      initializations: ['random'],
    });
    const repeated = run('Repeat', { in: weight, d0: sourced(a) });
    expect(numbers(repeated)).toEqual([]);
    const scaled = run('Mul', { a: repeated, b: sourced(c) });
    expect(numbers(scaled)).toEqual([3]);
    expect(numbers(run('Add', { a: scaled, b: sourced(a) }))).toEqual([1, 3]);
    expect(numbers(run('Add', { a: sourced(a), b: sourced(a) }))).toEqual([1]);
    const sameLabel = { ...b, label: a.label };
    expect(
      mergeLineages([valueLineage(sourced(a)), valueLineage(sourced(sameLabel))]).sources,
    ).toEqual([a, sameLabel]);
  });

  it.each(['NewZeros', 'ZerosLike', 'OnesScaled', 'Arange'])(
    '%s does not inherit reference tensor IDs',
    (type) => {
      expect(
        run(type, { in: sourced(a), device: sourced(b), d0: sourced(c) }).lineage,
      ).toMatchObject({ sources: [], origins: ['constant'], unknown: false });
    },
  );
  it.each(['TensorSize', 'TensorShapeSlice', 'GetDevice'])(
    '%s extracts metadata rather than input features',
    (type) => {
      expect(run(type, { in: sourced(a) }, { kind: 'int', value: 2 }).lineage).toMatchObject({
        sources: [],
        origins: ['metadata'],
        unknown: false,
      });
    },
  );
  it('excludes interpolation size references and shape-only metadata updates', () => {
    expect(
      numbers(
        run('CodeFunction', { in: sourced(a), like: sourced(c) }, tensor, {
          operation: 'Interpolate2DNearest',
        }),
      ),
    ).toEqual([1]);
    expect(
      numbers(
        run('UpdateImageMetaInputShape', {
          img_metas: sourced(b),
          input_shape: run('TensorShapeSlice', { in: sourced(a) }),
        }),
      ),
    ).toEqual([2]);
  });
  it('retains input IDs through parameterized and randomized operations without inventing sources', () => {
    for (const type of ['Linear', 'Conv2d'])
      expect(run(type, { in: sourced(a) }).lineage).toMatchObject({
        sources: [a],
        origins: ['parameter'],
      });
    expect(
      run('CodeFunction', { in: sourced(a) }, tensor, { operation: 'GpuPhotoMetricDistortion' })
        .lineage,
    ).toMatchObject({ sources: [a], origins: ['random'], unknown: false });
    expect(
      run('BufferSource', { d0: sourced(a) }, tensor, { initial: 'ones' }).lineage,
    ).toMatchObject({ sources: [], origins: ['state'], initializations: ['ones'] });
  });
  it('marks unsupported operators and symbolic while results unknown instead of copying accidental metadata', () => {
    for (const type of ['NotYetSupported', 'WhileLoop'])
      expect(run(type, { in: sourced(a) }, sourced(a)).lineage).toMatchObject({
        sources: [],
        unknown: true,
      });
    expect(
      run('Add', { a: run('NotYetSupported', { in: sourced(a) }), b: sourced(c) }).lineage,
    ).toMatchObject({ sources: [c], unknown: true });
  });
  it('preserves independent container fields/items and gives all fields of a root dictionary the same ID', () => {
    const first = sourced(a) as TensorSpec,
      second = sourced(b) as TensorSpec;
    const sequence = run('TensorSequencePack', {}, { kind: 'tuple', items: [first, second] });
    expect(numbers(sequence)).toEqual([1, 2]);
    expect(numbers(run('TensorSequenceGet', { in: sequence }, first))).toEqual([1]);
    const dictionary = sourced(c, {
      kind: 'dict',
      fields: { x: tensor, nested: { kind: 'dict', fields: { y: tensor } } },
    });
    if (!isDictionarySpec(dictionary)) throw Error('dictionary');
    expect(numbers(run('DictionaryGet', { in: dictionary }, dictionary.fields.x))).toEqual([3]);
    const empty = sourced(c, { kind: 'dict', fields: {} });
    expect(numbers(run('Identity', { in: empty }, empty))).toEqual([3]);
  });
  it('tracks parameters but never creates root IDs for Embedding/GetParameter/uniform initialization', () => {
    const p: Project = {
      schemaVersion: 1,
      name: 'params',
      variants: [],
      lineageEnabled: true,
      graph: {
        nodes: [
          registry.create('EmbeddingModule', { num_embeddings: 4, embedding_dim: 3 }, 'embedding'),
          registry.create('GetParameter', { name: 'weight' }, 'weight'),
          registry.create('UniformInitColumns', {}, 'init'),
        ],
        edges: [
          { id: 'a', source: 'embedding', sourcePort: 'out', target: 'weight', targetPort: 'in' },
          { id: 'b', source: 'weight', sourcePort: 'out', target: 'init', targetPort: 'in' },
        ],
      },
    };
    const r = inferProject(p, registry);
    expect(r.nodes.init.status).toBe('ok');
    expect(r.nodes.init.outputs.out.lineage).toMatchObject({
      sources: [],
      origins: ['parameter'],
      initializations: ['random'],
    });
    expect(r.nodes.weight.outputs.out).toHaveProperty('parameterRef');
  });
  it('tracks ForEach items separately while every nested interface inherits original root IDs', () => {
    const built = buildModel(loopRecipe());
    expect(built.ok).toBe(true);
    const p = built.project!;
    p.lineageEnabled = true;
    const r = inferProject(p, registry);
    expect(r.nodes.loop.status).toBe('ok');
    expect(numbers(r.nodes.loop.outputs.out)).toEqual([1, 2, 3, 4]);
    r.nodes.loop.iterations!.forEach((iteration, i) => {
      expect(numbers(iteration.outputs.out)).toEqual([i + 1]);
      expect(numbers(iteration.nodes.item.outputs.out)).toEqual([i + 1]);
      expect(iteration.nodes.item.outputs.out.lineage?.evidence).toBe('inferred');
    });
  });
  it('unions Q/K/V sources inside concrete attention subgraphs, not from wrapper input count', () => {
    const built = buildModel({
      name: 'attention',
      nodes: [
        { id: 'q', type: 'Input', params: { shape: '2,3,8' } },
        { id: 'kv', type: 'Input', params: { shape: '2,6,8' } },
        { id: 'attention', type: 'MultiheadAttention', params: { d_model: 8, heads: 2 } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['q.out', 'attention.q'],
        ['kv.out', 'attention.k'],
        ['kv.out', 'attention.v'],
        ['attention.out', 'out.in'],
      ],
    });
    expect(built.ok).toBe(true);
    const p = built.project!;
    p.lineageEnabled = true;
    const r = inferProject(p, registry);
    expect(r.nodes.attention.status).toBe('ok');
    expect(numbers(r.nodes.attention.outputs.out)).toEqual([1, 2]);
    expect(r.nodes.attention.outputs.out.lineage?.unknown).toBe(false);
  });
  it('preserves IDs across attribute SET/GET without treating GET as a new forward input', () => {
    const built = buildModel({
      name: 'attributes',
      nodes: [
        { id: 'value', type: 'Input', params: { shape: '2,1,3' } },
        { id: 'set', type: 'SetAttribute', params: { owner: '["head"]', name: 'scale' } },
        {
          id: 'head',
          type: 'Submodule',
          subgraph: {
            nodes: [
              { id: 'in', type: 'Inlet', params: { port: 'in' } },
              { id: 'get', type: 'GetAttribute', params: { owner: '["head"]', name: 'scale' } },
              { id: 'out', type: 'Outlet', params: { port: 'out' } },
            ],
            connections: [['get.out', 'out.in']],
          },
        },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['value.out', 'set.in'],
        ['value.out', 'head.in'],
        ['head.out', 'out.in'],
      ],
    });
    expect(built.ok).toBe(true);
    const p = built.project!;
    p.lineageEnabled = true;
    const r = inferProject(p, registry);
    expect(numbers(r.nodes.head.outputs.out)).toEqual([1]);
    expect(Object.keys(r.nodes.head.inputs)).toEqual(['in']);
  });
  it('numbers runtime input values without mutating snapshot data', () => {
    const p = fixture();
    p.lineageEnabled = true;
    const snapshot = {
      format: 'tensor-trail.runtime-snapshot' as const,
      version: 1 as const,
      id: 'run',
      name: 'run',
      capturedAt: '2026-11-12T00:00:00Z',
      context: {},
      bindings: [
        {
          nodePath: ['signal_a'],
          nodeType: 'Input' as const,
          value: { ...tensor, autograd: { requiresGrad: false, source: 'runtime' as const } },
        },
      ],
    };
    p.runtime = { active: 'run', snapshots: [{ id: 'run', name: 'run', file: 'runs/run.json' }] };
    const original = structuredClone(snapshot),
      r = inferProject(p, registry, snapshot);
    expect(r.nodes.signal_a.outputs.out).toMatchObject({
      autograd: { requiresGrad: false },
      lineage: { evidence: 'root' },
    });
    expect(numbers(r.nodes.merge.outputs.out)).toEqual([1, 2]);
    expect(snapshot).toEqual(original);
  });
  it('supports API toggle and legacy cleanup, preserving layouts and linked input identity', () => {
    const p = fixture(),
      original = structuredClone(p);
    const changed = patchModel(
      p,
      {
        operations: [
          { op: 'set_lineage_enabled', enabled: true },
          { op: 'duplicate_linked', source: 'signal_a', id: 'copy', position: { x: 70, y: 90 } },
        ],
      },
      { layout: 'preserve' },
    );
    expect(changed.ok).toBe(true);
    expect(p).toEqual(original);
    expect(numbers(changed.nodes.find((n) => n.path[0] === 'copy')!.outputs.out)).toEqual([1]);
    expect(changed.project!.graph.nodes[0].position).toEqual(original.graph.nodes[0].position);
    const legacy = changed.project!;
    legacy.graph.nodes[0].informationSource = { kind: 'image' };
    const reset = patchModel(legacy, {
      operations: [{ op: 'set_information_source', id: 'copy', source: null }],
    });
    expect(reset.ok).toBe(true);
    expect(reset.project!.graph.nodes[0].informationSource).toBeUndefined();
    expect(numbers(reset.nodes.find((n) => n.path[0] === 'copy')!.outputs.out)).toEqual([1]);
    const off = patchModel(reset.project!, {
      operations: [{ op: 'set_lineage_enabled', enabled: false }],
    });
    expect(off.nodes.find((n) => n.path[0] === 'signal_a')!.outputs.out.lineage).toBeUndefined();
    expect(JSON.stringify(getSchemas().patch)).not.toContain('"ego"');
    expect(
      patchModel(p, {
        operations: [{ op: 'set_information_source', id: 'signal_a', source: { kind: 'ego' } }],
      }).ok,
    ).toBe(false);
  });
  it.each([
    null,
    [],
    { kind: 'bogus' },
    { kind: 'image', extra: true },
    { kind: 'constant', initialization: 'bogus' },
  ])('rejects malformed legacy declaration %j', (source) => {
    expect(() => validateInformationSource(source)).toThrow();
  });
  it('rejects invalid stored settings and API operations atomically', () => {
    const p = fixture(),
      before = structuredClone(p);
    expect(validateModel({ ...p, lineageEnabled: 'yes' }).ok).toBe(false);
    expect(patchModel(p, { operations: [{ op: 'set_lineage_enabled', enabled: 'yes' }] }).ok).toBe(
      false,
    );
    expect(p).toEqual(before);
  });
});
