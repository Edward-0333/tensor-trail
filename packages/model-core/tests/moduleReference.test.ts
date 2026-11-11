import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import {
  isModuleSpec,
  shapeText,
  tensorValue,
  type ModuleSpec,
  type Params,
  type ValueSpec,
} from '../src/core/types';
import { registry } from '../src/modules';
import { moduleReferenceCode } from '../src/modules/moduleReference';

const infer = (type: string, params: Params = {}, inputs: Record<string, ValueSpec> = {}) => {
  const node = registry.create(type, params);
  return registry.get(type)!.inferValue!(inputs, node.params).outputs.out;
};
const module = (params: Params = {}) => infer('EmbeddingModule', params) as ModuleSpec;
const parameter = (source = module(), name = 'weight') =>
  tensorValue(infer('GetParameter', { name }, { in: source }));

describe('Module references and parameter access', () => {
  it('distinguishes an Embedding module from its registered Parameter', () => {
    const source = module();
    expect(isModuleSpec(source)).toBe(true);
    expect(shapeText(source)).toBe('Module · nn.Embedding');
    expect(source).not.toHaveProperty('shape');
    expect(source).not.toHaveProperty('autograd');
    expect(() => tensorValue(source)).toThrow(/需要 Tensor/);
    const weight = parameter(source);
    expect(weight).toMatchObject({
      shape: [1040, 3],
      dtype: 'float32',
      autograd: { requiresGrad: true, source: 'inferred' },
      parameterRef: { module: 'self.init_points', name: 'weight' },
    });
    weight.shape[0] = 1;
    expect(source.parameters.weight.shape).toEqual([1040, 3]);
  });

  it('reads any declared direct parameter, not only Embedding.weight', () => {
    const source = infer('ModuleReference', {
      reference: 'self.plan_head',
      module_type: 'nn.Linear',
      parameters: JSON.stringify({
        kind: 'dict',
        fields: {
          weight: {
            shape: [256, 8],
            dtype: 'float32',
            autograd: { requiresGrad: true, source: 'manual' },
          },
          bias: {
            shape: [256],
            dtype: 'float32',
            autograd: { requiresGrad: false, source: 'manual' },
          },
        },
      }),
    }) as ModuleSpec;
    expect(parameter(source, 'bias')).toMatchObject({
      shape: [256],
      autograd: { requiresGrad: false },
      parameterRef: { module: 'self.plan_head', name: 'bias' },
    });
    expect(parameter(source).shape).toEqual([256, 8]);
    expect(() => parameter(source, 'missing')).toThrow(/没有已声明/);
    const unknown = infer('ModuleReference') as ModuleSpec;
    expect(parameter(unknown).autograd).toBeUndefined();
    expect(moduleReferenceCode(registry.create('GetParameter', { name: 'bias' }))).toContain(
      'out = module.bias',
    );
  });

  it.each(['weight.data', '__proto__', '', 'weight;exec()', '0', 'constructor'])(
    'rejects an unsafe or nested attribute: %s',
    (name) => {
      expect(() => parameter(module(), name)).toThrow();
    },
  );

  it.each<ValueSpec>([
    { shape: [3], axes: ['C'], dtype: 'float32' },
    { kind: 'int', value: 3 },
    { kind: 'none' },
  ])('rejects non-module input %j', (input) => {
    expect(() => infer('GetParameter', {}, { in: input })).toThrow(/模块引用/);
  });

  it.each<Params>([
    { reference: 'self.x()' },
    { module_type: '' },
    { parameters: 'invalid' },
    { parameters: '{"kind":"dict","fields":{"weight":{"kind":"none"}}}' },
    { parameters: '{"kind":"dict","fields":{"weight":{"kind":"dict","fields":{}}}}' },
  ])('rejects malformed declarations %j', (params) => {
    expect(() => infer('ModuleReference', params)).toThrow();
  });

  it.each([true, false])(
    'retains parameter identity and requires_grad=%s across in-place initialization',
    (requires_grad) => {
      const node = registry.create(
        'EmbeddingXYZModuleInit',
        { requires_grad, num_embeddings: 7 },
        'init',
      );
      const result = inferGraph({ nodes: [node], edges: [] }, registry).nodes.init;
      expect(result.status, result.messages.join('\n')).toBe('ok');
      const source = result.outputs.out as ModuleSpec;
      expect(isModuleSpec(source)).toBe(true);
      expect(parameter(source)).toMatchObject({
        shape: [7, 3],
        autograd: { requiresGrad: requires_grad },
      });
      for (const id of [
        'get_weight',
        'registered_weight',
        'init_x',
        'init_yz',
        'initialized_weight',
      ])
        expect(result.child!.nodes[id].outputs.out).toEqual(parameter(source));
      expect(result.child!.nodes.weight.outputs.out).toEqual(source);
      expect(resolveForwardView({ nodes: [node], edges: [] }, ['init']).current.title).toContain(
        '初始化',
      );
    },
  );

  it('rejects derived, foreign, resized or gradient-changed parameters as in-place updates', () => {
    const source = module(),
      weight = parameter(source);
    expect(
      infer('ModuleParameterState', { name: 'weight' }, { module: source, parameter: weight }),
    ).toEqual(source);
    const foreign = parameter(module({ reference: 'self.other' }));
    const { parameterRef: _, ...derived } = weight;
    for (const update of [
      foreign,
      derived,
      { ...weight, shape: [4, 3] },
      { ...weight, autograd: { requiresGrad: false, source: 'inferred' as const } },
    ])
      expect(() =>
        infer('ModuleParameterState', { name: 'weight' }, { module: source, parameter: update }),
      ).toThrow();
  });

  it('supports API editing, linked module copies, and serialized reopening without changing layout', () => {
    const built = buildModel(
      {
        name: 'Explicit parameter read',
        nodes: [
          { id: 'init', type: 'EmbeddingXYZModuleInit', position: { x: 20, y: 30 } },
          { id: 'module', type: 'ValueVariable' },
          { id: 'read', type: 'GetParameter' },
          { id: 'weight', type: 'TensorVariable' },
          { id: 'view', type: 'Unsqueeze', params: { dim: 0 } },
        ],
        connections: [
          ['init.out', 'module.in'],
          ['module.out', 'read.in'],
          ['read.out', 'weight.in'],
          ['weight.out', 'view.in'],
        ],
      },
      { mode: 'draft' },
    );
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const result = patchModel(
      built.project!,
      {
        operations: [
          { op: 'duplicate_linked', source: 'module', id: 'nearby', position: { x: 200, y: 100 } },
          { op: 'disconnect', from: 'module.out', to: 'read.in' },
          { op: 'connect', from: 'nearby.out', to: 'read.in' },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const output = (id: string) => result.nodes.find((n) => n.path.join('/') === id)!.outputs.out;
    expect(output('read')).toEqual(output('weight'));
    expect(output('nearby')).toEqual(output('module'));
    expect(tensorValue(output('view')).shape).toEqual([1, 1040, 3]);
    expect(output('view')).not.toHaveProperty('parameterRef');
    expect(output('view')).not.toHaveProperty('autograd');
    for (const node of built.project!.graph.nodes)
      expect(result.project!.graph.nodes.find((n) => n.id === node.id)!.position).toEqual(
        node.position,
      );
    const reloaded = parseProject(serializeProject(result.project!), registry);
    expect(validateModel(reloaded, { mode: 'draft' }).nodes).toEqual(result.nodes);
    expect(JSON.stringify(describeModules())).toContain('GetParameter');
    const broken = structuredClone(reloaded);
    broken.graph.nodes[0].subgraph!.nodes = broken.graph.nodes[0].subgraph!.nodes.filter(
      (n) => n.type !== 'Outlet',
    );
    broken.graph.nodes[0].subgraph!.edges = [];
    expect(validateModel(broken, { mode: 'draft' }).ok).toBe(false);
  });
});
