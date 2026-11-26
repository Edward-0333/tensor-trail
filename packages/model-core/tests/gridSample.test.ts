import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, getSchemas, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { inferGraph } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Params, type Project, type RuntimeSnapshot, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const features: TensorSpec = { shape: [16, 32, 32, 8], axes: ['batch', 'channel', 'h', 'w'], dtype: 'float32' };
const grid: TensorSpec = { shape: [16, 256, 4, 2], axes: ['n', 'query', 'point', 'xy'], dtype: 'float32' };
const expected = { shape: [16, 32, 256, 4], axes: ['batch', 'channel', 'query', 'point'], dtype: 'float32' };
const defaults = { mode: 'bilinear', padding_mode: 'zeros', align_corners: false };
function infer(x: TensorSpec, coords: TensorSpec, params: Params = {}) {
  return registry.get('GridSample')!.infer!({ in: x, grid: coords }, registry.create('GridSample', params).params);
}
function recipe(params: Params = {}): Recipe {
  return {
    name: 'Plan A',
    nodes: [
      ...Object.entries({ features, grid }).map(([id, spec]) => ({
        id, type: 'Input', params: { shape: spec.shape.join(','), axes: spec.axes.join(','), dtype: spec.dtype },
      })),
      { id: 'sample', type: 'GridSample', presentation: 'compact', params },
      { id: 'output', type: 'Output' },
    ],
    connections: [['features.out', 'sample.in'], ['grid.out', 'sample.grid'], ['sample.out', 'output.in']],
  };
}

function activateInputSnapshot(project: Project, gradMode: unknown, xGrad: boolean, gridGrad: boolean): RuntimeSnapshot {
  const snapshot: RuntimeSnapshot = {
    format: 'tensor-trail.runtime-snapshot', version: 1, id: 'run', name: '采样', capturedAt: '2026-11-26T00:00:00Z',
    context: gradMode === undefined ? {} : { gradMode },
    bindings: [
      { nodePath: ['features'], nodeType: 'Input', value: { ...features, autograd: { requiresGrad: xGrad, source: 'runtime' } } },
      { nodePath: ['grid'], nodeType: 'Input', value: { ...grid, autograd: { requiresGrad: gridGrad, source: 'runtime' } } },
    ],
  };
  project.runtime = { active: snapshot.id, snapshots: [{ id: snapshot.id, name: snapshot.name, file: 'runs/run.json' }] };
  return snapshot;
}

describe('GridSample', () => {
  it.each(['bilinear', 'nearest', 'bicubic'])('samples the Plan A feature grid in 2D using %s', mode => {
    const x = { ...features, parameterRef: { module: 'encoder', name: 'weight' } };
    const coords = { ...grid, bufferRef: { reference: 'coords', persistent: true } };
    const result = infer(x, coords, { mode });
    expect(result.outputs.out).toEqual(expected);
    expect(result.warnings).toEqual([]);
  });

  it.each(['bilinear', 'nearest'])('samples a 3D volume with symbolic output sizes using %s', mode => {
    const x = { shape: ['B', 8, 7, 6, 5], axes: ['batch', 'channel', 'd', 'h', 'w'], dtype: 'float64' };
    const coords = { shape: ['B', 'D', 4, 'W', 3], axes: ['n', 'slice', 'row', 'col', 'xyz'], dtype: x.dtype };
    const result = infer(x, coords, { mode, padding_mode: 'reflection', align_corners: true });
    expect(result.warnings).toEqual([]);
    expect(result.outputs.out).toEqual({
      shape: ['B', 8, 'D', 4, 'W'], axes: ['batch', 'channel', 'slice', 'row', 'col'], dtype: 'float64',
    });
  });

  it('keeps named axes at their output positions when input names are absent', () => {
    expect(infer({ ...features, axes: [] }, grid).outputs.out.axes).toEqual(['axis0', 'axis1', 'query', 'point']);
    expect(infer(features, { ...grid, axes: [] }).outputs.out.axes).toEqual(['batch', 'channel', 'axis2', 'axis3']);
  });

  it('reports independent unresolved batch and coordinate constraints without losing the output', () => {
    const plan = recipe();
    plan.nodes[0].params!.shape = 'B,32,32,8';
    plan.nodes[1].params!.shape = 'N,Q,4,K';
    const result = buildModel(plan);
    expect(result.ok).toBe(true);
    expect(result.diagnostics).toHaveLength(2);
    for (const constraint of ['B = N', 'K = 2']) {
      expect(result.diagnostics).toContainEqual(expect.objectContaining({
        path: ['sample'], code: 'SHAPE_WARNING', severity: 'warning', message: expect.stringContaining(constraint),
      }));
    }
    expect(tensorValue(result.outputs.output).shape).toEqual(['B', 32, 'Q', 4]);
  });

  it.each([
    { x: '4,8,6,4', coords: 'K,5,7,K', components: 2 },
    { x: 'K,8,6,4', coords: '4,5,7,K', components: 2 },
    { x: '4,8,6,4,2', coords: 'K,5,7,2,K', components: 3 },
    { x: 'K,8,6,4,2', coords: '4,5,7,2,K', components: 3 },
  ])('rejects conflicting batch and coordinate constraints on K for $x and $coords', ({ x, coords, components }) => {
    const plan = recipe();
    plan.nodes[0].params = { shape: x, axes: '' };
    plan.nodes[1].params = { shape: coords, axes: '' };
    const result = buildModel(plan, { mode: 'strict' });
    expect(result.ok).toBe(false);
    expect(result.nodes.find(node => node.type === 'GridSample')).toMatchObject({ status: 'error', outputs: {} });
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: ['sample'], severity: 'error', message: expect.stringMatching(new RegExp(`K = 4.*K = ${components}`)),
    }));
  });

  it('keeps a shared batch and coordinate symbol valid when both constraints agree', () => {
    const result = infer({ ...features, shape: [2, 8, 6, 4] }, { ...grid, shape: ['K', 5, 7, 'K'] });
    expect(result.outputs.out.shape).toEqual([2, 8, 5, 7]);
    expect(result.warnings).toContainEqual(expect.stringContaining('K = 2'));
  });

  it.each([
    { x: [1, 2, 3], coords: [1, 2, 2], error: /4D 或 5D/ },
    { x: [1, 2, 3, 4], coords: [1, 1, 2, 3, 3], error: /grid.*同为 4D/ },
    { x: [1, 2, 3, 4], coords: [2, 1, 2, 2], error: /batch.*不匹配/ },
    { x: [1, 2, 3, 4], coords: [1, 2, 3, 3], error: /末维应为 2/ },
    { x: [1, 2, 3, 4, 5], coords: [1, 2, 3, 4, 2], error: /末维应为 3/ },
    { x: [1, 2, 0, 4], coords: [1, 2, 3, 2], error: /in 空间维.*正整数/ },
    { x: [1, 2, 3, 4, -1], coords: [1, 2, 3, 4, 3], error: /in 空间维.*正整数/ },
    { x: [1, 2, 3, 4], coords: [1, 2, 0, 2], error: /grid 输出空间维.*正整数/ },
    { x: [1, 2, 3, 4, 5], coords: [1, 1.5, 3, 4, 3], error: /grid 输出空间维.*正整数/ },
  ])('rejects incompatible shapes $x and $coords', ({ x, coords, error }) => {
    expect(() => infer({ ...features, shape: x }, { ...grid, shape: coords })).toThrow(error);
  });

  it('rejects bicubic for 5D inputs', () => {
    expect(() => infer({ ...features, shape: [1, 2, 3, 4, 5] }, { ...grid, shape: [1, 6, 7, 8, 3] },
      { mode: 'bicubic' })).toThrow(/bicubic.*仅支持 4D.*5D/);
  });

  it.each([
    ['int64', 'float32', /浮点/], ['float32', 'bool', /浮点/], ['float32', 'float16', /dtype 必须一致/],
    ['bfloat16', 'float16', /dtype 必须一致/],
  ] as const)('checks both dtypes (%s, %s)', (xType, gridType, error) => {
    expect(() => infer({ ...features, dtype: xType }, { ...grid, dtype: gridType })).toThrow(error);
  });

  it.each([
    { x: [1, 8, 6, 4], coords: [1, 5, 7, 2], out: [1, 8, 5, 7] },
    { x: [1, 8, 5, 6, 4], coords: [1, 3, 5, 7, 3], out: [1, 8, 3, 5, 7] },
  ])('supports matching bfloat16 inputs for $x and $coords', ({ x, coords, out }) => {
    const result = infer({ shape: x, dtype: 'bfloat16', axes: [] }, { shape: coords, dtype: 'bfloat16', axes: [] });
    expect(result.outputs.out).toMatchObject({ shape: out, dtype: 'bfloat16' });
    expect(result.warnings).toEqual([]);
  });

  it.each(['in', 'grid'])('reports a missing %s in strict and draft modes', port => {
    const plan = recipe();
    plan.connections = plan.connections.filter(([, to]) => to !== `sample.${port}`);
    for (const mode of ['strict', 'draft'] as const) {
      const result = buildModel(plan, { mode });
      expect(result.ok).toBe(mode === 'draft');
      expect(result.diagnostics).toContainEqual(expect.objectContaining({
        path: ['sample'], severity: mode === 'draft' ? 'warning' : 'error', message: expect.stringContaining(`输入：${port}`),
      }));
      expect(result.nodes.find(n => n.type === 'GridSample')).toMatchObject({
        status: mode === 'draft' ? 'incomplete' : 'blocked', outputs: {},
      });
    }
  });

  it.each([
    [true, false, true], [false, true, true], [false, false, false], [false, undefined, undefined],
    [true, undefined, true], [undefined, true, true], [undefined, false, undefined],
    [undefined, undefined, undefined], [true, true, true],
  ] as const)('preserves grad(%s, %s) = %s through inferGraph', (xGrad, gridGrad, outGrad) => {
    const graph = buildModel(recipe()).project!.graph;
    if (xGrad !== undefined) graph.nodes[0].tensorAutograd = { requiresGrad: xGrad, source: 'manual' };
    if (gridGrad !== undefined) graph.nodes[1].tensorAutograd = { requiresGrad: gridGrad, source: 'runtime' };
    const result = inferGraph(graph, registry);
    expect(result.nodes.sample.status).toBe('ok');
    for (const id of ['sample', 'output']) {
      expect(tensorValue(result.nodes[id].outputs.out).autograd).toEqual(
        outGrad === undefined ? undefined : { requiresGrad: outGrad, source: 'inferred' },
      );
    }
  });

  it.each([
    ['no_grad', true, false, false], ['no_grad', false, true, false],
    ['inference_mode', true, false, false], ['inference_mode', false, true, false],
    ['grad', true, false, true], ['grad', false, true, true],
    ['unknown', true, false, undefined], ['unknown', false, false, false],
  ] as const)('applies graph gradMode %s to inputs (%s, %s) in inferGraph and Agent validate', (gradMode, xGrad, gridGrad, expectedGrad) => {
    const project = buildModel(recipe()).project!;
    project.hyperparameters = { gradMode: { type: 'string', value: gradMode } };
    project.graph.nodes[0].tensorAutograd = { requiresGrad: xGrad, source: 'manual' };
    project.graph.nodes[1].tensorAutograd = { requiresGrad: gridGrad, source: 'manual' };
    const inferred = inferGraph(project.graph, registry, {}, 0, project.hyperparameters);
    expect(inferred.nodes.sample.status).toBe('ok');
    const validated = validateModel(project);
    expect(validated.ok).toBe(true);
    const expectedAutograd = expectedGrad === undefined ? undefined : { requiresGrad: expectedGrad, source: 'inferred' };
    for (const out of [inferred.nodes.sample.outputs.out, validated.outputs.output]) {
      expect(tensorValue(out).autograd).toEqual(expectedAutograd);
    }
  });

  it.each([
    [undefined, 'grad', true], [undefined, 'no_grad', false], [undefined, 'inference_mode', false],
    ['grad', 'no_grad', true], ['no_grad', 'grad', false], ['inference_mode', 'grad', false],
    ['unknown', 'grad', true], ['unknown', 'no_grad', false],
  ] as const)('resolves a manual grad:on annotation for two off inputs with snapshot %s and graph %s to %s', (snapshotMode, graphMode, expectedGrad) => {
    const project = buildModel(recipe()).project!;
    project.hyperparameters = { gradMode: { type: 'string', value: graphMode } };
    project.graph.nodes[0].tensorAutograd = { requiresGrad: false, source: 'manual' };
    project.graph.nodes[1].tensorAutograd = { requiresGrad: false, source: 'manual' };
    project.graph.nodes.find(node => node.id === 'sample')!.tensorAutograd = { requiresGrad: true, source: 'manual' };
    const snapshot = snapshotMode === undefined ? undefined : activateInputSnapshot(project, snapshotMode, false, false);
    const result = validateModel(project, { runtimeSnapshot: snapshot });
    expect(result.ok).toBe(true);
    expect(tensorValue(result.outputs.output).autograd).toEqual({
      requiresGrad: expectedGrad, source: expectedGrad ? 'manual' : 'inferred',
    });
  });

  it.each([
    ['no_grad', undefined, true, false, false], ['no_grad', undefined, false, true, false],
    ['inference_mode', undefined, true, false, false], ['inference_mode', undefined, false, true, false],
    ['no_grad', 'grad', true, false, false], ['inference_mode', 'grad', false, true, false],
    ['grad', 'no_grad', false, true, true], ['grad', 'inference_mode', false, false, false],
    [undefined, 'inference_mode', true, false, false], ['unknown', 'no_grad', false, true, false],
    [undefined, undefined, false, true, true], ['unknown', undefined, true, false, true],
    ['unknown', 'unknown', true, false, undefined], [false, 'grad', true, false, true],
  ] as const)('uses snapshot mode %s and graph mode %s for runtime inputs (%s, %s) → %s', (gradMode, graphMode, xGrad, gridGrad, expectedGrad) => {
    const project = buildModel(recipe()).project!;
    if (graphMode !== undefined) project.hyperparameters = { gradMode: { type: 'string', value: graphMode } };
    const snapshot = activateInputSnapshot(project, gradMode, xGrad, gridGrad);
    const inferred = inferGraph(project.graph, registry, {}, 0, project.hyperparameters, snapshot);
    expect(inferred.nodes.sample.status).toBe('ok');
    const validated = validateModel(project, { runtimeSnapshot: snapshot });
    expect(validated.ok).toBe(true);
    const expectedAutograd = expectedGrad === undefined ? undefined : { requiresGrad: expectedGrad, source: 'inferred' };
    expect.soft(tensorValue(inferred.nodes.sample.outputs.out).autograd).toEqual(expectedAutograd);
    expect.soft(tensorValue(validated.outputs.output).autograd).toEqual(expectedAutograd);
  });

  it('round-trips agent parameters, compact presentation and both root value dependencies', () => {
    const params = { mode: 'nearest', padding_mode: 'border', align_corners: true };
    const built = buildModel(recipe(params));
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(built.outputs.output).toEqual(expected);
    const project = built.project!;
    project.lineageEnabled = true;
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    expect(restored.graph.nodes.find(n => n.id === 'sample')).toMatchObject({ params, presentation: 'compact' });
    const result = validateModel(restored);
    expect(result.ok).toBe(true);
    expect(result.outputs.output).toMatchObject({ ...expected, lineage: {
      sources: [{ id: 'features', index: 1 }, { id: 'grid', index: 2 }], unknown: false, evidence: 'inferred',
    } });
  });

  it('describes the registered ports, defaults and supported recipe/patch schemas', () => {
    const module = describeModules().modules.find(m => m.type === 'GridSample')!;
    expect(registry.create('GridSample').params).toEqual(defaults);
    expect(module).toMatchObject({
      category: '基础层', inputs: [{ id: 'in' }, { id: 'grid' }], outputs: [{ id: 'out' }],
      paramsSchema: { additionalProperties: false, properties: {
        mode: { default: 'bilinear', enum: ['bilinear', 'nearest', 'bicubic'] },
        padding_mode: { default: 'zeros', enum: ['zeros', 'border', 'reflection'] },
        align_corners: { default: false, type: 'boolean' },
      } },
    });
    const schemas = getSchemas();
    for (const schema of [schemas.recipe, schemas.patch]) {
      expect(schema.$defs.node.oneOf).toContainEqual(expect.objectContaining({ properties: expect.objectContaining({
        type: { const: 'GridSample' }, params: module.paramsSchema, presentation: { enum: ['default', 'compact'] },
      }) }));
    }
    const invalidParams: Params[] = [{ mode: 'trilinear' }, { padding_mode: 'wrap' }, { align_corners: 'false' }];
    for (const params of invalidParams) {
      expect(buildModel(recipe(params)).ok).toBe(false);
    }
  });
});
