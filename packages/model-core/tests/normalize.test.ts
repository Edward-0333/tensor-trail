import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, getSchemas, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Params, type Project, type RuntimeSnapshot, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const image: TensorSpec = { shape: [3, 8, 16], axes: ['C', 'H', 'W'], dtype: 'float32' };
const defaults = { dim: -3, mean: '0.485, 0.456, 0.406', std: '0.229, 0.224, 0.225' };

function infer(x: TensorSpec, params: Params = {}) {
  return registry.get('Normalize')!.infer!({ in: x }, { ...defaults, ...params });
}

function recipe(params: Params = {}, shape = '3,8,16'): Recipe {
  return {
    name: 'Image normalization',
    nodes: [
      { id: 'image', type: 'Input', params: { shape, axes: 'C,H,W' } },
      { id: 'norm', type: 'Normalize', presentation: 'compact', params },
      { id: 'out', type: 'Output' },
    ],
    connections: [['image.out', 'norm.in'], ['norm.out', 'out.in']],
  };
}

function activateSnapshot(project: Project, gradMode: string, requiresGrad?: boolean): RuntimeSnapshot {
  project.runtime = { active: 'run', snapshots: [{ id: 'run', name: 'image', file: 'runs/run.json' }] };
  return {
    format: 'tensor-trail.runtime-snapshot', version: 1, id: 'run', name: 'image',
    capturedAt: '2026-11-26T00:00:00Z', context: { gradMode },
    bindings: [{ nodePath: ['image'], nodeType: 'Input', value: {
      ...image, ...(requiresGrad === undefined ? {} : { autograd: { requiresGrad, source: 'runtime' } }),
    } }],
  };
}

describe('Normalize', () => {
  it.each(['float16', 'bfloat16', 'float32', 'float64'])('preserves %s images with any leading axes', dtype => {
    for (const prefix of [[], ['B'], ['B', 'T']]) {
      const x = { dtype, shape: [...prefix, ...image.shape], axes: [...prefix, ...image.axes] };
      const result = infer(x);
      expect(result.warnings).toEqual([]);
      expect(result.outputs.out).toEqual(x);
      expect(result.outputs.out).not.toBe(x);
    }
  });

  it.each([
    { parameterRef: { module: 'encoder', name: 'weight' } },
    { bufferRef: { reference: 'encoder.image', persistent: true } },
  ])('returns a fresh value without its input registration %o', role => {
    expect(infer({ ...image, ...role }).outputs.out).toEqual(image);
  });

  it.each(['uint8', 'int64', 'bool', 'complex64', 'float8'])('rejects dtype %s', dtype => {
    expect(() => infer({ ...image, dtype })).toThrow(/浮点/);
  });

  it.each([
    ['float16', 1e-8], ['float16', 2 ** -25], ['float16', 2 ** -25 * (1 + Number.EPSILON)],
    ['bfloat16', 2 ** -134], ['bfloat16', 2 ** -134 * (1 + Number.EPSILON)],
    ['float32', 2 ** -150],
  ] as const)('rejects std %s %s when it rounds to zero', (dtype, value) => {
    expect(() => infer({ ...image, dtype }, { std: `1,${value},1` }))
      .toThrow(new RegExp(`std.*${dtype}.*0`));
  });

  it.each([
    ['float16', 65520], ['float16', 65519.99999999999],
    ['bfloat16', 2 ** 128 - 2 ** 119], ['bfloat16', (2 ** 128 - 2 ** 119) * (1 - Number.EPSILON)],
    ['float32', 1e39],
  ] as const)('rejects overflowing mean and std for %s at %s', (dtype, value) => {
    for (const [field, n] of [['mean', -value], ['mean', value], ['std', value]] as const) {
      expect(() => infer({ ...image, dtype }, { [field]: `1,${n},1` }))
        .toThrow(new RegExp(`${field}.*${dtype}.*溢出`));
    }
  });

  it.each([
    ['float16', 0, 1e-4],
    ['float16', -65519, 2 ** -25 + 2 ** -48],
    ['bfloat16', 2 ** 128 - 2 ** 120, 2 ** -134 + 2 ** -149],
    ['float32', 1e-100, 2 ** -149],
    ['float64', Number.MAX_VALUE, Number.MIN_VALUE],
  ] as const)('accepts %s statistics when mean %s stays finite and std %s stays positive', (dtype, mean, std) => {
    const x = { ...image, dtype };
    expect(infer(x, { mean: String(mean), std: String(std) })).toEqual({ outputs: { out: x }, warnings: [] });
  });

  it('uses the selected axis and accepts signed decimal and exponential statistics', () => {
    const x = { ...image, shape: [5, 8, 3], axes: ['H', 'W', 'C'] };
    const params = { dim: 2, mean: '-.5, +2e-2, 1.', std: '1.0, 2, .01' };
    expect(infer(x, params)).toEqual({ outputs: { out: x }, warnings: [] });
    expect(() => infer(x, { ...params, dim: 0 })).toThrow(/通道数.*5 与 3/);
  });

  it.each([3, -4, 0.5, '-3'])('rejects invalid dim %s', dim => {
    expect(() => infer(image, { dim })).toThrow(/轴|整数/);
    expect(buildModel(recipe({ dim })).ok).toBe(false);
  });

  it.each([
    ['mean', ''], ['mean', '  '], ['mean', '1, ,3'], ['mean', 'nan'], ['mean', '1a'],
    ['mean', '0x10'], ['mean', '1e999'], ['std', ''], ['std', '1,2,'], ['std', 'Infinity'],
    ['std', '1,0,3'], ['std', '1,-0.01,3'], ['std', '1,2'],
  ])('rejects %s = "%s" without an output', (field, value) => {
    const result = buildModel(recipe({ [field]: value }));
    expect(result.ok).toBe(false);
    expect(result.nodes.find(node => node.type === 'Normalize')).toMatchObject({ status: 'error', outputs: {} });
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      path: ['norm'], severity: 'error', message: expect.stringContaining(field),
    }));
  });

  it('keeps a symbolic channel constraint visible and the inferred output available', () => {
    const result = buildModel(recipe({}, 'C,8,16'));
    expect(result.ok).toBe(true);
    expect(result.diagnostics).toEqual([expect.objectContaining({
      path: ['norm'], code: 'SHAPE_WARNING', severity: 'warning', message: expect.stringContaining('C = 3'),
    })]);
    expect(tensorValue(result.outputs.out).shape).toEqual(['C', 8, 16]);
  });

  it.each([1, 7, 'C'])('broadcasts one pair over %s channels without constraining the size', channels => {
    const x = { ...image, shape: [channels, 8, 16] };
    expect(infer(x, { mean: '-1', std: '.5' })).toEqual({ outputs: { out: x }, warnings: [] });
  });

  it.each([
    // Project mode, active snapshot mode, expected effective mode
    [undefined, undefined, 'grad'], ['grad', undefined, 'grad'],
    ['no_grad', undefined, 'no_grad'], ['inference_mode', undefined, 'inference_mode'],
    ['unknown', undefined, 'unknown'], ['no_grad', 'grad', 'grad'],
    ['grad', 'no_grad', 'no_grad'], ['grad', 'inference_mode', 'inference_mode'],
    ['no_grad', 'unknown', 'no_grad'], ['grad', 'unknown', 'grad'],
    [undefined, 'unknown', 'grad'], ['unknown', 'unknown', 'unknown'],
  ] as const)('uses project %s and snapshot %s as %s through Agent validate', (graphMode, snapshotMode, effective) => {
    const project = buildModel(recipe()).project!;
    const [source, norm] = project.graph.nodes;
    if (graphMode) project.hyperparameters = { gradMode: { type: 'string', value: graphMode } };
    for (const inputGrad of [true, false, undefined]) {
      source.tensorAutograd = inputGrad === undefined ? undefined : { requiresGrad: inputGrad, source: 'manual' };
      const runtimeSnapshot = snapshotMode && activateSnapshot(project, snapshotMode, inputGrad);
      const disabled = effective === 'no_grad' || effective === 'inference_mode';
      const inferred = disabled || inputGrad === false ? false : effective === 'grad' ? inputGrad : undefined;
      for (const manual of [undefined, true, false]) {
        norm.tensorAutograd = manual === undefined ? undefined : { requiresGrad: manual, source: 'manual' };
        const result = validateModel(project, runtimeSnapshot ? { runtimeSnapshot } : {});
        expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
        const automatic = inferred === undefined ? undefined : { requiresGrad: inferred, source: 'inferred' };
        expect(tensorValue(result.outputs.out).autograd).toEqual(
          manual === undefined || (manual && disabled) ? automatic : norm.tensorAutograd,
        );
      }
    }
  });

  it('round-trips configured statistics, compact style and the input value source', () => {
    const params = { dim: 2, mean: '0,1', std: '.5,2' };
    const built = buildModel(recipe(params, '3,8,2'));
    expect(built.ok).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    expect(restored.graph.nodes[1]).toMatchObject({ type: 'Normalize', params, presentation: 'compact' });
    const result = validateModel(restored);
    expect(result.ok).toBe(true);
    expect(result.outputs.out).toMatchObject({
      shape: [3, 8, 2], axes: image.axes, dtype: image.dtype,
      lineage: { sources: [{ id: 'image', index: 1 }], unknown: false, evidence: 'inferred' },
    });
  });

  it('exposes native defaults, ports and presentation through the catalogue and schemas', () => {
    const module = describeModules().modules.find(m => m.type === 'Normalize')!;
    expect(registry.create('Normalize').params).toEqual(defaults);
    expect(module).toMatchObject({
      category: '基础层', inputs: [{ id: 'in' }], outputs: [{ id: 'out' }],
      paramsSchema: { additionalProperties: false, properties: {
        dim: { type: 'integer', default: defaults.dim, minimum: Number.MIN_SAFE_INTEGER },
        mean: { type: 'string', default: defaults.mean }, std: { type: 'string', default: defaults.std },
      } },
    });
    const schemas = getSchemas();
    for (const schema of [schemas.recipe, schemas.patch]) {
      expect(schema.$defs.node.oneOf).toContainEqual(expect.objectContaining({ properties: expect.objectContaining({
        type: { const: 'Normalize' }, params: module.paramsSchema, presentation: { enum: ['default', 'compact'] },
      }) }));
    }
  });
});
