import { describe, expect, it } from 'vitest';
import recipe from '../../../examples/standard-tensor-ops.recipe.json';
import { buildModel, describeModules, getSchemas, patchModel, validateModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Dim, type Params, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const types = ['Sigmoid', 'Log', 'Sqrt', 'Clamp', 'Chunk', 'TopK'];
function infer(type: string, shape: Dim[], params: Params = {}, dtype = 'float32') {
  const input: TensorSpec = { shape, axes: shape.map((_, i) => 'axis' + i), dtype };
  return registry.get(type)!.infer!({ in: input }, registry.create(type, params).params);
}

describe('Standard elementwise, splitting and selection operators', () => {
  it.each(['Sigmoid', 'Log', 'Sqrt'])('%s preserves scalar, symbolic and empty floating tensors', type => {
    for (const dtype of ['float16', 'bfloat16', 'float32', 'float64']) {
      for (const shape of [[], ['B', 0]]) {
        const { out } = infer(type, shape, {}, dtype).outputs;
        expect(out).toEqual({ shape, axes: shape.map((_, i) => 'axis' + i), dtype });
        expect(tensorValue(out).shape).not.toBe(shape);
      }
    }
    for (const dtype of ['bool', 'int64']) expect(() => infer(type, [1], {}, dtype)).toThrow('浮点');
  });

  it('keeps integer clamp dtype and reports empty bounds', () => {
    expect(infer('Clamp', [2], { max: '12' }, 'int64').outputs.out).toEqual({ shape: [2], axes: ['axis0'], dtype: 'int64' });
    expect(infer('Clamp', []).warnings?.join()).toContain('min');
  });

  it.each([
    [12, 6, [2, 2, 2, 2, 2, 2]], [11, 6, [2, 2, 2, 2, 2, 1]],
    [13, 6, [3, 3, 3, 3, 1]], [2, 9, [1, 1]], [0, 3, [0, 0, 0]],
  ] as const)('chunks axis length %i into up to %i pieces', (n, chunks, sizes) => {
    const { outputs } = infer('Chunk', [n, 7], { chunks, dim: -2 }, 'int64');
    expect(Object.keys(outputs)).toEqual(sizes.map((_, i) => 'out' + i));
    sizes.forEach((size, i) => expect(outputs['out' + i]).toEqual({ shape: [size, 7], axes: ['axis0', 'axis1'], dtype: 'int64' }));
  });

  it('represents symbolic widths and the possible short last piece', () => {
    const r = infer('Chunk', ['B', 'D']);
    expect(tensorValue(r.outputs.out0).shape).toEqual(['B', 'floor((D+1)/2)']);
    expect(tensorValue(r.outputs.out1).shape).toEqual(['B', 'D-floor((D+1)/2)']);
    expect(r.warnings).toHaveLength(1);
  });

  it('caps top-k for known and symbolic sizes and preserves unselected axes', () => {
    for (const [n, k, expected] of [[5, 2, 2], [2, 8, 2], [0, 8, 0], ['N', 8, 'min(8,N)'], ['N', 0, 0]] as const) {
      const r = infer('TopK', [n, 4], { k, largest: false, dim: -2 }, 'float64');
      for (const port of ['values', 'indices']) expect(r.outputs[port]).toMatchObject({
        shape: [expected, 4], axes: ['axis0', 'axis1'], dtype: port === 'values' ? 'float64' : 'int64',
      });
    }
    expect(tensorValue(infer('TopK', []).outputs.values).shape).toEqual([]);
    expect(() => infer('TopK', [1], { k: -1 })).toThrow('k');
    for (const type of ['TopK', 'Chunk']) expect(() => infer(type, [1], { dim: -2 })).toThrow('轴');
  });

  it('exposes registry defaults, bounded chunk ports and recipe/patch schemas', () => {
    const catalog = describeModules().modules;
    for (const schema of [getSchemas().recipe, getSchemas().patch]) {
      for (const type of types) {
        expect(schema.$defs.node.oneOf).toContainEqual(expect.objectContaining({
          properties: expect.objectContaining({ type: { const: type }, params: catalog.find(m => m.type === type)!.paramsSchema }),
        }));
      }
    }
    expect(catalog.find(m => m.type === 'Chunk')).toMatchObject({
      outputs: [{ id: 'out0' }, { id: 'out1' }], dynamicPorts: expect.stringContaining('out0'),
      paramsSchema: { properties: { chunks: { default: 2, minimum: 2, maximum: 9 } } },
    });
    expect(registry.create('Log').params).toEqual({ log1p: false });
    expect(registry.create('TopK').params).toEqual({ dim: -1, k: 8, largest: true });
    expect(registry.create('Clamp').params).toEqual({ min: '', max: '' });
    expect(ports(registry.create('Chunk', { chunks: 9 }), registry).outputs.at(-1)?.id).toBe('out8');
  });

  it.each([true, false, undefined])('propagates grad=%s and lineage through every output after save/load', requiresGrad => {
    const built = buildModel(recipe);
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    if (requiresGrad !== undefined) project.graph.nodes[0].tensorAutograd = { requiresGrad, source: 'manual' };
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    const result = validateModel(restored);
    expect(result.ok).toBe(true);
    for (const n of result.nodes.filter(n => types.includes(n.type))) {
      for (const [port, value] of Object.entries(n.outputs)) {
        expect(value.lineage).toMatchObject({ sources: [{ id: 'features' }], unknown: false });
        const grad = port === 'indices' ? false : requiresGrad;
        expect(tensorValue(value).autograd).toEqual(grad === undefined ? undefined : { requiresGrad: grad, source: 'inferred' });
      }
    }
  });

  it('rewires dynamic ports transactionally and blocks consumption of an absent chunk', () => {
    const project = buildModel(recipe).project!;
    const update = patchModel(project, { operations: [
      { op: 'set_params', id: 'split', params: { chunks: 3 } },
      { op: 'disconnect', from: 'split.out1', to: 'positive.in' },
      { op: 'connect', from: 'split.out2', to: 'positive.in' },
    ] });
    expect(update.ok, JSON.stringify(update.diagnostics)).toBe(true);
    expect(tensorValue(update.outputs.probability).shape).toEqual(['B', 4]);
    for (const chunks of [1, 2, 10, 2.5])
      expect(patchModel(update.project!, { operations: [{ op: 'set_params', id: 'split', params: { chunks } }] }).ok).toBe(false);
    const absent = patchModel(update.project!, { operations: [
      { op: 'set_params', id: 'features', params: { shape: 'B,2' } },
    ] }, { mode: 'draft' });
    expect(absent.ok).toBe(false);
    expect(absent.diagnostics.some(d => d.code === 'UPSTREAM_BLOCKED' && d.message.includes('out2'))).toBe(true);
  });
});
