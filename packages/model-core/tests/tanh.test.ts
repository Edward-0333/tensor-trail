import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { evaluateFunctionFormula, parseFunctionFormula } from '../src/core/functionFormula';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Project, type TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const spec = (dtype = 'float32'): TensorSpec => ({
  shape: [2, 1, 3],
  axes: ['batch', 'ego', 'xyz_scale'],
  dtype,
});

describe('Tanh and SparseWorld range scaling', () => {
  it.each(['float16', 'bfloat16', 'float32', 'float64'])(
    'preserves floating %s shape and axes',
    (dtype) => {
      const input = spec(dtype);
      const out = registry.get('Tanh')!.infer!({ in: input }, {}).outputs.out;
      expect(out).toEqual(input);
      expect(out).not.toBe(input);
      expect(out.shape).not.toBe(input.shape);
    },
  );

  it('preserves scalar, empty and symbolic shapes', () => {
    for (const shape of [[], [0, 3], ['B', 1, 3]]) {
      const input = { ...spec(), shape, axes: shape.map((_, i) => `axis${i}`) };
      expect(registry.get('Tanh')!.infer!({ in: input }, {}).outputs.out).toEqual(input);
    }
  });

  it.each(['int64', 'uint8', 'bool'])(
    'rejects %s rather than pretending dtype is preserved',
    (dtype) => {
      expect(() => registry.get('Tanh')!.infer!({ in: spec(dtype) }, {})).toThrow('浮点');
    },
  );

  it('round-trips and infers the exact range formula without fabricating gradients', () => {
    const input = registry.create(
      'Input',
      { shape: '2,1,3', axes: 'batch,ego,xyz_scale', dtype: 'float32' },
      'input',
    );
    input.tensorAutograd = { requiresGrad: true, source: 'manual' };
    const project: Project = {
      schemaVersion: 1,
      name: 'scale',
      variants: [],
      graph: {
        nodes: [
          input,
          registry.create('Tanh', {}, 'tanh'),
          registry.create(
            'Function',
            {
              argument: 'points_scale',
              expression: '(points_scale + 1) / 2 * (1.5 - 0.8) + 0.8',
            },
            'range',
          ),
          registry.create('TensorVariable', {}, 'assigned'),
        ],
        edges: [
          ['input', 'tanh'],
          ['tanh', 'range'],
          ['range', 'assigned'],
        ].map(([source, target]) => ({
          id: `${source}_${target}`,
          source,
          sourcePort: 'out',
          target,
          targetPort: 'in',
        })),
      },
    };
    const loaded = parseProject(serializeProject(project), registry);
    expect(loaded).toEqual(project);
    const result = inferProject(loaded, registry);
    expect(result.issues).toEqual([]);
    for (const id of ['tanh', 'range', 'assigned']) {
      expect(result.nodes[id].status).toBe('ok');
      expect(tensorValue(result.nodes[id].outputs.out)).toEqual(spec());
    }
    const formula = parseFunctionFormula(loaded.graph.nodes[2].params.expression, 'points_scale');
    expect(evaluateFunctionFormula(formula, -1)).toBeCloseTo(0.8);
    expect(evaluateFunctionFormula(formula, 0)).toBeCloseTo(1.15);
    expect(evaluateFunctionFormula(formula, 1)).toBeCloseTo(1.5);
  });
});
