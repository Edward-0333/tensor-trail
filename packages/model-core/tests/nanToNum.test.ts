import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import { inferGraph, inferProject } from '../src/core/engine';
import { inferForEach } from '../src/core/forEach';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Dim, Graph, TensorSequenceSpec, TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';

const tensor = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape, axes: shape.map((_, i) => `d${i}`), dtype,
});
const body: Graph = {
  nodes: [
    registry.create('Inlet', { port: 'item' }, 'item'),
    registry.create('NanToNum', {}, 'clean'),
    registry.create('Outlet', { port: 'out' }, 'out'),
  ],
  edges: [
    { id: 'e0', source: 'item', sourcePort: 'out', target: 'clean', targetPort: 'in' },
    { id: 'e1', source: 'clean', sourcePort: 'out', target: 'out', targetPort: 'in' },
  ],
};

describe('NanToNum', () => {
  it.each(['float16', 'bfloat16', 'float32', 'float64', 'int64', 'bool', 'complex64', 'complex128'])(
    'preserves %s, axes and scalar/empty/symbolic shapes without mutating inputs', (dtype) => {
      for (const shape of [[], [2, 1040, 48, 17], ['B', 0, 'D']] as Dim[][]) {
        const input = tensor(shape, dtype), before = structuredClone(input);
        const result = registry.get('NanToNum')!.infer!({ in: input }, {});
        expect(result.outputs.out).toEqual(input);
        expect(result.outputs.out.shape).not.toBe(input.shape);
        expect(result.outputs.out.axes).not.toBe(input.axes);
        expect(input).toEqual(before);
      }
    },
  );

  it('keeps unknown grad mode conservative and never labels the result as a parameter', () => {
    for (const requiresGrad of [true, false, undefined]) {
      const input: TensorSpec = {
        ...tensor([2, 3]),
        ...(requiresGrad === undefined ? {} : { autograd: { requiresGrad, source: 'runtime' } as const }),
      };
      const result = inferGraph(body, registry, { item: input });
      expect(result.issues).toEqual([]);
      const out = result.nodes.clean.outputs.out as TensorSpec;
      expect(out.autograd).toEqual(requiresGrad === false ? { requiresGrad: false, source: 'inferred' } : undefined);
      expect(out.parameterRef).toBeUndefined();
      expect(out.bufferRef).toBeUndefined();
    }
  });

  it('maps all six heterogeneous decoder outputs without stacking or changing their order', () => {
    for (const channels of [17, 3]) {
      const items = [1, 4, 16, 24, 32, 48].map(p => tensor([2, 1040, p, channels]));
      const input = { kind: 'list', items } as TensorSequenceSpec, before = structuredClone(input);
      const result = inferForEach(body, { items: input, initial: { kind: 'list', items: [] } },
        values => inferGraph(body, registry, values));
      expect(result.outputs.out).toEqual(input);
      expect(result.iterations).toHaveLength(6);
      expect(input).toEqual(before);
    }
  });

  it('round-trips through the API and preserves root input provenance', () => {
    const built = buildModel({ name: 'nan_to_num', nodes: [
      { id: 'x', type: 'Input', params: { shape: '2,3', axes: 'B,C' } },
      { id: 'clean', type: 'NanToNum' },
      { id: 'out', type: 'Output' },
    ], connections: [['x.out', 'clean.in'], ['clean.out', 'out.in']] });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    const reopened = parseProject(serializeProject(project), registry);
    expect(reopened).toEqual(project);
    expect(validateModel(reopened).ok).toBe(true);
    const result = inferProject(reopened, registry);
    expect(result.nodes.clean.outputs.out.lineage?.sources).toEqual(result.nodes.x.outputs.out.lineage?.sources);
    expect(result.nodes.clean.outputs.out.lineage?.unknown).toBe(false);
  });
});
