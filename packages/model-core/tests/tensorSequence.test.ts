import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec } from '../src/core/types';
import { isTensorSequenceSpec, isTensorSpec, shapeText } from '../src/core/types';
import { registry } from '../src/modules';
import { tensorSequenceCode } from '../src/modules/tensorSequence';

const tensors: TensorSpec[] = [64, 32, 16, 8].map((h, i) => ({
  shape: [60, 256, h, 176 / 2 ** i],
  axes: ['BN', 'C', 'H', 'W'],
  dtype: i === 0 ? 'float16' : 'float32',
  ...(i < 2 ? { autograd: { requiresGrad: i === 0, source: 'runtime' as const } } : {}),
}));
const inputs = Object.fromEntries(tensors.map((t, i) => [`item${i}`, t]));
const def = registry.get('TensorSequencePack')!;
const infer = (params: Params = {}) =>
  def.inferValue!(inputs, { kind: 'tuple', input_count: 4, ...params });

describe('Typed Tensor sequences', () => {
  it.each(['tuple', 'list'])(
    'keeps ordered, independent Tensor metadata in a %s without stacking',
    (kind) => {
      const before = structuredClone(inputs);
      const value = infer({ kind }).outputs.out;
      expect(value).toEqual({ kind, items: tensors });
      expect(isTensorSequenceSpec(value)).toBe(true);
      expect(isTensorSpec(value)).toBe(false);
      expect(value).not.toHaveProperty('autograd');
      expect(shapeText(value)).toBe(`${kind} · 4 个 Tensor`);
      expect(inputs).toEqual(before);
      if (isTensorSequenceSpec(value)) {
        value.items[0].shape[0] = 99;
        expect(inputs).toEqual(before);
      }
    },
  );

  it.each(['tuple', 'list'])(
    'initializes an empty %s without inputs or a zero-size Tensor',
    (kind) => {
      const value = def.inferValue!({}, { kind, input_count: 0 }).outputs.out;
      expect(value).toEqual({ kind, items: [] });
      expect(def.getPorts!({ kind, input_count: 0 }).inputs).toEqual([]);
      expect(shapeText(value)).toContain(kind === 'list' ? '[]' : '()');
      expect(isTensorSpec(value)).toBe(false);
    },
  );

  it.each<Params>([
    { kind: 'dict' },
    { input_count: -1 },
    { input_count: 17 },
    { input_count: 1.5 },
    { input_count: '4' },
    { input_count: Infinity },
    { input_count: NaN },
  ])('rejects invalid parameters but keeps ports and code inspection usable: %j', (params) => {
    expect(() => infer(params)).toThrow();
    const node = registry.create('TensorSequencePack', params);
    expect(tensorSequenceCode(node)).toContain('修正');
    expect(() => def.getPorts!(node.params)).not.toThrow();
  });

  it('rejects non-Tensor elements and missing inputs, and does not misclassify opaque list metadata', () => {
    expect(() =>
      def.inferValue!({ item0: { kind: 'int', value: 2 } }, { kind: 'tuple', input_count: 1 }),
    ).toThrow();
    expect(() => def.inferValue!({}, { kind: 'tuple', input_count: 1 })).toThrow();
    expect(isTensorSequenceSpec({ kind: 'list', text: 'img_metas: list[dict]' })).toBe(false);
    expect(tensorSequenceCode(registry.create('TensorSequencePack', { input_count: 1 }))).toContain(
      '(item0,)',
    );
    expect(
      tensorSequenceCode(registry.create('TensorSequencePack', { kind: 'list', input_count: 0 })),
    ).toContain('out = []');
  });

  it('returns one tuple through a nested Outlet and ValueVariable via draft/preserve API', () => {
    const built = buildModel(
      {
        name: 'function tuple return',
        nodes: [
          {
            id: 'fn',
            type: 'Submodule',
            position: { x: 81, y: 42 },
            subgraph: {
              nodes: tensors.map((_, i) => ({
                id: `p${i}`,
                type: 'Inlet',
                params: { port: `p${i}` },
              })),
              connections: [],
            },
          },
          ...tensors.map((t, i) => ({
            id: `input${i}`,
            type: 'Input',
            params: { shape: t.shape.join(','), axes: t.axes.join(','), dtype: t.dtype },
          })),
        ],
        connections: tensors.map((_, i) => [`input${i}.out`, `fn.p${i}`]),
      },
      { mode: 'draft' },
    );
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const result = patchModel(
      built.project!,
      {
        operations: [
          { op: 'add_node', path: ['fn'], node: { id: 'pack', type: 'TensorSequencePack' } },
          {
            op: 'add_node',
            path: ['fn'],
            node: { id: 'return', type: 'Outlet', params: { port: 'img_feats' } },
          },
          ...tensors.map((_, i) => ({
            op: 'connect' as const,
            path: ['fn'],
            from: `p${i}.out`,
            to: `pack.item${i}`,
          })),
          { op: 'connect', path: ['fn'], from: 'pack.out', to: 'return.in' },
          { op: 'add_node', node: { id: 'received', type: 'ValueVariable' } },
          { op: 'connect', from: 'fn.img_feats', to: 'received.in' },
          {
            op: 'add_node',
            node: {
              id: 'empty',
              type: 'TensorSequencePack',
              params: { kind: 'list', input_count: 0 },
            },
          },
          { op: 'add_node', node: { id: 'out', type: 'Output' } },
          { op: 'connect', from: 'received.out', to: 'out.in' },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.nodes[0].position).toEqual({ x: 81, y: 42 });
    const received = result.nodes.find((n) => n.path.join('/') === 'received')!.outputs.out;
    expect(received).toMatchObject({
      kind: 'tuple',
      items: tensors.map(({ shape, dtype }) => ({ shape, dtype })),
    });
    expect(result.nodes.find((n) => n.path.join('/') === 'empty')!.outputs.out).toEqual({
      kind: 'list',
      items: [],
    });
    expect(validateModel(parseProject(serializeProject(result.project!), registry)).ok).toBe(true);
  });
});
