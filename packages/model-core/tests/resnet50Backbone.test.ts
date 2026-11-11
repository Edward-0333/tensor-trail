import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, patchModel, validateModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { sequenceCode } from '../src/modules/sequenceOps';

const params = { operation: 'ResNet50Backbone' };
const spec = (shape: (string | number)[] = [60, 3, 256, 704], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: ['batch_views', 'channel', 'height', 'width'],
  dtype,
});
const infer = (x = spec()) => registry.get('CodeFunction')!.infer!({ in: x }, params);

describe('Named ResNet-50 backbone contract', () => {
  it('exposes four stage elements with the actual pre-FPN sizes, leaving the input unchanged', () => {
    const x = spec(),
      before = structuredClone(x),
      result = infer(x);
    expect(Object.keys(result.outputs)).toEqual(['c2', 'c3', 'c4', 'c5']);
    expect(Object.values(result.outputs).map((t) => t.shape)).toEqual([
      [60, 256, 64, 176],
      [60, 512, 32, 88],
      [60, 1024, 16, 44],
      [60, 2048, 8, 22],
    ]);
    expect(result.warnings).toEqual([]);
    expect(result.notes?.join()).toContain('tuple');
    expect(x).toEqual(before);
  });

  it('uses ceiling spatial downsampling for odd images and keeps batch size dynamic', () => {
    expect(Object.values(infer(spec([2, 3, 257, 705])).outputs).map((t) => t.shape)).toEqual([
      [2, 256, 65, 177],
      [2, 512, 33, 89],
      [2, 1024, 17, 45],
      [2, 2048, 9, 23],
    ]);
    expect(infer(spec([1, 3, 320, 800])).outputs.c5.shape).toEqual([1, 2048, 10, 25]);
    expect(infer(spec([1, 3, 1, 1])).outputs.c5.shape).toEqual([1, 2048, 1, 1]);
  });

  it('retains symbolic dimensions and reports unknown channels instead of assuming RGB', () => {
    const result = infer(spec(['BN', 'C', 'H', 'W']));
    expect(result.outputs.c2.shape).toEqual(['BN', 256, 'ceil((H)/4)', 'ceil((W)/4)']);
    expect(result.warnings?.length).toBeGreaterThan(0);
  });

  it.each(['float16', 'float32', 'float64'])('retains %s feature dtype', (dtype) => {
    for (const output of Object.values(infer(spec(undefined, dtype)).outputs))
      expect(output.dtype).toBe(dtype);
  });

  it.each([
    spec([3, 256, 704]),
    spec([60, 1, 256, 704]),
    spec([60, 4, 256, 704]),
    spec([60, 3, 0, 704]),
    spec([0, 3, 256, 704]),
    spec([60, 3, 2.5, 704]),
    spec(undefined, 'uint8'),
  ])('rejects invalid input contracts %j', (input) => expect(() => infer(input)).toThrow());

  it.each([true, false])(
    'does not infer output gradient state from input grad=%s',
    (requiresGrad) => {
      const result = infer({ ...spec(), autograd: { requiresGrad, source: 'runtime' } });
      for (const output of Object.values(result.outputs)) expect(output.autograd).toBeUndefined();
    },
  );

  it('provides a named block without expanding internals or changing other templates', () => {
    const node = registry.create('CodeFunction', params);
    expect(node.subgraph).toBeUndefined();
    expect(ports(node, registry).inputs.map((p) => p.id)).toEqual(['in']);
    expect(ports(node, registry).outputs.map((p) => p.id)).toEqual(['c2', 'c3', 'c4', 'c5']);
    expect(registry.get('CodeFunction')!.getFields!(params).map((f) => f.key)).toEqual([
      'operation',
    ]);
    expect(
      ports(registry.create('CodeFunction', { operation: 'ImageNormalize' }), registry).outputs.map(
        (p) => p.id,
      ),
    ).toEqual(['out']);
    const code = sequenceCode(node)!;
    expect(code).toContain('img_feats = self.img_backbone(img)');
    expect(code).toContain('use_grid_mask=False');
    expect(code).toContain('isinstance(img_feats, dict) 为 False');
    expect(code).toContain('frozen_stages=1');
    expect(
      describeModules().modules.find((m) => m.type === 'CodeFunction')?.fields[0].options,
    ).toContain('ResNet50Backbone');
  });

  it('supports API edits, four connections, serialization and changes of upstream resolution', () => {
    const base = buildModel(
      {
        name: 'backbone',
        nodes: [
          {
            id: 'img',
            type: 'Input',
            params: { shape: '60,3,256,704', axes: 'BN,C,H,W' },
            position: { x: 42, y: 66 },
          },
        ],
        connections: [],
      },
      { mode: 'draft' },
    );
    const original = structuredClone(base.project!);
    const result = patchModel(
      base.project!,
      {
        operations: [
          { op: 'add_node', node: { id: 'backbone', type: 'CodeFunction', params } },
          { op: 'connect', from: 'img.out', to: 'backbone.in' },
          ...['c2', 'c3', 'c4', 'c5'].flatMap((id) => [
            { op: 'add_node' as const, node: { id, type: 'Output' } },
            { op: 'connect' as const, from: `backbone.${id}`, to: `${id}.in` },
          ]),
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(base.project).toEqual(original);
    expect(result.project!.graph.nodes[0].position).toEqual(original.graph.nodes[0].position);
    const saved = parseProject(serializeProject(result.project!), registry);
    expect(validateModel(saved).ok).toBe(true);
    const changed = patchModel(saved, {
      operations: [{ op: 'set_params', id: 'img', params: { shape: '2,3,257,705' } }],
    });
    expect(changed.ok).toBe(true);
    expect(changed.nodes.find((n) => n.path[0] === 'c5')?.outputs.out).toMatchObject({
      shape: [2, 2048, 9, 23],
    });
  });
});
