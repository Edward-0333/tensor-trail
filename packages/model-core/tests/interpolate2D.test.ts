import { describe, expect, it } from 'vitest';
import { buildModel, patchModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { sequenceCode } from '../src/modules/sequenceOps';

const spec = (shape: (number | string)[] = [60, 256, 8, 22], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: ['BN', 'C', 'H', 'W'],
  dtype,
});
const params = { operation: 'Interpolate2DNearest' };
const infer = (x = spec(), like = spec([60, 256, 16, 44])) =>
  registry.get('CodeFunction')!.infer!({ in: x, like }, params);

describe('Nearest 2D interpolation to a reference shape', () => {
  it('preserves source batch/channels/dtype and uses only reference H/W', () => {
    const x = spec(),
      like = spec([2, 512, 16, 44], 'float64');
    const before = structuredClone({ x, like });
    expect(infer(x, like).outputs.out).toEqual(spec([60, 256, 16, 44]));
    expect({ x, like }).toEqual(before);
  });
  it.each([
    [17, 45],
    [7, 19],
    [8, 22],
  ])('uses exact target %j, not a hard-coded x2', (h, w) => {
    expect(infer(spec(), spec([60, 256, h, w])).outputs.out.shape).toEqual([60, 256, h, w]);
  });
  it('keeps target symbols and axis names', () => {
    const result = infer(spec(['BN', 256, 'H5', 'W5']), {
      ...spec(['N', 'C', 'H4', 'W4']),
      axes: ['N', 'C', 'target_h', 'target_w'],
    });
    expect(result.outputs.out.shape).toEqual(['BN', 256, 'H4', 'W4']);
    expect(result.outputs.out.axes).toEqual(['BN', 'C', 'target_h', 'target_w']);
  });
  it.each(['float16', 'float32', 'float64'])('retains source %s', (dtype) => {
    expect(infer(spec(undefined, dtype)).outputs.out.dtype).toBe(dtype);
  });
  it.each([
    [spec([3, 8, 22]), spec()],
    [spec(), spec([3, 16, 44])],
    [spec([60, 256, 0, 22]), spec()],
    [spec(), spec([60, 256, 0, 44])],
    [spec(), spec([60, 256, 1.5, 44])],
    [spec(undefined, 'uint8'), spec()],
  ])('rejects invalid contracts %#', (x, like) => expect(() => infer(x, like)).toThrow());
  it('does not invent gradient annotations', () => {
    expect(
      infer({ ...spec(), autograd: { requiresGrad: true, source: 'runtime' } }).outputs.out
        .autograd,
    ).toBeUndefined();
  });
  it('publishes the reference input and correct source code, leaving 1D interpolation unchanged', () => {
    const node = registry.create('CodeFunction', params);
    expect(ports(node, registry).inputs.map((p) => p.id)).toEqual(['in', 'like']);
    expect(ports(node, registry).outputs.map((p) => p.id)).toEqual(['out']);
    expect(registry.get('CodeFunction')!.getFields!(params).map((f) => f.key)).toEqual([
      'operation',
    ]);
    expect(sequenceCode(node)).toContain('size=like.shape[2:], mode="nearest"');
    expect(sequenceCode(node)).not.toContain('align_corners');
    expect(sequenceCode(registry.create('CodeFunction', { operation: 'Interpolate1D' }))).toContain(
      'mode="linear"',
    );
  });
  it('works in an API-built lateral/upsample/add/output-conv path and survives serialization', () => {
    const result = buildModel({
      name: 'FPN path',
      nodes: [
        { id: 'fine', type: 'Input', params: { shape: '60,1024,16,44', axes: 'BN,C,H,W' } },
        { id: 'coarse', type: 'Input', params: { shape: '60,256,8,22', axes: 'BN,C,H,W' } },
        {
          id: 'lateral',
          type: 'Conv2d',
          params: { in_channels: 1024, out_channels: 256, kernel_size: 1, padding: 0 },
        },
        { id: 'up', type: 'CodeFunction', params },
        { id: 'add', type: 'Add' },
        {
          id: 'smooth',
          type: 'Conv2d',
          params: { in_channels: 256, out_channels: 256, kernel_size: 3, padding: 1 },
        },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['fine.out', 'lateral.in'],
        ['coarse.out', 'up.in'],
        ['lateral.out', 'up.like'],
        ['lateral.out', 'add.a'],
        ['up.out', 'add.b'],
        ['add.out', 'smooth.in'],
        ['smooth.out', 'out.in'],
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.nodes.find((n) => n.path[0] === 'out')?.outputs.out).toMatchObject({
      shape: [60, 256, 16, 44],
    });
    const project = parseProject(serializeProject(result.project!), registry);
    const changed = patchModel(project, {
      operations: [{ op: 'set_params', id: 'fine', params: { shape: '60,1024,17,45' } }],
    });
    expect(changed.ok).toBe(true);
    expect(changed.nodes.find((n) => n.path[0] === 'out')?.outputs.out).toMatchObject({
      shape: [60, 256, 17, 45],
    });
  });
});
