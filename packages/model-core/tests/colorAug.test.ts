import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Project, TensorSpec } from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { registry } from '../src/modules/index';
import { sequenceCode } from '../src/modules/sequenceOps';

const params = { operation: 'GpuPhotoMetricDistortion' };
const spec = (shape: (number | string)[] = [60, 3, 256, 704], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: ['batch_views', 'channel', 'height', 'width'].slice(0, shape.length),
  dtype,
});
const infer = (x: TensorSpec) => registry.get('CodeFunction')!.infer!({ in: x }, params);

describe('SparseWorld color augmentation function block', () => {
  it.each(['float16', 'float32', 'float64'])(
    'preserves shape and %s without mutating input',
    (dtype) => {
      const x = spec([60, 3, 256, 704], dtype),
        before = structuredClone(x);
      const result = infer(x);
      expect(result.outputs.out).toEqual(x);
      expect(result.outputs.out).not.toBe(x);
      expect(x).toEqual(before);
      expect(result.warnings).toEqual([]);
      expect(result.notes?.join()).toContain('self.training=True');
      expect(result.notes?.join()).toContain('brightness_delta=32');
    },
  );

  it('keeps batch and spatial sizes dynamic', () => {
    expect(infer(spec(['B*N', 3, 'H', 'W'])).outputs.out).toEqual(spec(['B*N', 3, 'H', 'W']));
    expect(infer(spec([90, 3, 256, 704])).outputs.out.shape).toEqual([90, 3, 256, 704]);
    expect(infer(spec([60, 'C', 256, 704])).warnings?.length).toBeGreaterThan(0);
  });

  it.each([
    spec([3, 256, 704]),
    spec([60, 1, 256, 704]),
    spec([60, 4, 256, 704]),
    spec(undefined, 'uint8'),
  ])('rejects invalid image contracts: %j', (x) => expect(() => infer(x)).toThrow());

  it('uses one tensor port and shows only relevant fields and read-only source', () => {
    const node = registry.create('CodeFunction', params);
    expect(ports(node, registry).inputs.map((p) => p.id)).toEqual(['in']);
    expect(ports(node, registry).outputs.map((p) => p.id)).toEqual(['out']);
    expect(registry.get('CodeFunction')!.getFields!(node.params).map((f) => f.key)).toEqual([
      'operation',
    ]);
    const code = sequenceCode(node)!;
    expect(code).toContain('GpuPhotoMetricDistortion(');
    expect(code).toContain("self.data_aug['img_color_aug'] and self.training");
    expect(code).toContain('img = self.color_aug(img)');
    expect(code).toContain('hue_delta=18');
  });

  it('round-trips and does not turn input grad metadata into a runtime claim for the output', () => {
    const p: Project = {
      schemaVersion: 1,
      name: 'Color augmentation',
      variants: [],
      graph: {
        nodes: [
          {
            ...registry.create(
              'Input',
              { shape: '60,3,256,704', axes: 'batch_views,channel,height,width', dtype: 'float32' },
              'img',
            ),
            tensorAutograd: { requiresGrad: false, source: 'manual' },
          },
          registry.create('CodeFunction', params, 'color_aug'),
          registry.create('Output', {}, 'out'),
        ],
        edges: [
          { id: 'in', source: 'img', sourcePort: 'out', target: 'color_aug', targetPort: 'in' },
          { id: 'out', source: 'color_aug', sourcePort: 'out', target: 'out', targetPort: 'in' },
        ],
      },
    };
    const before = structuredClone(p);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const result = inferProject(p, registry);
    expect(result.nodes.color_aug.status).toBe('ok');
    expect(tensorValue(result.nodes.color_aug.outputs.out).shape).toEqual([60, 3, 256, 704]);
    expect(tensorValue(result.nodes.color_aug.outputs.out).autograd).toBeUndefined();
    expect(p).toEqual(before);
  });
});
