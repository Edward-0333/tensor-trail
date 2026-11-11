import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, Project, TensorSpec } from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { registry } from '../src/modules/index';
import { sequenceCode } from '../src/modules/sequenceOps';

const node = (p: Params = {}) =>
  registry.create('CodeFunction', { operation: 'ImageNormalize', ...p });
const spec = (shape: (number | string)[] = [60, 3, 256, 704], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const infer = (x = spec(), p: Params = {}) =>
  registry.get('CodeFunction')!.infer!({ in: x }, node(p).params);

describe('SparseWorld image normalization function block', () => {
  it('preserves the current image shape and float32 dtype without mutating the input', () => {
    const x = spec(),
      before = structuredClone(x),
      result = infer(x);
    expect(result.outputs.out).toEqual(x);
    expect(result.outputs.out).not.toBe(x);
    expect(result.warnings).toEqual([]);
    expect(x).toEqual(before);
    expect(result.notes?.join()).toContain('[1,3,1,1]');
    expect(result.notes?.join()).toContain('to_rgb=True');
  });

  it.each([
    ['float16', 'float32'],
    ['float32', 'float32'],
    ['float64', 'float64'],
  ])('promotes %s input with float32 channel statistics to %s', (input, output) => {
    expect(infer(spec(undefined, input)).outputs.out.dtype).toBe(output);
  });

  it('supports changed or symbolic batch/spatial sizes without hard-coding the debug batch', () => {
    for (const shape of [
      [90, 3, 256, 704],
      ['BN', 3, 'H', 'W'],
    ]) {
      expect(infer(spec(shape)).outputs.out.shape).toEqual(shape);
    }
    expect(infer(spec([60, 'C', 256, 704])).warnings?.length).toBeGreaterThan(0);
  });

  it.each([
    spec([3, 256, 704]),
    spec([60, 1, 256, 704]),
    spec([60, 4, 256, 704]),
    spec(undefined, 'uint8'),
  ])('rejects unsupported image contracts %j', (x) => expect(() => infer(x)).toThrow());

  it.each<Params>([
    { mean: 'not JSON' },
    { mean: '[1,2]' },
    { mean: '[1,2,3,4]' },
    { mean: '["1",2,3]' },
    { mean: '[null,2,3]' },
    { mean: '[true,2,3]' },
    { mean: '[1e400,2,3]' },
    { mean: '[1e100,2,3]' },
    { mean: '[1,2,3]' + ' '.repeat(256) },
    { std: '[1,0,3]' },
    { std: '[-1,2,3]' },
    { std: '[1e-100,2,3]' },
    { std: '[1,2]' },
    { std: '[1,2,3e100]' },
    { to_rgb: 'True' },
  ])('rejects invalid parameters without crashing the code inspector: %j', (p) => {
    expect(() => infer(spec(), p)).toThrow();
    expect(() => sequenceCode(node(p))).not.toThrow();
    expect(sequenceCode(node(p))).not.toContain('img = img /');
  });

  it('allows negative means and changes the channel-order setting and read-only code', () => {
    const p = { mean: '[-1, 0, 1]', std: '[1, 2, 3]', to_rgb: false };
    expect(infer(spec(), p).outputs.out.shape).toEqual([60, 3, 256, 704]);
    expect(infer(spec(), p).notes?.join()).toContain('to_rgb=False');
    const code = sequenceCode(node(p))!;
    expect(code).toContain('mean=[-1.0, 0.0, 1.0]');
    expect(code).toContain('to_rgb=False');
    expect(code).toContain("if img_norm_cfg['to_rgb']:");
  });

  it('shows the actual config lookup, same-device construction, channel indexing and broadcasting', () => {
    const code = sequenceCode(node())!;
    for (const line of [
      "img_norm_cfg = self.data_aug['img_norm_cfg']",
      "torch.tensor(img_norm_cfg['mean'], device=img.device)",
      "torch.tensor(img_norm_cfg['std'], device=img.device)",
      'img = img[:, [2, 1, 0], :, :]',
      'img = img - norm_mean.reshape(1, 3, 1, 1)',
      'img = img / norm_std.reshape(1, 3, 1, 1)',
    ])
      expect(code).toContain(line);
  });

  it('exposes only normalization fields and does not change existing function interfaces', () => {
    const fields = (operation: string) =>
      registry.get('CodeFunction')!.getFields!({ operation }).map((f) => f.key);
    expect(fields('ImageNormalize')).toEqual(['operation', 'mean', 'std', 'to_rgb']);
    expect(fields('NeighborhoodAttention1D')).toEqual([
      'operation',
      'dim',
      'num_heads',
      'kernel_size',
      'attn_drop',
      'proj_drop',
    ]);
    expect(fields('DropPath')).toEqual(['operation', 'p']);
    expect(fields('Interpolate1D')).toEqual(['operation']);
    expect(fields('GpuPhotoMetricDistortion')).toEqual(['operation']);
    expect(ports(node(), registry).inputs.map((p) => p.id)).toEqual(['in']);
    expect(ports(node(), registry).outputs.map((p) => p.id)).toEqual(['out']);
  });

  it('round-trips and infers the augmentation-to-normalization chain without inventing gradients', () => {
    const input = registry.create(
      'Input',
      { shape: '60,3,256,704', axes: 'BN,C,H,W', dtype: 'float32' },
      'img',
    );
    input.tensorAutograd = { requiresGrad: false, source: 'manual' };
    const p: Project = {
      schemaVersion: 1,
      name: 'Image preprocessing',
      variants: [],
      graph: {
        nodes: [
          input,
          registry.create('CodeFunction', { operation: 'GpuPhotoMetricDistortion' }, 'aug'),
          { ...node(), id: 'norm' },
          registry.create('Output', {}, 'out'),
        ],
        edges: [
          ['img', 'aug'],
          ['aug', 'norm'],
          ['norm', 'out'],
        ].map(([source, target]) => ({
          id: `${source}_${target}`,
          source,
          sourcePort: 'out',
          target,
          targetPort: 'in',
        })),
      },
    };
    const before = structuredClone(p);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const result = inferProject(p, registry);
    for (const n of Object.values(result.nodes)) expect(n.status).toBe('ok');
    const output = tensorValue(result.nodes.norm.outputs.out);
    expect(output.shape).toEqual([60, 3, 256, 704]);
    expect(output.dtype).toBe('float32');
    expect(output.autograd).toBeUndefined();
    expect(p).toEqual(before);
  });
});
