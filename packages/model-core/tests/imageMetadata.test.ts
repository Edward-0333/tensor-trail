import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { isTensorSpec, shapeText } from '../src/core/types';
import { registry } from '../src/modules';
import { imageMetadataCode } from '../src/modules/imageMetadata';

const image = (shape: (number | string)[] = [60, 3, 256, 704]): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype: 'float32',
  autograd: { requiresGrad: true, source: 'manual' },
});
const read = (x: ValueSpec = image(), p: Params = {}) =>
  registry.get('TensorShapeSlice')!.inferValue!({ in: x }, { start: '-2', end: '', ...p });
const metas: ValueSpec = { kind: 'forward', text: 'img_metas: list[dict], other fields unknown' };
const update = (size = read().outputs.out, input: ValueSpec = metas) =>
  registry.get('UpdateImageMetaInputShape')!.inferValue!(
    { img_metas: input, input_shape: size },
    {},
  );

describe('Image shape metadata', () => {
  it('reads H,W as a non-Tensor Size without mutating the image or propagating gradients', () => {
    const x = image(),
      before = structuredClone(x),
      result = read(x);
    expect(result.outputs.out).toEqual({ kind: 'size', dims: [256, 704] });
    expect(shapeText(result.outputs.out)).toBe('torch.Size([256, 704])');
    expect(isTensorSpec(result.outputs.out)).toBe(false);
    expect(result.outputs.out).not.toHaveProperty('autograd');
    expect(x).toEqual(before);
  });

  it.each([
    [
      [12, 3, 320, 800],
      [320, 800],
    ],
    [
      ['BN', 3, 'H', 'W'],
      ['H', 'W'],
    ],
  ])('derives changed or symbolic spatial dimensions: %j', (shape, dims) => {
    expect(read(image(shape)).outputs.out).toEqual({ kind: 'size', dims });
  });

  it.each([
    [{ start: '', end: '' }, [60, 3, 256, 704]],
    [{ start: '1', end: '-1' }, [3, 256]],
    [{ start: '-99', end: '99' }, [60, 3, 256, 704]],
    [{ start: '2', end: '1' }, []],
  ])('uses Python-style start/end shape slicing: %j', (p, dims) => {
    expect(read(image(), p).outputs.out).toEqual({ kind: 'size', dims });
  });

  it.each<Params>([
    { start: '1.5' },
    { end: 'x' },
    { start: '1e2' },
    { start: false },
    { start: '9'.repeat(40) },
  ])('rejects invalid bounds without breaking the code inspector: %j', (params) => {
    expect(() => read(image(), params)).toThrow();
    expect(imageMetadataCode(registry.create('TensorShapeSlice', params))).toContain('修正');
  });

  it('records an in-place per-sample update, without inventing a full dictionary or a batch of 60', () => {
    const before = structuredClone(metas),
      result = update();
    expect(result.outputs.out).toMatchObject({ kind: 'list' });
    expect(result.outputs.out).toHaveProperty(
      'text',
      expect.stringContaining('torch.Size([256, 704])'),
    );
    expect(result.outputs.out).toHaveProperty(
      'text',
      expect.stringContaining('other fields unknown'),
    );
    expect(result.outputs.out).not.toHaveProperty('fields');
    expect(result.notes?.join()).toContain('不推测未采集的列表长度');
    expect(metas).toEqual(before);
    expect(
      update(read(image([12, 3, 320, 800])).outputs.out, result.outputs.out).outputs.out,
    ).toHaveProperty('text', expect.stringContaining('torch.Size([320, 800])'));
  });

  it.each<ValueSpec>([
    image(),
    { kind: 'dict', fields: {} },
    { kind: 'int', value: 2 },
    { kind: 'none' },
  ])('rejects known non-list metadata %j', (value) =>
    expect(() => update(undefined, value)).toThrow(),
  );
  it.each<ValueSpec>([
    image(),
    { kind: 'size', dims: [1] },
    { kind: 'size', dims: [1, 2, 3] },
    { kind: 'int', value: 2 },
  ])('rejects non-HW size %j', (value) => expect(() => update(value)).toThrow());
  it('requires a Tensor for shape reading and shows the exact loop for the update', () => {
    expect(() => read(metas)).toThrow();
    expect(imageMetadataCode(registry.create('TensorShapeSlice'))).toContain('img.shape[-2:]');
    const code = imageMetadataCode(registry.create('UpdateImageMetaInputShape'));
    expect(code).toContain('for img_meta in img_metas:');
    expect(code).toContain('img_meta.update(input_shape=input_shape)');
  });

  it('is available through draft/preserve API, survives serialization, and responds to image changes', () => {
    const recipe: Recipe = {
      name: 'metadata branch',
      nodes: [
        {
          id: 'img',
          type: 'Input',
          params: { shape: '60,3,256,704', axes: 'BN,C,H,W' },
          position: { x: 42, y: 66 },
        },
        { id: 'metas', type: 'ForwardInput', params: { text: metas.text } },
      ],
      connections: [],
    };
    const original = buildModel(recipe, { mode: 'draft' }).project!;
    const result = patchModel(
      original,
      {
        operations: [
          { op: 'add_node', node: { id: 'size', type: 'TensorShapeSlice' } },
          { op: 'add_node', node: { id: 'update', type: 'UpdateImageMetaInputShape' } },
          { op: 'add_node', node: { id: 'after', type: 'ValueVariable' } },
          { op: 'connect', from: 'img.out', to: 'size.in' },
          { op: 'connect', from: 'size.out', to: 'update.input_shape' },
          { op: 'connect', from: 'metas.out', to: 'update.img_metas' },
          { op: 'connect', from: 'update.out', to: 'after.in' },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.nodes[0].position).toEqual(original.graph.nodes[0].position);
    expect(result.nodes.find((n) => n.path[0] === 'after')?.outputs.out).toMatchObject({
      kind: 'list',
    });
    const roundTrip = parseProject(serializeProject(result.project!), registry);
    expect(validateModel(roundTrip, { mode: 'draft' }).ok).toBe(true);
    const changed = patchModel(
      roundTrip,
      { operations: [{ op: 'set_params', id: 'img', params: { shape: '12,3,320,800' } }] },
      { mode: 'draft' },
    );
    expect(changed.nodes.find((n) => n.path[0] === 'size')?.outputs.out).toEqual({
      kind: 'size',
      dims: [320, 800],
    });
  });
});
