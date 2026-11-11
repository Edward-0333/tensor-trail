import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { parseDictionarySchema } from '../src/core/dictionary';
import { tensorValue, type Project } from '../src/core/types';
import { DEFAULT_TENSOR_PALETTE, tensorGradState, tensorInk } from '../src/core/tensorAppearance';
import { registry } from '../src/modules/index';

function project(): Project {
  return {
    schemaVersion: 1,
    name: 'gradient colors',
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,1,21', axes: '' }, 'x'),
        registry.create('View', { shape: '2,1,21' }, 'view'),
        registry.create('TensorVariable', {}, 'y'),
      ],
      edges: [
        { id: 'a', source: 'x', sourcePort: 'out', target: 'view', targetPort: 'in' },
        { id: 'b', source: 'view', sourcePort: 'out', target: 'y', targetPort: 'in' },
      ],
    },
  };
}
describe('Tensor appearance and conservative autograd metadata', () => {
  it('migrates old colors in the root, subgraphs and variants, without changing layouts', () => {
    const p = project();
    const block = registry.create('Transformer');
    block.subgraph!.nodes[0].tensorColor = '#123456';
    p.graph.nodes.push(block);
    p.variants.push({ id: 'v', name: 'variant', graph: structuredClone(block.subgraph!) });
    p.graph.nodes[0].tensorColor = '#abcdef';
    const restored = parseProject(serializeProject(p), registry);
    expect(restored.graph.nodes[0].tensorBorderColor).toBe('#abcdef');
    expect(restored.graph.nodes[0].tensorColor).toBeUndefined();
    expect(restored.graph.nodes[0].position).toEqual(p.graph.nodes[0].position);
    expect(restored.graph.edges).toEqual(p.graph.edges);
    for (const g of [restored.graph.nodes[3].subgraph!, restored.variants[0].graph])
      expect(g.nodes[0].tensorBorderColor).toBe('#123456');
    expect(parseProject(serializeProject(restored), registry)).toEqual(restored);
    expect(p.graph.nodes[0].tensorColor).toBe('#abcdef');
  });
  it('new explicit border wins over a legacy color; palette is persisted', () => {
    const p = project();
    Object.assign(p.graph.nodes[0], { tensorColor: '#ffffff', tensorBorderColor: '#000000' });
    p.tensorPalette = { ...DEFAULT_TENSOR_PALETTE, on: '#445566' };
    const restored = parseProject(serializeProject(p), registry);
    expect(restored.tensorPalette).toEqual(p.tensorPalette);
    expect(restored.graph.nodes[0].tensorBorderColor).toBe('#000000');
  });
  it.each(['red', 'url(bad)', '#fff', '#00000000'])('rejects invalid colors %s', (color) => {
    const p = project();
    p.tensorPalette = { ...DEFAULT_TENSOR_PALETTE, on: color };
    expect(() => parseProject(serializeProject(p), registry)).toThrow('统一背景色');
    delete p.tensorPalette;
    p.graph.nodes[0].tensorBorderColor = color;
    expect(() => parseProject(serializeProject(p), registry)).toThrow('边框');
  });
  it('keeps unknown separate from false and propagates a known false through View', () => {
    const p = project();
    let r = inferProject(p, registry);
    expect(tensorGradState(tensorValue(r.nodes.x.outputs.out))).toBe('unknown');
    expect(tensorGradState(tensorValue(r.nodes.y.outputs.out))).toBe('unknown');
    p.graph.nodes[0].tensorAutograd = { requiresGrad: false, source: 'runtime' };
    r = inferProject(p, registry);
    expect(tensorValue(r.nodes.x.outputs.out).autograd?.source).toBe('runtime');
    expect(tensorValue(r.nodes.y.outputs.out).autograd).toEqual({
      requiresGrad: false,
      source: 'inferred',
    });
    expect(tensorValue(r.nodes.y.outputs.out).shape).toEqual([2, 1, 21]);
    expect(r.nodes.x.outputs.out).not.toBe(r.nodes.view.outputs.out);
  });
  it('does not guess grad mode, trainable parameters or copy metadata through unknown operators', () => {
    const p = project();
    p.graph.nodes[0].tensorAutograd = { requiresGrad: true, source: 'runtime' };
    expect(tensorGradState(tensorValue(inferProject(p, registry).nodes.y.outputs.out))).toBe(
      'unknown',
    );
    p.graph.nodes[1] = registry.create('Linear', { in_features: 21, out_features: 10 }, 'view');
    p.graph.nodes[0].tensorAutograd.requiresGrad = false;
    expect(tensorGradState(tensorValue(inferProject(p, registry).nodes.y.outputs.out))).toBe(
      'unknown',
    );
    p.graph.nodes[1] = registry.create('ReLU', {}, 'view');
    expect(tensorValue(inferProject(p, registry).nodes.view.outputs.out).autograd).toBeUndefined();
  });
  it('accepts explicit output annotations without mutating the upstream tensor', () => {
    const p = project();
    p.graph.nodes[0].tensorAutograd = { requiresGrad: false, source: 'runtime' };
    p.graph.nodes[2].tensorAutograd = { requiresGrad: true, source: 'manual' };
    const r = inferProject(p, registry);
    expect(tensorGradState(tensorValue(r.nodes.view.outputs.out))).toBe('off');
    expect(tensorGradState(tensorValue(r.nodes.y.outputs.out))).toBe('on');
  });
  it('validates dictionary metadata and preserves integer-time lookup metadata', () => {
    const p = project();
    const schema = JSON.stringify({
      kind: 'dict',
      keyType: 'int',
      fields: {
        0: {
          shape: [2, 1, 21],
          dtype: 'float32',
          autograd: { requiresGrad: false, source: 'runtime' },
        },
      },
    });
    p.graph.nodes[0] = registry.create('ForwardInput', { schema }, 'x');
    p.graph.nodes[1] = registry.create('DictionaryGet', { path: '[0]' }, 'view');
    expect(tensorValue(inferProject(p, registry).nodes.y.outputs.out).autograd).toEqual({
      requiresGrad: false,
      source: 'runtime',
    });
    expect(() => parseDictionarySchema(schema.replace('false', '"false"'))).toThrow('requiresGrad');
    expect(() =>
      parseDictionarySchema(schema.replace('false', 'true').replace('float32', 'int64')),
    ).toThrow('不能');
  });
  it('integer tensors are off without treating Python integers as TensorSpecs', () => {
    expect(tensorGradState({ shape: [], axes: [], dtype: 'int64' })).toBe('off');
    const p = project();
    p.graph.nodes[0].params.dtype = 'bool';
    p.graph.nodes[0].tensorAutograd = { requiresGrad: true, source: 'manual' };
    expect(() => parseProject(serializeProject(p), registry)).toThrow('不能');
    expect(inferProject(p, registry).nodes.x.status).toBe('error');
  });
  it('chooses contrasting ink for custom backgrounds', () => {
    expect(tensorInk('#000000')).toBe('#ffffff');
    expect(tensorInk('#ffffff')).toBe('#000000');
    expect(tensorInk(DEFAULT_TENSOR_PALETTE.on)).toBe('#000000');
  });
});
