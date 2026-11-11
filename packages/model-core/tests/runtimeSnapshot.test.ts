import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject, validateProjectObject } from '../src/core/serialization';
import {
  parseRuntimeSnapshot,
  portableProject,
  safeSnapshotPath,
} from '../src/core/runtimeSnapshot';
import { tensorValue, type Project, type RuntimeSnapshot } from '../src/core/types';

const snapshot = (id = 'run2', size = 2): RuntimeSnapshot => ({
  format: 'tensor-trail.runtime-snapshot',
  version: 1,
  id,
  name: id,
  capturedAt: '2026-11-10T03:00:00Z',
  context: { batch_size: size },
  bindings: [
    {
      nodePath: ['x'],
      nodeType: 'Input',
      value: {
        shape: [size, 1, 21],
        axes: ['batch', 'ego', 'feature'],
        dtype: 'float32',
        autograd: { requiresGrad: false, source: 'runtime' },
      },
    },
  ],
});
const project = (): Project => ({
  schemaVersion: 1,
  name: 'snapshot',
  variants: [],
  runtime: {
    active: 'run2',
    snapshots: [
      { id: 'run2', name: 'batch2', file: 'runs/run2.json' },
      { id: 'run3', name: 'batch3', file: 'runs/run3.json' },
    ],
  },
  graph: {
    nodes: [
      registry.create('Input', { shape: '1,1,21', axes: '' }, 'x'),
      registry.create('TensorVariable', {}, 'y'),
    ],
    edges: [{ id: 'edge', source: 'x', sourcePort: 'out', target: 'y', targetPort: 'in' }],
  },
});

describe('Independent runtime snapshots', () => {
  it('changes observations without changing the static contract, layout or serialized project', () => {
    const p = project(),
      before = serializeProject(p);
    const r = inferProject(p, registry, snapshot());
    expect(r.issues).toEqual([]);
    expect(tensorValue(r.nodes.y.outputs.out).shape).toEqual([2, 1, 21]);
    expect(tensorValue(r.nodes.y.outputs.out).autograd?.source).toBe('runtime');
    expect(serializeProject(p)).toBe(before);
    p.runtime!.active = 'run3';
    expect(
      tensorValue(inferProject(p, registry, snapshot('run3', 3)).nodes.y.outputs.out).shape,
    ).toEqual([3, 1, 21]);
    delete p.runtime!.active;
    expect(tensorValue(inferProject(p, registry, snapshot()).nodes.y.outputs.out).shape).toEqual([
      1, 1, 21,
    ]);
  });
  it('does not silently treat missing or mismatched runs as runtime data', () => {
    expect(inferProject(project(), registry).issues.join()).toContain('尚未加载');
    expect(inferProject(project(), registry, snapshot('wrong')).issues.join()).toContain(
      '尚未加载',
    );
    const broken = snapshot();
    broken.bindings[0].nodePath = ['missing'];
    expect(inferProject(project(), registry, broken).issues.join()).toContain('不存在');
  });
  it('portable exports contain only the selected run and require no sidecar', () => {
    const p = project();
    expect(() => portableProject(p)).toThrow('尚未加载');
    const exported = parseProject(serializeProject(portableProject(p, snapshot())), registry);
    expect(exported.runtime?.snapshots).toHaveLength(1);
    expect(inferProject(exported, registry).issues).toEqual([]);
    expect(p.runtime?.embedded).toBeUndefined();
    delete p.runtime!.active;
    expect(portableProject(p).runtime).toBeUndefined();
  });
  it.each([
    '../bad.json',
    '/tmp/a.json',
    'a/../../b.json',
    'https://x/a.json',
    'a\\b.json',
    'a/./b.json',
    'a.txt',
  ])('rejects unsafe reference %s', (file) => {
    expect(safeSnapshotPath(file)).toBe(false);
    const p = project();
    p.runtime!.snapshots[0].file = file;
    expect(() => parseProject(serializeProject(p), registry)).toThrow('相对');
  });
  it('validates snapshots before binding and forces provenance to runtime', () => {
    const s = snapshot();
    tensorValue(s.bindings[0].value).autograd = { requiresGrad: false, source: 'manual' };
    expect(parseRuntimeSnapshot(s).bindings[0].value).toHaveProperty('autograd.source', 'runtime');
    expect(() => parseRuntimeSnapshot({ ...s, bindings: [...s.bindings, ...s.bindings] })).toThrow(
      '重复',
    );
    expect(() => parseRuntimeSnapshot({ ...s, capturedAt: 'not a date' })).toThrow('时间');
    expect(() =>
      parseRuntimeSnapshot({ ...s, bindings: [{ ...s.bindings[0], nodeType: 'Linear' }] }),
    ).toThrow('仅支持');
  });
  it('migrates valid old schema strings to objects without mutating caller data', () => {
    const p = project();
    delete p.runtime;
    const schema = {
      kind: 'dict',
      fields: { x: { shape: [2], dtype: 'float32', axes: ['batch'] } },
    };
    p.graph.nodes[0] = registry.create('ForwardInput', {}, 'x');
    p.graph.nodes[0].params.schema = JSON.stringify(schema);
    const loaded = parseProject(serializeProject(p), registry);
    expect(loaded.graph.nodes[0].params.schema).toEqual(schema);
    expect(typeof p.graph.nodes[0].params.schema).toBe('string');
    expect(serializeProject(loaded)).not.toContain('\\"kind\\"');
    expect(parseProject(serializeProject(loaded), registry)).toEqual(loaded);
  });
  it('direct draft validation still enforces size and cycle safety', () => {
    expect(() =>
      validateProjectObject({ ...project(), extra: 'x'.repeat(3_000_001) }, registry),
    ).toThrow('3 MB');
    const p = project() as Project & { cycle?: unknown };
    p.cycle = p;
    expect(() => validateProjectObject(p, registry)).toThrow('循环');
  });
  it('accepts dictionary inputs up to the same depth as static schemas', () => {
    let value: unknown = { shape: [2], dtype: 'float32' };
    for (let i = 0; i < 8; i++) value = { kind: 'dict', fields: { x: value } };
    const s = parseRuntimeSnapshot({
      ...snapshot(),
      bindings: [{ nodePath: ['kwargs'], nodeType: 'ForwardInput', value }],
    });
    expect(s.bindings).toHaveLength(1);
  });
});
