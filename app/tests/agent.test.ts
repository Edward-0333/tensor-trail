import { describe, expect, it } from 'vitest';
import {
  buildModel,
  describeModules,
  getSchemas,
  patchModel,
  validateModel,
  type Recipe,
} from '../src/agent';
import { createCanvasApi } from '../src/agent/canvas';
import { graphAt } from '../src/core/graph';
import { parseProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { clone, type Project } from '../src/core/types';
import mixed from '../examples/cnn-transformer.recipe.json';
import swiglu from '../examples/swiglu.patch.json';

const recipe = (): Recipe => ({
  recipeVersion: 1,
  name: 'Transformer test',
  nodes: [
    { id: 'input', type: 'Input', params: { shape: '2, 12, 64' } },
    { id: 'encoder', type: 'Transformer' },
    { id: 'output', type: 'Output' },
  ],
  connections: [
    ['input.out', 'encoder.in'],
    ['encoder.out', 'output.in'],
  ],
});
const project = () => buildModel(recipe()).project!;

describe('AI recipes and diagnostics', () => {
  it('builds an expanded, deterministic, reloadable Transformer with node shapes and defaults', () => {
    const result = buildModel(recipe());
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.outputs.output.shape).toEqual([2, 12, 64]);
    expect(
      result.nodes.find((n) => n.path.join('/') === 'encoder/attention/q_heads')?.outputs.out.shape,
    ).toEqual([2, 8, 12, 8]);
    expect(result.defaultsApplied).toContainEqual({
      path: ['encoder'],
      parameter: 'heads',
      value: 8,
    });
    expect(result.project).toEqual(buildModel(recipe()).project);
    expect(parseProject(JSON.stringify(result.project), registry)).toEqual(result.project);
    expect(result.project!.graph.nodes[2].position.x).toBeGreaterThan(
      result.project!.graph.nodes[0].position.x,
    );
  });
  it('builds the CNN → Transformer → bidirectional LSTM example with named hidden ports', () => {
    const result = buildModel(mixed);
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.outputs.output.shape).toEqual([8, 100, 64]);
    expect(result.nodes.find((n) => n.path[0] === 'cnn')?.outputs.out.shape).toEqual([8, 128, 100]);
    expect(result.nodes.find((n) => n.type === 'LSTM')?.outputs.hidden.shape).toEqual([2, 8, 32]);
  });
  it('builds custom nested blocks and preserves their boundary ports', () => {
    const r = recipe();
    r.nodes[1] = {
      id: 'encoder',
      type: 'Block',
      subgraph: {
        nodes: [
          { id: 'input', type: 'Inlet' },
          { id: 'norm', type: 'RMSNorm' },
          { id: 'output', type: 'Outlet' },
        ],
        connections: [
          ['input.out', 'norm.in'],
          ['norm.out', 'output.in'],
        ],
      },
    };
    const result = buildModel(r);
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.nodes[1].customized).toBe(true);
    expect(result.outputs.output.shape).toEqual([2, 12, 64]);
  });
  it.each([
    ['typo module', { id: 'encoder', type: 'Transfomer' }, 'UNKNOWN_MODULE'],
    [
      'typo param',
      { id: 'encoder', type: 'Transformer', params: { n_heads: 8 } },
      'UNKNOWN_PARAMETER',
    ],
    [
      'wrong type',
      { id: 'encoder', type: 'Transformer', params: { heads: '8' } },
      'INVALID_PARAMETER',
    ],
    [
      'zero heads',
      { id: 'encoder', type: 'Transformer', params: { heads: 0 } },
      'INVALID_PARAMETER',
    ],
    ['unsafe ID', { id: '__proto__', type: 'Transformer' }, 'INVALID_ID'],
    ['unknown field', { id: 'encoder', type: 'Transformer', code: 'run()' }, 'UNKNOWN_FIELD'],
  ])('rejects %s without executing a graph', (_, node, code) => {
    const r = recipe();
    r.nodes[1] = node;
    const result = buildModel(r);
    expect(result.ok).toBe(false);
    expect(result.diagnostics[0].code).toBe(code);
    expect(result.project).toBeUndefined();
  });
  it('returns deep shape errors and an inspectable draft for a feature mismatch', () => {
    const r = recipe();
    r.nodes[0].params!.shape = '2, 12, 32';
    const result = buildModel(r);
    expect(result.ok).toBe(false);
    expect(result.project).toBeDefined();
    expect(
      result.diagnostics.some((d) => d.path.join('/') === 'encoder' && d.code === 'SHAPE_ERROR'),
    ).toBe(true);
    const deep = patchModel(project(), {
      operations: [
        { op: 'set_params', path: ['encoder', 'attention'], id: 'q', params: { in_features: 32 } },
      ],
    });
    expect(deep.ok).toBe(false);
    expect(
      deep.diagnostics.some(
        (d) => d.path.join('/') === 'encoder/attention/q' && d.code === 'SHAPE_ERROR',
      ),
    ).toBe(true);
  });
  it('rejects cycles, duplicate connections, missing endpoints and unsupported versions', () => {
    const r = recipe();
    expect(buildModel({ ...r, recipeVersion: 2 }).ok).toBe(false);
    expect(
      buildModel({ ...r, connections: [...r.connections, ['encoder.out', 'encoder.in']] }).ok,
    ).toBe(false);
    expect(buildModel({ ...r, connections: [...r.connections, r.connections[0]] }).ok).toBe(false);
    expect(buildModel({ ...r, connections: [['input.out', 'missing.in']] }).ok).toBe(false);
  });
  it('counts auto-expanded nodes toward the resource limit', () => {
    const r = {
      name: 'large',
      nodes: Array.from({ length: 2000 }, (_, i) => ({ id: `t${i}`, type: 'Transformer' })),
      connections: [],
    };
    expect(buildModel(r).diagnostics[0].code).toBe('LIMIT_EXCEEDED');
  });
  it('describes internal boundaries on request and exposes versioned schemas', () => {
    expect(describeModules().modules.some((m) => m.type === 'Inlet')).toBe(false);
    expect(describeModules(true).modules.some((m) => m.type === 'Inlet')).toBe(true);
    expect(getSchemas().recipe.$schema).toContain('2020-12');
    expect(getSchemas().patch.properties.patchVersion).toEqual({ const: 1 });
  });
});

describe('Atomic model patches', () => {
  it('rebuilds a generated Transformer into RMSNorm/SwiGLU and preserves the original', () => {
    const original = buildModel(mixed).project!,
      before = clone(original);
    const result = patchModel(original, swiglu);
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(original).toEqual(before);
    const encoder = result.project!.graph.nodes.find((n) => n.id === 'encoder')!;
    expect(encoder.subgraph!.nodes.find((n) => n.id === 'norm1')!.type).toBe('RMSNorm');
    expect(
      encoder
        .subgraph!.nodes.find((n) => n.id === 'ffn')!
        .subgraph!.nodes.some((n) => n.type === 'Mul'),
    ).toBe(true);
  });
  it('supports deep edits, marks ancestor customization, and prevents generator overwrites', () => {
    const original = project();
    const result = patchModel(original, {
      operations: [
        { op: 'set_params', path: ['encoder', 'attention'], id: 'weights', params: { dim: -1 } },
      ],
    });
    expect(result.ok).toBe(true);
    expect(result.project!.graph.nodes[1].customized).toBe(true);
    expect(
      graphAt(result.project!.graph, ['encoder']).nodes.find((n) => n.id === 'attention')!
        .customized,
    ).toBe(true);
    const reject = patchModel(result.project, {
      operations: [{ op: 'set_params', id: 'encoder', params: { heads: 4 } }],
    });
    expect(reject.diagnostics[0].code).toBe('CUSTOMIZED_COMPOSITE');
  });
  it('inserts a node by disconnecting and reconnecting as one complete transaction', () => {
    const original = project();
    const result = patchModel(original, {
      operations: [
        { op: 'disconnect', from: 'encoder.out', to: 'output.in' },
        { op: 'add_node', node: { id: 'relu', type: 'ReLU' } },
        { op: 'connect', from: 'encoder.out', to: 'relu.in' },
        { op: 'connect', from: 'relu.out', to: 'output.in' },
      ],
    });
    expect(result.ok).toBe(true);
    expect(result.outputs.output.shape).toEqual([2, 12, 64]);
    expect(original.graph.nodes).toHaveLength(3);
  });
  it('rejects a later failed operation without partial mutation', () => {
    const original = project(),
      before = clone(original);
    const result = patchModel(original, {
      operations: [
        { op: 'rename_project', name: 'Changed' },
        { op: 'set_params', id: 'missing', params: {} },
      ],
    });
    expect(result.ok).toBe(false);
    expect(original).toEqual(before);
    expect(result.project).toBeUndefined();
  });
  it('protects boundary nodes and rejects malformed paths/unknown operations', () => {
    expect(
      patchModel(project(), { operations: [{ op: 'remove_node', path: ['encoder'], id: 'input' }] })
        .diagnostics[0].code,
    ).toBe('PROTECTED_BOUNDARY');
    expect(patchModel(project(), { operations: [{ op: 'layout', path: 'encoder' }] }).ok).toBe(
      false,
    );
    expect(patchModel(project(), { operations: [{ op: 'eval', code: 'anything' }] }).ok).toBe(
      false,
    );
  });
  it('validates full project JSON and rejects graphs with no model output', () => {
    expect(validateModel(project()).ok).toBe(true);
    const p = project();
    p.graph.nodes.pop();
    p.graph.edges.pop();
    expect(validateModel(p).diagnostics.some((d) => d.code === 'MISSING_OUTPUT')).toBe(true);
  });
});

describe('Canvas API revisions', () => {
  function host() {
    let current = project(),
      changes = 0;
    const listeners = new Set<() => void>();
    const commit = (p: Project) => {
      current = clone(p);
      changes++;
      listeners.forEach((fn) => fn());
    };
    const bridge = createCanvasApi({
      read: () => current,
      commit,
      subscribe: (fn) => {
        listeners.add(fn);
        return () => {
          listeners.delete(fn);
        };
      },
    });
    return { ...bridge, changes: () => changes, commit };
  }
  it('returns independent snapshots and rejects stale previews after a manual edit', () => {
    const h = host(),
      before = h.api.getModel();
    before.project.name = 'snapshot edit';
    expect(h.api.getModel().project.name).not.toBe('snapshot edit');
    h.commit(project());
    const result = h.api.applyModel({ project: before.project, expectedRevision: before.revision });
    expect(result.diagnostics.some((d) => d.code === 'REVISION_CONFLICT')).toBe(true);
    expect(h.changes()).toBe(1);
    h.dispose();
  });
  it('applies once with a valid revision and never commits shape errors', () => {
    const h = host(),
      before = h.api.getModel();
    const invalid = recipe();
    invalid.nodes[0].params!.shape = '2, 12, 7';
    expect(h.api.applyModel({ project: invalid, expectedRevision: before.revision }).ok).toBe(
      false,
    );
    expect(h.changes()).toBe(0);
    const result = h.api.applyModel({ project: before.project, expectedRevision: before.revision });
    expect(result.ok).toBe(true);
    expect(h.changes()).toBe(1);
    expect(result.revision).not.toBe(before.revision);
    h.dispose();
  });
  it('requires explicit acknowledgement for symbolic-shape warnings', () => {
    const h = host(),
      before = h.api.getModel(),
      symbolic = recipe();
    symbolic.nodes[0].params!.shape = 'B, T, D';
    const result = buildModel(symbolic);
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.diagnostics.some((d) => d.severity === 'warning')).toBe(true);
    expect(
      h.api.applyModel({ project: result.project, expectedRevision: before.revision }).ok,
    ).toBe(false);
    expect(
      h.api.applyModel({
        project: result.project,
        expectedRevision: before.revision,
        allowWarnings: true,
      }).ok,
    ).toBe(true);
    h.dispose();
  });
});
