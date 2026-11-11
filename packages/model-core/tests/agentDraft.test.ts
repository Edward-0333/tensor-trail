import { describe, expect, it } from 'vitest';
import {
  buildModel,
  validateModel,
  patchModel,
  getSchemas,
  describeModules,
} from '../src/agent/index';
import type { PatchOptions, Recipe, ValidationOptions } from '../src/agent/types';
import { createCanvasApi } from '../src/agent/canvas';
import type { Project, RuntimeSnapshot } from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { registry } from '../src/modules/index';

const draft = { mode: 'draft' } as const;
const recipe = (): Recipe => ({
  name: 'incremental',
  nodes: [
    { id: 'x', type: 'Input', params: { shape: '2,3', axes: 'B,C' } },
    { id: 'relu', type: 'ReLU' },
  ],
  connections: [],
});
const project = () => buildModel(recipe(), draft).project!;
const run: RuntimeSnapshot = {
  format: 'tensor-trail.runtime-snapshot',
  version: 1,
  id: 'r',
  name: 'run',
  capturedAt: '2026-11-10T00:00:00Z',
  context: {},
  bindings: [
    {
      nodePath: ['x'],
      nodeType: 'Input',
      value: { shape: [7, 3], axes: ['B', 'C'], dtype: 'float32' },
    },
  ],
};

describe('Explicit draft validation', () => {
  it('keeps strict validation as the default and reports draft incompleteness without inventing outputs', () => {
    expect(buildModel(recipe()).ok).toBe(false);
    const r = buildModel(recipe(), draft);
    expect(r.ok).toBe(true);
    expect(r.mode).toBe('draft');
    expect(r.diagnostics).toContainEqual(
      expect.objectContaining({ code: 'MISSING_OUTPUT', severity: 'warning' }),
    );
    expect(r.nodes.find((n) => n.path[0] === 'relu')).toMatchObject({
      status: 'incomplete',
      outputs: {},
    });
    expect(validateModel(r.project).ok).toBe(false);
    expect(validateModel(r.project, draft).ok).toBe(true);
    expect(
      buildModel({ name: 'empty', nodes: [], connections: [] }, draft).diagnostics.map(
        (d) => d.code,
      ),
    ).toEqual(['MISSING_INPUT', 'MISSING_OUTPUT']);
  });
  it('permits independently adding a node and then connecting it in a later patch', () => {
    const p = project(),
      before = structuredClone(p);
    const added = patchModel(
      p,
      { operations: [{ op: 'add_node', node: { id: 'pending', type: 'Identity' } }] },
      draft,
    );
    expect(added.ok).toBe(true);
    const connected = patchModel(
      added.project,
      {
        operations: [
          { op: 'connect', from: 'x.out', to: 'relu.in' },
          { op: 'connect', from: 'relu.out', to: 'pending.in' },
        ],
      },
      draft,
    );
    expect(connected.ok).toBe(true);
    expect(
      tensorValue(connected.nodes.find((n) => n.path[0] === 'pending')!.outputs.out).shape,
    ).toEqual([2, 3]);
    expect(p).toEqual(before);
  });
  it('propagates incompleteness through nested subgraphs, downstream nodes and value aliases', () => {
    const p = buildModel(
      {
        name: 'nested',
        nodes: [
          recipe().nodes[0],
          {
            id: 'block',
            type: 'Submodule',
            subgraph: {
              nodes: [
                { id: 'input', type: 'Inlet' },
                { id: 'pending', type: 'ReLU' },
                { id: 'output', type: 'Outlet' },
              ],
              connections: [['pending.out', 'output.in']],
            },
          },
          { id: 'after', type: 'Identity' },
        ],
        connections: [
          ['x.out', 'block.in'],
          ['block.out', 'after.in'],
        ],
      },
      draft,
    );
    expect(p.ok, JSON.stringify(p.diagnostics)).toBe(true);
    for (const id of ['block', 'after'])
      expect(p.nodes.find((n) => n.path.join('/') === id)?.status).toBe('incomplete');
    const q = project();
    q.graph.nodes.push(registry.create('TensorVariable', {}, 'pending_value'));
    q.graph.edges.push({
      id: 'to_value',
      source: 'relu',
      sourcePort: 'out',
      target: 'pending_value',
      targetPort: 'in',
    });
    q.graph.nodes.push({
      ...registry.create('TensorVariable', {}, 'alias'),
      valueSource: 'pending_value',
    });
    expect(validateModel(q, draft).nodes.find((n) => n.path[0] === 'alias')?.status).toBe(
      'incomplete',
    );
  });
  it('keeps real shape errors and their blocked dependents as errors, even beside a missing input', () => {
    const r = buildModel(
      {
        name: 'bad',
        nodes: [
          recipe().nodes[0],
          { id: 'bad', type: 'Linear', params: { in_features: 4, out_features: 5 } },
          { id: 'after', type: 'Add' },
        ],
        connections: [
          ['x.out', 'bad.in'],
          ['bad.out', 'after.a'],
        ],
      },
      draft,
    );
    expect(r.ok).toBe(false);
    expect(r.diagnostics).toContainEqual(
      expect.objectContaining({ path: ['bad'], code: 'SHAPE_ERROR', severity: 'error' }),
    );
    expect(r.diagnostics).toContainEqual(
      expect.objectContaining({ path: ['after'], code: 'UPSTREAM_BLOCKED', severity: 'error' }),
    );
  });
  it.each([
    { operations: [{ op: 'connect', from: 'x.out', to: 'missing.in' }] },
    { operations: [{ op: 'connect', from: 'x.typo', to: 'relu.in' }] },
    { operations: [{ op: 'connect', from: 'relu.out', to: 'relu.in' }] },
    {
      operations: [
        { op: 'connect', from: 'x.out', to: 'relu.in' },
        { op: 'connect', from: 'x.out', to: 'relu.in' },
      ],
    },
    { operations: [{ op: 'add_node', node: { id: 'x', type: 'ReLU' } }] },
    { operations: [{ op: 'add_node', node: { id: 'typo', type: 'Unknown' } }] },
    {
      operations: [
        { op: 'add_node', node: { id: 'bad', type: 'Linear', params: { in_features: 0 } } },
      ],
    },
  ])('does not downgrade structural or parameter errors: %j', (patch) => {
    const p = project(),
      before = structuredClone(p);
    expect(patchModel(p, patch, draft).ok).toBe(false);
    expect(p).toEqual(before);
  });
  it('warns about unloaded sidecars without removing references or embedding runtime data', () => {
    const p = project();
    p.runtime = { active: 'r', snapshots: [{ id: 'r', name: 'run', file: 'runs/r.json' }] };
    const before = structuredClone(p);
    expect(validateModel(p).ok).toBe(false);
    const r = validateModel(p, draft);
    expect(r.ok).toBe(true);
    expect(r.diagnostics).toContainEqual(
      expect.objectContaining({ code: 'RUNTIME_NOT_LOADED', severity: 'warning' }),
    );
    expect(r.project).toEqual(before);
    const observed = validateModel(p, { ...draft, runtimeSnapshot: run });
    expect(tensorValue(observed.nodes.find((n) => n.path[0] === 'x')!.outputs.out).shape).toEqual([
      7, 3,
    ]);
    expect(observed.diagnostics.some((d) => d.code === 'RUNTIME_NOT_LOADED')).toBe(false);
    expect(observed.project!.runtime!.embedded).toBeUndefined();
    const bad = structuredClone(run);
    bad.bindings[0].nodePath = ['removed'];
    expect(validateModel(p, { ...draft, runtimeSnapshot: bad }).ok).toBe(false);
    expect(p).toEqual(before);
  });
  it.each([{ mode: 'wrong' }, { mode: ['draft'] }, { mode: true }, { unexpected: true }])(
    'rejects invalid policy %j',
    (settings) => {
      expect(validateModel(project(), settings as ValidationOptions).ok).toBe(false);
    },
  );
});

describe('Layout-preserving patches', () => {
  function placed(): Project {
    const r = recipe();
    r.connections = [['x.out', 'relu.in']];
    const p = buildModel(r, draft).project!;
    p.graph.nodes[0].position = { x: -19.25, y: 57.3 };
    p.graph.nodes[1].position = { x: 784.1, y: -25.7 };
    p.graph.edges[0].waypoints = [
      { x: 200, y: 100 },
      { x: 600, y: 100 },
    ];
    p.graph.edges[0].labelPosition = 0.43;
    return p;
  }
  it('preserves all existing geometry by default and places only new nodes deterministically', () => {
    const p = placed(),
      before = structuredClone(p),
      patch = {
        operations: [
          { op: 'add_node', node: { id: 'new', type: 'Identity' } },
          { op: 'connect', from: 'relu.out', to: 'new.in' },
        ],
      };
    const r = patchModel(p, patch, draft);
    expect(r.ok).toBe(true);
    expect(r.project!.graph.nodes.slice(0, 2)).toEqual(p.graph.nodes);
    expect(r.project!.graph.edges[0]).toEqual(p.graph.edges[0]);
    expect(r.project!.graph.nodes[2].position.x).toBeGreaterThan(784.1);
    expect(r.project).toEqual(patchModel(p, patch, draft).project);
    expect(p).toEqual(before);
  });
  it('honors explicit positions including zero and decimals; moves only the requested node', () => {
    const p = placed();
    const r = patchModel(
      p,
      {
        operations: [
          { op: 'add_node', node: { id: 'new', type: 'Identity', position: { x: 0, y: 0 } } },
          { op: 'set_position', id: 'new', position: { x: -40.5, y: 90.25 } },
        ],
      },
      draft,
    );
    expect(r.ok).toBe(true);
    expect(r.project!.graph.nodes[2].position).toEqual({ x: -40.5, y: 90.25 });
    expect(r.project!.graph.nodes.slice(0, 2)).toEqual(p.graph.nodes);
    expect(r.project!.graph.edges).toEqual(p.graph.edges);
    const built = buildModel(
      {
        name: 'positions',
        nodes: [{ ...recipe().nodes[0], position: { x: 25, y: 50 } }],
        connections: [],
      },
      draft,
    );
    expect(built.project!.graph.nodes[0].position).toEqual({ x: 25, y: 50 });
  });
  it('keeps nested geometry and marks semantic edits, not moves, as customized', () => {
    const r = buildModel(
      {
        name: 'nested',
        nodes: [recipe().nodes[0], { id: 'sub', type: 'Submodule' }],
        connections: [['x.out', 'sub.data']],
      },
      draft,
    );
    const before = structuredClone(r.project!);
    const moved = patchModel(
      r.project,
      {
        operations: [{ op: 'set_position', path: ['sub'], id: 'data', position: { x: 20, y: 30 } }],
      },
      draft,
    );
    expect(moved.project!.graph.nodes[1].customized).toBeUndefined();
    const added = patchModel(
      moved.project,
      {
        operations: [
          { op: 'add_node', path: ['sub'], node: { id: 'new', type: 'Identity' } },
          { op: 'connect', path: ['sub'], from: 'data.out', to: 'new.in' },
        ],
      },
      draft,
    );
    expect(added.ok).toBe(true);
    expect(added.project!.graph.nodes[1].customized).toBe(true);
    expect(added.project!.graph.nodes[1].subgraph!.nodes[0].position).toEqual({ x: 20, y: 30 });
    expect(added.project!.graph.nodes[1].position).toEqual(before.graph.nodes[1].position);
  });
  it('retains generator-internal geometry across parameter changes', () => {
    const p = buildModel(
      {
        name: 'generated',
        nodes: [
          { id: 'x', type: 'Input' },
          { id: 't', type: 'Transformer' },
        ],
        connections: [['x.out', 't.in']],
      },
      draft,
    ).project!;
    const g = p.graph.nodes[1].subgraph!;
    g.nodes[0].position = { x: 1234, y: 5678 };
    g.edges[0].waypoints = [{ x: 555, y: 666 }];
    g.edges[0].labelPosition = 0.6;
    const changed = patchModel(
      p,
      { operations: [{ op: 'set_params', id: 't', params: { d_ff: 512 } }] },
      draft,
    );
    expect(changed.ok).toBe(true);
    const next = changed.project!.graph.nodes[1].subgraph!;
    expect(next.nodes[0].position).toEqual(g.nodes[0].position);
    expect(next.edges[0].waypoints).toEqual(g.edges[0].waypoints);
    expect(next.edges[0].labelPosition).toBe(0.6);
  });
  it('reflows only when explicitly requested', () => {
    const p = placed();
    const auto = patchModel(
      p,
      { operations: [{ op: 'add_node', node: { id: 'new', type: 'Identity' } }] },
      { ...draft, layout: 'auto' },
    );
    expect(auto.project!.graph.nodes[0].position).not.toEqual(p.graph.nodes[0].position);
    const explicit = patchModel(p, { operations: [{ op: 'layout' }] }, draft);
    expect(explicit.project!.graph.nodes[0].position).not.toEqual(p.graph.nodes[0].position);
  });
  it.each([{ x: '3', y: 1 }, { x: null, y: 0 }, { x: 1 }, { x: 1, y: 2, z: 3 }])(
    'rejects invalid positions atomically %j',
    (position) => {
      const p = placed(),
        before = structuredClone(p);
      expect(
        patchModel(p, { operations: [{ op: 'set_position', id: 'x', position }] }, draft).ok,
      ).toBe(false);
      expect(p).toEqual(before);
    },
  );
  it('publishes options and positioning in the machine-readable catalogue/schema', () => {
    expect(describeModules().editing.defaultPatchLayout).toBe('preserve');
    expect(getSchemas().patchOptions).toMatchObject({
      properties: { layout: { default: 'preserve' } },
    });
    expect(getSchemas().patch).toMatchObject({
      properties: {
        operations: {
          items: {
            oneOf: expect.arrayContaining([
              expect.objectContaining({
                properties: expect.objectContaining({ op: { const: 'set_position' } }),
              }),
            ]),
          },
        },
      },
    });
    expect(
      patchModel(placed(), { operations: [{ op: 'layout' }] }, {
        layout: ['auto'],
      } as unknown as PatchOptions).ok,
    ).toBe(false);
  });
});

describe('Applying drafts to a canvas', () => {
  it('requires explicit draft mode, warning acknowledgement and a fresh revision', () => {
    let p = project(),
      commits = 0;
    const listeners = new Set<() => void>();
    const h = createCanvasApi({
      read: () => p,
      commit: (next) => {
        p = next;
        commits++;
        listeners.forEach((fn) => fn());
      },
      subscribe: (fn) => {
        listeners.add(fn);
        return () => listeners.delete(fn);
      },
    });
    const snapshot = h.api.getModel();
    expect(
      h.api.applyModel({ project: p, expectedRevision: snapshot.revision, allowWarnings: true }).ok,
    ).toBe(false);
    expect(
      h.api.applyModel({ project: p, expectedRevision: snapshot.revision, mode: 'draft' }).ok,
    ).toBe(false);
    expect(commits).toBe(0);
    expect(
      h.api.applyModel({
        project: p,
        expectedRevision: snapshot.revision,
        mode: 'draft',
        allowWarnings: true,
      }).ok,
    ).toBe(true);
    expect(commits).toBe(1);
    expect(
      h.api
        .applyModel({
          project: p,
          expectedRevision: snapshot.revision,
          mode: 'draft',
          allowWarnings: true,
        })
        .diagnostics.some((d) => d.code === 'REVISION_CONFLICT'),
    ).toBe(true);
    expect(commits).toBe(1);
    h.dispose();
  });
});
