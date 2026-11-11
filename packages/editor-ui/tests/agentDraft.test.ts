import { describe, expect, it } from 'vitest';
import { canvasApi } from '../src/editor/agentBridge';
import { useEditor } from '../src/editor/store';
import { buildModel } from '@tensor-trail/model-core/agent/index';
import type { RuntimeSnapshot } from '@tensor-trail/model-core/core/types';
import { tensorValue } from '@tensor-trail/model-core/core/types';

const runtime: RuntimeSnapshot = {
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
function load() {
  const p = buildModel(
    {
      name: 'draft',
      nodes: [
        {
          id: 'x',
          type: 'Input',
          params: { shape: '2,3', axes: 'B,C' },
          position: { x: 22, y: 45 },
        },
      ],
      connections: [],
    },
    { mode: 'draft' },
  ).project!;
  p.runtime = { active: 'r', snapshots: [{ id: 'r', name: 'run', file: 'runs/r.json' }] };
  useEditor.getState().load(p);
  useEditor.getState().setRuntimeSnapshot(runtime);
}
describe('Draft canvas integration', () => {
  it('applies one undoable patch, preserves document/runtime state and uses the loaded sidecar', () => {
    load();
    const before = canvasApi.getModel(),
      state = useEditor.getState();
    const changed = canvasApi.patchModel(
      before.project,
      {
        operations: [
          { op: 'add_node', node: { id: 'y', type: 'Identity' } },
          { op: 'connect', from: 'x.out', to: 'y.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(changed.ok).toBe(true);
    expect(tensorValue(changed.nodes.find((n) => n.path[0] === 'y')!.outputs.out).shape).toEqual([
      7, 3,
    ]);
    expect(changed.diagnostics.some((d) => d.code === 'RUNTIME_NOT_LOADED')).toBe(false);
    const applied = canvasApi.applyModel({
      project: changed.project,
      expectedRevision: before.revision,
      mode: 'draft',
      allowWarnings: true,
    });
    expect(applied.ok).toBe(true);
    expect(useEditor.getState().past.length).toBe(state.past.length + 1);
    expect(useEditor.getState().documentId).toBe(state.documentId);
    expect(useEditor.getState().runtimeSnapshot).toEqual(state.runtimeSnapshot);
    expect(useEditor.getState().project.runtime).toEqual(before.project.runtime);
    expect(useEditor.getState().project.graph.nodes[0].position).toEqual({ x: 22, y: 45 });
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before.project);
    expect(useEditor.getState().runtimeSnapshot).toEqual(state.runtimeSnapshot);
  });
  it('invalidates a preview when runtime inputs change', () => {
    load();
    const before = canvasApi.getModel();
    const next = structuredClone(runtime);
    next.bindings[0].value = { shape: [9, 3], axes: ['B', 'C'], dtype: 'float32' };
    useEditor.getState().setRuntimeSnapshot(next);
    const applied = canvasApi.applyModel({
      project: before.project,
      expectedRevision: before.revision,
      mode: 'draft',
      allowWarnings: true,
    });
    expect(applied.ok).toBe(false);
    expect(applied.diagnostics.some((d) => d.code === 'REVISION_CONFLICT')).toBe(true);
  });
});
