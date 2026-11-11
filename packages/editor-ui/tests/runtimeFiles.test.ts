import { describe, expect, it, vi } from 'vitest';
import type { Project, RuntimeSnapshot } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { serializeProject } from '@tensor-trail/model-core/core/serialization';
import { ProjectFiles, type ProjectFileHost } from '../src/editor/projectFiles';
import { useEditor } from '../src/editor/store';

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
      value: { shape: [2], axes: ['batch'], dtype: 'float32' },
    },
  ],
};
const make = (): Project => ({
  schemaVersion: 1,
  name: 'run',
  variants: [],
  graph: { nodes: [registry.create('Input', {}, 'x')], edges: [] },
  runtime: { active: 'r', snapshots: [{ id: 'r', name: 'run', file: 'runs/r.json' }] },
});
describe('Runtime file workflow', () => {
  it('loads sidecars read-only, exports one portable file and preserves source association', async () => {
    let doc = { project: make(), documentId: 'old', hasProject: true };
    let targetText = '';
    const sourceWrite = vi.fn(async () => true);
    const source = {
      name: 'source.json',
      key: 'source',
      readText: async () => serializeProject(make()),
      writeText: sourceWrite,
      readRelatedText: vi.fn(async () => JSON.stringify(run)),
    };
    const target = {
      name: 'portable.json',
      readText: async () => targetText,
      writeText: async (text: string) => {
        targetText = text;
        return true;
      },
    };
    const host: ProjectFileHost = {
      read: () => doc,
      load: (project) => {
        doc = { ...doc, project, documentId: 'opened' };
      },
      toast: vi.fn(),
      download: vi.fn(),
      pickOpen: async () => source,
      pickSave: async () => target,
    };
    const files = new ProjectFiles(host);
    await files.open();
    const loaded = await files.readRuntime(doc.project.runtime!.snapshots[0]);
    expect(loaded.id).toBe('r');
    expect(sourceWrite).not.toHaveBeenCalled();
    expect(await files.exportPortable(loaded)).toBe(true);
    expect(JSON.parse(targetText).runtime.embedded.id).toBe('r');
    expect(files.getSnapshot().fileName).toBe('source.json');
    expect(doc.project.runtime!.embedded).toBeUndefined();
    expect(await files.save(true)).toBe(false);
    expect(host.toast).toHaveBeenCalledWith(expect.stringContaining('外部快照'));
  });
  it('does not reuse snapshots across new documents or undo into another document', () => {
    const s = () => useEditor.getState();
    s().load(make());
    s().setRuntimeSnapshot(run);
    expect(s().runtimeSnapshot?.id).toBe('r');
    s().newProject();
    expect(s().runtimeSnapshot).toBeUndefined();
    s().undo();
    expect(s().runtimeSnapshot).toBeUndefined();
    s().setRuntimeSnapshot(run);
    s().selectRuntime();
    expect(s().runtimeSnapshot).toBeUndefined();
    expect(s().project.runtime?.active).toBeUndefined();
  });
  it('does not serialize again on busy/selection notifications for immutable editor revisions', async () => {
    const p = make();
    delete p.runtime;
    let reads = 0;
    const shape = p.graph.nodes[0].params.shape;
    Object.defineProperty(p.graph.nodes[0].params, 'shape', {
      enumerable: true,
      get() {
        reads++;
        return shape;
      },
    });
    const host: ProjectFileHost = {
      read: () => ({ project: p, documentId: 'doc', hasProject: true, revision: 1 }),
      load: vi.fn(),
      toast: vi.fn(),
      download: vi.fn(),
      pickSave: async () => undefined,
    };
    const files = new ProjectFiles(host);
    await files.save();
    const after = reads;
    await files.save();
    await files.save();
    expect(reads).toBe(after);
  });
  it('retains an embedded portable run when temporarily switching to the static graph', () => {
    const p = make();
    p.runtime!.embedded = run;
    useEditor.getState().load(p);
    useEditor.getState().selectRuntime();
    expect(useEditor.getState().project.runtime?.active).toBeUndefined();
    expect(useEditor.getState().project.runtime?.embedded?.id).toBe('r');
    useEditor.getState().selectRuntime('r');
    expect(useEditor.getState().project.runtime?.embedded?.id).toBe('r');
  });
});
