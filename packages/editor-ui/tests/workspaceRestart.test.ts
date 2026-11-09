import { browserProjectFile } from '../../../apps/web/src/platform/browser';
import { describe, expect, it, vi } from 'vitest';
import { serializeProject } from '@tensor-trail/model-core/core/serialization';
import { example } from '@tensor-trail/model-core/modules/examples';
import type { ProjectFileHost } from '../src/editor/projectFiles';
import { ProjectFiles } from '../src/editor/projectFiles';

describe('每次启动后手动打开项目', () => {
  it('即使旧草稿存在也启动为空，选择 JSON 后关联，重启必须重新打开文件', async () => {
    const memory = new Map([
      ['tensor-trail.project.v1', serializeProject(example())],
      [
        'tensor-trail.workspace.v2',
        JSON.stringify({ version: 2, project: example(), documentId: 'old-file' }),
      ],
    ]);
    const storage = {
      getItem: vi.fn((key: string) => memory.get(key) ?? null),
      setItem: vi.fn((key: string, value: string) => {
        memory.set(key, value);
      }),
      removeItem: vi.fn((key: string) => {
        memory.delete(key);
      }),
    };
    vi.stubGlobal('localStorage', storage);
    let disk = serializeProject({ ...example(), name: '从 JSON 读取的项目' }),
      pending = '';
    const file = {
      name: 'chosen-network.json',
      getFile: async () => ({ size: disk.length, text: async () => disk }),
      createWritable: async () => ({
        write: async (text: string) => {
          pending = text;
        },
        close: async () => {
          disk = pending;
        },
        abort: async () => {},
      }),
    };
    const resource = browserProjectFile(file);
    async function start() {
      vi.resetModules();
      const { useEditor } = await import('../src/editor/store');
      const pickSave = vi.fn(async () => resource);
      const host: ProjectFileHost = {
        read: () => useEditor.getState(),
        load: (project) => useEditor.getState().load(project),
        toast: (message) => useEditor.getState().toast(message),
        pickOpen: async () => resource,
        pickSave,
        download: vi.fn(),
      };
      const files = new ProjectFiles(host);
      const unsubscribe = useEditor.subscribe((state, previous) => {
        if (state.project !== previous.project || state.documentId !== previous.documentId)
          files.changed();
      });
      return { editor: useEditor, files, pickSave, unsubscribe };
    }
    const first = await start();
    expect(first.editor.getState().hasProject).toBe(false);
    expect(first.editor.getState().project).toMatchObject({
      graph: { nodes: [], edges: [] },
      variants: [],
    });
    expect(first.editor.getState().past).toHaveLength(0);
    expect(first.files.getSnapshot().fileName).toBe('');
    await first.files.save();
    expect(first.pickSave).not.toHaveBeenCalled();
    first.editor.getState().add('Linear');
    expect(first.editor.getState().project.graph.nodes).toHaveLength(0);
    await first.files.open();
    expect(first.editor.getState().project.name).toBe('从 JSON 读取的项目');
    expect(first.editor.getState().hasProject).toBe(true);
    expect(first.files.getSnapshot()).toMatchObject({
      fileName: 'chosen-network.json',
      dirty: false,
    });
    first.editor.getState().renameProject('已写入 JSON 的修改');
    await first.files.save();
    expect(JSON.parse(disk).name).toBe('已写入 JSON 的修改');
    expect(first.pickSave).not.toHaveBeenCalled();
    first.editor.getState().renameProject('未保存的临时编辑');
    first.unsubscribe();
    const second = await start();
    expect(second.editor.getState().hasProject).toBe(false);
    expect(second.editor.getState().project.graph.nodes).toHaveLength(0);
    expect(second.files.getSnapshot().fileName).toBe('');
    await second.files.open();
    expect(second.editor.getState().project.name).toBe('已写入 JSON 的修改');
    second.editor.getState().undo();
    expect(second.editor.getState().hasProject).toBe(false);
    expect(second.files.getSnapshot().fileName).toBe('');
    second.editor.getState().redo();
    expect(second.editor.getState().hasProject).toBe(true);
    expect(second.files.getSnapshot().fileName).toBe('chosen-network.json');
    second.editor.getState().newProject();
    expect(second.editor.getState().hasProject).toBe(true);
    expect(second.files.getSnapshot().fileName).toBe('');
    expect(storage.getItem).not.toHaveBeenCalled();
    expect(storage.setItem).not.toHaveBeenCalled();
    expect(storage.removeItem).not.toHaveBeenCalled();
    second.unsubscribe();
    vi.unstubAllGlobals();
  });
});
