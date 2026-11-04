import { describe, expect, it, vi } from 'vitest';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';
import { ProjectFiles, type ProjectFileHost } from '../src/editor/projectFiles';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function file(name = 'source.json', content = serializeProject(example())) {
  const disk = { text: content };
  let pending = '';
  const stream = {
    write: vi.fn(async (text: string) => {
      pending = text;
    }),
    close: vi.fn(async () => {
      disk.text = pending;
    }),
    abort: vi.fn(async () => {}),
  };
  const handle = {
    name,
    getFile: vi.fn(async () => ({ size: disk.text.length, text: async () => disk.text })),
    requestPermission: vi.fn(async () => 'granted'),
    createWritable: vi.fn(async () => stream),
  };
  return { handle, stream, disk };
}
async function setup(overrides: Partial<ProjectFileHost> = {}) {
  let doc = { project: example(), documentId: 'initial', hasProject: true };
  const source = file();
  const target = file('new.json');
  const host: ProjectFileHost = {
    read: () => doc,
    load: (project) => {
      doc = { project, documentId: `${doc.documentId}-opened`, hasProject: true };
    },
    toast: vi.fn(),
    pickOpen: vi.fn(async () => source.handle),
    pickSave: vi.fn(async () => target.handle),
    download: vi.fn(),
    ...overrides,
  };
  const files = new ProjectFiles(host);
  await files.ready;
  return {
    files,
    host,
    source,
    target,
    rename: (name: string) => {
      doc = { ...doc, project: { ...doc.project, name } };
      files.changed();
    },
    switchDocument: () => {
      doc = { project: example('blank'), documentId: `${doc.documentId}-new`, hasProject: true };
      files.changed();
    },
    restoreDocument: (snapshot: typeof doc) => {
      doc = snapshot;
      files.changed();
    },
  };
}

describe('项目源文件保存', () => {
  it('手动打开旧字典输入文件时转换为 Forward 输入，Ctrl+S 写回所选文件', async () => {
    const legacy = {
      schemaVersion: 1 as const,
      name: '旧输入项目',
      variants: [],
      graph: {
        nodes: [
          {
            id: 'data',
            type: 'DictInput',
            definitionVersion: 1 as const,
            label: '字典输入 Dict',
            params: { text: '原备注' },
            position: { x: 0, y: 0 },
          },
        ],
        edges: [],
      },
    };
    const oldText = serializeProject(legacy);
    expect(parseProject(oldText, registry).graph.nodes[0].type).toBe('ForwardInput');
    const source = file('original.json', oldText);
    const s = await setup({
      pickOpen: async () => source.handle,
    });
    await s.files.open();
    expect(s.files.getSnapshot()).toMatchObject({ fileName: 'original.json', dirty: false });
    await s.files.save();
    expect(s.host.pickSave).not.toHaveBeenCalled();
    expect(JSON.parse(source.disk.text).graph.nodes[0].type).toBe('ForwardInput');
    expect(s.files.getSnapshot().dirty).toBe(false);
  });
  it('打开后多次保存直接覆盖源文件，不调用另存为选择器', async () => {
    const s = await setup();
    await s.files.open();
    expect(s.files.getSnapshot()).toMatchObject({ fileName: 'source.json', dirty: false });
    s.rename('修改一');
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('修改一');
    s.rename('修改二');
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('修改二');
    expect(s.source.stream.close).toHaveBeenCalledTimes(2);
    expect(s.host.pickSave).not.toHaveBeenCalled();
    expect(s.host.download).not.toHaveBeenCalled();
    expect(s.files.getSnapshot().dirty).toBe(false);
  });
  it('新项目第一次选择位置，后续保存复用同一个目标', async () => {
    const s = await setup();
    await s.files.save();
    s.rename('首次保存后的修改');
    await s.files.save();
    expect(s.host.pickSave).toHaveBeenCalledTimes(1);
    expect(JSON.parse(s.target.disk.text).name).toBe('首次保存后的修改');
  });
  it('另存为成功后后续保存使用新文件，原文件不受影响', async () => {
    const s = await setup();
    await s.files.open();
    const original = s.source.disk.text;
    s.rename('另存副本');
    await s.files.save(true);
    s.rename('继续修改副本');
    await s.files.save();
    expect(s.host.pickSave).toHaveBeenCalledTimes(1);
    expect(s.source.disk.text).toBe(original);
    expect(JSON.parse(s.target.disk.text).name).toBe('继续修改副本');
    expect(s.files.getSnapshot().fileName).toBe('new.json');
  });
  it('取消另存为保留原文件关联，不报告保存成功', async () => {
    const s = await setup({
      pickSave: vi.fn(async () => {
        throw { name: 'AbortError' };
      }),
    });
    await s.files.open();
    s.rename('待保存');
    vi.mocked(s.host.toast).mockClear();
    await s.files.save(true);
    expect(s.host.toast).not.toHaveBeenCalled();
    expect(s.files.getSnapshot()).toMatchObject({
      fileName: 'source.json',
      dirty: true,
      busy: false,
    });
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('待保存');
  });
  it('拒绝权限不会下载副本或弹出另存为，重试仍写原文件', async () => {
    const s = await setup();
    await s.files.open();
    s.rename('等待授权');
    s.source.handle.requestPermission.mockResolvedValueOnce('denied');
    await s.files.save();
    expect(s.host.toast).toHaveBeenLastCalledWith(expect.stringContaining('写入权限'));
    expect(s.source.handle.createWritable).not.toHaveBeenCalled();
    expect(s.host.pickSave).not.toHaveBeenCalled();
    expect(s.host.download).not.toHaveBeenCalled();
    expect(s.files.getSnapshot().dirty).toBe(true);
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('等待授权');
  });
  it.each(['write', 'close'] as const)('%s 失败时中止写入且不切换保存目标', async (method) => {
    const s = await setup();
    await s.files.open();
    s.rename('未完成的另存为');
    s.target.stream[method].mockRejectedValueOnce(new Error('磁盘写入失败'));
    await s.files.save(true);
    expect(s.target.stream.abort).toHaveBeenCalledTimes(1);
    expect(s.files.getSnapshot()).toMatchObject({ fileName: 'source.json', dirty: true });
    expect(s.host.toast).toHaveBeenLastCalledWith(expect.stringContaining('保存失败'));
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('未完成的另存为');
  });
  it('选择器未关闭时重复保存不重复打开选择器', async () => {
    const pick = deferred<ReturnType<typeof file>['handle']>();
    const s = await setup({ pickSave: vi.fn(() => pick.promise) });
    const first = s.files.save();
    await s.files.save();
    await s.files.save(true);
    expect(s.host.pickSave).toHaveBeenCalledTimes(1);
    pick.resolve(s.target.handle);
    await first;
    expect(s.target.stream.close).toHaveBeenCalledTimes(1);
  });
  it('选择位置期间切换项目则取消写入', async () => {
    const pick = deferred<ReturnType<typeof file>['handle']>();
    const s = await setup({ pickSave: () => pick.promise });
    const saving = s.files.save();
    s.switchDocument();
    pick.resolve(s.target.handle);
    await saving;
    expect(s.target.handle.createWritable).not.toHaveBeenCalled();
    expect(s.files.getSnapshot().fileName).toBe('');
  });
  it('写入期间的后续编辑仍标记为待保存', async () => {
    const s = await setup();
    await s.files.open();
    s.rename('本次快照');
    const write = deferred<void>();
    const started = deferred<void>();
    s.source.stream.write.mockImplementationOnce(async () => {
      started.resolve();
      await write.promise;
    });
    const saving = s.files.save();
    await started.promise;
    s.rename('写入期间的更新');
    write.resolve();
    await saving;
    expect(s.files.getSnapshot().dirty).toBe(true);
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('写入期间的更新');
    expect(s.files.getSnapshot().dirty).toBe(false);
  });
  it('新建项目解除目标，撤销可恢复原项目及其目标', async () => {
    const s = await setup();
    await s.files.open();
    const opened = s.host.read();
    s.switchDocument();
    expect(s.files.getSnapshot().fileName).toBe('');
    await s.files.save();
    expect(s.files.getSnapshot().fileName).toBe('new.json');
    s.restoreDocument(opened);
    s.rename('撤销返回原项目');
    await s.files.save();
    expect(JSON.parse(s.source.disk.text).name).toBe('撤销返回原项目');
    expect(s.host.pickSave).toHaveBeenCalledTimes(1);
  });
  it('打开非法文件不替换当前项目或文件关联', async () => {
    const s = await setup();
    await s.files.open();
    const before = s.host.read();
    s.source.disk.text = '{"schemaVersion":99}';
    await s.files.open();
    expect(s.host.read()).toBe(before);
    expect(s.files.getSnapshot().fileName).toBe('source.json');
    expect(s.host.toast).toHaveBeenLastCalledWith(expect.stringContaining('打开失败'));
  });

  it('不支持源文件写入的浏览器仅在显式另存为时下载副本', async () => {
    const s = await setup({ pickOpen: undefined, pickSave: undefined });
    await s.files.save();
    expect(s.host.download).not.toHaveBeenCalled();
    expect(s.host.toast).toHaveBeenLastCalledWith(expect.stringContaining('不支持保存回源文件'));
    await s.files.save(true);
    expect(s.host.download).toHaveBeenCalledTimes(1);
    expect(s.files.getSnapshot().fileName).toBe('');
  });
});
