import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { DocumentFiles, readLimited, replaceFile } from '../apps/desktop/src/main/files';
import { documentState, version } from '../apps/desktop/src/main/validation';
import { serializeProject } from '@tensor-trail/model-core/core/serialization';
import { example } from '@tensor-trail/model-core/modules/examples';
import type { DocumentState } from '@tensor-trail/platform-api';

let directory: string;
const state = (documentId = 'document-a', revision = 0): DocumentState => ({
  documentId,
  revision,
  projectName: '测试',
  hasProject: true,
  dirty: true,
  busy: false,
  fileName: '',
});
const project = (name: string) => serializeProject({ ...example('mixed'), name });
beforeEach(async () => {
  directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tt-desktop-files-'));
});
afterEach(async () => {
  const target = path.resolve(directory);
  if (
    !target.startsWith(path.resolve(os.tmpdir()) + path.sep) ||
    !path.basename(target).startsWith('tt-desktop-files-')
  )
    throw new Error('Unexpected fixture cleanup path');
  await fs.rm(target, { recursive: true, force: true });
});

describe('桌面主进程文件会话', () => {
  it('only reads declared, project-local snapshots for the owning document', async()=>{
    const p=example('mixed');
    p.runtime={active:'r',snapshots:[{id:'r',name:'run',file:'runs/r.json'}]};
    const source=path.join(directory,'project.json');
    await fs.writeFile(source,serializeProject(p));
    await fs.mkdir(path.join(directory,'runs'));
    const snapshot={format:'tensor-trail.runtime-snapshot',version:1,id:'r',name:'run',capturedAt:'2026-11-10T00:00:00Z',context:{},bindings:[{nodePath:['input'],nodeType:'Input',value:{shape:[2],dtype:'float32'}}]};
    await fs.writeFile(path.join(directory,'runs/r.json'),JSON.stringify(snapshot));
    const files=new DocumentFiles();files.update(state());
    const token=await files.issue(source,'open',state());
    await expect(files.readRelated(token.key,'runs/r.json')).rejects.toMatchObject({code:'DOCUMENT_OWNER'});
    files.update(state('opened',1));files.update({...state('opened',1),fileKey:token.key});
    expect(JSON.parse(await files.readRelated(token.key,'runs/r.json')).id).toBe('r');
    await expect(files.readRelated(token.key,'../outside.json')).rejects.toMatchObject({code:'SNAPSHOT_PATH'});
    await expect(files.readRelated(token.key,'other.json')).rejects.toMatchObject({code:'SNAPSHOT_REFERENCE'});
    await fs.unlink(path.join(directory,'runs/r.json'));
    await fs.symlink(source,path.join(directory,'runs/r.json'));
    await expect(files.readRelated(token.key,'runs/r.json')).rejects.toMatchObject({code:'SNAPSHOT_PATH'});
    files.update(state('other',2));
    await expect(files.readRelated(token.key,'runs/r.json')).rejects.toMatchObject({code:'DOCUMENT_OWNER'});
  });
  it('已验证的打开文件若内容改变，不能复用旧的格式校验结论', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const source = path.join(directory, 'validated.json');
    await fs.writeFile(source, project('有效项目'));
    const token = await files.issue(source, 'open', state());
    await fs.writeFile(source, '{"schemaVersion":999}');
    await expect(files.read(token.key)).rejects.toMatchObject({ code: 'FILE_CHANGED' });
  });
  it('另存为选择的现有文件，未校验前不能当成项目读取；提交后才有效', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const target = path.join(directory, 'save-target.json');
    await fs.writeFile(target, '{"schemaVersion":999}');
    const token = await files.issue(target, 'save', state());
    await expect(files.read(token.key)).rejects.toMatchObject({ code: 'INVALID_PROJECT' });
    await files.write(token.key, project('合法替换'), state());
    expect(JSON.parse(await files.read(token.key)).name).toBe('合法替换');
  });
  it('新会话为空，打开中文空格路径与 BOM 后才绑定并直接保存原文件', async () => {
    const files = new DocumentFiles();
    expect(files.state).toBeUndefined();
    files.update(state());
    const source = path.join(directory, '中文 项目.json');
    await fs.writeFile(source, '\uFEFF' + project('原项目'));
    const token = await files.issue(source, 'open', state());
    expect(token).not.toHaveProperty('path');
    expect((await files.read(token.key)).startsWith('{')).toBe(true);
    files.update(state('opened', 1));
    files.update({ ...state('opened', 1), fileKey: token.key, fileName: token.name });
    await files.write(token.key, project('直接保存'), state('opened', 1));
    expect(JSON.parse(await fs.readFile(source, 'utf8')).name).toBe('直接保存');
  });
  it('新建、跨文档写入被隔离，撤销恢复原文档后可继续保存', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const target = path.join(directory, 'first.json');
    const token = await files.issue(target, 'save', state());
    await files.write(token.key, project('第一次'), state());
    files.update(state('new-document', 2));
    await expect(
      files.write(token.key, project('错误项目'), state('new-document', 2)),
    ).rejects.toMatchObject({ code: 'DOCUMENT_OWNER' });
    files.update({ ...state('document-a', 3), fileKey: token.key });
    await files.write(token.key, project('恢复原项目'), state('document-a', 3));
    expect(JSON.parse(await fs.readFile(target, 'utf8')).name).toBe('恢复原项目');
  });
  it('另存为选择目标不写文件，成功写入后才关联，原文件保持原样', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const old = path.join(directory, 'old.json');
    const other = path.join(directory, 'other.json');
    const first = await files.issue(old, 'save', state());
    await files.write(first.key, project('原件'), state());
    const second = await files.issue(other, 'save', state());
    await expect(fs.stat(other)).rejects.toMatchObject({ code: 'ENOENT' });
    await files.write(second.key, project('副本'), state());
    expect(JSON.parse(await fs.readFile(old, 'utf8')).name).toBe('原件');
    expect(JSON.parse(await fs.readFile(other, 'utf8')).name).toBe('副本');
  });
  it('外部程序改动后拒绝覆盖，保留外部文件及当前编辑', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const target = path.join(directory, 'external.json');
    const token = await files.issue(target, 'save', state());
    await files.write(token.key, project('原件'), state());
    const external = project('其他程序的更新');
    await fs.writeFile(target, external);
    await expect(files.write(token.key, project('画布修改'), state())).rejects.toMatchObject({
      code: 'FILE_CHANGED',
    });
    expect(await fs.readFile(target, 'utf8')).toBe(external);
  });
  it('Windows 替换失败时原文件完整，临时文件清理，重试可成功', async () => {
    const target = path.join(directory, 'locked.json');
    const original = project('磁盘原件');
    await fs.writeFile(target, original);
    const hash = createHash('sha256').update(original).digest('hex');
    await expect(
      replaceFile(target, project('修改'), hash, async () => {
        throw Object.assign(new Error('fixture lock'), { code: 'EPERM' });
      }),
    ).rejects.toMatchObject({ code: 'EPERM' });
    expect(await fs.readFile(target, 'utf8')).toBe(original);
    expect(await fs.readdir(directory)).toEqual(['locked.json']);
    await replaceFile(target, project('重试保存'), hash);
    expect(JSON.parse(await fs.readFile(target, 'utf8')).name).toBe('重试保存');
  });
  it('只读文件不通过删除原件回退，解除只读后可保存', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const target = path.join(directory, 'readonly.json');
    await fs.writeFile(target, project('只读原件'));
    const token = await files.issue(target, 'save', state());
    await fs.chmod(target, 0o444);
    try {
      await expect(files.write(token.key, project('修改'), state())).rejects.toMatchObject({
        code: 'FILE_READONLY',
      });
    } finally {
      await fs.chmod(target, 0o666);
    }
    expect(JSON.parse(await fs.readFile(target, 'utf8')).name).toBe('只读原件');
    await files.write(token.key, project('允许保存'), state());
  });
  it('无效、超限 JSON 不建立可读文件关联', async () => {
    const files = new DocumentFiles();
    files.update(state());
    const target = path.join(directory, 'invalid.json');
    await fs.writeFile(target, '{"schemaVersion":999}');
    await expect(files.issue(target, 'open', state())).rejects.toMatchObject({
      code: 'INVALID_PROJECT',
    });
    await fs.writeFile(target, 'x'.repeat(3_000_001));
    await expect(readLimited(target)).rejects.toMatchObject({ code: 'FILE_SIZE' });
    await expect(files.read('unissued-key')).rejects.toMatchObject({ code: 'FILE_TOKEN' });
    expect(files.state?.documentId).toBe('document-a');
  });
  it('IPC 状态和版本字段必须通过类型及长度校验', () => {
    for (const input of [
      null,
      {},
      { documentId: 'a', revision: -1 },
      { documentId: 'a', revision: 1.5 },
    ])
      expect(() => version(input)).toThrow();
    expect(() => documentState({ ...state(), busy: 'false' })).toThrow();
    expect(documentState(state())).toEqual(state());
  });
});
