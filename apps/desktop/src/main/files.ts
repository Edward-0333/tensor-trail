import * as fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import type { DocumentState, DocumentVersion } from '@tensor-trail/platform-api';
import { parseProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules';
import type { FileToken } from '../shared/protocol';
import { parseRuntimeSnapshot, safeSnapshotPath } from '@tensor-trail/model-core/core/runtimeSnapshot';

const MAX = 3_000_000;
export class FileFault extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}
export function checkProject(text: string) {
  if (typeof text !== 'string' || Buffer.byteLength(text, 'utf8') > MAX)
    throw new FileFault('FILE_SIZE', '项目文件超过 3 MB。');
  const clean = text.replace(/^\uFEFF/, '');
  try {
    parseProject(clean, registry);
  } catch (error) {
    throw new FileFault(
      'INVALID_PROJECT',
      `项目格式无效：${(error as Error).message.slice(0, 400)}`,
    );
  }
  return clean;
}
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

export async function readLimited(file: string) {
  const info = await fs.lstat(file);
  if (!info.isFile() || info.isSymbolicLink())
    throw new FileFault('FILE_TYPE', '请选择实际的 JSON 项目文件。');
  if (info.size > MAX) throw new FileFault('FILE_SIZE', '项目文件超过 3 MB。');
  const handle = await fs.open(file, 'r');
  try {
    const chunks: Buffer[] = [];
    let total = 0;
    while (true) {
      const buffer = Buffer.alloc(Math.min(65536, MAX + 1 - total));
      const { bytesRead } = await handle.read(buffer);
      if (!bytesRead) break;
      total += bytesRead;
      if (total > MAX) throw new FileFault('FILE_SIZE', '项目文件超过 3 MB。');
      chunks.push(buffer.subarray(0, bytesRead));
    }
    return Buffer.concat(chunks);
  } finally {
    await handle.close();
  }
}
async function fingerprint(file: string): Promise<string | null> {
  try {
    return hash(await readLimited(file));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw error;
  }
}
function decode(bytes: Buffer) {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    throw new FileFault('FILE_ENCODING', '项目文件不是有效的 UTF-8 文本。');
  }
}

/** Replaces only after the adjacent temporary file is flushed and the source is rechecked. */
export async function replaceFile(
  file: string,
  text: string,
  expected: string | null,
  rename: typeof fs.rename = fs.rename,
) {
  if ((await fingerprint(file)) !== expected)
    throw new FileFault(
      'FILE_CHANGED',
      '文件已被其他程序修改或删除，请重新打开，或另存为以保留当前编辑。',
    );
  if (expected !== null && !((await fs.stat(file)).mode & 0o200))
    throw new FileFault('FILE_READONLY', '文件为只读，请解除只读或另存为。');
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${randomUUID()}.tmp`);
  let created = false;
  try {
    const handle = await fs.open(temporary, 'wx', 0o600);
    created = true;
    try {
      await handle.writeFile(text, 'utf8');
      await handle.sync();
    } finally {
      await handle.close();
    }
    if ((await fingerprint(file)) !== expected)
      throw new FileFault('FILE_CHANGED', '保存期间源文件发生变化，请重新打开或另存为。');
    // Never delete the original as a fallback when Windows denies replacement.
    await rename(temporary, file);
    created = false;
  } finally {
    if (created) await fs.unlink(temporary).catch(() => {});
  }
}

interface FileRecord {
  path: string;
  fingerprint: string | null;
  owner?: string;
  openedFrom?: string;
  adoption?: string;
  validated: boolean;
}

/** Paths and versions live exclusively in the owning window's main-process session. */
export class DocumentFiles {
  state?: DocumentState;
  private records = new Map<string, FileRecord>();
  private writing = false;

  current(version: DocumentVersion) {
    if (
      !this.state ||
      version.documentId !== this.state.documentId ||
      version.revision > this.state.revision
    )
      throw new FileFault('DOCUMENT_CHANGED', '当前项目已切换，请重新操作。');
  }
  update(state: DocumentState) {
    if (this.state && this.state.documentId !== state.documentId)
      for (const file of this.records.values())
        if (!file.owner && !file.adoption && file.openedFrom === this.state.documentId)
          file.adoption = state.documentId;
    if (state.fileKey) {
      const file = this.record(state.fileKey);
      if (file.owner ? file.owner !== state.documentId : file.adoption !== state.documentId)
        throw new FileFault('DOCUMENT_OWNER', '文件不属于当前项目会话。');
      file.owner = state.documentId;
    }
    this.state = state;
  }
  private record(key: string) {
    const file = this.records.get(key);
    if (!file) throw new FileFault('FILE_TOKEN', '文件关联已失效，请重新打开。');
    return file;
  }
  async issue(
    filePath: string,
    mode: 'open' | 'save',
    version: DocumentVersion,
  ): Promise<FileToken> {
    this.current(version);
    const selected = path.resolve(filePath);
    const bytes = await readLimited(selected).catch((error) => {
      if (mode === 'save' && (error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw error;
    });
    if (mode === 'open' && bytes) checkProject(decode(bytes));
    this.current(version);
    const key = randomUUID();
    this.records.set(key, {
      path: selected,
      fingerprint: bytes ? hash(bytes) : null,
      validated: mode === 'open',
      ...(mode === 'open' ? { openedFrom: version.documentId } : { owner: version.documentId }),
    });
    return { key, name: path.basename(selected) };
  }
  async read(key: string) {
    const file = this.record(key);
    const bytes = await readLimited(file.path);
    if (hash(bytes) !== file.fingerprint)
      throw new FileFault('FILE_CHANGED', '选择后文件已发生变化，请重新打开。');
    const text = decode(bytes);
    if (!file.validated) {
      checkProject(text);
      file.validated = true;
    }
    return text.replace(/^\uFEFF/, '');
  }
  async readRelated(key: string, relativePath: string) {
    const file = this.record(key);
    if (!this.state || file.owner !== this.state.documentId) throw new FileFault('DOCUMENT_OWNER','快照不属于当前项目。');
    if (!safeSnapshotPath(relativePath)) throw new FileFault('SNAPSHOT_PATH','快照路径必须是项目目录内的相对 JSON 路径。');
    const project = parseProject(await this.read(key), registry);
    const reference = project.runtime?.snapshots.find(ref=>ref.file===relativePath);
    if (!reference) throw new FileFault('SNAPSHOT_REFERENCE','项目文件没有声明这个快照，请先保存引用。');
    const base = await fs.realpath(path.dirname(file.path));
    let resolved = base;
    for (const component of relativePath.split('/')) {
      resolved = path.join(resolved,component);
      if ((await fs.lstat(resolved)).isSymbolicLink()) throw new FileFault('SNAPSHOT_PATH','快照路径不能包含符号链接。');
    }
    const real = await fs.realpath(resolved);
    if (!real.startsWith(base + path.sep)) throw new FileFault('SNAPSHOT_PATH','快照超出项目目录。');
    const text = decode(await readLimited(resolved));
    if (parseRuntimeSnapshot(text).id !== reference.id) throw new FileFault('SNAPSHOT_ID','快照 ID 与项目引用不匹配。');
    if (file.owner !== this.state?.documentId) throw new FileFault('DOCUMENT_CHANGED','读取期间项目已切换。');
    return text;
  }
  async write(key: string, text: string, version: DocumentVersion) {
    this.current(version);
    const file = this.record(key);
    if (file.owner !== version.documentId)
      throw new FileFault('DOCUMENT_OWNER', '不能保存到其他项目的文件。');
    if (this.writing) throw new FileFault('FILE_BUSY', '已有保存操作正在进行。');
    const checked = checkProject(text);
    this.writing = true;
    try {
      await replaceFile(file.path, checked, file.fingerprint);
      file.fingerprint = hash(Buffer.from(checked, 'utf8'));
      file.validated = true;
    } finally {
      this.writing = false;
    }
  }
}
