import type { Project } from '@tensor-trail/model-core/core/types';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules/index';
import type { RuntimeReference, RuntimeSnapshot } from '@tensor-trail/model-core/core/types';
import {
  parseRuntimeSnapshot,
  portableProject,
} from '@tensor-trail/model-core/core/runtimeSnapshot';

import type { ProjectFile, DiscardDecision } from '@tensor-trail/platform-api';
export type { ProjectFile } from '@tensor-trail/platform-api';

interface DocumentSnapshot {
  project: Project;
  documentId: string;
  hasProject: boolean;
  revision?: number;
}
export interface ProjectFileHost {
  read: () => DocumentSnapshot;
  load: (project: Project) => void;
  toast: (message: string) => void;
  pickOpen?: () => Promise<ProjectFile | undefined>;
  pickSave?: (name: string) => Promise<ProjectFile | undefined>;
  download: (text: string, name: string) => void;
  confirmDiscard?: () => Promise<DiscardDecision>;
}
export function projectFileName(project: Project) {
  return `${project.name.replace(/[<>:"/\\|?*]/g, '_')}.tensor-trail.json`;
}

/** Owns file handles outside Project JSON and keeps save targets separate across undo/new/open. */
export class ProjectFiles {
  readonly ready = Promise.resolve();
  private associations = new Map<string, { handle: ProjectFile; savedText: string }>();
  private listeners = new Set<() => void>();
  private view = {
    fileName: '',
    fileKey: undefined as string | undefined,
    dirty: true,
    busy: false,
  };
  private leaving = false;
  private texts = new WeakMap<Project, string>();
  private text(doc: DocumentSnapshot): string {
    // Only the editor host opts into immutable revision caching. Generic hosts may mutate.
    if (doc.revision === undefined) return serializeProject(doc.project);
    let text = this.texts.get(doc.project);
    if (text === undefined) {
      text = serializeProject(doc.project);
      this.texts.set(doc.project, text);
    }
    return text;
  }

  constructor(private host: ProjectFileHost) {
    this.publish();
  }

  getSnapshot = () => this.view;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private publish(update: Partial<typeof this.view> = {}) {
    const doc = this.host.read(),
      association = this.associations.get(doc.documentId);
    this.view = {
      ...this.view,
      fileName: association?.handle.name ?? '',
      fileKey: association?.handle.key,
      dirty: doc.hasProject !== false && (!association || association.savedText !== this.text(doc)),
      ...update,
    };
    this.listeners.forEach((listener) => listener());
  }

  changed = () => {
    this.publish();
  };

  private begin() {
    if (this.view.busy) return false;
    this.publish({ busy: true });
    return true;
  }

  private report(error: unknown, action: string) {
    if ((error as { name?: string })?.name === 'AbortError') return;
    this.host.toast(`${action}失败：${(error as Error).message || String(error)}`);
  }

  private sameDocument(documentId: string) {
    if (this.host.read().documentId === documentId) return true;
    this.host.toast('项目已切换，已取消本次文件操作，请在当前项目重新操作。');
    return false;
  }

  open = async () => {
    if (!this.host.pickOpen || this.view.busy) return;
    if (this.host.confirmDiscard && !(await this.confirmLeave())) return;
    if (!this.begin()) return;
    const before = this.host.read();
    const { documentId } = before;
    const previousText = this.host.confirmDiscard ? this.text(before) : undefined;
    try {
      const handle = await this.host.pickOpen();
      if (!handle) return;
      const project = parseProject((await handle.readText()).replace(/^\uFEFF/, ''), registry);
      if (!this.sameDocument(documentId)) return;
      if (previousText !== undefined && this.text(this.host.read()) !== previousText) {
        this.host.toast('选择文件期间当前项目发生变化，已取消打开，请重试。');
        return;
      }
      this.host.load(project);
      this.associations.set(this.host.read().documentId, {
        handle,
        savedText: this.text(this.host.read()),
      });
      this.host.toast(`已打开「${handle.name}」，Ctrl+S 将保存回此文件。`);
    } catch (error) {
      this.report(error, '打开');
    } finally {
      this.publish({ busy: false });
    }
  };

  save = async (saveAs = false): Promise<boolean> => {
    if (!this.begin()) return false;
    const doc = this.host.read(),
      text = this.text(doc);
    const previous = this.associations.get(doc.documentId);
    try {
      if (doc.hasProject === false) {
        this.host.toast('请先打开项目或新建项目。');
        return false;
      }
      if (saveAs && doc.project.runtime?.snapshots.length && !doc.project.runtime.embedded) {
        this.host.toast(
          '此项目引用外部快照；请使用“运行数据 → 导出当前运行单文件”，或整体复制项目与快照目录。',
        );
        return false;
      }
      let handle = saveAs ? undefined : previous?.handle;
      if (!handle) {
        if (!this.host.pickSave) {
          if (saveAs) {
            this.host.download(text, projectFileName(doc.project));
            this.host.toast(
              '已下载项目副本；当前浏览器不支持保存回源文件，请使用 Chrome 或 Edge。',
            );
          } else
            this.host.toast(
              '当前浏览器不支持保存回源文件，请使用 Chrome 或 Edge 打开项目，或点击“另存为”下载副本。',
            );
          return false;
        }
        // Open the picker directly in the user gesture, before any asynchronous work.
        handle = await this.host.pickSave(previous?.handle.name ?? projectFileName(doc.project));
      }
      if (!this.sameDocument(doc.documentId)) return false;
      if (!handle) return false;
      if (
        !(await handle.writeText(text, () => this.sameDocument(doc.documentId), {
          documentId: doc.documentId,
          revision: doc.revision ?? 0,
        }))
      )
        return false;
      // The adapter commits atomically; keep the previous target on cancellation/failure.
      this.associations.set(doc.documentId, { handle, savedText: text });
      this.host.toast(`已保存到「${handle.name}」。`);
      return true;
    } catch (error) {
      this.report(error, '保存');
      return false;
    } finally {
      this.publish({ busy: false });
    }
  };

  /** Native hosts guard destructive navigation; web behavior remains unchanged. */
  confirmLeave = async (): Promise<boolean> => {
    if (this.view.busy || this.leaving) return false;
    if (!this.host.confirmDiscard || !this.view.dirty) return true;
    const before = this.host.read();
    const text = this.text(before);
    this.leaving = true;
    try {
      const decision = await this.host.confirmDiscard();
      const current = this.host.read();
      if (current.documentId !== before.documentId || this.text(current) !== text) return false;
      if (decision === 'cancel') return false;
      if (decision === 'discard') return true;
      return (await this.save()) && !this.view.dirty && this.sameDocument(before.documentId);
    } catch (error) {
      this.report(error, '检查未保存修改');
      return false;
    } finally {
      this.leaving = false;
    }
  };

  async readRuntime(reference: RuntimeReference): Promise<RuntimeSnapshot> {
    const before = this.host.read();
    const handle = this.associations.get(before.documentId)?.handle;
    if (!handle?.readRelatedText)
      throw Error('当前打开方式不能自动读取旁边的文件，请手动选择所引用的快照 JSON。');
    const snapshot = parseRuntimeSnapshot(await handle.readRelatedText(reference.file));
    if (!this.sameDocument(before.documentId)) throw Error('读取期间项目已切换。');
    if (snapshot.id !== reference.id) throw Error('快照 ID 与项目引用不匹配。');
    return snapshot;
  }

  async exportPortable(snapshot?: RuntimeSnapshot): Promise<boolean> {
    if (!this.begin()) return false;
    const doc = this.host.read();
    try {
      if (!doc.hasProject) throw Error('请先打开或新建项目。');
      const text = serializeProject(portableProject(doc.project, snapshot));
      const name = projectFileName(doc.project).replace(
        '.tensor-trail.json',
        '.portable.tensor-trail.json',
      );
      if (!this.host.pickSave) {
        this.host.download(text, name);
        return true;
      }
      const handle = await this.host.pickSave(name);
      if (!handle || !this.sameDocument(doc.documentId)) return false;
      if (
        handle.name.toLowerCase() ===
        this.associations.get(doc.documentId)?.handle.name.toLowerCase()
      )
        throw Error('便携副本请使用与源项目不同的文件名。');
      const saved = await handle.writeText(text, () => this.sameDocument(doc.documentId), {
        documentId: doc.documentId,
        revision: doc.revision ?? 0,
      });
      if (saved) this.host.toast('已导出包含当前快照的单文件副本；原项目关联不变。');
      return saved;
    } catch (error) {
      this.report(error, '导出单文件');
      return false;
    } finally {
      this.publish({ busy: false });
    }
  }
}
