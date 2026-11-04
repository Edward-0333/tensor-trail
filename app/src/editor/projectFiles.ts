import type { Project } from '../core/types';
import { parseProject, serializeProject } from '../core/serialization';
import { registry } from '../modules';

export interface ProjectFile {
  name: string;
  getFile: () => Promise<{ size: number; text: () => Promise<string> }>;
  requestPermission?: (options: { mode: 'readwrite' }) => Promise<string>;
  createWritable: () => Promise<{
    write: (text: string) => Promise<void>;
    close: () => Promise<void>;
    abort: () => Promise<void>;
  }>;
}
interface DocumentSnapshot {
  project: Project;
  documentId: string;
  hasProject: boolean;
}
export interface ProjectFileHost {
  read: () => DocumentSnapshot;
  load: (project: Project) => void;
  toast: (message: string) => void;
  pickOpen?: () => Promise<ProjectFile>;
  pickSave?: (name: string) => Promise<ProjectFile>;
  download: (text: string, name: string) => void;
}
export function projectFileName(project: Project) {
  return `${project.name.replace(/[<>:"/\\|?*]/g, '_')}.tensor-trail.json`;
}

/** Owns file handles outside Project JSON and keeps save targets separate across undo/new/open. */
export class ProjectFiles {
  readonly ready = Promise.resolve();
  private associations = new Map<string, { handle: ProjectFile; savedText: string }>();
  private listeners = new Set<() => void>();
  private view = { fileName: '', dirty: true, busy: false };

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
      dirty:
        doc.hasProject !== false &&
        (!association || association.savedText !== serializeProject(doc.project)),
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
    if (!this.host.pickOpen || !this.begin()) return;
    const { documentId } = this.host.read();
    try {
      const handle = await this.host.pickOpen();
      const file = await handle.getFile();
      if (file.size > 3_000_000) throw new Error('文件超过 3 MB。');
      const project = parseProject(await file.text(), registry);
      if (!this.sameDocument(documentId)) return;
      this.host.load(project);
      this.associations.set(this.host.read().documentId, {
        handle,
        savedText: serializeProject(project),
      });
      this.host.toast(`已打开「${handle.name}」，Ctrl+S 将保存回此文件。`);
    } catch (error) {
      this.report(error, '打开');
    } finally {
      this.publish({ busy: false });
    }
  };

  save = async (saveAs = false) => {
    if (!this.begin()) return;
    const doc = this.host.read(),
      text = serializeProject(doc.project);
    const previous = this.associations.get(doc.documentId);
    let writable: Awaited<ReturnType<ProjectFile['createWritable']>> | undefined;
    try {
      if (doc.hasProject === false) {
        this.host.toast('请先打开项目或新建项目。');
        return;
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
          return;
        }
        // Open the picker directly in the user gesture, before any asynchronous work.
        handle = await this.host.pickSave(previous?.handle.name ?? projectFileName(doc.project));
      }
      if (!this.sameDocument(doc.documentId)) return;
      if (
        handle.requestPermission &&
        (await handle.requestPermission({ mode: 'readwrite' })) !== 'granted'
      )
        throw new Error('未获得文件写入权限，请允许访问后重试。');
      if (!this.sameDocument(doc.documentId)) return;
      writable = await handle.createWritable();
      if (!this.sameDocument(doc.documentId)) {
        await writable.abort();
        writable = undefined;
        return;
      }
      await writable.write(text);
      await writable.close();
      writable = undefined;
      // Only close() commits the file. Keep the previous target on cancellation/failure.
      this.associations.set(doc.documentId, { handle, savedText: text });
      this.host.toast(`已保存到「${handle.name}」。`);
    } catch (error) {
      if (writable) await writable.abort().catch(() => {});
      this.report(error, '保存');
    } finally {
      this.publish({ busy: false });
    }
  };
}
