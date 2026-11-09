import type { EditorPlatform, ProjectFile } from '@tensor-trail/platform-api';

export interface BrowserFileHandle {
  name: string;
  getFile(): Promise<{ size: number; text(): Promise<string> }>;
  requestPermission?: (options: { mode: 'readwrite' }) => Promise<string>;
  createWritable(): Promise<{
    write(text: string): Promise<void>;
    close(): Promise<void>;
    abort(): Promise<void>;
  }>;
}
interface PickerWindow {
  showOpenFilePicker?: (options: object) => Promise<BrowserFileHandle[]>;
  showSaveFilePicker?: (options: object) => Promise<BrowserFileHandle>;
}

/** Browser transaction details are deliberately kept outside the shared editor. */
export function browserProjectFile(handle: BrowserFileHandle): ProjectFile {
  return {
    name: handle.name,
    async readText() {
      const file = await handle.getFile();
      if (file.size > 3_000_000) throw new Error('文件超过 3 MB。');
      return file.text();
    },
    async writeText(text, stillCurrent) {
      if (handle.requestPermission &&
          await handle.requestPermission({ mode: 'readwrite' }) !== 'granted')
        throw new Error('未获得文件写入权限，请允许访问后重试。');
      if (!stillCurrent()) return false;
      const writable = await handle.createWritable();
      try {
        if (!stillCurrent()) { await writable.abort(); return false; }
        await writable.write(text);
        await writable.close();
        return true;
      } catch (error) {
        await writable.abort().catch(() => {});
        throw error;
      }
    },
  };
}

export function createBrowserPlatform(): EditorPlatform {
  const picker = window as Window & PickerWindow;
  const types = [{ description: 'Tensor Trail 项目', accept: { 'application/json': ['.json'] } }];
  return {
    kind: 'web',
    files: {
      pickOpen: picker.showOpenFilePicker ? async () => {
        const [handle] = await picker.showOpenFilePicker!({ types, multiple: false, id: 'tensor-trail' });
        return handle && browserProjectFile(handle);
      } : undefined,
      pickSave: picker.showSaveFilePicker ? async suggestedName =>
        browserProjectFile(await picker.showSaveFilePicker!({ types, suggestedName, id: 'tensor-trail' }))
        : undefined,
      download(text, name) {
        const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = name;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      },
    },
    preferences: {
      getTheme() {
        try { return localStorage.getItem('tensor-trail-theme') === 'light' ? 'light' : 'dark'; }
        catch { return 'dark'; }
      },
      setTheme(theme) {
        try { localStorage.setItem('tensor-trail-theme', theme); } catch { /* Session theme still works. */ }
      },
    },
  };
}
