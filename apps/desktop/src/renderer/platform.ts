import type { DesktopApi, FileToken, Reply } from '../shared/protocol';
import type { DocumentState, EditorPlatform, ProjectFile } from '@tensor-trail/platform-api';

declare global {
  interface Window {
    tensorTrailDesktop: DesktopApi;
  }
}
async function value<T>(response: Promise<Reply<T>>): Promise<T> {
  const result = await response;
  if (!result.ok) throw new Error(result.error.message);
  return result.value;
}
export async function desktopPlatform(api = window.tensorTrailDesktop): Promise<EditorPlatform> {
  if (!api) throw new Error('请通过 Tensor Trail 桌面程序打开此界面。');
  let preference = (await value(api.preferences())).theme;
  let current: DocumentState;
  const resource = (token: FileToken): ProjectFile => ({
    name: token.name,
    key: token.key,
    readText: () => value(api.readFile(token.key)),
    async writeText(text, stillCurrent, version) {
      if (!stillCurrent()) return false;
      if (!version) throw new Error('缺少文档版本，请重新打开项目。');
      await value(api.writeFile(token.key, text, version));
      return true;
    },
  });
  return {
    kind: 'desktop',
    files: {
      async pickOpen() {
        const token = await value(api.pickOpen(current));
        return token && resource(token);
      },
      async pickSave(name) {
        const token = await value(api.pickSave(name, current));
        return token && resource(token);
      },
      download() {
        throw new Error('请使用桌面的“另存为”。');
      },
    },
    preferences: {
      getTheme: () => preference,
      setTheme(theme) {
        preference = theme;
        void value(api.setTheme(theme)).catch((error) => console.error(error));
      },
    },
    lifecycle: {
      documentChanged(state) {
        current = state;
        void value(api.documentChanged(state)).catch((error) => console.error(error));
      },
      onAction: (listener) => api.onAction(listener),
      confirmDiscard: () => value(api.confirmDiscard()),
      closeResponse: (id, state, approved) => value(api.closeResponse(id, state, approved)),
    },
  };
}
