import { contextBridge, ipcRenderer } from 'electron';
import { CHANNEL, type DesktopApi } from '../shared/protocol';
import type { EditorAction } from '@tensor-trail/platform-api';

// No ipcRenderer, channel selection, filesystem paths, or process APIs cross this boundary.
const api: DesktopApi = {
  preferences: () => ipcRenderer.invoke(CHANNEL.preferences),
  setTheme: (theme) => ipcRenderer.invoke(CHANNEL.theme, theme),
  documentChanged: (state) => ipcRenderer.invoke(CHANNEL.state, state),
  pickOpen: (state) => ipcRenderer.invoke(CHANNEL.open, state),
  pickSave: (name, state) => ipcRenderer.invoke(CHANNEL.saveAs, name, state),
  readFile: (key) => ipcRenderer.invoke(CHANNEL.read, key),
  writeFile: (key, text, state) => ipcRenderer.invoke(CHANNEL.write, key, text, state),
  confirmDiscard: () => ipcRenderer.invoke(CHANNEL.discard),
  closeResponse: (requestId, state, approved) =>
    ipcRenderer.invoke(CHANNEL.close, requestId, state, approved),
  onAction: (listener) => {
    const handler = (_event: Electron.IpcRendererEvent, action: EditorAction) => listener(action);
    ipcRenderer.on(CHANNEL.action, handler);
    return () => {
      ipcRenderer.removeListener(CHANNEL.action, handler);
    };
  },
};
contextBridge.exposeInMainWorld('tensorTrailDesktop', api);
