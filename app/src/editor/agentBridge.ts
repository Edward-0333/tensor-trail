import { createCanvasApi } from '../agent/canvas';
import { useEditor } from './store';

const bridge = createCanvasApi({
  read: () => useEditor.getState().project,
  commit: (project) => useEditor.getState().load(project),
  subscribe: (changed) =>
    useEditor.subscribe((state, previous) => {
      if (state.project !== previous.project) changed();
    }),
});
export const canvasApi = bridge.api;
declare global {
  interface Window {
    tensorTrail: typeof canvasApi;
  }
}
if (typeof window !== 'undefined')
  Object.defineProperty(window, 'tensorTrail', {
    value: canvasApi,
    configurable: true,
    writable: false,
  });
if (import.meta.hot) import.meta.hot.dispose(() => bridge.dispose());
