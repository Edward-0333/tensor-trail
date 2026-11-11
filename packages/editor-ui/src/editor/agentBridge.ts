import { createCanvasApi } from '@tensor-trail/model-core/agent/canvas';
import { useEditor } from './store';

const bridge = createCanvasApi({
  read: () => useEditor.getState().project,
  readRuntime: () => useEditor.getState().runtimeSnapshot,
  commit: (project) => {
    const previous = useEditor.getState();
    const runtime =
      JSON.stringify(previous.project.runtime) === JSON.stringify(project.runtime)
        ? previous.runtimeSnapshot
        : undefined;
    previous.load(project, true);
    if (runtime) useEditor.getState().setRuntimeSnapshot(runtime);
  },
  subscribe: (changed) =>
    useEditor.subscribe((state, previous) => {
      if (state.project !== previous.project || state.runtimeSnapshot !== previous.runtimeSnapshot)
        changed();
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
