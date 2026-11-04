import { ProjectFiles, type ProjectFile } from './projectFiles';
import { useEditor } from './store';

interface FilePickerWindow {
  showOpenFilePicker?: (options: object) => Promise<ProjectFile[]>;
  showSaveFilePicker?: (options: object) => Promise<ProjectFile>;
}
const pickerWindow = window as Window & FilePickerWindow;
const types = [{ description: 'Tensor Trail 项目', accept: { 'application/json': ['.json'] } }];
export const canOpenSourceFile = typeof pickerWindow.showOpenFilePicker === 'function';
export const projectFiles = new ProjectFiles({
  read: () => useEditor.getState(),
  load: (project) => useEditor.getState().load(project),
  toast: (message) => useEditor.getState().toast(message),
  pickOpen: canOpenSourceFile
    ? async () => {
        const [handle] = await pickerWindow.showOpenFilePicker!({
          types,
          multiple: false,
          id: 'tensor-trail',
        });
        return handle;
      }
    : undefined,
  pickSave:
    typeof pickerWindow.showSaveFilePicker === 'function'
      ? (suggestedName) =>
          pickerWindow.showSaveFilePicker!({ types, suggestedName, id: 'tensor-trail' })
      : undefined,
  download: (text, name) => {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
});
const unsubscribe = useEditor.subscribe((state, previous) => {
  if (state.project !== previous.project || state.documentId !== previous.documentId)
    projectFiles.changed();
});
if (import.meta.hot) import.meta.hot.dispose(unsubscribe);
