import type { DocumentState, EditorAction, EditorPlatform } from '@tensor-trail/platform-api';
import type { Project } from '@tensor-trail/model-core/core/types';
import { ProjectFiles } from './projectFiles';
import { useEditor } from './store';

export let editorPlatform: EditorPlatform;
export let projectFiles: ProjectFiles;
export const canOpenSourceFile = () => Boolean(editorPlatform.files.pickOpen);
let revision = 0;
const documentState = (): DocumentState => {
  const state = useEditor.getState();
  return {
    documentId: state.documentId,
    revision,
    projectName: state.project.name,
    hasProject: state.hasProject,
    ...projectFiles.getSnapshot(),
  };
};

function flushPendingEdit() {
  if (typeof document !== 'undefined' && document.activeElement instanceof HTMLElement)
    document.activeElement.blur();
}
export function saveProject(saveAs = false) {
  if (typeof document !== 'undefined' && document.querySelector('dialog[open]')) {
    useEditor.getState().toast('请先完成或关闭当前对话框，再保存项目。');
    return Promise.resolve(false);
  }
  flushPendingEdit();
  return projectFiles.save(saveAs);
}
export function openSourceProject() {
  flushPendingEdit();
  return projectFiles.open();
}

export async function replaceEditorProject(project?: Project) {
  flushPendingEdit();
  if (editorPlatform.lifecycle && !(await projectFiles.confirmLeave())) return;
  project ? useEditor.getState().load(project) : useEditor.getState().newProject();
}

export async function runEditorAction(action: EditorAction) {
  if (action.type === 'open') return openSourceProject();
  if (action.type === 'new') return replaceEditorProject();
  if (action.type === 'save' || action.type === 'save-as')
    return saveProject(action.type === 'save-as');
  if (action.type === 'close') {
    flushPendingEdit();
    const approved = await projectFiles.confirmLeave();
    await editorPlatform.lifecycle?.closeResponse(action.requestId, documentState(), approved);
    return;
  }
  const active = typeof document === 'undefined' ? null : document.activeElement;
  if (
    typeof HTMLElement !== 'undefined' &&
    active instanceof HTMLElement &&
    (active.closest('input,textarea') || active.isContentEditable)
  )
    document.execCommand(action.type);
  else action.type === 'undo' ? useEditor.getState().undo() : useEditor.getState().redo();
}

/** Configure once in each renderer entry, before mounting App. */
export function configureEditor(platform: EditorPlatform) {
  editorPlatform = platform;
  revision = 0;
  projectFiles = new ProjectFiles({
    read: () => ({ ...useEditor.getState(), revision }),
    load: (project) => useEditor.getState().load(project),
    toast: (message) => useEditor.getState().toast(message),
    ...platform.files,
    confirmDiscard: platform.lifecycle?.confirmDiscard,
  });
  const report = () => platform.lifecycle?.documentChanged(documentState());
  const unsubscribeFile = projectFiles.subscribe(report);
  const unsubscribeState = useEditor.subscribe((state, previous) => {
    if (state.project !== previous.project) revision += 1;
    if (state.project !== previous.project || state.documentId !== previous.documentId)
      projectFiles.changed();
  });
  const unsubscribeActions = platform.lifecycle?.onAction((action) => {
    void runEditorAction(action).catch((error) => useEditor.getState().toast(String(error)));
  });
  report();
  return () => {
    unsubscribeState();
    unsubscribeFile();
    unsubscribeActions?.();
  };
}
