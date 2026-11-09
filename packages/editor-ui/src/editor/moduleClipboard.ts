import { useEditor } from './store';
import { clipboardText, readClipboard } from '@tensor-trail/model-core/editing/clipboard';

/** In-page copies work without system clipboard permissions or native paste data.
 * Native events still exchange fragments across tabs. After focus leaves this page,
 * let the next native paste inspect external content before falling back to the buffer.
 */
export function createModuleClipboard() {
  let localCopy = false;
  const paste = () => {
    const state = useEditor.getState();
    if (!state.clipboard?.nodes.length) {
      state.toast('请先选中模块并复制，再粘贴。');
      return false;
    }
    const before = state.project;
    state.paste();
    return useEditor.getState().project !== before;
  };
  return {
    copy() {
      useEditor.getState().copy();
      localCopy = true;
      return useEditor.getState().clipboard;
    },
    paste,
    pasteText(text: string) {
      // A failed empty-selection copy must never restore an older system fragment.
      if (localCopy && !useEditor.getState().clipboard?.nodes.length) return false;
      try {
        if (!text.trim()) return paste();
        const fragment = readClipboard(text);
        if (!fragment) return false;
        const state = useEditor.getState();
        if (!state.clipboard || clipboardText(state.clipboard) !== clipboardText(fragment))
          useEditor.setState({ clipboard: fragment, clipboardOrigin: null, pasteCount: 0 });
        localCopy = true;
        return paste();
      } catch (error) {
        useEditor.getState().toast(`粘贴失败：${(error as Error).message}`);
        return false;
      }
    },
    ownsShortcut: () => localCopy,
    release: () => {
      localCopy = false;
    },
    discard: () => {
      localCopy = false;
      useEditor.setState({ clipboard: null, clipboardOrigin: null, pasteCount: 0 });
    },
  };
}
