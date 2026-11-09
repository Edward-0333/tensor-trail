import type {
  DiscardDecision,
  DocumentState,
  DocumentVersion,
  EditorAction,
  Theme,
} from '@tensor-trail/platform-api';

export const CHANNEL = {
  preferences: 'tt:preferences',
  theme: 'tt:theme',
  state: 'tt:state',
  open: 'tt:open',
  saveAs: 'tt:save-as',
  read: 'tt:read',
  write: 'tt:write',
  discard: 'tt:discard',
  close: 'tt:close-response',
  action: 'tt:action',
} as const;
export interface FileToken {
  key: string;
  name: string;
}
export type Reply<T> =
  | { ok: true; value: T }
  | { ok: false; error: { code: string; message: string } };
export interface DesktopApi {
  preferences(): Promise<Reply<{ theme: Theme }>>;
  setTheme(theme: Theme): Promise<Reply<void>>;
  documentChanged(state: DocumentState): Promise<Reply<void>>;
  pickOpen(version: DocumentVersion): Promise<Reply<FileToken | undefined>>;
  pickSave(name: string, version: DocumentVersion): Promise<Reply<FileToken | undefined>>;
  readFile(key: string): Promise<Reply<string>>;
  writeFile(key: string, text: string, version: DocumentVersion): Promise<Reply<void>>;
  confirmDiscard(): Promise<Reply<DiscardDecision>>;
  closeResponse(requestId: string, state: DocumentState, approved: boolean): Promise<Reply<void>>;
  onAction(listener: (action: EditorAction) => void): () => void;
}
