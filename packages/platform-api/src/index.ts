export type Theme = 'dark' | 'light';

/** An opaque selected file; browser handles and native paths stay in the adapter. */
export interface ProjectFile {
  readonly key?: string;
  readonly name: string;
  readText(): Promise<string>;
  /** Return false if the caller's document changed before the write could start. */
  writeText(text: string, stillCurrent: () => boolean, version?: DocumentVersion): Promise<boolean>;
}
export interface DocumentVersion {
  documentId: string;
  revision: number;
}
export interface DocumentState extends DocumentVersion {
  projectName: string;
  hasProject: boolean;
  fileKey?: string;
  fileName: string;
  dirty: boolean;
  busy: boolean;
}
export type EditorAction =
  | { type: 'new' | 'open' | 'save' | 'save-as' | 'undo' | 'redo' }
  | { type: 'close'; requestId: string };
export type DiscardDecision = 'save' | 'discard' | 'cancel';
export interface FileServices {
  pickOpen?: () => Promise<ProjectFile | undefined>;
  pickSave?: (suggestedName: string) => Promise<ProjectFile | undefined>;
  download(text: string, name: string): void;
}
export interface Preferences {
  getTheme(): Theme;
  setTheme(theme: Theme): void;
}
export interface RuntimeInfo {
  available: boolean;
  pythonVersion?: string;
  torchVersion?: string;
  cpuOnly?: boolean;
}
export interface EditorPlatform {
  readonly kind: 'web' | 'desktop';
  readonly files: FileServices;
  readonly preferences: Preferences;
  readonly runtime?: { probe(): Promise<RuntimeInfo> };
  readonly lifecycle?: {
    documentChanged(state: DocumentState): void;
    onAction(listener: (action: EditorAction) => void): () => void;
    confirmDiscard(): Promise<DiscardDecision>;
    closeResponse(requestId: string, state: DocumentState, approved: boolean): Promise<void>;
  };
}
