import type { DocumentState, DocumentVersion, Theme } from '@tensor-trail/platform-api';
import { FileFault } from './files';

export const invalid = () => {
  throw new FileFault('INVALID_REQUEST', '桌面请求无效。');
};
export function text(value: unknown, max = 100): string {
  if (typeof value !== 'string' || !value.length || value.length > max || value.includes('\0'))
    return invalid();
  return value;
}
export function version(value: unknown): DocumentVersion {
  if (!value || typeof value !== 'object') return invalid();
  const data = value as Record<string, unknown>;
  if (!Number.isSafeInteger(data.revision) || (data.revision as number) < 0) return invalid();
  return { documentId: text(data.documentId), revision: data.revision as number };
}
export function documentState(value: unknown): DocumentState {
  if (!value || typeof value !== 'object') return invalid();
  const data = value as Record<string, unknown>;
  if (['hasProject', 'dirty', 'busy'].some((key) => typeof data[key] !== 'boolean'))
    return invalid();
  if (typeof data.fileName !== 'string' || data.fileName.length > 512) return invalid();
  return {
    ...version(value),
    projectName: text(data.projectName, 200),
    hasProject: data.hasProject as boolean,
    dirty: data.dirty as boolean,
    busy: data.busy as boolean,
    fileName: data.fileName,
    fileKey: data.fileKey === undefined ? undefined : text(data.fileKey),
  };
}
export function theme(value: unknown): Theme {
  if (value !== 'dark' && value !== 'light') return invalid();
  return value;
}
