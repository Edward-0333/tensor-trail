export const MAX_PROJECT_BYTES = 10_000_000;
export const MAX_EXPANDED_NODES = 6_000;
export const MAX_EDGES_PER_GRAPH = 24_000;
export const MAX_NODE_VISITS = 60_000;

const encoder = new TextEncoder();
export function utf8Bytes(text: string): number {
  return /[^\x00-\x7f]/.test(text) ? encoder.encode(text).byteLength : text.length;
}
export function fitsProjectFile(text: string): boolean {
  // A UTF-16 code unit takes at most three bytes in UTF-8.
  return text.length <= MAX_PROJECT_BYTES / 3 ||
    (text.length <= MAX_PROJECT_BYTES && utf8Bytes(text) <= MAX_PROJECT_BYTES);
}

export function shortText(value: unknown, limit: number): value is string {
  return (
    typeof value === 'string' &&
    (value.length <= limit || (value.length <= limit * 2 && Array.from(value).length <= limit))
  );
}
