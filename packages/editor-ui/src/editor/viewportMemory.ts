import type { Viewport } from '@xyflow/react';

// Browsing state belongs to the editor session, not the project or undo history.
export function createViewportMemory() {
  const views = new Map<string, Viewport>();
  const key = (documentId: string, path: readonly string[]) => JSON.stringify([documentId, path]);
  return {
    get(documentId: string, path: readonly string[]): Viewport | undefined {
      const viewport = views.get(key(documentId, path));
      return viewport && { ...viewport };
    },
    set(documentId: string, path: readonly string[], viewport: Viewport) {
      if (![viewport.x, viewport.y, viewport.zoom].every(Number.isFinite) || viewport.zoom <= 0) return;
      views.set(key(documentId, path), { ...viewport });
    },
  };
}
