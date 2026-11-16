import { describe, expect, it } from 'vitest';
import { createViewportMemory } from '../src/editor/viewportMemory';
import { useEditor } from '../src/editor/store';
import { example } from '@tensor-trail/model-core/modules/examples';

describe('Session viewport memory', () => {
  it('remembers root, nested, and sibling views independently', () => {
    const memory = createViewportMemory();
    expect(memory.get('document', [])).toBeUndefined();
    const root = { x: -1200, y: 350, zoom: 1.3 };
    const child = { x: 42, y: -200, zoom: .45 };
    memory.set('document', [], root);
    memory.set('document', ['block', 'attention'], child);
    expect(memory.get('document', [])).toEqual(root);
    expect(memory.get('document', ['block', 'attention'])).toEqual(child);
    expect(memory.get('document', ['block'])).toBeUndefined();
    expect(memory.get('document', ['sibling', 'attention'])).toBeUndefined();
    expect(memory.get('other-document', [])).toBeUndefined();
  });

  it('does not confuse IDs containing path separators with nested paths', () => {
    const memory = createViewportMemory(), view = { x: 1, y: 2, zoom: .6 };
    memory.set('document', ['a/b'], view);
    expect(memory.get('document', ['a', 'b'])).toBeUndefined();
    expect(memory.get('document', ['a/b'])).toEqual(view);
  });

  it('copies viewport values and replaces the last view after pan, zoom, or fit', () => {
    const memory = createViewportMemory(), view = { x: 10, y: 20, zoom: 2 };
    memory.set('document', [], view);
    view.x = 90;
    memory.get('document', [])!.y = 90;
    expect(memory.get('document', [])).toEqual({ x: 10, y: 20, zoom: 2 });
    memory.set('document', [], view);
    expect(memory.get('document', [])).toEqual(view);
  });

  it('ignores invalid viewports without corrupting the last view', () => {
    const memory = createViewportMemory(), view = { x: 0, y: 0, zoom: 1 };
    memory.set('document', [], view);
    for (const invalid of [{ x: NaN }, { y: Infinity }, { zoom: 0 }, { zoom: -1 }])
      memory.set('document', [], { ...view, ...invalid });
    expect(memory.get('document', [])).toEqual(view);
  });

  it('survives graph edits and undo without modifying the model or history', () => {
    const memory = createViewportMemory(), state = () => useEditor.getState();
    state().load(example());
    const original = state(), view = { x: -700, y: 30, zoom: .8 };
    memory.set(original.documentId, [], view);
    expect(state()).toBe(original);
    state().renameNode('input', 'renamed');
    expect(memory.get(state().documentId, [])).toEqual(view);
    state().undo();
    expect(memory.get(state().documentId, [])).toEqual(view);
    state().load(example());
    expect(memory.get(state().documentId, [])).toBeUndefined();
    state().undo();
    expect(memory.get(state().documentId, [])).toEqual(view);
  });
});
