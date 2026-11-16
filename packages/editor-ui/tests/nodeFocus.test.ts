import { beforeEach, describe, expect, it } from 'vitest';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { serializeProject } from '@tensor-trail/model-core/core/serialization';
import { example } from '@tensor-trail/model-core/modules/examples';
import { useEditor } from '../src/editor/store';

const state = () => useEditor.getState();
beforeEach(() => state().load(example()));

describe('Explicit node focus', () => {
  it('selects only the requested node without editing the model or history', () => {
    state().duplicateLinked('input');
    const alias = state().selected!;
    state().createLayoutRegion([alias]);
    state().selectMany(['input', alias]);
    const before = state(), saved = serializeProject(before.project);
    state().focusNode(alias);
    expect(state().selected).toBe(alias);
    expect(state().selectedIds).toEqual([alias]);
    expect(state().nodeFocus).toEqual({ nodeId: alias, path: [], documentId: before.documentId });
    expect(state().project).toBe(before.project);
    expect(serializeProject(state().project)).toBe(saved);
    expect(state().past).toBe(before.past);
    expect(state().future).toBe(before.future);
    expect(state().canvasVersion).toBe(before.canvasVersion);
  });

  it('ordinary selection does not request a viewport move', () => {
    state().select('input');
    expect(state().nodeFocus).toBeUndefined();
    state().focusNode('input');
    state().select('output');
    expect(state().nodeFocus).toBeUndefined();
  });

  it('can refocus the same node and does not consume a newer request', () => {
    state().focusNode('input');
    const first = state().nodeFocus!;
    state().focusNode('input');
    const second = state().nodeFocus!;
    expect(second).not.toBe(first);
    state().finishNodeFocus(first);
    expect(state().nodeFocus).toBe(second);
    state().finishNodeFocus(second);
    expect(state().nodeFocus).toBeUndefined();
  });

  it('resolves targets in the current nested graph rather than the root', () => {
    const path = ['transformer', 'attention'];
    state().navigate(path);
    const { graph } = resolveForwardView(state().project.graph, path);
    const target = graph.nodes[0].id;
    state().focusNode(target);
    expect(state().nodeFocus).toEqual({ nodeId: target, path, documentId: state().documentId });
    expect(state().path).toEqual(path);
    state().navigate([]);
    expect(state().nodeFocus).toBeUndefined();
  });

  it('ignores a stale target without clearing the current selection', () => {
    state().focusNode('input');
    const before = state();
    state().focusNode('deleted-node');
    expect(state()).toBe(before);
  });

  it.each(['load', 'undo', 'redo', 'newDocument'] as const)('%s clears pending focus', action => {
    state().renameNode('input', 'renamed');
    if (action === 'redo') state().undo();
    state().focusNode('input');
    if (action === 'load') state().load(example());
    if (action === 'undo') state().undo();
    if (action === 'redo') state().redo();
    if (action === 'newDocument') state().startNewDocument();
    expect(state().nodeFocus).toBeUndefined();
  });
});
