import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules';
import { useEditor } from '../src/editor/store';
import { levelInputs } from '../src/views/levelInputs';
import { Inspector } from '../src/editor/Inspector';
import { unlinkValueWithConfirmation, valueUnlinkWarning } from '../src/editor/valueLinkActions';

// Static rendering otherwise sees Zustand's initial (empty-workspace) server snapshot.
vi.mock('../src/editor/store', async importOriginal => {
  const actual = await importOriginal<typeof import('../src/editor/store')>();
  return { ...actual, useEditor: Object.assign(() => actual.useEditor.getState(), actual.useEditor) };
});

const state = () => useEditor.getState();
beforeEach(() => {
  const input = registry.create('ForwardInput', {}, 'img_metas'); input.label = 'img_metas';
  state().load({ schemaVersion: 1, name: 'Shared forward input', variants: [], graph: { nodes: [
    registry.create('Input', {}, 'img'), input, registry.create('ForwardInput', {}, 'kwargs'),
    { ...structuredClone(input), id: 'near_head', valueSource: 'img_metas' },
  ], edges: [] } });
});

describe('Region membership is not value identity', () => {
  it('preserves the three forward inputs throughout region operations and undo/redo', () => {
    const check = () => {
      expect(levelInputs(state().project.graph, true).map(n => n.id)).toEqual(['img', 'img_metas', 'kwargs']);
      expect(state().project.graph.nodes.find(n => n.id === 'near_head')!.valueSource).toBe('img_metas');
    };
    state().createLayoutRegion(['img', 'near_head']); const id = state().selected!; check();
    state().assignLayoutRegion(['img_metas'], id); check();
    state().resizeLayoutRegion(id, { x: -100, y: -100, width: 1500, height: 900 }); check();
    state().move(id, { x: 100, y: 200 }); check();
    state().assignLayoutRegion(['near_head'], null); check();
    state().undo(); check(); state().redo(); check();
    state().removeNodes([id]); check();
  });

  it('requires confirmation to turn an input alias into an independent fourth input', () => {
    const original = structuredClone(state().project), cancel = vi.fn(() => false);
    unlinkValueWithConfirmation('near_head', cancel);
    expect(cancel).toHaveBeenCalledWith(expect.stringContaining('输入数量增加 1'));
    expect(state().project).toEqual(original);
    unlinkValueWithConfirmation('near_head', () => true);
    expect(levelInputs(state().project.graph, true)).toHaveLength(4);
    state().undo(); expect(levelInputs(state().project.graph, true)).toHaveLength(3);
  });

  it('warns equally when unlinking the primary input, but not for an unlinked value', () => {
    expect(valueUnlinkWarning(state().project.graph, true, 'img_metas')).toContain('输入数量增加 1');
    expect(valueUnlinkWarning(state().project.graph, true, 'img')).toBeUndefined();
  });

  it('visually separates region removal from unlinking values in the inspector', () => {
    state().createLayoutRegion(['near_head']); state().select('near_head');
    const html = renderToStaticMarkup(createElement(Inspector, { graph: state().project.graph }));
    expect(html).toContain('解除分区归属'); expect(html).toContain('解除值关联');
    expect(html).toContain('新增一个独立 forward 输入');
    expect(html.indexOf('所属分区')).toBeLessThan(html.indexOf('aria-label="值关联"'));
  });
});
