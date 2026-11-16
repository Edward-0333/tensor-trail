import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { LayoutRegionNode } from '../src/views/LayoutRegionNode';
import { LayoutRegionPanel } from '../src/editor/LayoutRegionPanel';
import { RegionImportance } from '../src/views/RegionImportance';
import { useEditor } from '../src/editor/store';

describe('Region importance UI', () => {
  it.each([1, 2, 3, 4, 5])('renders five stars with exactly %i filled and explicit local scope', importance => {
    const model = registry.create('LayoutRegion', { importance, importance_note: '同层核心计算' });
    const html = renderToStaticMarkup(createElement(RegionImportance, { model }));
    expect(html.match(/<svg/g)).toHaveLength(5);
    expect(html.match(/class="[^"]*filled[^"]*"/g)).toHaveLength(importance);
    expect(html).toContain(`本层重要性：${importance}/5星`);
    expect(html).toContain('不跨层级比较'); expect(html).toContain('同层核心计算');
  });

  it('displays ungraded, not a fake low score, for old or new ungraded regions', () => {
    const model = registry.create('LayoutRegion'); delete model.params.importance;
    const html = renderToStaticMarkup(createElement(RegionImportance, { model }));
    expect(html).toContain('未评级'); expect(html).not.toContain('filled');
    expect(html).not.toContain('0/5');
  });

  it('provides native keyboard-operable ratings, reason editing and a clear action', () => {
    const node = registry.create('LayoutRegion', { importance: 3 });
    const html = renderToStaticMarkup(createElement(LayoutRegionPanel, { graph: { nodes: [node], edges: [] }, node }));
    expect(html.match(/type="radio"/g)).toHaveLength(5);
    expect(html.match(/checked=""/g)).toHaveLength(1);
    expect(html).toContain('重要性 · 仅本层比较');
    expect(html).toContain('清除评级'); expect(html).toContain('aria-label="评级依据"');
    expect(html).toContain('低星不表示可以删除');
  });

  it('keeps the existing region header, note and color while adding a read-only badge', () => {
    const model = registry.create('LayoutRegion', { importance: 5, note: '特征融合', color: '#38b2ac' });
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(LayoutRegionNode, { model, selected: false })));
    expect(html).toContain('layout-region-heading'); expect(html).toContain('--region-color:#38b2ac');
    expect(html).toContain('特征融合'); expect(html).toContain('本层重要性：5/5星');
    expect(html).not.toContain('type="radio"');
  });

  it('undoes, redoes and clears ratings without changing geometry or inference', () => {
    const state = () => useEditor.getState();
    const input = registry.create('Input', {}, 'input'), region = registry.create('LayoutRegion', { members: 'input' }, 'region');
    state().load({ schemaVersion: 1, name: 'importance', variants: [], graph: { nodes: [input, region], edges: [] } });
    const before = structuredClone(state().project), specs = inferProject(before, registry).nodes.input.outputs;
    expect(state().setParam('region', 'importance', 5)).toBe(true);
    const expected = structuredClone(before); expected.graph.nodes[1].params.importance = 5;
    expect(state().project).toEqual(expected);
    expect(inferProject(state().project, registry).nodes.input.outputs).toEqual(specs);
    state().undo(); expect(state().project).toEqual(before);
    state().redo(); expect(state().project).toEqual(expected);
    expect(state().setParam('region', 'importance', 6)).toBe(false);
    expect(state().project).toEqual(expected);
    state().setParam('region', 'importance', 0); expect(state().project).toEqual(before);
  });
});
