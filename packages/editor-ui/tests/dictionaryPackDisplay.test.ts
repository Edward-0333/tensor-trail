import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult, Port } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { ModuleGlyph } from '../src/views/ModuleGlyph';

function render(options: {
  keys?: string; reverse?: boolean; portLayout?: ModelNode['portLayout'];
  result?: NodeResult; selected?: boolean; active?: boolean; dimmed?: boolean;
  declaredPorts?: { inputs: Port[]; outputs: Port[] };
} = {}) {
  const model = registry.create('DictionaryPack', { keys: options.keys ?? 'cls_score,refine_pts,outs' }, 'pack');
  model.label = 'outputs = dict(...)';
  model.portLayout = options.portLayout;
  const before = structuredClone(model);
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, result: options.result, category: '组合计算', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...(options.declaredPorts ?? ports(model, registry)), open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(model).toEqual(before);
  return html;
}

describe('Compact DictionaryPack', () => {
  it('shows a field-to-dictionary glyph and dict() without the large generic card', () => {
    const html = render({ result: { status: 'ok', inputs: {}, outputs: {
      out: { kind: 'dict', fields: {
        cls_score: { shape: [2, 720, 48, 17], axes: ['b', 'q', 'p', 'c'], dtype: 'float32' },
      } },
    }, messages: [], notes: [] } });
    expect(html).toContain('compact-operation-node dictionary-pack-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'DictionaryPack' })));
    expect(html).toContain('不堆叠或融合张量');
    expect(html).toContain('<code>dict()</code>');
    expect(html).toContain('min-height:72px');
    for (const removed of ['node-heading', 'node-footer', 'node-axis-labels', '组合子网络', 'grad:', '[2, 720, 48, 17]'])
      expect(html).not.toContain(removed);
  });

  it('preserves field order, real handle IDs and descriptive labels', () => {
    const html = render();
    expect(html.match(/data-handleid=/g)).toHaveLength(4);
    for (const key of ['cls_score', 'refine_pts', 'outs']) {
      expect(html).toContain(`aria-label="输入 ${key}"`);
      expect(html).toContain(`>${key}</span>`);
    }
    expect(html.indexOf('data-handleid="cls_score"')).toBeLessThan(html.indexOf('data-handleid="refine_pts"'));
    expect(html).toContain('data-handleid="out"');
    expect(html).toContain('aria-label="输出 out"');
    expect(html).toContain('>dict</span>');
    expect(html).toContain('dict(cls_score, refine_pts, outs)');
  });

  it('supports reversed and same-side ports without changing their identity', () => {
    const reverse = render({ reverse: true });
    expect(reverse).toMatch(/react-flow__handle-right[^>]+aria-label="输入 cls_score"/);
    expect(reverse).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const same = render({ portLayout: { input: 'left', output: 'left' } });
    expect(same.match(/react-flow__handle-left/g)).toHaveLength(4);
    expect(same).toContain('min-height:90px');
  });

  it('reserves space for top/bottom port labels and a four-field head output', () => {
    const html = render({ keys: 'init_points,all_cls_scores,all_refine_pts,query_feat',
      portLayout: { input: 'top', output: 'bottom' } });
    expect(html).toMatch(/react-flow__handle-top[^>]+aria-label="输入 all_refine_pts"/);
    expect(html).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    expect(html).toContain('padding-top:24px');
    expect(html).toContain('padding-bottom:24px');
    expect(html).toContain('min-width:465px');
  });

  it.each([1, 4, 32])('scales to %i fields without dropping handles', count => {
    const html = render({ keys: Array.from({ length: count }, (_, i) => `field${i}`).join(',') });
    expect(html.match(/data-handleid=/g)).toHaveLength(count + 1);
    expect(html).toContain(`min-height:${Math.max(40, (count + 1) * 18)}px`);
  });

  it('bounds space for long keys but preserves their full name in tooltips and handles', () => {
    const key = 'x'.repeat(128);
    const html = render({ keys: key });
    expect(html).toContain(`title="${key}"`);
    expect(html).toContain(`data-handleid="${key}"`);
    expect(html).toContain('padding-left:136px');
  });

  it('renders incomplete/error states without parsing keys a second time', () => {
    const html = render({ keys: '', declaredPorts: { inputs: [], outputs: [{ id: 'out', label: 'out' }] },
      result: { status: 'error', inputs: {}, outputs: {}, messages: ['字段名无效'], notes: [] } });
    expect(html).toContain('invalid');
    expect(html).toContain('字段名无效');
    expect(html).toContain('<code>dict()</code>');
  });

  it.each(['error', 'blocked'] as const)('keeps %s feedback and selection/stepping styles', status => {
    const html = render({ selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['请连接输入'], notes: [] } });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('请连接输入');
  });
});
