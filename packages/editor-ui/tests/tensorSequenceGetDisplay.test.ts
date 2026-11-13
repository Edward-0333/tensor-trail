import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { ModuleGlyph } from '../src/views/ModuleGlyph';

function render(options: {
  index?: string; reverse?: boolean; portLayout?: ModelNode['portLayout'];
  result?: NodeResult; selected?: boolean; active?: boolean; dimmed?: boolean;
} = {}) {
  const model = registry.create('TensorSequenceGet', { index: options.index ?? '-1' }, 'get');
  model.label = 'all_refine_pts[-1]';
  model.portLayout = options.portLayout;
  const before = structuredClone(model);
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, result: options.result, category: '索引与切片', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...ports(model, registry), open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(model).toEqual(before);
  return html;
}

describe('Compact TensorSequenceGet', () => {
  it('uses a list-to-element glyph and index instead of a generic network card', () => {
    const html = render({ result: {
      status: 'ok', inputs: {}, outputs: { out: { shape: [2, 1040, 48, 3],
        axes: ['batch', 'query', 'point', 'xyz'], dtype: 'float32' } }, messages: [], notes: [],
    } });
    expect(html).toContain('compact-operation-node tensor-sequence-get-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'TensorSequenceGet' })));
    expect(html).toContain('<code>[-1]</code>');
    expect(html).toContain('从 list/tuple 中按索引取一个 Tensor');
    expect(html).toContain('不切片 Tensor 轴、不复制或 detach');
    for (const removed of ['node-heading', 'node-footer', 'node-axis-labels', 'node-port-label', '组合子网络', 'grad:', '[2, 1040, 48, 3]'])
      expect(html).not.toContain(removed);
  });

  it.each([['0', '[0]'], [' 3 ', '[3]'], ['num_layers - 1', '[num_layers - 1]'], ['', '[?]']])(
    'shows index %s without hard-coding the current diagram', (index, label) => {
      expect(render({ index })).toContain(`<code>${label}</code>`);
    },
  );

  it('escapes index text and keeps invalid parameters editable', () => {
    const html = render({ index: '<script>' });
    expect(html).toContain('<code>[&lt;script&gt;]</code>');
    expect(html).not.toContain('<script>');
  });

  it('retains two accessible ports in standard, reverse, vertical and same-side layouts', () => {
    const normal = render();
    expect(normal.match(/data-handleid=/g)).toHaveLength(2);
    expect(normal).toContain('aria-label="输入 in"');
    expect(normal).toContain('aria-label="输出 out"');
    const reverse = render({ reverse: true });
    expect(reverse).toMatch(/react-flow__handle-right[^>]+aria-label="输入 in"/);
    expect(reverse).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const vertical = render({ portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 in"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    const sameSide = render({ portLayout: { input: 'left', output: 'left' } });
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(2);
    expect(sameSide).toContain('min-height:42px');
  });

  it.each(['error', 'blocked'] as const)('preserves %s feedback, selection and stepping state', status => {
    const html = render({ selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['索引超出序列范围'], notes: [] } });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('索引超出序列范围');
    expect(html).toContain('aria-label="all_refine_pts[-1]: list/tuple[-1]"');
  });
});
