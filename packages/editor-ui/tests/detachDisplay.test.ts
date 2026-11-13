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
  reverse?: boolean; portLayout?: ModelNode['portLayout']; result?: NodeResult;
  selected?: boolean; active?: boolean; dimmed?: boolean;
} = {}) {
  const model = registry.create('Detach', {}, 'detach');
  model.label = 'query_points.detach()';
  model.portLayout = options.portLayout;
  const before = structuredClone(model);
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, result: options.result, category: '维度操作', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...ports(model, registry), open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(model).toEqual(before);
  return html;
}
describe('Compact Detach', () => {
  it('shows a gradient-stop glyph and detach() without a generic network card', () => {
    const html = render({ result: {
      status: 'ok', inputs: {}, outputs: { out: { shape: [2,720,48,3],
        axes: ['batch','query','point','xyz'], dtype: 'float32',
        autograd: { requiresGrad: false, source: 'inferred' } } }, messages: [], notes: [],
    } });
    expect(html).toContain('compact-operation-node detach-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'Detach' })));
    expect(html).toContain('<code>detach()</code>');
    expect(html).toContain('数据继续传递，梯度在此截断');
    expect(html).toContain('requires_grad=False');
    expect(html).toContain('与输入共享存储，不是 clone');
    for (const removed of ['node-heading', 'node-footer', 'node-axis-labels', 'node-port-label', '组合子网络', 'grad:off', '[2, 720, 48, 3]'])
      expect(html).not.toContain(removed);
  });
  it('retains accessible ports in standard, reverse, vertical and same-side layouts', () => {
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
  it.each(['error','blocked'] as const)('preserves %s diagnostics and selection/stepping styles', status => {
    const html = render({ selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['需要 Tensor 输入'], notes: [] } });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('需要 Tensor 输入');
    expect(html).toContain('aria-label="query_points.detach(): detach()"');
  });
});
