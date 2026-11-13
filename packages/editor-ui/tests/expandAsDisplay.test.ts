import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

function render(options: {
  reverse?: boolean;
  portLayout?: ModelNode['portLayout'];
  result?: NodeResult;
  selected?: boolean;
  active?: boolean;
  dimmed?: boolean;
} = {}) {
  const model = registry.create('ExpandAs', {}, 'expand');
  model.label = 'expand_as(lidar2img)';
  model.portLayout = options.portLayout;
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, result: options.result, category: '维度操作', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...ports(model, registry), open: () => {},
    },
  };
  return renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
}

describe('Compact expand_as display', () => {
  it('uses outward expansion arrows and an operation name without a generic module card', () => {
    const html = render();
    expect(html).toContain('compact-operation-node expand-as-node');
    expect(html).toContain('lucide-expand');
    expect(html).toContain('<code>expand_as()</code>');
    expect(html).toContain('aria-label="expand_as(lidar2img): expand_as()"');
    expect(html.match(/data-handleid=/g)).toHaveLength(3);
    for (const removed of ['node-heading', 'node-operation', 'node-footer', 'node-axis-labels', 'module-glyph', 'grad:'])
      expect(html).not.toContain(removed);
  });

  it('keeps distinct data and shape-reference ports and explains the reference without implying fusion', () => {
    const html = render();
    for (const id of ['in', 'like']) {
      expect(html).toContain(`aria-label="输入 ${id}"`);
      expect(html).toMatch(new RegExp(`class="node-port-label [^"]+"[^>]*>${id}</span>`));
    }
    expect(html).toContain('aria-label="输出 out"');
    expect(html).toContain('目标 shape 参考');
    expect(html).toContain('like 只提供目标 shape，不融合其数值');
  });

  it('keeps reversed, vertical and same-side layouts with space for port labels', () => {
    const reversed = render({ reverse: true });
    expect(reversed).toMatch(/react-flow__handle-right[^>]+aria-label="输入 like"/);
    expect(reversed).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    expect(reversed).toContain('padding-right:36px');
    const vertical = render({ portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 like"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    expect(vertical).toContain('padding-top:22px');
    expect(vertical).toContain('padding-bottom:22px');
    const sameSide = render({ portLayout: { input: 'left', output: 'left' } });
    expect(sameSide).toContain('min-height:64px');
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(3);
  });

  it.each(['error', 'blocked'] as const)('keeps %s messages, selection and step states', status => {
    const html = render({
      selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['Incompatible target shape'], notes: [] },
    });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('Incompatible target shape');
    expect(html).toContain('title="expand_as()');
  });
});
