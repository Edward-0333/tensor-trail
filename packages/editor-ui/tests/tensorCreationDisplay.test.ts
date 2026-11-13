import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

function render(mode: 'tensor' | 'new_tensor' = 'new_tensor', options: {
  reverse?: boolean;
  portLayout?: ModelNode['portLayout'];
  result?: NodeResult;
  selected?: boolean;
  active?: boolean;
  dimmed?: boolean;
} = {}) {
  const model = registry.create('ArrayToTensor', { mode }, 'create');
  model.label = 'query_feat.new_tensor(ego2lidar)';
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

describe('Compact tensor creation display', () => {
  it.each([
    ['new_tensor', 'new_tensor()', 4],
    ['tensor', 'torch.tensor()', 3],
  ] as const)('shows the actual %s mode with a small creation icon', (mode, expression, handleCount) => {
    const html = render(mode);
    expect(html).toContain('compact-operation-node tensor-create-node');
    expect(html).toContain('lucide-square-plus');
    expect(html).toContain(`<code>${expression}</code>`);
    expect(html).toContain(`aria-label="query_feat.new_tensor(ego2lidar): ${expression}"`);
    expect(html.match(/data-handleid=/g)).toHaveLength(handleCount);
    for (const removed of ['node-heading', 'node-operation', 'node-footer', 'node-axis-labels', 'module-glyph', 'grad:'])
      expect(html).not.toContain(removed);
  });

  it('keeps data and metadata ports separately labelled instead of adding a reference Tensor input', () => {
    const html = render();
    for (const id of ['in', 'device', 'dtype']) {
      expect(html).toContain(`aria-label="输入 ${id}"`);
      expect(html).toMatch(new RegExp(`class="node-port-label [^"]+"[^>]*>${id}</span>`));
    }
    expect(html).toContain('NumPy 数组（数据）');
    expect(html).toContain('device（元数据）');
    expect(html).toContain('dtype（元数据）');
    expect(html).toContain('aria-label="输出 out"');
    expect(html).not.toContain('aria-label="输入 reference"');
    expect(render('tensor')).not.toContain('aria-label="输入 dtype"');
  });

  it('keeps reversed, vertical and same-side port layouts with room for port labels', () => {
    const reversed = render('new_tensor', { reverse: true });
    expect(reversed).toMatch(/react-flow__handle-right[^>]+aria-label="输入 dtype"/);
    expect(reversed).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    expect(reversed).toContain('padding-right:46px');
    const vertical = render('new_tensor', { portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 dtype"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    expect(vertical).toContain('padding-top:22px');
    expect(vertical).toContain('padding-bottom:22px');
    const sameSide = render('new_tensor', { portLayout: { input: 'left', output: 'left' } });
    expect(sameSide).toContain('min-height:80px');
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(4);
  });

  it.each(['error', 'blocked'] as const)('keeps %s messages, selection and step states', status => {
    const html = render('new_tensor', {
      selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['Expected dtype metadata'], notes: [] },
    });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('Expected dtype metadata');
    expect(html).toContain('title="new_tensor()');
  });
});
