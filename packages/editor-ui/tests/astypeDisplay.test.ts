import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

function render(dtype = 'float32', options: {
  reverse?: boolean;
  portLayout?: ModelNode['portLayout'];
  result?: NodeResult;
  selected?: boolean;
  active?: boolean;
  dimmed?: boolean;
} = {}) {
  const model = registry.create('NumpyAstype', { dtype }, 'astype');
  model.label = 'astype(np.float32)';
  model.portLayout = options.portLayout;
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, result: options.result, category: '维度操作', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      inputs: [{ id: 'in', label: '数组' }], outputs: [{ id: 'out', label: '数组' }],
      open: () => {},
    },
  };
  return renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
}

describe('Compact NumPy astype display', () => {
  it.each(['float16', 'float32', 'float64', 'int32', 'int64', 'uint8', 'bool'])(
    'shows a conversion icon and the actual target dtype %s', dtype => {
      const html = render(dtype);
      expect(html).toContain('compact-operation-node astype-node');
      expect(html).toContain('lucide-replace');
      expect(html).toContain('<code>astype</code>');
      expect(html).toContain(`<code class="astype-dtype">${dtype}</code>`);
      expect(html).toContain(`aria-label="astype(np.float32): astype(np.${dtype})"`);
      expect(html).toContain('aria-label="输入 in"');
      expect(html).toContain('aria-label="输出 out"');
      for (const removed of ['node-heading', 'node-operation', 'node-footer', 'node-axis-labels', 'grad:'])
        expect(html).not.toContain(removed);
    },
  );

  it('preserves reversed and custom port layouts', () => {
    const reversed = render('float32', { reverse: true });
    expect(reversed).toMatch(/react-flow__handle-right[^>]+aria-label="输入 in"/);
    expect(reversed).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const vertical = render('float32', { portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 in"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
  });

  it('preserves selection, stepping, dimming and validation messages', () => {
    const html = render('float32', {
      selected: true, active: true, dimmed: true,
      result: { status: 'error', inputs: {}, outputs: {}, messages: ['Expected NumPy array'], notes: [] },
    });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('Expected NumPy array');
    expect(html).toContain('title="astype(np.float32)');
  });
});
