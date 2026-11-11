import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

function render(name = 'weight', reverse = false) {
  const source = registry.create('ModuleReference', {}, 'module');
  const model = registry.create('GetParameter', { name }, 'read');
  const result = inferGraph({
    nodes: [source, model],
    edges: [{ id: 'edge', source: 'module', sourcePort: 'out', target: 'read', targetPort: 'in' }],
  }, registry).nodes.read;
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: false, dragging: false, zIndex: 0,
    selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, result, category: '输入输出', reverse, active: false, dimmed: false,
      inputs: [{ id: 'in', label: '模块' }], outputs: [{ id: 'out', label: '参数' }],
      open: () => {},
    },
  };
  return renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
}

describe('Compact GetParameter display', () => {
  it.each(['weight', 'bias'])('shows the actual parameter name %s without a generic module card', name => {
    const html = render(name);
    expect(html).toContain('compact-operation-node parameter-read-node');
    expect(html).toContain(`<code>读取 .${name}</code>`);
    expect(html).toContain('aria-label="输入 in"');
    expect(html).toContain('aria-label="输出 out"');
    for (const removed of ['node-heading', 'node-operation', 'node-footer', 'node-axis-labels', '<svg'])
      expect(html).not.toContain(removed);
  });

  it('keeps input and output handles when flow direction is reversed', () => {
    const html = render('weight', true);
    expect(html).toMatch(/react-flow__handle-right[^>]+aria-label="输入 in"/);
    expect(html).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
  });

  it('keeps validation errors discoverable in the compact form', () => {
    const html = render('weight.data');
    expect(html).toContain('invalid');
    expect(html).toContain('<code>参数名待修正</code>');
    expect(html).toContain('参数名需要单个 Python 属性名称');
    expect(html).not.toContain('node-footer');
  });
});
