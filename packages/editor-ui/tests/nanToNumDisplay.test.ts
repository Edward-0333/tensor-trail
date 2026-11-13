import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

it('renders nan_to_num as a compact unary operation rather than a neural layer', () => {
  const model = registry.create('NanToNum', {}, 'clean');
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: false, dragging: false, zIndex: 0,
    selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, category: '数学运算', reverse: false, active: false, dimmed: false,
      ...ports(model, registry), open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(html).toContain('compact-operation-node feature-operation-node');
  expect(html).toContain('<code>nan_to_num()</code>');
  expect(html).toContain('aria-label="输入 in"');
  expect(html).toContain('aria-label="输出 out"');
  expect(html).not.toContain('node-footer');
});
