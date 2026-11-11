import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

it('shows reweight as an expandable function instead of an opaque operation or neural forward', () => {
  const model = registry.create('Reweight', {}, 'reweight');
  const props: NodeProps<TrailNode> = {
    id: model.id,
    type: 'module',
    selected: false,
    dragging: false,
    zIndex: 0,
    selectable: true,
    deletable: true,
    draggable: true,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    data: {
      model,
      ...ports(model, registry),
      category: '组合计算',
      reverse: false,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
  const html = renderToStaticMarkup(
    createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
  );
  expect(html).toContain('function-node');
  expect(html).toContain('function-symbol');
  expect(html).toContain('<strong>reweight()</strong>');
  expect(html).toContain('双击进入函数');
  expect(html).toContain('lucide-arrow-up-right');
  expect(html).not.toContain('forward()');
  expect(html).not.toContain('sort_rows_by_max()');
});
