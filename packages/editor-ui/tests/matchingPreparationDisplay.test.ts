import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

describe('Matching preparation operation labels', () => {
  it.each([
    ['OnesScaled', 'ones() * -1'],
    ['Arange', 'arange(end)'],
    ['SortRowsByMax', 'sort_rows_by_max()'],
    ['MaxWithIndices', 'max(-1)'],
    ['Sort', 'sort(asc)'],
    ['RowIndex', 'x[indices]'],
  ])('shows %s as a compact operation, not a network layer', (type, label) => {
    const model = registry.create(type, {}, 'operation');
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
        category: registry.get(type)!.category,
        reverse: false,
        active: false,
        dimmed: false,
        open: () => {},
      },
    };
    const html = renderToStaticMarkup(
      createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
    );
    expect(html).toContain('compact-operation-node feature-operation-node');
    expect(html).toContain(`<code>${label}</code>`);
    expect(html).not.toContain('node-footer');
  });
});
