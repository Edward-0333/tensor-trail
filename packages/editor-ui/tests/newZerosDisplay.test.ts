import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

function render(type: 'NewZeros' | 'TensorVariable') {
  const source = registry.create('Input', { shape: '2,1040,1,3', axes: '' }, 'reference');
  source.tensorAutograd = { requiresGrad: true, source: 'manual' };
  const factory = registry.create(
    'NewZeros',
    { shape: '2,1040,256', axes: 'batch,query,embedding' },
    'factory',
  );
  const value = registry.create('TensorVariable', {}, 'query_feat');
  const result = inferGraph(
    {
      nodes: [source, factory, value],
      edges: [
        {
          id: 'metadata',
          source: 'reference',
          sourcePort: 'out',
          target: 'factory',
          targetPort: 'in',
        },
        {
          id: 'created',
          source: 'factory',
          sourcePort: 'out',
          target: 'query_feat',
          targetPort: 'in',
        },
      ],
    },
    registry,
  );
  const model = type === 'NewZeros' ? factory : value;
  const nodePorts = ports(model, registry);
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
      result: result.nodes[model.id],
      ...nodePorts,
      category: '维度操作',
      reverse: false,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
  return renderToStaticMarkup(
    createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
  );
}

describe('NewZeros visualization', () => {
  it('uses a compact new_zeros() operation instead of a generic network block', () => {
    const html = render('NewZeros');
    expect(html).toContain('compact-operation-node feature-operation-node');
    expect(html).toContain('<code>new_zeros()</code>');
    expect(html).toContain('参考 Tensor（仅继承 dtype/device）');
    expect(html).not.toContain('node-footer');
  });
  it('shows the created query features as grad:off, not Parameter or inherited grad:on', () => {
    const html = render('TensorVariable');
    expect(html).toContain('[2, 1040, 256]');
    expect(html).toContain('grad:off');
    expect(html).not.toContain('grad:on');
    expect(html).not.toContain('parameter-role-badge');
    expect(html).toContain('静态推导');
  });
});
