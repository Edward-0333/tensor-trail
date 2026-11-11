import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModuleSpec, NodeResult, ValueSpec } from '@tensor-trail/model-core/core/types';
import { shapeText } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorView } from '../src/views/TensorView';
import { canvasValueLabel } from '../src/views/canvasValueLabel';

const module: ModuleSpec = {
  kind: 'module', reference: 'self.init_points', moduleType: 'nn.Embedding',
  parameters: { weight: { shape: [1040, 3], axes: ['Q', 'xyz'], dtype: 'float32', parameterRef: { module: 'self.init_points', name: 'weight' } } },
};
const result: NodeResult = { inputs: { in: module }, outputs: { out: module }, status: 'ok', notes: [], messages: [] };

describe('Uncluttered module reference display', () => {
  it.each(['ValueVariable', 'Outlet'])('shows only the variable name and ports for %s', type => {
    const model = registry.create(type, type === 'Outlet' ? { port: 'out' } : {}, 'ref');
    model.label = 'self.init_points';
    const props: NodeProps<TrailNode> = {
      id: model.id, type: 'module', selected: false, dragging: false, zIndex: 0,
      selectable: true, deletable: true, draggable: true, isConnectable: true,
      positionAbsoluteX: 0, positionAbsoluteY: 0,
      data: { model, result, inputs: [{ id: 'in', label: '模块' }], outputs: [{ id: 'out', label: '模块' }],
        category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {} },
    };
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
    expect(html).toContain('self.init_points');
    expect(html).toContain('module-reference-node');
    expect(html).toContain('aria-label="输入 in"');
    expect(html).toContain('aria-label="输出 out"');
    for (const text of ['nn.Embedding', 'tensor-node-shape', 'tensor-grad-badge', 'node-axis-labels'])
      expect(html).not.toContain(text);
  });

  it('hides module labels on wires without changing type information in the inspector or API', () => {
    expect(canvasValueLabel(module)).toBeUndefined();
    expect(shapeText(module)).toBe('Module · nn.Embedding');
    const html = renderToStaticMarkup(createElement(TensorView, { result }));
    expect(html).toContain('nn.Embedding');
    expect(html).toContain('self.init_points');
    expect(html).toContain('Parameter [1040, 3]');
  });

  it.each<[ValueSpec | undefined, string]>([
    [undefined, '待推导'],
    [module.parameters.weight, '[1040, 3]'],
    [{ kind: 'int', value: 2 }, 'int · 2'],
    [{ kind: 'size', dims: [256, 704] }, 'torch.Size([256, 704])'],
    [{ kind: 'list', items: [module.parameters.weight] }, 'list · 1 个 Tensor'],
  ])('preserves non-module labels %j', (value, label) => {
    expect(canvasValueLabel(value)).toBe(label);
  });
});
