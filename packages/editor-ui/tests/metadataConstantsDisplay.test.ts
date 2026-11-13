import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorView } from '../src/views/TensorView';

describe('Compact metadata constants', () => {
  it.each(['DeviceConstant', 'DtypeConstant'])('shows %s as a small value with no input port or gradient badge', type => {
    const model = registry.create(type, type === 'DeviceConstant' ? { device: 'cuda:0' } : {}, 'metadata');
    const inferred = registry.get(type)!.inferValue!({}, model.params);
    const result = { status: 'ok' as const, inputs: {}, outputs: inferred.outputs, notes: inferred.notes ?? [], messages: [] };
    const props: NodeProps<TrailNode> = {
      id: model.id, type: 'module', selected: false, dragging: false, zIndex: 0,
      selectable: true, deletable: true, draggable: true, isConnectable: true,
      positionAbsoluteX: 0, positionAbsoluteY: 0,
      data: { model, result, ...ports(model, registry), category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {} },
    };
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
    expect(html).toContain('metadata-constant-node');
    expect(html).toContain(type === 'DeviceConstant' ? '<code>cuda:0</code>' : '<code>float32</code>');
    expect(html).toContain(type === 'DeviceConstant' ? '>DEVICE<' : '>DTYPE<');
    expect(html).toContain('aria-label="输出 out"');
    for (const forbidden of ['aria-label="输入', 'tensor-grad-badge', 'tensor-shape', 'node-heading', 'grad:']) expect(html).not.toContain(forbidden);
    if (type === 'DeviceConstant') {
      const inspector = renderToStaticMarkup(createElement(TensorView, { result }));
      expect(inspector).toContain('固定设备声明');
      expect(inspector).not.toContain('不推断为 cuda:0');
    }
  });
});
