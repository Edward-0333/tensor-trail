import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { NodeResult } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorView } from '../src/views/TensorView';
import { valueEdgeStyle } from '../src/views/valueEdgeStyle';

const device = { kind: 'device' as const, reference: 'maps.device' };
const result: NodeResult = {
  status: 'ok',
  inputs: {},
  outputs: { out: device },
  notes: [],
  messages: [],
};
function render(type: string, hasResult = true) {
  const model = registry.create(type, {}, 'device');
  model.label = 'maps.device';
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
      result: hasResult ? result : undefined,
      ...ports(model, registry),
      category: '输入输出',
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
describe('Device metadata presentation', () => {
  it.each(['DeviceVariable', 'ValueVariable', 'Inlet', 'Outlet'])(
    'renders %s as a small DEVICE value without Tensor shape/grad colors',
    (type) => {
      const html = render(type);
      expect(html).toContain('device-value-node');
      expect(html).toContain('device-role-badge');
      expect(html).toContain('maps.device');
      expect(html).not.toContain('tensor-grad-badge');
      expect(html).not.toContain('--tensor-bg');
      expect(html).not.toContain('grad:');
      expect(html).not.toContain('tensor-shape');
      expect(html).not.toContain('cuda:0');
    },
  );
  it('shows an unconnected device without implying unknown Tensor gradients', () => {
    const html = render('DeviceVariable', false);
    expect(html).toContain('DEVICE');
    expect(html).not.toContain('grad:');
  });
  it('renders GetDevice as the .device read operation', () => {
    expect(render('GetDevice')).toContain('<code>.device</code>');
  });
  it('explains unobserved device metadata in the inspector rather than Tensor axes or autograd status', () => {
    const html = renderToStaticMarkup(createElement(TensorView, { result }));
    expect(html).toContain('设备元数据');
    expect(html).toContain('实际设备未采集');
    expect(html).not.toContain('tensor-axes');
    expect(html).not.toContain('grad:');
  });
  it('uses dashed gray device edges, retains error color and leaves data/size edges solid', () => {
    expect(valueEdgeStyle(device)).toMatchObject({
      stroke: 'var(--text-secondary)',
      strokeDasharray: '5 4',
    });
    expect(valueEdgeStyle(device, true)).toMatchObject({
      stroke: 'var(--danger)',
      strokeDasharray: '5 4',
    });
    expect(valueEdgeStyle({ shape: [1040], axes: ['query'], dtype: 'int64' })).not.toHaveProperty(
      'strokeDasharray',
    );
    expect(valueEdgeStyle({ kind: 'int', value: 1040 })).not.toHaveProperty('strokeDasharray');
  });
});
