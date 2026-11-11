import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { NodeResult } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorView } from '../src/views/TensorView';

const value = {
  shape: [2, 1, 3],
  axes: ['batch', 'ego', 'xyz'],
  dtype: 'float32',
  autograd: { requiresGrad: true, source: 'manual' as const },
};
function render(type: string, reverse = false) {
  const model = registry.create(
    type,
    { owner: '["pts_bbox_head"]', name: 'points_scale' },
    'attribute',
  );
  const write = type === 'SetAttribute';
  const result: NodeResult = {
    status: 'ok',
    inputs: write ? { in: value } : {},
    outputs: write ? {} : { out: value },
    messages: [],
    notes: [],
  };
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
      result,
      inputs: write ? [{ id: 'in', label: '值' }] : [],
      outputs: write ? [] : [{ id: 'out', label: '值' }],
      category: '输入输出',
      reverse,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
  return renderToStaticMarkup(
    createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
  );
}

describe('Attribute node presentation', () => {
  it.each(['SetAttribute', 'GetAttribute'])(
    'distinguishes %s from Tensor variables and Parameters',
    (type) => {
      const html = render(type);
      expect(html).toContain('attribute-node');
      expect(html).toContain(type === 'SetAttribute' ? '>SET<' : '>GET<');
      expect(html).toContain('points_scale');
      expect(html).toContain('self.pts_bbox_head');
      expect(html).toContain('[2, 1, 3]');
      expect(html).toContain('grad:on');
      expect(html).toContain('--tensor-bg:');
      expect(html).not.toContain('parameter-role-badge');
      expect(html).not.toContain('module-node');
      expect(html).not.toContain('node-footer');
    },
  );

  it('SET has only an input and GET only an output, including reversed flow', () => {
    expect(render('SetAttribute')).toContain('aria-label="输入 in"');
    expect(render('SetAttribute')).not.toContain('aria-label="输出 out"');
    expect(render('GetAttribute')).toContain('aria-label="输出 out"');
    expect(render('GetAttribute')).not.toContain('aria-label="输入 in"');
    expect(render('GetAttribute', true)).toMatch(
      /react-flow__handle-left[^>]+aria-label="输出 out"/,
    );
  });

  it('inspects the value being written rather than pretending SET has a Tensor output', () => {
    const result: NodeResult = {
      status: 'ok',
      inputs: { in: value },
      outputs: {},
      messages: [],
      notes: [],
    };
    const html = renderToStaticMarkup(createElement(TensorView, { result, observeInput: true }));
    expect(html).toContain('属性写入值（输入 Tensor）');
    expect(html).toContain('[2, 1, 3]');
    expect(html).not.toContain('暂时没有可用输出');
    expect(html).toContain('grad:on');
  });
});
