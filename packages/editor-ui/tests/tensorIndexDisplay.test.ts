import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { Params, TensorSpec } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

it.each<[number[] | undefined, Params, string]>([
  [[1040, 8], {}, '[:, :-1]'],
  [[2, 1040, 8], {}, '[:, :-1, ...]'],
  [undefined, {}, '[:, :-1, ...]'],
  [[1040, 8], { mode: 'index', index: '0' }, '[:, 0]'],
  [[1040], { mode: 'index', dim: '0', index: '0' }, '[0]'],
])('renders TensorIndex using input rank %j rather than output rank', (shape, params, label) => {
  const model = registry.create('TensorIndex', {
    mode: 'slice', dim: '1', start: '', stop: '-1', step: 1, ...params,
  }, 'slice');
  const input: TensorSpec | undefined = shape && {
    shape, axes: shape.map((_, i) => `axis${i}`), dtype: 'float32',
  };
  const inferred = input ? registry.get('TensorIndex')!.infer!({ in: input }, model.params) : undefined;
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: false, dragging: false, zIndex: 0,
    selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, ...ports(model, registry), category: '索引与切片', reverse: false,
      active: false, dimmed: false, open: () => {},
      result: inferred && input ? { status: 'ok', inputs: { in: input }, outputs: inferred.outputs, notes: [], messages: [] } : undefined,
    },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(html).toContain(`<code>${label}</code>`);
  expect(html).toContain(`aria-label="索引与切片：${label}"`);
  if (inferred && model.params.mode === 'slice') expect(inferred.notes).toContain(label);
});
