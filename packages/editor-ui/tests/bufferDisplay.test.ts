import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorView } from '../src/views/TensorView';

function fixture() {
  const buffer = registry.create('BufferSource', { shape: '1040,7', axes: 'query,time' }, 'buffer');
  const cast = registry.create('Cast', { dtype: 'float32' }, 'cast');
  const value = registry.create('TensorVariable', {}, 'value');
  const result = inferGraph(
    {
      nodes: [buffer, cast, value],
      edges: [
        { id: 'cast', source: 'buffer', sourcePort: 'out', target: 'cast', targetPort: 'in' },
        { id: 'value', source: 'cast', sourcePort: 'out', target: 'value', targetPort: 'in' },
      ],
    },
    registry,
  );
  return { buffer, value, result };
}
function render(kind: 'buffer' | 'value') {
  const f = fixture(),
    model = f[kind];
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
      result: f.result.nodes[model.id],
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
describe('Buffer registration appearance', () => {
  it('uses a BUFFER role and green border while retaining the shared grad:off background', () => {
    const html = render('buffer');
    expect(html).toContain('buffer-role-badge');
    expect(html).toContain('>BUFFER<');
    expect(html).toContain('--tensor-border:#059669');
    expect(html).toContain('grad:off');
    expect(html).not.toContain('parameter-role-badge');
    expect(html).toContain('[1040, 7]');
  });
  it('does not label float-converted temporary tensors as registered buffers', () => {
    const html = render('value');
    expect(html).toContain('grad:off');
    expect(html).not.toContain('buffer-role-badge');
    expect(html).not.toContain('parameter-role-badge');
  });
  it('explains state ownership and unobserved current values in the inspector', () => {
    const { result } = fixture();
    const html = renderToStaticMarkup(createElement(TensorView, { result: result.nodes.buffer }));
    expect(html).toContain('不是 Tensor 子类或 Parameter');
    expect(html).toContain('当前数值未采集');
    expect(html).toContain('state_dict');
  });
});
