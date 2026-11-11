import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '@tensor-trail/model-core/agent';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import {
  shapeText,
  type NodeResult,
  type TensorSpec,
  type ValueSpec,
} from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { TensorAxes, TensorView } from '../src/views/TensorView';
import { canvasValueLabel } from '../src/views/canvasValueLabel';
import { isScalarTensor, valueDisplayText } from '../src/views/valueDisplayText';
import { DEFAULT_TENSOR_PALETTE } from '@tensor-trail/model-core/core/tensorAppearance';
import { useEditor } from '../src/editor/store';

const tensor = (shape: number[] = [], dtype = 'int64'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const result = (value: ValueSpec): NodeResult => ({
  status: 'ok',
  inputs: {},
  outputs: { out: value },
  notes: [],
  messages: [],
});
const inspect = (value: ValueSpec) =>
  renderToStaticMarkup(createElement(TensorView, { result: result(value) }));

describe('Scalar presentation without changing Tensor semantics', () => {
  it.each(['int64', 'float32', 'bool', 'complex64'])(
    'displays rank-zero %s as a scalar, while raw shape remains []',
    (dtype) => {
      const value = tensor([], dtype),
        before = structuredClone(value);
      expect(isScalarTensor(value)).toBe(true);
      expect(valueDisplayText(value)).toBe('标量（0D）');
      expect(canvasValueLabel(value, 'node')).toBe('标量（0D）');
      expect(canvasValueLabel(value)).toBe('标量');
      expect(shapeText(value)).toBe('[]');
      const html = inspect(value);
      expect(html).toContain('<div class="tensor-shape">标量（0D）</div>');
      expect(html).toContain('零维标量张量，包含 1 个元素');
      expect(html).toContain('shape: torch.Size([])');
      expect(html).toContain(dtype);
      expect(html).not.toContain('class="tensor-axes"');
      expect(value).toEqual(before);
    },
  );

  it.each<[ValueSpec, string]>([
    [tensor([1]), '[1]'],
    [tensor([0]), '[0]'],
    [tensor([0, 7]), '[0, 7]'],
    [{ kind: 'int', value: 2 }, 'int · 2'],
    [{ kind: 'bool', value: true }, 'bool · True'],
    [{ kind: 'size', dims: [] }, 'torch.Size([])'],
    [{ kind: 'list', items: [] }, 'list · 0 个 Tensor []'],
  ])('does not mislabel non-scalars %j', (value, text) => {
    expect(isScalarTensor(value)).toBe(false);
    expect(valueDisplayText(value)).toBe(text);
    expect(canvasValueLabel(value)).toBe(text);
    expect(inspect(value)).not.toContain('零维标量张量，包含 1 个元素');
    expect(inspect(value)).not.toContain('class="scalar-tensor-details"');
  });

  it('retains the axis card for a length-one or empty vector', () => {
    for (const value of [tensor([1]), tensor([0])]) {
      const html = renderToStaticMarkup(createElement(TensorAxes, { spec: value }));
      expect(html).toContain('AXIS 0');
      expect(html).not.toContain('scalar-tensor-details');
    }
    expect(canvasValueLabel()).toBe('待推导');
    expect(valueDisplayText()).toBe('待连接');
  });

  it.each([false, true])(
    'keeps gradient palette and badges unchanged for scalar requires_grad=%s',
    (grad) => {
      const value: TensorSpec = {
        ...tensor([], 'float32'),
        autograd: { requiresGrad: grad, source: 'manual' },
      };
      const model = registry.create('TensorVariable', {}, 'scalar');
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
          result: result(value),
          ...ports(model, registry),
          category: '输入输出',
          reverse: false,
          active: false,
          dimmed: false,
          open: () => {},
        },
      };
      const html = renderToStaticMarkup(
        createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
      );
      const state = grad ? 'on' : 'off';
      const palette = useEditor.getState().project.tensorPalette ?? DEFAULT_TENSOR_PALETTE;
      expect(html).toContain('<code title="标量（0D）');
      expect(html).toContain(`data-grad-state="${state}"`);
      expect(html).toContain(`--tensor-bg:${palette[state]}`);
      expect(html).not.toContain('>标量</code>');
    },
  );

  it('displays scalar elements consistently in dictionaries, sequences, parameters and input summaries', () => {
    const scalar = tensor();
    const containers: ValueSpec[] = [
      { kind: 'dict', fields: { index: scalar } },
      { kind: 'tuple', items: [scalar] },
      {
        kind: 'module',
        reference: 'self.mod',
        moduleType: 'nn.Module',
        parameters: { weight: scalar },
      },
    ];
    for (const v of containers) expect(inspect(v)).toContain('标量（0D）');
    const html = renderToStaticMarkup(
      createElement(TensorView, {
        result: { ...result({ kind: 'int', value: 'ind' }), inputs: { in: scalar } },
      }),
    );
    expect(html).toContain('int · 符号整数');
    expect(html).toContain('<code>标量（0D）</code>');
    expect(html).not.toContain('class="scalar-tensor-details"');
  });

  it('preserves Argmax -> scalar Tensor -> item() semantics and serialization', () => {
    const built = buildModel({
      name: 'Scalar example',
      nodes: [
        { id: 'scores', type: 'Input', params: { shape: '7', axes: 'time', dtype: 'float32' } },
        { id: 'argmax', type: 'Argmax' },
        { id: 'scalar', type: 'TensorVariable' },
        { id: 'item', type: 'TensorItemInt', params: { symbol: 'ind', cast: false } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['scores.out', 'argmax.in'],
        ['argmax.out', 'scalar.in'],
        ['scalar.out', 'item.in'],
        ['item.out', 'out.in'],
      ],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!,
      before = serializeProject(project),
      inference = inferProject(project, registry);
    expect(inference.nodes.argmax.outputs.out).toEqual({
      shape: [],
      axes: [],
      dtype: 'int64',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(inference.nodes.item.outputs.out).toEqual({ kind: 'int', value: 'ind' });
    inspect(inference.nodes.argmax.outputs.out);
    expect(serializeProject(project)).toBe(before);
    expect(validateModel(parseProject(before, registry)).nodes).toEqual(built.nodes);
  });
});
