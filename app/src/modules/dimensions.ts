import type { ModuleDefinition, Params, Port, IntegerSpec } from '../core/types';
import { tensorValue } from '../core/types';
import { text, input } from './primitives';
import { axisIndex, requireThat } from './math';

const parts = (value: unknown): string[] =>
  typeof value === 'string' ? value.split(',').map((v) => v.trim()) : [];

/** Keep existing connections stable during incomplete edits and output renames. */
export function dimensionPorts(params: Params): { inputs: Port[]; outputs: Port[] } {
  const count = Math.min(8, Math.max(1, parts(params.axes).length));
  const names = parts(params.names);
  return {
    inputs: input,
    outputs: Array.from({ length: count }, (_, i) => ({
      id: `dim${i}`,
      label: names[i] || `dim${i}`,
    })),
  };
}

function selections(params: Params) {
  const axes = parts(params.axes),
    names = parts(params.names);
  requireThat(
    axes.length > 0 &&
      axes.length <= 8 &&
      axes.every((axis) => /^-?\d+$/.test(axis) && Number.isSafeInteger(Number(axis))),
    '读取维度需要 1–8 个整数轴，以逗号分隔，例如 0, 1；支持负轴。',
  );
  requireThat(typeof params.names === 'string', '输出名称需要是逗号分隔的文本。');
  requireThat(
    names.slice(axes.length).every((name) => !name),
    '输出名称数量不能多于读取的轴数。',
  );
  const outputs = dimensionPorts(params).outputs;
  requireThat(
    outputs.every(
      (port) =>
        /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(port.label) &&
        !['__proto__', 'constructor', 'prototype'].includes(port.label),
    ),
    '输出名称应为变量名，例如 bs, A；留空使用默认名称。',
  );
  requireThat(
    new Set(outputs.map((port) => port.label)).size === outputs.length,
    '输出名称不能重复。',
  );
  return axes.map((axis, i) => ({ axis: Number(axis), ...outputs[i] }));
}

export const dimensionModules: ModuleDefinition[] = [
  {
    type: 'TensorSize',
    label: '读取维度 Size',
    category: '形状查询',
    description:
      '读取 Tensor 指定轴的尺寸，每个轴分别输出一个整数；例如轴 0, 1 输出 bs、A，对应 bs, A = x.shape[0:2]。不读取张量元素或参与梯度计算。',
    fields: [
      { ...text('axes', '读取轴（逗号分隔）', '0'), maxLength: 256 },
      { ...text('names', '输出名称（逗号分隔）', ''), maxLength: 512 },
    ],
    inputs: input,
    outputs: [{ id: 'dim0', label: 'dim0' }],
    getPorts: dimensionPorts,
    inferValue: ({ in: value }, params) => {
      const tensor = tensorValue(value);
      const requested = selections(params);
      const outputs: Record<string, IntegerSpec> = {};
      const notes: string[] = [];
      for (const selection of requested) {
        const axis = axisIndex(selection.axis, tensor.shape.length);
        const size = tensor.shape[axis];
        outputs[selection.id] = { kind: 'int', value: size };
        notes.push(
          `${selection.label} = input.shape[${selection.axis}] = ${size}；输出为整数尺寸。`,
        );
      }
      return { outputs, notes };
    },
  },
];
