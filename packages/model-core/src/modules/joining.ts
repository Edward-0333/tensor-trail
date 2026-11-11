import type { Dim, ModuleDefinition, Params, TensorSpec } from '../core/types';
import { tensorValue } from '../core/types';
import { resultDtype } from './elementwise';
import { axisIndex, requireThat, same, tensor } from './math';
import { num, output, text } from './primitives';

const dtypes = new Set([
  'bool',
  'uint8',
  'int8',
  'int16',
  'int32',
  'int64',
  'float16',
  'bfloat16',
  'float32',
  'float64',
]);
const portId = (index: number) => String.fromCharCode(97 + index);
const configuredCount = (params: Params) =>
  params.input_count === undefined ? 2 : params.input_count;

/** Keep even incomplete inspector edits safe to render; inference validates the actual value. */
export function joiningPorts(params: Params) {
  const raw = configuredCount(params);
  const count =
    typeof raw === 'number' && Number.isFinite(raw)
      ? Math.max(1, Math.min(16, Math.trunc(raw)))
      : 2;
  return {
    inputs: Array.from({ length: count }, (_, index) => ({
      id: portId(index),
      label: `输入 ${portId(index).toUpperCase()}`,
    })),
    outputs: output,
  };
}

function readInputs(inputs: Record<string, TensorSpec>, params: Params): TensorSpec[] {
  const count = configuredCount(params);
  requireThat(
    typeof count === 'number' && Number.isSafeInteger(count) && count >= 1 && count <= 16,
    '输入数量必须是 1–16 的整数。',
  );
  return Array.from({ length: count }, (_, index) => {
    const id = portId(index);
    requireThat(inputs[id], `请连接输入 ${id}。`);
    const value = tensorValue(inputs[id]);
    requireThat(dtypes.has(value.dtype), `暂不支持 dtype=${value.dtype}。`);
    return value;
  });
}

function dimension(params: Params): number {
  requireThat(
    typeof params.dim === 'number' && Number.isSafeInteger(params.dim),
    'dim 必须是安全范围内的整数。',
  );
  return params.dim;
}

function joinedDtype(inputs: TensorSpec[]): string {
  return inputs.slice(1).reduce(
    (current, next) => ({
      ...current,
      dtype: resultDtype(current, next),
    }),
    inputs[0],
  ).dtype;
}

function summedDimensions(dimensions: Dim[]): Dim {
  const terms = dimensions.filter((dim) => dim !== 0);
  if (!terms.length) return 0;
  if (terms.every((dim) => typeof dim === 'number')) {
    const result = (terms as number[]).reduce((sum, dim) => sum + dim, 0);
    requireThat(Number.isSafeInteger(result), '拼接后的维度超过安全整数范围。');
    return result;
  }
  return terms.length === 1 ? terms[0] : `(${terms.join('+')})`;
}

function checkSharedAxes(
  inputs: TensorSpec[],
  warnings: string[],
  message: string,
  skippedAxis?: number,
): void {
  inputs[0].shape.forEach((first, axis) => {
    if (axis === skippedAxis) return;
    const dimensions = inputs.map((value) => value.shape[axis]);
    // A symbolic first input must not hide incompatible concrete sizes in later inputs.
    const reference = dimensions.find((size) => typeof size === 'number') ?? first;
    for (const size of dimensions) same(reference, size, message, warnings);
  });
}

const fields = [
  // Inference checks the rank-dependent bounds; a fixed minimum would reject valid deep stacks.
  num('dim', '操作轴', -1, Number.MIN_SAFE_INTEGER),
  { ...num('input_count', '输入数量', 2, 1), help: '1–16 路输入，按 a、b、c… 的顺序拼接或堆叠。' },
];
const base = {
  category: '分支运算',
  fields,
  // Older saved Concat nodes contain only dim; they still mean exactly two inputs.
  getFields: (params: Params) =>
    params.input_count === undefined
      ? fields.filter((field) => field.key !== 'input_count')
      : fields,
  inputs: joiningPorts({}).inputs,
  outputs: output,
  getPorts: joiningPorts,
};
const concatAxisName = {
  ...text('axis_name', '拼接轴含义（可选）', ''),
  maxLength: 128,
  help: '仅覆盖输出拼接轴的语义标签，不改变形状、数值或拼接规则；留空沿用第一个输入。',
};

export const joiningModules: ModuleDefinition[] = [
  {
    ...base,
    type: 'Concat',
    label: 'Cat',
    description: '对应 torch.cat，沿已有轴连接 1–16 路 Tensor；其它轴必须相等，不进行广播。',
    fields: [...fields, concatAxisName],
    getFields: (params) =>
      params.axis_name === undefined
        ? base.getFields(params)
        : [...base.getFields(params), concatAxisName],
    infer: (values, params) => {
      const inputs = readInputs(values, params),
        dim = dimension(params);
      const axisName = params.axis_name ?? '';
      requireThat(typeof axisName === 'string' && axisName.length <= 128, '拼接轴含义需要短文本。');
      requireThat(
        inputs.every((value) => value.shape.length > 0),
        'Cat 不能拼接零维 Tensor，请使用 Stack。',
      );
      const dtype = joinedDtype(inputs),
        warnings: string[] = [];
      // torch.cat accepts one-dimensional empty tensors alongside any otherwise compatible rank.
      const active = inputs.filter((value) => !(value.shape.length === 1 && value.shape[0] === 0));
      if (!active.length)
        return {
          outputs: {
            out: tensor([0], axisName.trim() ? [axisName.trim()] : [...inputs[0].axes], dtype),
          },
          notes: ['所有输入均为一维空 Tensor，Cat 输出一维空 Tensor；dtype 仍按全部输入提升。'],
        };
      const reference = active[0],
        axis = axisIndex(dim, reference.shape.length);
      for (const value of active.slice(1)) {
        requireThat(
          value.shape.length === reference.shape.length,
          'Cat 需要相同维数，只有一维 [0] 空 Tensor 可忽略。',
        );
      }
      checkSharedAxes(active, warnings, '非拼接轴必须相等，Cat 不进行广播', axis);
      const shape = [...reference.shape];
      shape[axis] = summedDimensions(active.map((value) => value.shape[axis]));
      const axes = [...reference.axes];
      if (axisName.trim()) axes[axis] = axisName.trim();
      return {
        outputs: { out: tensor(shape, axes, dtype) },
        warnings,
        notes: [`沿第 ${axis} 轴拼接 ${inputs.length} 路 Tensor，其余维度不变。`],
      };
    },
  },
  {
    ...base,
    type: 'Stack',
    label: 'Stack',
    description:
      '对应 torch.stack，将 1–16 路同形 Tensor 堆叠到一个新轴；新轴长度为输入数量，不进行广播。',
    infer: (values, params) => {
      const inputs = readInputs(values, params),
        dim = dimension(params);
      const reference = inputs[0],
        axis = axisIndex(dim, reference.shape.length + 1);
      const warnings: string[] = [];
      for (const value of inputs.slice(1)) {
        requireThat(
          value.shape.length === reference.shape.length,
          'Stack 需要每路 Tensor 的形状相同。',
        );
      }
      checkSharedAxes(inputs, warnings, 'Stack 各轴必须相等，不进行广播');
      const shape = [...reference.shape],
        axes = [...reference.axes];
      shape.splice(axis, 0, inputs.length);
      axes.splice(axis, 0, 'stack');
      return {
        outputs: { out: tensor(shape, axes, joinedDtype(inputs)) },
        warnings,
        notes: [`在第 ${axis} 轴插入大小为 ${inputs.length} 的新维度，原有轴保持不变。`],
      };
    },
  },
];
