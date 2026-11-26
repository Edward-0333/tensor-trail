import { shortText } from '../core/limits';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { tensorGradState } from '../core/tensorAppearance';
import type { Hyperparameters, ModuleDefinition, TensorSpec } from '../core/types';
import { isTensorSpec } from '../core/types';
import { requireThat } from './math';
import { select, text } from './primitives';

const dtypes = ['auto', 'float32', 'float16', 'float64', 'int64', 'uint8', 'bool'];
const gradModes = ['auto', 'on', 'off'];

function portNames(value: unknown, max: number): string[] | undefined {
  if (typeof value !== 'string') return;
  const ids = value.split(',').map((id) => id.trim());
  if (
    ids.length <= max &&
    new Set(ids).size === ids.length &&
    ids.every(
      (id) =>
        /^[A-Za-z_]\w{0,99}$/.test(id) && !['__proto__', 'constructor', 'prototype'].includes(id),
    )
  )
    return ids;
}
const namedPorts = (ids: string[]) => ids.map((id) => ({ id, label: id }));

const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

function outputSpecs(
  raw: unknown,
  outputs: string[],
): Map<string, Record<string, unknown>> | undefined {
  requireThat(
    shortText(raw, 4000),
    '多输出规格需要 4000 字以内的 JSON 文本。',
  );
  if (!raw.trim()) return;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw Error('多输出规格 JSON 格式无效。');
  }
  requireThat(isObject(parsed), '多输出规格需要按端口名填写 JSON 对象。');
  const specs = new Map<string, Record<string, unknown>>();
  for (const [port, spec] of Object.entries(parsed)) {
    requireThat(outputs.includes(port), `多输出规格引用了未声明的端口 ${port}。`);
    requireThat(
      isObject(spec) &&
        'shape' in spec &&
        Object.keys(spec).every((key) => ['shape', 'axes', 'dtype', 'requires_grad'].includes(key)),
      `输出 ${port} 需要包含 shape 的规格对象。`,
    );
    specs.set(port, spec);
  }
  return specs;
}

function outputTensor(
  spec: Record<string, unknown>,
  tensors: TensorSpec[],
  hyperparameters?: Hyperparameters,
): TensorSpec {
  const { shape, axes = '', dtype = 'auto', requires_grad = 'auto' } = spec;
  requireThat(typeof shape === 'string' && typeof axes === 'string', 'shape 和轴含义需要文本。');
  requireThat(typeof dtype === 'string' && dtypes.includes(dtype), '输出 dtype 选项无效。');
  const grad = requires_grad === true ? 'on' : requires_grad === false ? 'off' : requires_grad;
  requireThat(
    typeof grad === 'string' && gradModes.includes(grad),
    '输出 requires_grad 选项无效。',
  );
  const primary = tensors[0];
  const declared = shape.trim();
  requireThat(declared || primary, '没有 Tensor 输入可继承，请声明输出 shape。');
  const dims = !declared
    ? [...primary.shape]
    : declared === '[]'
      ? []
      : declared
          .replace(/^\[(.*)\]$/, '$1')
          .split(',')
          .map((token) => resolveIntegerExpression(token.trim(), hyperparameters));
  requireThat(
    dims.length <= 8 && dims.every((dim) => typeof dim !== 'number' || dim >= 0),
    '输出 shape 最多 8 维，尺寸须为非负整数。',
  );
  const labels = axes.trim()
    ? axes.split(',').map((axis) => axis.trim())
    : declared
      ? dims.map((_, index) => `axis${index}`)
      : [...primary.axes];
  requireThat(
    labels.length === dims.length && labels.every(Boolean),
    '轴名称数量必须与输出 shape 的维数一致。',
  );
  const output = {
    shape: dims,
    axes: labels,
    dtype: dtype === 'auto' ? (primary?.dtype ?? 'float32') : dtype,
  };
  const floating = /^(?:b?float)/.test(output.dtype);
  requireThat(grad !== 'on' || floating, '只有浮点输出可设 requires_grad=on。');
  if (grad !== 'auto')
    return { ...output, autograd: { requiresGrad: grad === 'on', source: 'manual' } };
  if (!floating || (tensors.length && tensors.every((t) => tensorGradState(t) === 'off')))
    return { ...output, autograd: { requiresGrad: false, source: 'inferred' } };
  return output;
}

export const semanticOp: ModuleDefinition = {
  type: 'SemanticOp',
  label: '语义步骤',
  category: '组合计算',
  composite: false,
  description: '用摘要、输入依赖和输出 Tensor 规格描述一个语义步骤。',
  fields: [
    { ...text('summary', '语义说明', '语义计算步骤'), maxLength: 240 },
    { ...text('formula', '公式 / 代码摘要（可选）', ''), maxLength: 240 },
    text('inputs', '输入端口（逗号分隔）', 'x'),
    text('outputs', '输出端口（逗号分隔）', 'out'),
    text('shape', '单输出 shape（留空继承首输入）', ''),
    text('axes', '单输出轴含义（留空继承）', ''),
    select('dtype', '单输出 dtype', 'auto', dtypes),
    select('requires_grad', '梯度状态', 'auto', gradModes),
    { ...text('output_specs', '多输出规格 JSON（可选）', ''), maxLength: 4000 },
  ],
  inputs: namedPorts(['x']),
  outputs: namedPorts(['out']),
  getPorts: (p) => ({
    inputs: namedPorts(portNames(p.inputs, 12) ?? ['x']),
    outputs: namedPorts(portNames(p.outputs, 8) ?? ['out']),
  }),
  inferValue: (values, p, context) => {
    const { summary, formula } = p;
    requireThat(
      shortText(summary, 240) && summary.trim(),
      '语义说明需填写 1–240 个字符。',
    );
    requireThat(
      shortText(formula, 240),
      '精简表达式最多 240 个字符。',
    );
    const inputs = portNames(p.inputs, 12),
      outputs = portNames(p.outputs, 8);
    requireThat(inputs && outputs, '端口名需要唯一的标识符，输入 1–12 个，输出 1–8 个。');
    const tensors = inputs
      .map((id) => {
        requireThat(Object.hasOwn(values, id), `请连接输入 ${id}。`);
        return values[id];
      })
      .filter(isTensorSpec);
    const specs = outputSpecs(p.output_specs, outputs);
    return {
      outputs: Object.fromEntries(
        outputs.map((id) => {
          const spec = specs ? (specs.get(id) ?? { shape: '' }) : p;
          return [id, outputTensor(spec, tensors, context?.hyperparameters)];
        }),
      ),
      notes: [summary, ...(formula.trim() ? [formula] : [])],
    };
  },
};
