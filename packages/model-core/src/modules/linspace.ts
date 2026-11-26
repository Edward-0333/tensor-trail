import { ParameterError, type ModelNode, type ModuleDefinition, type Params } from '../core/types';
import { shortText } from '../core/limits';
import { decimalLiteral, tensor } from './math';
import { num, output, select, text } from './primitives';

const int64Min = -(1n << 63n);
const int64Max = (1n << 63n) - 1n;
const uint64Max = (1n << 64n) - 1n;
const ranges: Record<string, readonly [number | bigint, number | bigint]> = {
  float16: [-65504, 65504],
  bfloat16: [-3.3895313892515355e38, 3.3895313892515355e38],
  float32: [-3.4028234663852886e38, 3.4028234663852886e38],
  float64: [-Number.MAX_VALUE, Number.MAX_VALUE],
  int32: [-2147483648, 2147483647],
  int64: [int64Min, int64Max],
};

function bindEndpoint(field: 'start' | 'end', text: unknown) {
  const literal = decimalLiteral(text, field);
  // Python binds integer literals exactly and decimal/exponent literals as binary64.
  const value = /[.eE]/.test(literal) ? Number(literal) : BigInt(literal);
  if (typeof value === 'bigint' && (value < int64Min || value > uint64Max))
    throw new ParameterError(field, `Linspace 的 ${field} 整数字面量超出标量绑定范围 [${int64Min}, ${uint64Max}]。`);
  return { field, literal, value };
}

function linspaceParams(p: Params) {
  const { steps, dtype, axis } = p;
  if (typeof steps !== 'number' || !Number.isSafeInteger(steps) || steps < 0)
    throw new ParameterError('steps', 'Linspace 的 steps 必须为非负安全整数。');
  if (typeof dtype !== 'string' || !Object.hasOwn(ranges, dtype))
    throw new ParameterError('dtype', 'Linspace 的 dtype 无效。');
  if (!shortText(axis, 64))
    throw new ParameterError('axis', 'Linspace 的 axis 最多 64 个字符。');

  // Bind both arguments even for an empty sequence. Only consumed endpoints are cast.
  const start = bindEndpoint('start', p.start), end = bindEndpoint('end', p.end);
  const [min, max] = ranges[dtype];
  for (const endpoint of [start, end].slice(0, steps)) {
    // Check before truncating fractional values. Mixed bigint/number comparisons are exact.
    if (endpoint.value < min || endpoint.value > max)
      throw new ParameterError(endpoint.field, `Linspace 的 ${endpoint.field} 超出 ${dtype} 的有限范围。`);
  }
  return { start: start.literal, end: end.literal, steps, dtype, axis: axis.trim() || 'step' };
}

export function linspaceCall(params: Params, withDtype: boolean) {
  const { start, end, steps, dtype } = linspaceParams(params);
  return `linspace(${start}, ${end}, ${steps}${withDtype ? `, dtype=torch.${dtype}` : ''})`;
}

export function linspaceCode(node: ModelNode): string | undefined {
  if (node.type !== 'Linspace') return;
  try {
    return `out = torch.${linspaceCall(node.params, true)}\n# 静态推导一维采样轴及 dtype。`;
  } catch (error) {
    return `# ${(error as Error).message}`;
  }
}

export const linspace: ModuleDefinition = {
  type: 'Linspace', label: 'Linspace', category: '维度操作',
  description: 'torch.linspace(start, end, steps) 在两端点间生成指定步数的一维等距序列。',
  inputs: [], outputs: output,
  fields: [
    text('start', '起始值', '0.0'),
    text('end', '结束值', '1.0'),
    num('steps', '步数', 4, 0),
    select('dtype', '输出类型', 'float32', Object.keys(ranges)),
    { ...text('axis', '轴名', 'step'), maxLength: 64 },
  ],
  infer: (_, params) => {
    const { steps, axis, dtype } = linspaceParams(params);
    const out = tensor([steps], [axis], dtype);
    out.autograd = { requiresGrad: false, source: 'inferred' };
    return { outputs: { out } };
  },
};
