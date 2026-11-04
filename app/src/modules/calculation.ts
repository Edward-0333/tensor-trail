import type { ModuleDefinition, TensorSpec } from '../core/types';
import { axisIndex, requireThat, tensor } from './math';
import { input, output, num, text } from './primitives';

export type Calculation =
  | { op: 'invert' }
  | { op: 'any'; dim: number | null; keepdim: boolean }
  | { op: 'slice'; dim: number; start: string; stop: string; step: number };
export const defaultSlice: Calculation = { op: 'slice', dim: -1, start: '', stop: '2', step: 1 };
export const calculationLabel = (op: Calculation) =>
  op.op === 'invert'
    ? '~'
    : op.op === 'any'
      ? `any(${op.dim ?? ''}${op.keepdim ? ', keepdim' : ''})`
      : `${op.dim === -1 ? '[..., ' : `[轴${op.dim}: `}${op.start}:${op.stop}${op.step === 1 ? '' : `:${op.step}`}]`;

function bound(value: unknown): string {
  requireThat(
    typeof value === 'string' && (value === '' || /^-?\d+$/.test(value)),
    '切片边界应为整数或留空。',
  );
  requireThat(value === '' || Number.isSafeInteger(Number(value)), '切片边界超出安全整数范围。');
  return value;
}
export function parseCalculations(value: unknown): Calculation[] {
  requireThat(typeof value === 'string' && value.length < 2000, '计算步骤必须为短 JSON 文本。');
  const ops: unknown = JSON.parse(value);
  requireThat(
    Array.isArray(ops) && ops.length > 0 && ops.length <= 16,
    '计算模块需要 1–16 个运算。',
  );
  return ops.map((value) => {
    requireThat(value && typeof value === 'object' && !Array.isArray(value), '计算步骤格式无效。');
    const v = value as Record<string, unknown>;
    const allowed =
      v.op === 'invert'
        ? ['op']
        : v.op === 'any'
          ? ['op', 'dim', 'keepdim']
          : ['op', 'dim', 'start', 'stop', 'step'];
    requireThat(
      Object.keys(v).every((k) => allowed.includes(k)),
      '计算步骤存在未知字段。',
    );
    if (v.op === 'invert') return { op: 'invert' };
    requireThat(
      (v.dim === null && v.op === 'any') || Number.isSafeInteger(v.dim),
      '运算轴必须为整数，any 可选全部轴。',
    );
    if (v.op === 'any') {
      requireThat(typeof v.keepdim === 'boolean', 'keepdim 必须为开关值。');
      return { op: 'any', dim: v.dim as number | null, keepdim: v.keepdim };
    }
    requireThat(v.op === 'slice', '不支持此运算；可选 invert、any、slice。');
    requireThat(Number.isSafeInteger(v.step) && Number(v.step) > 0, '切片步长必须为正整数。');
    return {
      op: 'slice',
      dim: v.dim as number,
      start: bound(v.start),
      stop: bound(v.stop),
      step: Number(v.step),
    };
  });
}

export function calculate(x: TensorSpec, op: Calculation): TensorSpec {
  if (op.op === 'invert') {
    requireThat(
      ['bool', 'int64'].includes(x.dtype),
      '~ 取反需要 bool 或 int64 输入；浮点 Tensor 不支持按位取反。',
    );
    return structuredClone(x);
  }
  if (op.op === 'any') {
    if (op.dim === null)
      return tensor(op.keepdim ? x.shape.map(() => 1) : [], op.keepdim ? [...x.axes] : [], 'bool');
    const axis = axisIndex(op.dim, x.shape.length);
    return tensor(
      op.keepdim
        ? x.shape.map((d, i) => (i === axis ? 1 : d))
        : x.shape.filter((_, i) => i !== axis),
      op.keepdim ? [...x.axes] : x.axes.filter((_, i) => i !== axis),
      'bool',
    );
  }
  requireThat(x.shape.length > 0, '标量 Tensor 没有可切片的轴。');
  const axis = axisIndex(op.dim, x.shape.length),
    size = x.shape[axis];
  const shape = [...x.shape];
  if (typeof size === 'number') {
    const normalize = (v: string, fallback: number) =>
      v === ''
        ? fallback
        : Math.min(size, Math.max(0, Number(v) < 0 ? size + Number(v) : Number(v)));
    const start = normalize(op.start, 0),
      stop = normalize(op.stop, size);
    shape[axis] = Math.max(0, Math.ceil((stop - start) / op.step));
  } else shape[axis] = `slice(${size},${op.start}:${op.stop}:${op.step})`;
  return tensor(shape, [...x.axes], x.dtype);
}

export const calculationModules: ModuleDefinition[] = [
  {
    type: 'Slice',
    label: '切片 Slice',
    category: '维度操作',
    description: '沿指定轴切片，默认 [..., :2]；负数边界从轴末尾计数，留空表示起点或终点。',
    inputs: input,
    outputs: output,
    fields: [
      num('dim', '切片轴', -1, -8),
      text('start', '开始（留空从头）', ''),
      text('stop', '结束（不含，留空到末尾）', '2'),
      num('step', '步长', 1),
    ],
    infer: ({ in: x }, p) => {
      const op = parseCalculations(JSON.stringify([{ op: 'slice', ...p }]))[0];
      const out = calculate(x, op);
      return {
        outputs: { out },
        notes: [calculationLabel(op)],
        warnings: out.shape.some((d) => typeof d === 'string' && d.startsWith('slice('))
          ? ['符号维度的切片长度需要具体尺寸验证。']
          : [],
      };
    },
  },
  {
    type: 'Compute',
    label: '计算模块',
    category: '基础层',
    description: '按顺序组合切片、~ 取反和 any()；当前仅推导形状与类型，不执行任意代码。',
    inputs: input,
    outputs: output,
    fields: [
      {
        ...text('operations', '计算步骤', JSON.stringify([defaultSlice])),
        help: 'JSON 数组，1–16 步。invert: {"op":"invert"}；any: {"op":"any","dim":null,"keepdim":false}；slice: {"op":"slice","dim":-1,"start":"","stop":"2","step":1}。',
      },
    ],
    infer: ({ in: x }, p) => {
      let out = structuredClone(x);
      const notes: string[] = [],
        warnings: string[] = [];
      for (const [i, op] of parseCalculations(p.operations).entries()) {
        try {
          out = calculate(out, op);
        } catch (e) {
          throw new Error(`第 ${i + 1} 步 ${calculationLabel(op)}：${(e as Error).message}`);
        }
        notes.push(`${i + 1}. ${calculationLabel(op)} → [${out.shape.join(', ')}] · ${out.dtype}`);
      }
      if (out.shape.some((d) => typeof d === 'string' && d.startsWith('slice(')))
        warnings.push('符号维度的切片长度需要具体尺寸验证。');
      return { outputs: { out }, notes, warnings };
    },
  },
];
