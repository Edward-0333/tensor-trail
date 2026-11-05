import type { ModuleDefinition, Params } from '../core/types';
import { calculate, calculationLabel, type Calculation } from './calculation';
import { input, output, text, bool } from './primitives';
import { requireThat } from './math';

export function anyOperation(params: Params): Extract<Calculation, { op: 'any' }> {
  requireThat(typeof params.dim === 'string', '归约轴需填写整数，或留空表示全部轴。');
  const dim = params.dim.trim();
  requireThat(
    dim === '' || (/^-?\d+$/.test(dim) && Number.isSafeInteger(Number(dim))),
    '归约轴需填写整数，或留空表示全部轴。',
  );
  requireThat(typeof params.keepdim === 'boolean', 'keepdim 必须为开关值。');
  return { op: 'any', dim: dim === '' ? null : Number(dim), keepdim: params.keepdim };
}

export function anyLabel(params: Params): string {
  return calculationLabel(anyOperation(params));
}

export const logicalModules: ModuleDefinition[] = [
  {
    type: 'Any',
    label: '任意为真 Any',
    category: '逻辑与归约',
    description:
      '沿指定轴判断是否存在非零（True）元素。dim 留空表示全部轴；keepdim 保留被归约的轴。输出通常为 bool，uint8 输入按 PyTorch 约定保留 uint8。',
    inputs: input,
    outputs: output,
    fields: [
      { ...text('dim', '归约轴（留空表示全部）', '-1'), maxLength: 32 },
      bool('keepdim', '保留归约轴', false),
    ],
    infer: ({ in: source }, params) => ({
      outputs: { out: calculate(source, anyOperation(params)) },
      notes: ['任一元素非零则结果为真；空轴归约结果为假。当前仅推导形状和类型。'],
    }),
  },
  {
    type: 'Invert',
    label: '取反 ~',
    category: '逻辑与归约',
    description:
      '对应 PyTorch 的 ~：bool 输入交换 True/False，整数输入逐位取反（不是取负数）。保持形状和数据类型，浮点 Tensor 不支持。',
    inputs: input,
    outputs: output,
    fields: [],
    infer: ({ in: source }) => ({
      outputs: { out: calculate(source, { op: 'invert' }) },
      notes: [
        source.dtype === 'bool' ? 'True ↔ False；形状不变。' : '逐位取反，保留整数类型及形状。',
      ],
    }),
  },
];
