import { effectiveGradMode, inferAutogradFromInputs } from '../core/tensorAppearance';
import type { ModuleDefinition, Params } from '../core/types';
import { axisIndex, requireThat, same, tensor } from './math';
import { input, num, output, text } from './primitives';

// At half the smallest subnormal, ties round to zero. At the midpoint between
// the largest finite value and the next power of two, ties round to infinity.
const roundingMidpoints: Partial<Record<string, { zero: number; overflow: number }>> = {
  float16: { zero: 2 ** -25, overflow: 65520 },
  bfloat16: { zero: 2 ** -134, overflow: 2 ** 128 - 2 ** 119 },
};

function statisticCount(value: unknown, field: 'mean' | 'std', dtype: string): number {
  requireThat(typeof value === 'string', `Normalize 的 ${field} 需要逗号分隔的浮点数。`);
  const items = value.split(',');
  const bounds = roundingMidpoints[dtype];
  // PyTorch converts list elements through float32 on the way to float16 or bfloat16.
  const roundToFloat32 = dtype === 'float32' || bounds !== undefined;
  for (const item of items) {
    const token = item.trim();
    requireThat(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(token),
      `Normalize 的 ${field} 每项必须是有限浮点数，当前项为 "${token}"。`);
    const n = Number(token);
    const magnitude = Math.abs(roundToFloat32 ? Math.fround(n) : n);
    requireThat(magnitude < (bounds?.overflow ?? Infinity),
      `Normalize 的 ${field} 项 "${token}" 转为 ${dtype} 时溢出。`);
    if (field === 'std') {
      requireThat(magnitude > (bounds?.zero ?? 0),
        `Normalize 的 std 项 "${token}" 转为 ${dtype} 后为 0，标准差必须大于 0。`);
      requireThat(n > 0, 'Normalize 的 std 每项必须大于 0。');
    }
  }
  return items.length;
}

export function normalizeChannelCount(params: Params, dtype = 'float64'): number {
  const count = statisticCount(params.mean, 'mean', dtype);
  requireThat(count === statisticCount(params.std, 'std', dtype), 'Normalize 的 mean 和 std 长度必须一致。');
  return count;
}

export const normalize: ModuleDefinition = {
  type: 'Normalize',
  label: 'Normalize',
  category: '基础层',
  description: '对应 torchvision.transforms.Normalize(mean, std)，沿通道轴逐项计算 (x - mean) / std。',
  inputs: input,
  outputs: output,
  fields: [
    num('dim', '通道轴', -3, Number.MIN_SAFE_INTEGER),
    text('mean', '通道均值', '0.485, 0.456, 0.406'),
    text('std', '通道标准差', '0.229, 0.224, 0.225'),
  ],
  infer: ({ in: x }, params, context) => {
    requireThat(['float16', 'bfloat16', 'float32', 'float64'].includes(x.dtype),
      `Normalize 需要浮点 Tensor，当前为 ${x.dtype}。`);
    requireThat(typeof params.dim === 'number', 'Normalize 的 dim 必须是整数。');
    const axis = axisIndex(params.dim, x.shape.length);
    const count = normalizeChannelCount(params, x.dtype);
    const warnings: string[] = [];
    if (count > 1) same(x.shape[axis], count, 'Normalize 的通道数必须与统计量长度一致', warnings);

    const out = tensor([...x.shape], [...x.axes], x.dtype);
    const gradMode = effectiveGradMode(context);
    const autograd = inferAutogradFromInputs([x], gradMode);
    if (autograd) out.autograd = autograd;
    return { outputs: { out }, warnings };
  },
};
