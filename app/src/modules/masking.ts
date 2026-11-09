import type { Dim, TensorSpec } from '../core/types';
import { product, requireThat, same } from './math';

/** A boolean mask consumes matching leading axes, not broadcast axes. */
export function maskedSelection(
  x: TensorSpec,
  mask: TensorSpec,
  symbol: unknown,
  warnings: string[],
) {
  const rank = mask.shape.length;
  requireThat(rank >= 1 && rank <= x.shape.length, 'bool 掩码需要 1 到输入 rank 个维度。');
  requireThat(mask.dtype === 'bool', '掩码必须是 bool Tensor。');
  requireThat(
    typeof symbol === 'string' && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(symbol),
    '有效数量需要合法的符号名称。',
  );
  mask.shape.forEach((d, i) => same(d, x.shape[i], `掩码第 ${i} 轴必须等于目标对应轴`, warnings));
  const capacity = mask.shape.includes(0) ? 0 : product(mask.shape);
  const count: Dim = capacity === 0 ? 0 : symbol;
  return {
    shape: [count, ...x.shape.slice(rank)],
    axes: [rank === 1 ? 'valid_agent' : 'selected', ...x.axes.slice(rank)],
    capacity,
    count,
  };
}

export function assignmentShape(
  target: Dim[],
  values: Dim[],
  broadcast: boolean,
  capacity: Dim,
  warnings: string[],
) {
  if (!broadcast)
    requireThat(values.length === target.length, '回填值 rank 必须等于索引结果；广播请显式开启。');
  requireThat(values.length <= target.length, '回填值 rank 不能超过索引结果。');
  const padded = [...Array(target.length - values.length).fill(1), ...values];
  padded.forEach((v, i) => {
    if (broadcast && v === 1) return;
    if (i === 0 && typeof v === 'number') {
      requireThat(Number.isSafeInteger(v) && v >= 0, '回填行数必须为非负整数。');
      if (typeof capacity === 'number') requireThat(v <= capacity, '回填行数不能超过目标行数。');
    }
    same(
      v,
      target[i],
      i === 0 ? '回填行数必须等于 count_nonzero(mask)' : '回填值尾轴必须匹配目标',
      warnings,
    );
  });
}
