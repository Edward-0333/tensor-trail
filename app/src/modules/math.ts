import type { Dim, TensorSpec } from '../core/types';

export function requireThat(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
export function numericList(value: unknown): number[] {
  const tokens = String(value)
    .replace(/[\[\]()]/g, '')
    .split(',')
    .map((s) => s.trim());
  requireThat(
    tokens.length > 0 && tokens.every((s) => /^-?\d+$/.test(s)),
    '请输入逗号分隔的整数，例如 0, 2, 1。',
  );
  const values = tokens.map(Number);
  requireThat(values.every(Number.isSafeInteger), '整数超出安全范围。');
  return values;
}
export function dimensions(value: unknown): Dim[] {
  const tokens = String(value)
    .replace(/[\[\]()]/g, '')
    .split(',')
    .map((s) => s.trim());
  requireThat(tokens.length > 0 && tokens.length <= 8, 'Tensor 需有 1–8 个维度。');
  return tokens.map((s) => {
    if (/^\d+$/.test(s)) {
      const n = Number(s);
      requireThat(n > 0 && Number.isSafeInteger(n), '维度必须是安全范围内的正整数。');
      return n;
    }
    requireThat(/^[A-Za-z][A-Za-z0-9_]*$/.test(s), '维度填写正整数或符号，例如 8, T, 64。');
    return s;
  });
}
export const tensor = (shape: Dim[], axes: string[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes,
  dtype,
});
export function rank(x: TensorSpec, expected: number) {
  requireThat(
    x.shape.length === expected,
    `需要 ${expected} 维 Tensor，实际为 [${x.shape.join(', ')}]。`,
  );
}
export function floating(x: TensorSpec) {
  requireThat(
    ['float16', 'float32', 'float64'].includes(x.dtype),
    `该模块需要浮点 Tensor，当前为 ${x.dtype}。`,
  );
}
export function same(a: Dim, b: Dim, message: string, warnings: string[]) {
  if (a === b) return;
  if (typeof a === 'number' && typeof b === 'number')
    throw new Error(`${message}（${a} 与 ${b} 不匹配）。`);
  warnings.push(`${message}：需要 ${a} = ${b}，符号维度尚待验证。`);
}
export function product(dims: Dim[]): Dim {
  if (dims.every((d) => typeof d === 'number')) {
    const n = (dims as number[]).reduce((a, b) => a * b, 1);
    requireThat(Number.isSafeInteger(n), '元素数量超出安全整数范围。');
    return n;
  }
  return (
    dims
      .filter((d) => d !== 1)
      .map((d) => (typeof d === 'string' ? `(${d})` : d))
      .join('×') || 1
  );
}
export function axisIndex(axis: number, rank: number) {
  const idx = axis < 0 ? rank + axis : axis;
  requireThat(
    Number.isInteger(idx) && idx >= 0 && idx < rank,
    `轴 ${axis} 超出 ${rank} 维 Tensor 的范围。`,
  );
  return idx;
}
export function broadcast(a: TensorSpec, b: TensorSpec, warnings: string[]): TensorSpec {
  requireThat(a.dtype === b.dtype, '两路 Tensor 的 dtype 必须一致。');
  const size = Math.max(a.shape.length, b.shape.length),
    shape: Dim[] = [],
    axes: string[] = [];
  for (let i = 0; i < size; i++) {
    const ai = a.shape.length - size + i,
      bi = b.shape.length - size + i;
    const x = a.shape[ai] ?? 1,
      y = b.shape[bi] ?? 1;
    if (x !== 1 && y !== 1) same(x, y, '逐元素运算需要对应维度相等或其中一个为 1', warnings);
    shape.push(x === 1 ? y : x);
    axes.push((x === 1 ? b.axes[bi] : a.axes[ai]) ?? `axis${i}`);
  }
  return tensor(shape, axes, a.dtype);
}
