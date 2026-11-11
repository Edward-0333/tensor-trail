import type { Dim, Params, ValueSpec } from '../core/types';
import { isIntegerSpec } from '../core/types';
import { requireThat } from './math';

export function factoryShape(values: Record<string, ValueSpec>, params: Params, name: string) {
  requireThat(
    typeof params.shape === 'string' && params.shape.length <= 256,
    `${name} 需要有效的目标形状。`,
  );
  const tokens = params.shape.split(',').map((token) => token.trim());
  requireThat(
    tokens.length >= 1 && tokens.length <= 8 && tokens.every(Boolean),
    `${name} 需要 1–8 个非空尺寸。`,
  );
  const shape: Dim[] = tokens.map((token) => {
    if (/^d[0-7]$/.test(token)) {
      const value = values[token];
      requireThat(value && isIntegerSpec(value), `${token} 需要整数尺寸，不是 Tensor。`);
      requireThat(
        typeof value.value === 'number'
          ? Number.isSafeInteger(value.value) && value.value >= 0
          : typeof value.value === 'string' && value.value.trim().length > 0,
        `${name} 的尺寸必须是非负安全整数或符号尺寸。`,
      );
      return value.value;
    }
    requireThat(
      /^\+?\d+$/.test(token) && Number.isSafeInteger(Number(token)),
      `${name} 仅接受非负整数或 d0–d7，不接受 -1 或任意表达式。`,
    );
    return Number(token);
  });
  requireThat(typeof params.axes === 'string' && params.axes.length <= 256, '轴标签需要短文本。');
  const axes = params.axes.trim()
    ? params.axes.split(',').map((axis) => axis.trim())
    : shape.map((_, i) => `axis${i}`);
  requireThat(
    axes.length === shape.length && axes.every(Boolean),
    '轴标签数量必须与目标维数一致，且不能有空标签。',
  );
  return { shape, axes };
}
