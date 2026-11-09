import type { ModuleDefinition, Params } from '../core/types';
import { input, output, num } from './primitives';
import { requireThat, tensor } from './math';

export const unsqueezeLabel = (params: Params) => `unsqueeze(${params.dim})`;

export const unsqueeze: ModuleDefinition = {
  type: 'Unsqueeze',
  label: '增加维度 Unsqueeze',
  category: '维度操作',
  description:
    '在 dim 指定的位置插入一个大小为 1 的维度，保留元素和数据类型。-1 表示末尾；例如 [B,N] → [B,N,1]。',
  inputs: input,
  outputs: output,
  fields: [
    {
      ...num('dim', '插入位置', -1, -8),
      help: '输入为 n 维时，dim 可取 -n-1 到 n；0 插在开头，-1 插在末尾。',
    },
  ],
  infer: ({ in: source }, params) => {
    const rank = source.shape.length;
    requireThat(rank < 8, '当前画布最多支持 8 维 Tensor，无法继续增加维度。');
    const dim = params.dim;
    requireThat(
      typeof dim === 'number' && Number.isSafeInteger(dim) && dim >= -rank - 1 && dim <= rank,
      `Unsqueeze 的 dim 必须是 [-${rank + 1}, ${rank}] 范围内的整数。`,
    );
    const axis = dim < 0 ? rank + 1 + dim : dim;
    const shape = [...source.shape],
      axes = [...source.axes];
    shape.splice(axis, 0, 1);
    axes.splice(axis, 0, 'singleton');
    return {
      outputs: { out: tensor(shape, axes, source.dtype) },
      notes: [`在第 ${axis} 轴插入大小为 1 的维度；元素数量、数据类型及原有轴含义不变。`],
    };
  },
};
