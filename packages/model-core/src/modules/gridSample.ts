import { effectiveGradMode, inferAutogradFromInputs } from '../core/tensorAppearance';
import type { Dim, ModuleDefinition } from '../core/types';
import { requireThat, same, tensor } from './math';
import { bool, output, select } from './primitives';

const samplingDtypes = ['float16', 'bfloat16', 'float32', 'float64'];

function checkSpatialDims(dims: Dim[], label: string) {
  requireThat(
    dims.every(dim => typeof dim !== 'number' || (Number.isInteger(dim) && dim > 0)),
    `GridSample 的 ${label}必须是正整数，实际为 [${dims.join(', ')}]。`,
  );
}

export const gridSample: ModuleDefinition = {
  type: 'GridSample',
  label: 'GridSample',
  category: '基础层',
  description: '对应 F.grid_sample，按 grid 的归一化坐标对 2D 或 3D 特征图采样。',
  inputs: [{ id: 'in', label: '特征张量' }, { id: 'grid', label: '采样网格坐标' }],
  outputs: output,
  fields: [
    select('mode', '插值模式', 'bilinear', ['bilinear', 'nearest', 'bicubic']),
    select('padding_mode', '越界填充', 'zeros', ['zeros', 'border', 'reflection']),
    bool('align_corners', '对齐角像素中心', false),
  ],
  infer: ({ in: x, grid }, params, context) => {
    const rank = x.shape.length;
    requireThat(rank === 4 || rank === 5, 'GridSample 的 in 需要 4D 或 5D 特征张量。');
    requireThat(grid.shape.length === rank, `GridSample 的 grid 必须与 in 同为 ${rank}D。`);
    requireThat(rank !== 5 || params.mode !== 'bicubic', 'GridSample 的 bicubic 插值仅支持 4D 输入，5D 请选择 bilinear 或 nearest。');
    for (const input of [x, grid]) {
      requireThat(samplingDtypes.includes(input.dtype), `GridSample 需要浮点 Tensor，当前为 ${input.dtype}。`);
    }
    requireThat(x.dtype === grid.dtype, 'GridSample 的 in 与 grid 的 dtype 必须一致。');

    const warnings: string[] = [];
    const inBatch = x.shape[0], gridBatch = grid.shape[0];
    const last = grid.shape[rank - 1], expectedLast = rank - 2;
    if (typeof last === 'string' && (last === inBatch || last === gridBatch)) {
      const otherBatch = last === inBatch ? gridBatch : inBatch;
      requireThat(typeof otherBatch !== 'number' || otherBatch === expectedLast,
        `GridSample 的 batch 维要求 ${last} = ${otherBatch}，grid 末维要求 ${last} = ${expectedLast}，两项约束冲突。`);
    }
    same(inBatch, gridBatch, 'in 与 grid 的 batch 维必须相等', warnings);
    same(last, expectedLast, `grid 末维应为 ${expectedLast} 个坐标分量`, warnings);
    const outputSpatial = grid.shape.slice(1, -1);
    checkSpatialDims(x.shape.slice(2), 'in 空间维');
    checkSpatialDims(outputSpatial, 'grid 输出空间维');

    const shape = [...x.shape.slice(0, 2), ...outputSpatial];
    const axes = shape.map((_, i) => (i < 2 ? x.axes[i] : grid.axes[i - 1]) ?? `axis${i}`);
    const out = tensor(shape, axes, x.dtype);
    const gradMode = effectiveGradMode(context);
    const autograd = inferAutogradFromInputs([x, grid], gradMode);
    if (autograd) out.autograd = autograd;
    return { outputs: { out }, warnings };
  },
};
