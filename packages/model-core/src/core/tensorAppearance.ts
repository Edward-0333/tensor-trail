import type {
  ModelNode,
  TensorAutograd,
  TensorGradState,
  TensorPalette,
  TensorSpec,
  ValueSpec,
} from './types';
import { isTensorSpec } from './types';

export const DEFAULT_TENSOR_PALETTE: Readonly<TensorPalette> = {
  on: '#ccece6',
  off: '#e2e8f0',
  unknown: '#f5e6c8',
};
export const isHexColor = (value: unknown): value is string =>
  typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value);
export const canRequireGrad = (dtype: string): boolean => /^(float|bfloat|complex)/.test(dtype);
export function validateAutograd(value: unknown, dtype?: string): asserts value is TensorAutograd {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('自动求导状态必须包含 requiresGrad 和 source。');
  const v = value as Record<string, unknown>;
  if (
    Object.keys(v).some((key) => !['requiresGrad', 'source'].includes(key)) ||
    typeof v.requiresGrad !== 'boolean' ||
    !['manual', 'runtime', 'inferred'].includes(String(v.source))
  )
    throw new Error('自动求导状态需要布尔 requiresGrad，来源为 manual、runtime 或 inferred。');
  if (v.requiresGrad && dtype && !canRequireGrad(dtype))
    throw new Error('整数或 bool Tensor 不能设置 requires_grad=True。');
}
export function tensorGradState(spec?: TensorSpec): TensorGradState {
  if (!spec) return 'unknown';
  if (!canRequireGrad(spec.dtype)) return 'off';
  return spec.autograd ? (spec.autograd.requiresGrad ? 'on' : 'off') : 'unknown';
}
export const gradLabel = (state: TensorGradState): string =>
  `grad:${state === 'unknown' ? '?' : state}`;
export const gradSourceLabel = (spec: TensorSpec): string =>
  spec.autograd
    ? { manual: '手工标注', runtime: '运行时采集', inferred: '静态推导' }[spec.autograd.source]
    : !canRequireGrad(spec.dtype)
      ? '数据类型约束'
      : '尚未确定';
export function tensorInk(background: string): string {
  const rgb = [1, 3, 5].map((offset) => {
    const c = parseInt(background.slice(offset, offset + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722 > 0.179 ? '#000000' : '#ffffff';
}

const valueReads = new Set([
  'Input',
  'Inlet',
  'Outlet',
  'Output',
  'Identity',
  'TensorVariable',
  'ValueVariable',
  'DictionaryGet',
  'TensorSequenceGet',
  // Construction declares requires_grad; in-place initialization does not change it.
  'EmbeddingWeight',
  'UniformInitColumns',
  'GetParameter',
  'GetAttribute',
  // This factory declares its own requires_grad, independently of its reference.
  'NewZeros',
  'BufferSource',
  'OnesScaled',
  'Arange',
  'TensorRowAt', 'Argmax', 'ScalarIndexWrite', 'TensorEqualInt',
  'BranchValue', 'ZeroTailColumn', 'SliceRowsFrom', 'WriteTailRows', 'MergeBranches',
]);
const shapeOnly = new Set([
  'View',
  'Reshape',
  'Permute',
  'Contiguous',
  'Flatten',
  'Unsqueeze',
  'Squeeze',
  'Repeat',
  'Slice',
  'TensorIndex',
  'AlignTrailingDims',
  'Cast',
]);
const parameterFreeMath = new Set(['ReduceSum', 'Divide', 'Concat', 'SortRowsByMax', 'MaxWithIndices', 'Sort', 'RowIndex']);

/** Conservative metadata propagation, not a PyTorch autograd simulator. */
export function annotateTensorOutputs(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  outputs: Record<string, ValueSpec>,
): Record<string, ValueSpec> {
  const result = Object.fromEntries(
    Object.entries(outputs).map(([port, value]) => {
      if (!isTensorSpec(value) || node.subgraph || valueReads.has(node.type)) return [port, value];
      const source = inputs.in;
      if (node.type === 'Cast' && source && isTensorSpec(source) && source.dtype === value.dtype)
        return [port, { ...value,
          ...(source.autograd ? { autograd: source.autograd } : {}),
          ...(source.parameterRef ? { parameterRef: source.parameterRef } : {}),
          ...(source.bufferRef ? { bufferRef: source.bufferRef } : {}),
        }];
      const { autograd: _previous, parameterRef: _parameter, bufferRef: _buffer, ...tensor } = value;
      if (['MaxWithIndices', 'Sort'].includes(node.type) && port === 'indices')
        return [port, { ...tensor, autograd: { requiresGrad: false, source: 'inferred' as const } }];
      const input = inputs.in;
      // Known False is safe through shape/dtype operations and parameter-free arithmetic.
      // True cannot be inferred without knowing grad mode and view/checkpoint context.
      if (
        (shapeOnly.has(node.type) &&
        input &&
        isTensorSpec(input) &&
        input.autograd?.requiresGrad === false) ||
        (parameterFreeMath.has(node.type) && Object.values(inputs).length > 0 &&
          Object.values(inputs).every(value => isTensorSpec(value) &&
            (value.autograd?.requiresGrad === false || !canRequireGrad(value.dtype))))
      )
        return [
          port,
          { ...tensor, autograd: { requiresGrad: false, source: 'inferred' as const } },
        ];
      return [port, tensor];
    }),
  );
  if (node.tensorAutograd) {
    const out = result.out;
    if (!out || !isTensorSpec(out)) throw new Error('自动求导标注仅适用于 out Tensor。');
    validateAutograd(node.tensorAutograd, out.dtype);
    result.out = { ...out, autograd: { ...node.tensorAutograd } };
  }
  return result;
}
