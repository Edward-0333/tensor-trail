import type {
  InferenceContext,
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
export function effectiveGradMode(context?: InferenceContext) {
  return context?.runtimeGradMode ?? context?.hyperparameters.gradMode?.value ?? 'grad';
}

function isGradDisabled(gradMode: unknown): boolean {
  return gradMode === 'no_grad' || gradMode === 'inference_mode';
}
/** For fresh, parameter-free Tensor results that are differentiable in every input. */
export function inferAutogradFromInputs(inputs: TensorSpec[], gradMode: unknown): TensorAutograd | undefined {
  if (isGradDisabled(gradMode) || inputs.every(input => tensorGradState(input) === 'off'))
    return { requiresGrad: false, source: 'inferred' };
  if (gradMode === 'grad' && inputs.some(input => tensorGradState(input) === 'on'))
    return { requiresGrad: true, source: 'inferred' };
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

// These layers infer their own output gradients under the effective grad mode.
const gradModeAware = new Set(['GridSample', 'Normalize']);
// An explicit grad:on requires a floating output and a known grad-enabled context.
const gradOnRequiresGradMode = new Set(['Linspace', 'OnesLike', 'ZerosLike', 'Maximum']);
const valueReads = new Set([
  ...gradModeAware,
  'SemanticOp', // Each output declares or infers its own autograd state.
  'TensorFieldContract', 'ArrayToTensorExplicit', 'SelectValue', 'TensorToDevice', 'ZerosLike', 'OnesLike', 'Linspace',
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
  'NewOnes',
  'BufferSource',
  'OnesScaled',
  'Arange',
  'Detach',
  'NoGradResult',
  'BatchMetaTensor',
  'ArrayToTensor',
  'TensorRowAt', 'Argmax', 'ScalarIndexWrite', 'TensorEqualInt',
  'BranchValue', 'ZeroTailColumn', 'SliceRowsFrom', 'WriteTailRows', 'MergeBranches',
]);
const shapeOnly = new Set([
  'Unflatten', 'DynamicSlice', 'ReshapeLike',
  'BooleanIndexAxis',
  'PointRangeTransform',
  'ReduceMean',
  'View',
  'Reshape',
  'Permute',
  'Contiguous',
  'ExpandAs',
  'Flatten',
  'Unsqueeze',
  'Squeeze',
  'Repeat',
  'Slice',
  'TensorIndex',
  'AlignTrailingDims',
  'Cast',
]);
const parameterFreeMath = new Set(['MatrixInverse', 'Exp', 'Softplus', 'L2Norm', 'Add', 'Mul', 'Subtract', 'Dropout', 'Function', 'NanToNum', 'MatMul', 'ReduceSum', 'Divide', 'Maximum', 'Concat', 'SortRowsByMax', 'MaxWithIndices', 'Sort', 'RowIndex']);
const differentiableUnary = new Set(['Sigmoid', 'Log', 'Sqrt', 'Clamp', 'Chunk', 'TopK']);

/** Conservative metadata propagation, not a PyTorch autograd simulator. */
export function annotateTensorOutputs(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  outputs: Record<string, ValueSpec>,
  gradMode: unknown = 'grad',
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
      if (['MaxWithIndices', 'Sort', 'TopK'].includes(node.type) && port === 'indices')
        return [port, { ...tensor, autograd: { requiresGrad: false, source: 'inferred' as const } }];
      if (differentiableUnary.has(node.type) && source && isTensorSpec(source)) {
        const state = tensorGradState(source);
        if (state !== 'unknown')
          return [port, { ...tensor, autograd: { requiresGrad: state === 'on', source: 'inferred' as const } }];
      }
      if (shapeOnly.has(node.type) && source && isTensorSpec(source) && source.autograd?.requiresGrad === false)
        return [port, { ...tensor, autograd: { requiresGrad: false, source: 'inferred' as const } }];
      if (parameterFreeMath.has(node.type)) {
        const operands = Object.values(inputs);
        if (operands.length && operands.every(isTensorSpec)) {
          // Older arithmetic rules establish only known-off. Maximum models both input gradients.
          const autograd = inferAutogradFromInputs(operands, node.type === 'Maximum' ? gradMode : undefined);
          if (autograd) return [port, { ...tensor, autograd }];
        }
      }
      return [port, tensor];
    }),
  );
  if (node.tensorAutograd) {
    const out = result.out;
    if (!out || !isTensorSpec(out)) throw new Error('自动求导标注仅适用于 out Tensor。');
    const wouldEnableDisabledGrad = node.tensorAutograd.requiresGrad && (
      (gradModeAware.has(node.type) && isGradDisabled(gradMode)) ||
      (gradOnRequiresGradMode.has(node.type) && (gradMode !== 'grad' || !canRequireGrad(out.dtype)))
    );
    validateAutograd(node.tensorAutograd, wouldEnableDisabledGrad ? undefined : out.dtype);
    if (!wouldEnableDisabledGrad)
      result.out = { ...out, autograd: { ...node.tensorAutograd } };
  }
  return result;
}
