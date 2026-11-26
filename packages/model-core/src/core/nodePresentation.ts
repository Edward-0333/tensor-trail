import type { ModelNode, NodePresentation } from './types';
import { isCodeRegion } from './ifRegions';
import { isLayoutRegion } from './layoutRegions';

export const nodePresentations = ['default', 'compact', 'edge'] as const;
export const compactJunctionTypes = new Set([
  'Add',
  'Mul',
  'Subtract',
  'Divide',
  'BitwiseAnd',
  'Maximum',
  'MatMul',
  'Concat',
  'Stack',
]);
export const compactLayerTypes = new Set([
  'Linspace',
  'OnesLike',
  'ZerosLike',
  'GridSample',
  'Normalize',
  'Linear',
  'LinearByDimensions',
  'LayerNorm',
  'RMSNorm',
  'Conv1d',
  'Conv2d',
  'FFN',
  'MMCVFFNContract',
  'LSTM',
  'GRU',
  'RNN',
  'MultiheadAttention',
  'SelfAttention',
  'Transformer',
  'ReLU',
  'GELU',
  'SiLU',
  'Tanh',
  'Softmax',
]);
export const edgeOperatorTypes = new Set([
  'Cos',
  'Sin',
  'Function',
  'TensorCompareScalar',
  'OnesLike',
  'ZerosLike',
  'TensorIndex',
  'Slice',
  'Linear',
  'Reshape',
  'View',
  'Permute',
  'Unsqueeze',
  'Squeeze',
  'Flatten',
  'Unflatten',
  'LayerNorm',
  'Repeat',
  'RepeatBatch',
  'Contiguous',
  'Detach',
  'ReLU',
  'GELU',
  'SiLU',
  'Sigmoid',
  'Softmax',
  'Tanh',
  'Scale',
  'Cast',
  'NanToNum',
  'Clamp',
  'Dropout',
  'Exp',
  'Log',
  'Sqrt',
  'Softplus',
  'L2Norm',
  'Identity',
  'SemanticOp',
]);

export const supportsCompactPresentation = (node: Pick<ModelNode, 'type'>) =>
  compactJunctionTypes.has(node.type) || compactLayerTypes.has(node.type);
export const supportsCompactLayerPresentation = (node: Pick<ModelNode, 'type'>) =>
  compactLayerTypes.has(node.type);
export const supportsEdgePresentation = (node: Pick<ModelNode, 'type' | 'params'>) =>
  edgeOperatorTypes.has(node.type);

export function validateNodePresentation(
  node: Pick<ModelNode, 'type' | 'params'>,
  value: unknown,
): asserts value is NodePresentation {
  if (isCodeRegion(node) || isLayoutRegion(node))
    throw Error('范围框与布局分区不支持节点画布样式。');
  if (!nodePresentations.some((mode) => mode === value))
    throw Error('节点画布样式需要 default、compact 或 edge。');
  if (
    (value === 'compact' && !supportsCompactPresentation(node)) ||
    (value === 'edge' && !supportsEdgePresentation(node))
  )
    throw Error(`${node.type} 不支持画布样式 ${value}。`);
}

export function setNodePresentation(node: ModelNode, value: unknown) {
  const mode = value === null ? 'default' : value;
  validateNodePresentation(node, mode);
  if (mode === 'default') delete node.presentation;
  else node.presentation = mode;
}
