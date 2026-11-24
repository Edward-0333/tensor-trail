import type { ModelNode, ModuleDefinition, Registry } from '../core/types';

export const calculationCategories = [
  '索引与切片',
  '形状查询',
  '数学运算',
  '逻辑与归约',
  '组合计算',
  '维度操作',
  '类型转换',
  '分支运算',
  '注意力算子',
];
// Shared by the library and drag-to-merge eligibility; persisted categories stay intact.
export const libraryCategory = (definition: ModuleDefinition) =>
  ['Inlet', 'Outlet'].includes(definition.type)
    ? '输入输出'
    : definition.type === 'Compute'
      ? '组合计算'
      : definition.type === 'Slice'
        ? '索引与切片'
        : definition.category;

export const isCalculation = (definition?: ModuleDefinition) =>
  !!definition &&
  (definition.type === 'CalculationGroup' ||
    calculationCategories.includes(libraryCategory(definition)));

/** Only these leaf calculations may disappear during automatic simplification. */
const glueTypes = new Set([
  'DictionaryGet', 'TensorSequenceGet', 'TensorIndex', 'Slice', 'Unsqueeze', 'Squeeze', 'Reshape',
  'ReshapeLike', 'View', 'Permute', 'Flatten', 'Unflatten', 'Contiguous', 'Repeat', 'RepeatBatch',
  'AlignTrailingDims', 'ZerosLike', 'Zeros', 'NewZeros', 'NewOnes', 'OnesScaled', 'Cast',
  'TensorCompareScalar', 'LogicalAnd', 'BitwiseAnd', 'Any', 'BooleanNot', 'Where', 'Clamp',
  'Add', 'Subtract', 'Mul', 'Divide', 'Function', 'Sin', 'Cos', 'Exp', 'Log', 'Sqrt',
  'NanToNum', 'NoGradResult', 'Compute',
]);

export function isGlueCalculation(node: ModelNode, registry: Registry): boolean {
  return !node.subgraph && glueTypes.has(node.type) && isCalculation(registry.get(node.type));
}
