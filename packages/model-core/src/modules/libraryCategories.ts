import type { ModuleDefinition } from '../core/types';

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
