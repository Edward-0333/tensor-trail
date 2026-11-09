import type { ModuleDefinition } from '../core/types';

export const calculationGroup: ModuleDefinition = {
  type: 'CalculationGroup',
  label: '计算组合',
  category: '内部',
  description: '由相邻计算节点合并而成，保留各操作参数与数据依赖。双击进入或拆开组合继续编辑。',
  fields: [],
  inputs: [],
  outputs: [],
  composite: true,
};
