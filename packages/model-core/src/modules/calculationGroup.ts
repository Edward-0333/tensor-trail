import type { ModuleDefinition } from '../core/types';
import { text } from './primitives';

export const calculationGroup: ModuleDefinition = {
  type: 'CalculationGroup',
  label: '计算组合',
  category: '内部',
  description: '由相邻计算节点合并而成，保留各操作参数与数据依赖。双击进入或拆开组合继续编辑。',
  fields: [
    { ...text('summary', '语义摘要', ''), maxLength: 240 },
    { ...text('formula', '精简表达式', ''), maxLength: 240 },
  ],
  inputs: [],
  outputs: [],
  composite: true,
};
