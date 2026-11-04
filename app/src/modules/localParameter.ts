import type { ModuleDefinition } from '../core/types';
import { clone, isIntegerSpec } from '../core/types';
import { requireThat } from './math';
import { input, output } from './primitives';

export const localParameter: ModuleDefinition = {
  type: 'LocalParameter',
  label: '局部参数',
  category: '输入输出',
  description:
    '为上游计算得到的整数或符号尺寸命名，例如 bs、A。画布只显示名称，数值和类型在属性中查看；右侧端口可继续传递该值。',
  fields: [],
  inputs: input,
  outputs: output,
  inferValue: ({ in: value }) => {
    requireThat(
      value && isIntegerSpec(value),
      '局部参数需要上游整数尺寸；Tensor 请先通过“读取维度 Size”提取尺寸。',
    );
    return {
      outputs: { out: clone(value) },
      notes: ['值随上游计算更新；这是数据流中的局部值，不会加入“模型超参数”。'],
    };
  },
};
