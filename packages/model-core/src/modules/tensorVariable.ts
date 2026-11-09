import type { ModuleDefinition } from '../core/types';
import { clone, isTensorSpec } from '../core/types';
import { requireThat } from './math';
import { input, output } from './primitives';

export const tensorVariable: ModuleDefinition = {
  type: 'TensorVariable',
  label: '张量变量',
  category: '输入输出',
  description:
    '为中间 Tensor 命名并继续传递，例如 agent_pos、agent_heading。显示名称和形状，尺寸与类型随上游更新；本节点不执行张量计算，也没有可训练参数。',
  fields: [],
  inputs: input,
  outputs: output,
  inferValue: ({ in: value }) => {
    requireThat(
      value && isTensorSpec(value),
      '张量变量需要上游 Tensor；字典请先读取字段，整数尺寸请使用“局部参数”。',
    );
    return {
      outputs: { out: clone(value) },
      notes: [
        '为中间 Tensor 命名，原样传递其形状、轴含义和 dtype；最终返回值请使用“输出 Tensor”。',
      ],
    };
  },
};
