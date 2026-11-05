import type { ModuleDefinition } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { text, output } from './primitives';

export const integerParameter: ModuleDefinition = {
  type: 'IntegerParameter',
  label: '整数参数读取',
  category: '输入输出',
  description:
    '从模型超参数或整数表达式读取值，可接到局部参数或计算节点。修改超参数后数值同步更新。',
  fields: [{ ...text('expression', '整数表达式', ''), maxLength: 256 }],
  inputs: [],
  outputs: output,
  inferValue: (_, params, context) => {
    const value = resolveIntegerExpression(params.expression, context?.hyperparameters);
    return {
      outputs: { out: { kind: 'int', value } },
      notes: [`读取 ${params.expression} = ${value}；数值来自项目超参数，不新增全局变量。`],
    };
  },
};
