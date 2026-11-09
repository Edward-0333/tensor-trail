import type { ModuleDefinition } from '../core/types';
import { clone } from '../core/types';
import { parseFunctionFormula } from '../core/functionFormula';
import { floating } from './math';
import { input, output, text } from './primitives';

export const functionModule: ModuleDefinition = {
  type: 'Function',
  label: '函数',
  category: '数学运算',
  description:
    '用数学公式定义单输入、单输出的逐元素函数。画布只显示函数名称，公式在右侧编辑；输入与输出保持相同形状和浮点类型。',
  inputs: input,
  outputs: output,
  fields: [
    { ...text('argument', '输入变量', 'x'), maxLength: 64 },
    {
      ...text('expression', '数学公式', 'x'),
      maxLength: 1000,
      help: '只填写等号右侧；支持输入变量、数值、math.pi（或 pi）、+ - * / % 和括号。% 使用 Python/PyTorch 取模语义。',
    },
  ],
  infer: ({ in: source }, params) => {
    floating(source);
    parseFunctionFormula(params.expression, params.argument);
    return {
      outputs: { out: clone(source) },
      notes: [
        '对输入各元素应用同一公式；保持 shape、轴含义和 dtype。',
        '当前画布进行公式校验与形状推导，不执行训练张量。',
      ],
    };
  },
};
