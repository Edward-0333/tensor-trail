import type { ModuleDefinition } from '../core/types';
import { requireThat } from './math';
import { text } from './primitives';

export { isForwardInput as isContainerInput } from '../core/inputTypes';

export const containerInputs: ModuleDefinition[] = [
  {
    type: 'ForwardInput',
    label: 'Forward输入',
    category: '输入输出',
    description: '用一个文字框标记 forward 的输入，无需指定形状。双击画布文字即可修改。',
    inputs: [],
    outputs: [{ id: 'out', label: '输入' }],
    fields: [text('text', '备注', '')],
    inferValue: (_, params) => {
      requireThat(
        typeof params.text === 'string' && params.text.length < 2000,
        '备注最多 1999 个字符。',
      );
      return {
        outputs: { out: { kind: 'forward', text: params.text } },
        notes: ['Forward 输入标记，不要求填写形状。'],
      };
    },
  },
];
