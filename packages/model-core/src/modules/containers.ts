import type { ModuleDefinition } from '../core/types';
import { requireThat } from './math';
import { text } from './primitives';
import {
  DICTIONARY_SCHEMA_LIMIT,
  emptyDictionary,
  parseDictionarySchema,
} from '../core/dictionary';

export { isForwardInput as isContainerInput } from '../core/inputTypes';

export const containerInputs: ModuleDefinition[] = [
  {
    type: 'DictionaryInput',
    label: '字典输入',
    category: '输入输出',
    description: '定义 forward 的嵌套字典输入。字段可以是字典或 Tensor；输出端口传递整份字典。',
    inputs: [],
    outputs: [{ id: 'out', label: '字典' }],
    fields: [
      {
        ...text('schema', '字典字段', JSON.stringify(emptyDictionary())),
        maxLength: DICTIONARY_SCHEMA_LIMIT,
        help: '嵌套字典结构描述：kind=dict，fields 中的值是子字典或带 shape、dtype、可选 axes 的 Tensor 描述。',
      },
    ],
    inferValue: (_, params) => ({
      outputs: { out: parseDictionarySchema(params.schema) },
      notes: ['整份字典沿连线传递；此处描述字段、形状和类型，不保存真实张量。'],
    }),
  },
  {
    type: 'ForwardInput',
    label: '网络输入',
    category: '输入输出',
    description: '表示 forward 的输入。画布只显示名称，数据类型和各字段尺寸在属性中查看或编辑。',
    inputs: [],
    outputs: [{ id: 'out', label: '输入' }],
    fields: [
      {
        ...text('schema', '输入数据', ''),
        maxLength: DICTIONARY_SCHEMA_LIMIT,
        help: '可选的嵌套字典结构，保存各 Tensor 字段的 shape 和 dtype；留空时不指定输入类型。',
      },
      text('text', '备注', ''),
    ],
    inferValue: (_, params) => {
      const remark = params.text ?? '';
      requireThat(typeof remark === 'string' && remark.length < 2000, '备注最多 1999 个字符。');
      if (params.schema !== undefined && params.schema !== '')
        return {
          outputs: { out: parseDictionarySchema(params.schema) },
          notes: ['输入类型为 dict；各 Tensor 的尺寸和 dtype 保存在字段中，整份字典沿连线传递。'],
        };
      return {
        outputs: { out: { kind: 'forward', text: remark } },
        notes: ['网络输入标记，不要求填写形状。'],
      };
    },
  },
];
