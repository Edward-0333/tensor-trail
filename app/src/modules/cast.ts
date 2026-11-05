import type { ModuleDefinition, Params } from '../core/types';
import { tensorValue } from '../core/types';
import { requireThat, tensor } from './math';
import { input, output, select } from './primitives';

const castDtypes = [
  'float16',
  'float32',
  'float64',
  'int8',
  'int16',
  'int32',
  'int64',
  'uint8',
  'bool',
];

function targetDtype(params: Params): string {
  requireThat(
    typeof params.dtype === 'string' && castDtypes.includes(params.dtype),
    '类型转换的 dtype 无效，请选择支持的数据类型。',
  );
  return params.dtype;
}

export function castLabel(params: Params): string {
  const dtype = targetDtype(params);
  return dtype === 'int64' ? 'long()' : dtype === 'float32' ? 'float()' : `to(${dtype})`;
}

export const castModule: ModuleDefinition = {
  type: 'Cast',
  label: '类型转换',
  category: '类型转换',
  description:
    '转换 Tensor 的数据类型，保留形状与轴含义；int64 对应 long()，float32 对应 float()。',
  inputs: input,
  outputs: output,
  fields: [select('dtype', '目标数据类型', 'int64', castDtypes)],
  infer: ({ in: source }, params) => {
    const x = tensorValue(source),
      dtype = targetDtype(params);
    return {
      outputs: { out: tensor([...x.shape], [...x.axes], dtype) },
      notes: [`${castLabel(params)} · ${x.dtype} → ${dtype}`],
    };
  },
};
