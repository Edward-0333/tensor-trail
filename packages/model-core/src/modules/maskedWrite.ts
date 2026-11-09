import type { Dim, ModelNode, ModuleDefinition, Params } from '../core/types';
import { clone, isIntegerSpec, tensorValue } from '../core/types';
import { requireThat, tensor } from './math';
import { maskedSelection, assignmentShape } from './masking';
import { output, select, text, bool } from './primitives';
import { viewPorts } from './featureMath';

const shapeDefault = 'd0, d1';
const dtypes = ['float16', 'bfloat16', 'float32', 'float64', 'int32', 'int64', 'bool'];
const zerosPorts = (p: Params) => ({
  inputs: [
    ...viewPorts(p).inputs.filter((port) => port.id !== 'in'),
    { id: 'device_like', label: '设备参考 Tensor' },
  ],
  outputs: output,
});

export function maskedWriteCode(node: ModelNode): string | undefined {
  if (node.type === 'Zeros')
    return `def forward(self, device_like, **sizes):\n    # d0、d1 等由尺寸输入接口传入。\n    shape = (${String(
      node.params.shape,
    )
      .split(',')
      .map((s) => (/^d[0-7]$/.test(s.trim()) ? `sizes["${s.trim()}"]` : s.trim()))
      .join(
        ', ',
      )},)\n    return torch.zeros(shape, device=device_like.device, dtype=torch.${node.params.dtype})\n\n# dtype 记录当前 torch 默认类型；不继承 device_like.dtype。`;
  if (node.type === 'BooleanAssign')
    return 'def forward(self, x, mask, values):\n    # mask 匹配 x 的前若干轴；values 广播到 [mask.sum(), ...剩余轴]\n    x[mask] = values\n    return x\n\n# 输出表示赋值后的 x；未选中的行保留原值。\n# 这是索引写入，不是乘法或相加；梯度可传回 values。';
}

export const maskedWriteModules: ModuleDefinition[] = [
  {
    type: 'Zeros',
    label: '零张量 Zeros',
    category: '维度操作',
    description:
      '按尺寸创建零张量。device_like 只提供设备，dtype 单独设置；不继承参考张量形状或类型。',
    fields: [
      {
        ...text('shape', '目标形状', shapeDefault),
        maxLength: 256,
        help: '非负整数或 d0–d7 尺寸接口，例如 d0, d1。不支持 -1。',
      },
      {
        ...select('dtype', '数据类型', 'float32', dtypes),
        help: '记录运行时默认 dtype；PLUTO 当前为 float32。',
      },
    ],
    ...zerosPorts({ shape: shapeDefault }),
    getPorts: zerosPorts,
    inferValue: (values, params) => {
      tensorValue(values.device_like);
      requireThat(
        typeof params.shape === 'string' && params.shape.length <= 256,
        '零张量需要有效的目标形状。',
      );
      const tokens = params.shape.split(',').map((s) => s.trim());
      requireThat(
        tokens.length >= 1 && tokens.length <= 8 && tokens.every(Boolean),
        '零张量需要 1–8 个维度。',
      );
      const shape: Dim[] = tokens.map((token) => {
        if (/^d[0-7]$/.test(token)) {
          const value = values[token];
          requireThat(value && isIntegerSpec(value), `${token} 需要局部整数尺寸。`);
          requireThat(
            typeof value.value === 'number'
              ? Number.isSafeInteger(value.value) && value.value >= 0
              : typeof value.value === 'string' && value.value.trim().length > 0,
            '零张量的尺寸必须是非负安全整数或符号。',
          );
          return value.value;
        }
        requireThat(
          /^\+?\d+$/.test(token) && Number.isSafeInteger(Number(token)),
          '零张量仅支持非负整数或 d0–d7，不允许 -1。',
        );
        return Number(token);
      });
      requireThat(dtypes.includes(String(params.dtype)), '零张量的数据类型不受支持。');
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, i) => `axis${i}`),
            String(params.dtype),
          ),
        },
        notes: [
          '创建全零 Tensor；device = device_like.device，dtype 由本节点参数记录。',
          '画布未记录实际设备，不分配内存；连接的参考 Tensor 仅表达设备依赖。',
        ],
      };
    },
  },
  {
    type: 'BooleanAssign',
    label: '布尔索引赋值',
    category: '索引与切片',
    description:
      '将 values 写入 x[mask]，保留未选中位置。支持多维前缀 bool 掩码；可显式开启赋值广播。',
    fields: [
      text('count_symbol', '有效数量符号', 'N_valid'),
      bool('allow_broadcast', '允许赋值广播', false),
    ],
    getFields: (p) => [
      text('count_symbol', '有效数量符号', 'N_valid'),
      ...(p.allow_broadcast === undefined ? [] : [bool('allow_broadcast', '允许赋值广播', false)]),
    ],
    inputs: [
      { id: 'in', label: '待写入 Tensor' },
      { id: 'mask', label: '布尔掩码' },
      { id: 'values', label: '回填值' },
    ],
    outputs: output,
    infer: ({ in: x, mask, values }, params) => {
      const warnings: string[] = [];
      const selection = maskedSelection(x, mask, params.count_symbol, warnings);
      requireThat(x.dtype === values.dtype, '索引写入的目标与回填值 dtype 必须一致。');
      assignmentShape(
        selection.shape,
        values.shape,
        params.allow_broadcast === true,
        selection.capacity,
        warnings,
      );
      return {
        outputs: { out: clone(x) },
        warnings,
        notes: [
          '输出为赋值后的目标 Tensor；未选中位置保持原值。',
          `${params.count_symbol} 表示 count_nonzero(mask)，实际计数和设备一致性需在运行时验证。`,
          params.allow_broadcast
            ? 'values 按 PyTorch 规则广播到索引结果；目标形状不会因广播扩大。'
            : '当前不广播；画布用数据流版本表达原地写入。',
        ],
      };
    },
  },
];
