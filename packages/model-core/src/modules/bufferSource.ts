import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { validateAutograd } from '../core/tensorAppearance';
import { bool, output, select, text } from './primitives';
import { factoryShape } from './factoryShape';
import { viewPorts } from './featureMath';
import { moduleReferenceName } from './moduleReference';
import { requireThat, tensor } from './math';

const dtypes = [
  'float16',
  'bfloat16',
  'float32',
  'float64',
  'int8',
  'uint8',
  'int16',
  'int32',
  'int64',
  'bool',
];
const getPorts = (params: Params) => ({
  inputs: viewPorts(params).inputs.filter((p) => p.id !== 'in'),
  outputs: output,
});

export function bufferSourceCode(node: ModelNode): string | undefined {
  if (node.type !== 'BufferSource') return;
  const reference = String(node.params.reference),
    split = reference.lastIndexOf('.');
  const owner = split >= 0 ? reference.slice(0, split) : 'self',
    name = split >= 0 ? reference.slice(split + 1) : reference;
  return `# 构造声明：仅初始化时注册，不在每次 forward 重新创建。\n${owner}.register_buffer(${JSON.stringify(name)}, torch.${node.params.initial}((${node.params.shape},), dtype=torch.${node.params.dtype}${node.params.requires_grad ? ', requires_grad=True' : ''}), persistent=${node.params.persistent ? 'True' : 'False'})\n\n# 本节点 out 表示读取已持有的 buffer 引用，不是 register_buffer 的返回值。\nout = ${reference}\n# 初值不等于当前值；checkpoint 或手动累加可能已覆盖它。\n# 不分配真实 Tensor、不读取 checkpoint，不是数据集输入或 nn.Parameter。${node.params.context ? `\n# ${node.params.context}` : ''}`;
}

export const bufferSource: ModuleDefinition = {
  type: 'BufferSource',
  label: '注册缓冲区 Buffer',
  category: '输入输出',
  description:
    '声明并读取模型已注册的 buffer。构造初值、形状与 dtype 是静态契约；不是 forward 形参，不会每次重新初始化，不保证当前数值仍等于初值。',
  fields: [
    { ...text('reference', 'buffer 属性路径', 'self.num_stamps_all'), maxLength: 256 },
    {
      ...text('shape', 'buffer 形状', 'd0,d1'),
      maxLength: 256,
      help: '非负整数或 d0–d7 的整数尺寸接口；支持 1–8 维，不使用 batch 维度除非源码确有该维度。',
    },
    { ...text('axes', '轴含义', ''), maxLength: 256 },
    select('dtype', 'buffer dtype', 'int64', dtypes),
    select('initial', '构造初值（不是当前数值）', 'ones', ['ones', 'zeros']),
    bool('persistent', '随 state_dict 保存', true),
    bool('requires_grad', 'buffer requires_grad', false),
    text('context', '执行上下文（仅说明）', ''),
  ],
  ...getPorts({ shape: 'd0,d1' }),
  getPorts,
  inferValue: (values, params) => {
    const reference = moduleReferenceName(params.reference);
    const { shape, axes } = factoryShape(values, params, 'Buffer');
    requireThat(dtypes.includes(String(params.dtype)), 'buffer dtype 不受支持。');
    requireThat(['ones', 'zeros'].includes(String(params.initial)), 'buffer 初值声明无效。');
    requireThat(typeof params.persistent === 'boolean', 'persistent 必须是布尔值。');
    const autograd = { requiresGrad: params.requires_grad, source: 'inferred' };
    validateAutograd(autograd, String(params.dtype));
    return {
      outputs: {
        out: {
          ...tensor(shape, axes, String(params.dtype)),
          autograd,
          bufferRef: { reference, persistent: params.persistent },
        },
      },
      notes: [
        `读取已注册 buffer ${reference}；构造初值为 ${params.initial}，不声称当前值仍为初值。`,
        'buffer 是模块状态的注册身份，不是特殊 Tensor 数据类型或可训练 Parameter；不会由优化器作为参数直接更新。',
        '浮点 buffer 可以单独开启梯度；整数 buffer 不能开启。设备跟随模型，静态画布不验证实际设备。',
        params.persistent
          ? '默认随 state_dict 保存；checkpoint 可覆盖构造初值。'
          : 'persistent=False，不随 state_dict 保存。',
        ...(params.context ? [String(params.context)] : []),
      ],
    };
  },
};
