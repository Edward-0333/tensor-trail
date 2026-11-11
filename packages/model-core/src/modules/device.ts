import type { DeviceSpec, ModelNode, ModuleDefinition, ValueSpec } from '../core/types';
import { clone, isDeviceSpec, tensorValue } from '../core/types';
import { requireThat } from './math';
import { moduleReferenceName } from './moduleReference';
import { input, output, text } from './primitives';

export function deviceValue(value: ValueSpec | undefined): DeviceSpec {
  requireThat(
    value && isDeviceSpec(value),
    '需要 Device 设备值；请先用 GetDevice 读取 Tensor.device，不能直接传入 Tensor。',
  );
  requireThat(
    typeof value.reference === 'string' && value.reference.endsWith('.device'),
    '设备来源需要 Tensor.device 表达式。',
  );
  moduleReferenceName(value.reference);
  return value;
}

export function deviceCode(node: ModelNode): string | undefined {
  if (node.type === 'GetDevice')
    return `device = ${node.params.tensor_name}.device\n# 读取 torch.device 元数据，不读取张量元素、不复制张量。\n# device 不是 Tensor，没有 shape、dtype 或 requires_grad。\n# 来源名称仅作表达式标注，依赖由输入连线决定。\n# 静态图没有实际设备观测，不把来源推断成 cuda:0 或 cpu。`;
  if (node.type === 'DeviceVariable')
    return '# 为同一个设备值命名或创建关联副本；不是 Tensor 拷贝，也不分配设备。\n# 可连接工厂算子的 device 接口，灰色虚线表示设备元数据依赖。';
}

export const deviceModules: ModuleDefinition[] = [
  {
    type: 'GetDevice',
    label: '读取设备 GetDevice',
    category: '形状查询',
    description:
      '显式读取 Tensor.device，输出独立的 Device 元数据；不把整个 Tensor 当作后续工厂的数据输入。',
    fields: [
      {
        ...text('tensor_name', '来源张量名称（表达式标注）', 'x'),
        maxLength: 249,
        help: '例如 maps，对应 maps.device。名称只用于说明；实际依赖必须由 Tensor 输入连线提供。',
      },
    ],
    inputs: [{ id: 'in', label: '来源 Tensor（读取 device）' }],
    outputs: [{ id: 'out', label: 'device（设备元数据）' }],
    inferValue: ({ in: value }, params) => {
      tensorValue(value);
      const name = moduleReferenceName(params.tensor_name);
      requireThat(name.length <= 249, '来源张量名称最多 249 字符。');
      return {
        outputs: { out: { kind: 'device', reference: `${name}.device` } },
        notes: [
          '只读取设备属性，不使用张量元素，不产生 autograd 运算。',
          'Device 不是 Tensor，没有 shape、dtype 或 requires_grad；来源仍由图中的 Tensor 连线决定。',
          '实际设备未采集；显示 maps.device 这样的来源表达式，不声称设备已确定为 CUDA 或 CPU。',
        ],
      };
    },
  },
  {
    type: 'DeviceVariable',
    label: '设备值 Device',
    category: '输入输出',
    description:
      '为 Device 元数据命名。显示 DEVICE 标签，不显示 Tensor 尺寸或梯度标记；支持关联副本。',
    fields: [],
    inputs: input,
    outputs: output,
    inferValue: ({ in: value }) => ({
      outputs: { out: clone(deviceValue(value)) },
      notes: ['同一个设备值的命名或引用，不是张量数据流；实际设备未采集。'],
    }),
  },
];
