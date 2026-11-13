import type { DeviceSpec, ModelNode, ModuleDefinition, ValueSpec } from '../core/types';
import { clone, isDeviceSpec, tensorValue } from '../core/types';
import { requireThat } from './math';
import { moduleReferenceName } from './moduleReference';
import { input, output, select, text } from './primitives';

const devicePattern = /^(cpu|cuda|mps|xpu|meta)(?::(?:0|[1-9]\d{0,5}))?$/;
const dtypeNames = ['float16', 'bfloat16', 'float32', 'float64', 'bool', 'uint8',
  'int8', 'int16', 'int32', 'int64', 'complex64', 'complex128'];

/** Only accept a literal constructor expression; never evaluate arbitrary code. */
export function literalDeviceName(reference: string): string | undefined {
  const match = /^torch\.device\("([^"\n]+)"\)$/.exec(reference);
  return match && devicePattern.test(match[1]) ? match[1] : undefined;
}

export function deviceValue(value: ValueSpec | undefined): DeviceSpec {
  requireThat(
    value && isDeviceSpec(value),
    '需要 Device 设备值；请用 GetDevice 读取 Tensor.device 或使用 DeviceConstant，不能直接传入 Tensor。',
  );
  if (typeof value.reference === 'string' && literalDeviceName(value.reference)) return value;
  requireThat(
    typeof value.reference === 'string' && value.reference.endsWith('.device'),
    '设备来源需要 Tensor.device 表达式。',
  );
  moduleReferenceName(value.reference);
  return value;
}

export function deviceCode(node: ModelNode): string | undefined {
  if (node.type === 'DeviceConstant' || node.type === 'DtypeConstant') {
    const expression = node.type === 'DeviceConstant'
      ? `torch.device(${JSON.stringify(node.params.device)})` : `torch.${node.params.dtype}`;
    return `# 当前配置的固定元数据值，不执行 .device/.dtype 读取。\nout = ${expression}\n# 不是 Tensor，不分配张量，不产生输入来源或梯度。\n# 改变设备、精度或运行配置时需要同步修改，不会自动跟随参考张量。${node.params.context ? `\n# ${node.params.context}` : ''}`;
  }
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
      notes: ['同一个设备值的命名或引用，不是张量数据流；固定设备声明不代表已实际执行。'],
    }),
  },
  {
    type: 'DeviceConstant', label: '设备常量 Device', category: '输入输出',
    description: '直接提供已知设备值，无 Tensor 输入。固定当前配置，不自动跟随张量设备；支持关联副本。',
    fields: [text('device', '设备', 'cpu'), text('context', '依据 / 使用范围', '')],
    inputs: [], outputs: [{ id: 'out', label: 'device' }],
    inferValue: (_, p) => {
      requireThat(typeof p.device === 'string' && devicePattern.test(p.device), '设备需为 cpu、cuda、cuda:0、mps、xpu 或 meta 等有效名称。');
      return { outputs: { out: { kind: 'device', reference: `torch.device(${JSON.stringify(p.device)})` } },
        notes: ['固定设备声明，不读取张量，不分配内存，也不验证硬件可用性；配置变化时需同步更新。', String(p.context || '')].filter(Boolean) };
    },
  },
  {
    type: 'DtypeConstant', label: '类型常量 Dtype', category: '输入输出',
    description: '直接提供已知 torch.dtype，无 Tensor 输入。不是 astype/cast，不转换数据；支持关联副本。',
    fields: [select('dtype', '数据类型', 'float32', dtypeNames), text('context', '依据 / 使用范围', '')],
    inputs: [], outputs: [{ id: 'out', label: 'dtype' }],
    inferValue: (_, p) => {
      requireThat(dtypeNames.includes(String(p.dtype)), '不支持的 torch.dtype。');
      return { outputs: { out: { kind: 'dtype', value: String(p.dtype) } },
        notes: ['固定类型声明，不读取或转换张量，也不传递梯度；精度配置变化时需同步更新。', String(p.context || '')].filter(Boolean) };
    },
  },
];
