import type { ModelNode, ModuleDefinition } from '../core/types';
import {
  clone,
  isDictionarySpec,
  isIntegerSpec,
  isNumpyArraySpec,
  isTensorSpec,
  tensorValue,
} from '../core/types';
import { factoryShape } from './factoryShape';
import { viewPorts } from './featureMath';
import { product, requireThat, same, tensor } from './math';
import { bool, input, output, select, text } from './primitives';

export function forecastSupportCode(n: ModelNode): string | undefined {
  if (n.type === 'SingleIterationPreview')
    return '# 绘图范围校验，不是模型代码。\n# 父画布当前展开 range(1) 的唯一一轮 interval=0。\n# 次数不是1时阻止输出，避免把首轮结果误标为完整循环结果。';
  if (n.type === 'TensorFieldContract')
    return `out = data[${JSON.stringify(n.params.field)}]\n# 缺少字段规格时使用显式声明 [${n.params.shape}] / ${n.params.dtype}。\n# 部分快照没有采集此字段；声明不是运行观测，不修改快照。\n# 字段若存在则校验声明并保留其真实规格。`;
  if (n.type === 'ReshapeLike')
    return 'out = x.reshape(like.shape)\n# like 仅提供 shape，不融合其数值、dtype 或梯度。';
  if (n.type === 'BatchMetaStack')
    return `out = np.stack([meta[${JSON.stringify(n.params.field)}] for meta in img_metas])\n# 输出 ndarray；shape 是显式字段契约，dtype 未观测，不猜测原类型。`;
  if (n.type === 'ArrayToTensorExplicit')
    return `out = torch.tensor(array, device=${JSON.stringify(n.params.device)}, dtype=torch.${n.params.dtype})\n# dtype/device 为源码显式常量，不继承其它张量的值。\n# 默认 requires_grad=False；只推导规格，不实际分配设备内存。`;
}

export const forecastSupport: ModuleDefinition[] = [
  {
    type: 'SingleIterationPreview',
    label: '单轮展开校验',
    category: '逻辑与约束',
    description:
      '保护父画布的单轮代码展开：必须只有1轮，否则要求使用真实循环或补画余下轮次。不执行模型。',
    fields: [],
    inputs: [{ id: 'iterations', label: 'num_fu_frames' }],
    outputs: output,
    inferValue: ({ iterations }) => {
      requireThat(
        iterations && isIntegerSpec(iterations) && iterations.value === 1,
        '此父画布仅完整展开1轮；num_fu_frames 必须为1。多轮请使用 ForLoop 携带状态推导，不能复用首轮结果冒充最终输出。',
      );
      return {
        outputs: { out: { kind: 'int', value: 0 } },
        notes: ['绘图校验，非新增网络运算；唯一迭代 interval=0。'],
      };
    },
  },
  {
    type: 'TensorFieldContract',
    label: '读取字段（补充规格）',
    category: '索引与切片',
    description:
      '从字典或不透明参数读取 Tensor；部分快照未覆盖字段时使用显式声明，不伪造运行观测。',
    fields: [
      text('field', '字段', 'temporal_trajs'),
      text('shape', '声明 shape', '2,6,2'),
      text('axes', '轴含义', 'batch,time,xy'),
      select('dtype', '声明 dtype', 'float32', ['float16', 'float32', 'float64', 'int64', 'bool']),
      bool('requires_grad', '缺失字段的 requires_grad 声明', false),
    ],
    inputs: input,
    outputs: output,
    getPorts: (p) => ({
      inputs: [...input, ...viewPorts(p).inputs.filter((port) => port.id !== 'in')],
      outputs: output,
    }),
    inferValue: (v, p) => {
      requireThat(
        typeof p.field === 'string' && /^[A-Za-z_]\w*$/.test(p.field),
        '需要普通字段名。',
      );
      const data = v.in;
      requireThat(
        data && (isDictionarySpec(data) || (!isTensorSpec(data) && data.kind === 'forward')),
        '需要字典或不透明 forward 参数，不能输入 Tensor。',
      );
      const { shape, axes } = factoryShape(v, p, 'TensorFieldContract');
      const existing = isDictionarySpec(data) ? data.fields[String(p.field)] : undefined;
      if (existing) {
        const x = tensorValue(existing),
          warnings: string[] = [];
        requireThat(
          x.shape.length === shape.length && x.dtype === p.dtype,
          '字段 rank/dtype 与声明不一致。',
        );
        shape.forEach((d, i) => same(d, x.shape[i], '字段 shape 与声明不一致', warnings));
        return {
          outputs: { out: clone(x) },
          warnings,
          notes: ['保留已有字段规格；不覆盖其梯度信息。'],
        };
      }
      requireThat(typeof p.requires_grad === 'boolean', 'requires_grad 必须是布尔声明。');
      requireThat(
        !p.requires_grad || String(p.dtype).startsWith('float'),
        '整数/bool 不能需要梯度。',
      );
      return {
        outputs: {
          out: {
            ...tensor(shape, axes, String(p.dtype)),
            autograd: { requiresGrad: p.requires_grad, source: 'manual' },
          },
        },
        notes: ['此字段未在输入规格或部分快照中出现；使用明确声明，不是运行时采集。'],
      };
    },
  },
  {
    type: 'ReshapeLike',
    label: 'reshape(like.shape)',
    category: '维度操作',
    description: '按参考张量的 shape 重排，参考只提供尺寸，不参与数值融合。',
    fields: [],
    inputs: [
      { id: 'in', label: 'Tensor' },
      { id: 'like', label: 'shape 参考' },
    ],
    outputs: output,
    infer: ({ in: x, like }) => {
      const warnings: string[] = [];
      same(product(x.shape), product(like.shape), 'reshape 前后元素总数必须相同', warnings);
      return { outputs: { out: tensor([...like.shape], [...like.axes], x.dtype) }, warnings };
    },
  },
  {
    type: 'BatchMetaStack',
    label: '提取字段并 np.stack',
    category: '组合计算',
    description: 'list[dict] 的字段按 batch 堆叠成 ndarray；不预设原始数组 dtype。',
    fields: [
      text('field', '字段', 'ego2lidar'),
      text('shape', '声明 shape', 'd0,4,4'),
      text('axes', '轴含义', 'batch,row,column'),
    ],
    inputs: [],
    outputs: output,
    getPorts: (p) => ({
      inputs: [
        { id: 'img_metas', label: 'img_metas' },
        ...viewPorts(p).inputs.filter((port) => port.id !== 'in'),
      ],
      outputs: output,
    }),
    inferValue: (v, p) => {
      const metas = v.img_metas;
      requireThat(
        metas && !isTensorSpec(metas) && ['forward', 'list'].includes(metas.kind),
        '需要 list[dict] 元数据。',
      );
      requireThat(typeof p.field === 'string' && /^[A-Za-z_]\w*$/.test(p.field), '字段名无效。');
      const { shape, axes } = factoryShape(v, p, 'BatchMetaStack');
      return {
        outputs: { out: { kind: 'ndarray', dims: shape, axes, dtype: 'unknown' } },
        notes: ['shape 是数据字段契约，未执行 np.stack、未采集矩阵数值和原始 dtype。'],
      };
    },
  },
  {
    type: 'ArrayToTensorExplicit',
    label: 'torch.tensor',
    category: '维度操作',
    description: '从 ndarray 构造 Tensor，dtype 和 device 使用显式参数。',
    fields: [
      select('dtype', 'dtype', 'float32', ['float16', 'float32', 'float64', 'int64', 'bool']),
      select('device', 'device（源码常量）', 'cuda', ['cuda', 'cpu']),
    ],
    inputs: input,
    outputs: output,
    inferValue: ({ in: x }, p) => {
      requireThat(x && isNumpyArraySpec(x), '需要 NumPy ndarray。');
      return {
        outputs: {
          out: {
            ...tensor([...x.dims], [...x.axes], String(p.dtype)),
            autograd: { requiresGrad: false, source: 'inferred' },
          },
        },
        notes: [`明确构造 ${p.dtype} / ${p.device} Tensor；未实际执行设备拷贝。`],
      };
    },
  },
];
