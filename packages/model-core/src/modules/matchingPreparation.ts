import type { Field, ModelNode, ModuleDefinition, Params, ValueSpec } from '../core/types';
import { tensorValue } from '../core/types';
import { factoryShape } from './factoryShape';
import { viewPorts } from './featureMath';
import { requireThat, tensor } from './math';
import { bool, input, num, output, select, text } from './primitives';
import { deviceValue } from './device';

const deviceSource = select('device_source', '设备来源', 'device', ['device', 'tensor_legacy']);
const explicitDevice = (params: Params) => params.device_source === 'device';
const devicePort = (params: Params) =>
  explicitDevice(params)
    ? { id: 'device', label: 'device（Device 设备值）' }
    : { id: 'in', label: '旧版参考 Tensor（仅使用 device）' };
const activeFields = (fields: Field[], params: Params) =>
  params.device_source === undefined ? fields.filter((f) => f.key !== 'device_source') : fields;
function checkDevice(values: Record<string, ValueSpec>, params: Params) {
  requireThat(
    params.device_source === undefined ||
      ['device', 'tensor_legacy'].includes(String(params.device_source)),
    '设备来源模式无效。',
  );
  if (explicitDevice(params)) deviceValue(values.device);
  else tensorValue(values.in);
}
const deviceExpression = (params: Params) =>
  explicitDevice(params) ? 'device' : 'reference.device';
const factoryPorts = (params: Params) => ({
  inputs: viewPorts(params).inputs.map((port) => (port.id === 'in' ? devicePort(params) : port)),
  outputs: output,
});
const noGrad = { requiresGrad: false, source: 'inferred' as const };

export function matchingPreparationLabel(node: ModelNode): string | undefined {
  if (node.type === 'OnesScaled') return `ones() * ${node.params.value}`;
  if (node.type === 'Arange') return 'arange(end)';
  if (node.type === 'SortRowsByMax') return 'sort_rows_by_max()';
}

export function matchingPreparationCode(node: ModelNode): string | undefined {
  if (node.type === 'OnesScaled')
    return `out = torch.ones((${node.params.shape},), device=${deviceExpression(node.params)}) * (${node.params.value})\n# 对应先创建全 1 再乘常数，合并展示；不是 torch.full 的整数 dtype 推断。\n# 源码未指定 dtype；静态假设 torch.get_default_dtype() == torch.${node.params.default_dtype}。\n# ${explicitDevice(node.params) ? 'device 接口只接收 GetDevice 提取的设备值，不接收 Tensor。' : '旧版兼容：仅使用 reference.device。'}不继承 dtype、数值、形状或梯度。\n# 新建普通 Tensor，requires_grad=False，不是 Parameter / buffer。`;
  if (node.type === 'Arange')
    return `out = torch.arange(end, device=${deviceExpression(node.params)})\n# end 为非负整数，不含 end：0, 1, ..., end - 1。\n# dtype=int64，requires_grad=False。\n# ${explicitDevice(node.params) ? 'device 接口接收 Device 元数据，不接收 Tensor。' : '旧版兼容：参考 Tensor 仅提供 device。'}\n# 形状为 [end]，不把索引伪装成浮点概率或模型参数。`;
  if (node.type === 'SortRowsByMax') {
    const columns = Number(node.params.trailing_columns);
    return `scores = ${columns === 0 ? 'x' : `x[:, :-${columns}]`}\nmax_map = scores.max(dim=-1).values\nsort_index = torch.sort(max_map, descending=${node.params.descending ? 'True' : 'False'})[1]\nout = x[sort_index]\n# 整行移动，不归一化、不改变输入 x。\n# ${columns === 0 ? '当前所有列参与评分。' : `最后 ${columns} 列不参与评分；默认配置对应 reweight(maps)，原始 query 编号随行保留。`}\n# 并列分数不保证稳定顺序；静态图仅推导规格，不声称已算出实际排序索引。`;
  }
}

export const matchingPreparationModules: ModuleDefinition[] = [
  {
    type: 'OnesScaled',
    label: '常量初始化 Ones × value',
    category: '维度操作',
    description:
      '合并展示 torch.ones(size, device=device) * value。device 接口接收 GetDevice 输出；dtype 使用声明的 PyTorch 默认浮点类型，不是 full_like。旧图保留 Tensor 参考兼容模式。',
    fields: [
      { ...text('shape', '目标形状', 'd0'), maxLength: 256 },
      { ...text('axes', '轴含义', ''), maxLength: 256 },
      num('value', '乘以的常数', -1, -Number.MAX_VALUE),
      select('default_dtype', '默认浮点类型（静态假设）', 'float32', ['float32', 'float64']),
      deviceSource,
    ],
    getFields(params) {
      return activeFields(this.fields, params);
    },
    ...factoryPorts({ shape: 'd0', device_source: 'device' }),
    getPorts: factoryPorts,
    inferValue: (values, params) => {
      checkDevice(values, params);
      const { shape, axes } = factoryShape(values, params, 'ones');
      requireThat(
        typeof params.value === 'number' && Number.isFinite(params.value),
        '常数必须是有限数值。',
      );
      requireThat(
        ['float32', 'float64'].includes(String(params.default_dtype)),
        '需要声明有效的默认浮点类型。',
      );
      return {
        outputs: {
          out: { ...tensor(shape, axes, String(params.default_dtype)), autograd: { ...noGrad } },
        },
        notes: [
          explicitDevice(params)
            ? 'device 为独立设备元数据，不是 Tensor 数据输入。'
            : '旧版兼容：Tensor 参考仅用于 device；建议改用 GetDevice 和 device 接口。',
          `源码省略 dtype；默认浮点类型按 ${params.default_dtype} 静态建模，不从运行快照猜测。`,
          `构造语义为全 1 乘 ${params.value}；默认 requires_grad=False，不向参考输入回传梯度。`,
          '静态图不分配真实 Tensor，也未验证实际设备或采集元素数值。',
        ],
      };
    },
  },
  {
    type: 'Arange',
    label: '整数序列 Arange',
    category: '维度操作',
    description:
      '对应 torch.arange(end, device=device)，生成从 0 到 end-1 的 int64 索引序列。device 接口只接收 Device；旧图保留 Tensor 参考兼容模式。',
    fields: [{ ...text('axis', '轴含义', 'query'), maxLength: 256 }, deviceSource],
    getFields(params) {
      return activeFields(this.fields, params);
    },
    inputs: [devicePort({ device_source: 'device' }), { id: 'end', label: 'end（非负整数）' }],
    getPorts: (params) => ({
      inputs: [devicePort(params), { id: 'end', label: 'end（非负整数）' }],
      outputs: output,
    }),
    outputs: output,
    inferValue: (values, params) => {
      checkDevice(values, params);
      const { shape, axes } = factoryShape(
        { d0: values.end },
        { shape: 'd0', axes: params.axis },
        'arange',
      );
      return {
        outputs: { out: { ...tensor(shape, axes, 'int64'), autograd: { ...noGrad } } },
        notes: [
          '按 end 创建 [0, end) 的整数序列，dtype=int64，requires_grad=False。',
          explicitDevice(params)
            ? 'device 为独立设备元数据，不从 Tensor 数据输入继承属性；end=0 时为空序列。'
            : '旧版兼容：Tensor 参考仅提供 device；建议改用 GetDevice 和 device 接口。',
          '静态图只描述序列规则与尺寸，不分配数组或声称采集了当前设备。',
        ],
      };
    },
  },
  {
    type: 'SortRowsByMax',
    label: '按最大分数重排行',
    category: '索引与切片',
    description:
      '将二维 Tensor 按每行最大分数排序。排除末尾指定数量的元数据列，随后整行重排；对应 SparseWorld 的 reweight(maps)。',
    fields: [
      num('trailing_columns', '末尾非评分列数量', 1, 0),
      bool('descending', '降序排列', true),
    ],
    inputs: input,
    outputs: output,
    infer: ({ in: x }, params) => {
      requireThat(x.shape.length === 2, '按行分数排序需要二维 Tensor。');
      requireThat(
        [
          'float16',
          'bfloat16',
          'float32',
          'float64',
          'int8',
          'uint8',
          'int16',
          'int32',
          'int64',
        ].includes(x.dtype),
        '按分数排序仅支持实数数值 Tensor。',
      );
      const trailing = params.trailing_columns;
      requireThat(
        typeof trailing === 'number' && Number.isSafeInteger(trailing) && trailing >= 0,
        '非评分列数量必须为非负安全整数。',
      );
      requireThat(typeof params.descending === 'boolean', 'descending 必须为布尔值。');
      const columns = x.shape[1];
      requireThat(
        typeof columns !== 'number' || columns > trailing,
        '排除元数据后必须至少保留一列分数。',
      );
      return {
        outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) },
        notes: [
          `排除末尾 ${trailing} 列后取各行最大值，按最大值${params.descending ? '降序' : '升序'}排列整行。`,
          '保持 shape/dtype，但改变行顺序；原始 query 编号随整行移动，输入 Tensor 不被原地修改。',
          'torch.sort 未开启 stable；并列顺序不能从规格推断。不是逐元素函数或概率归一化。',
          '当前只验证形状与规则，不执行实际排序、不伪造 sort_index；需真实数值才能确定顺序。',
        ],
        warnings:
          typeof columns === 'number' ? [] : ['评分列数量为符号，运行时必须至少有一列分数。'],
      };
    },
  },
];
