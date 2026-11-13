import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { isDtypeSpec, isNumpyArraySpec, isTensorSpec, tensorValue } from '../core/types';
import { deviceValue } from './device';
import { factoryShape } from './factoryShape';
import { viewPorts } from './featureMath';
import { requireThat, tensor } from './math';
import { input, output, select, text } from './primitives';

const dtypes = ['float16', 'float32', 'float64', 'int32', 'int64', 'uint8', 'bool'];
const modes = ['tensor', 'new_tensor'];
const arrayOperations = ['asarray_astype', 'asarray'];
const arrayFields = [
  text('field', '字段名', 'lidar2img'),
  { ...text('shape', '声明的数组形状（非实测）', '2,30,4,4'),
    help: '不透明元数据不能自动推导真实字段形状；请按数据配置更新。可连接 d0–d7 尺寸端口。' },
  text('axes', '轴含义', ''),
  select('operation', '处理步骤', 'asarray_astype', arrayOperations),
  select('dtype', 'astype 类型', 'float32', dtypes),
  select('source_dtype', 'asarray 输出类型（未采集可选 unknown）', 'unknown', ['unknown', ...dtypes]),
];
const conversionPorts = (p: Params) => ({
  inputs: [
    { id: 'in', label: 'NumPy 数组（数据）' },
    { id: 'device', label: 'device（元数据）' },
    ...(p.mode === 'new_tensor' ? [{ id: 'dtype', label: 'dtype（元数据）' }] : []),
  ],
  outputs: output,
});

export function arrayConversionCode(node: ModelNode): string | undefined {
  const p = node.params;
  if (node.type === 'BatchMetaArray') return `values = np.asarray([m[${JSON.stringify(p.field)}] for m in img_metas])${p.operation === 'asarray' ? '' : `.astype(np.${p.dtype})`}
# 输出是 NumPy ndarray，不是 torch.Tensor，没有 requires_grad。
# shape=[${p.shape}] 是元数据字段的显式规格声明，不是运行时读取结果。
# ${p.operation === 'asarray' ? '此节点只提取字段并按 batch 组织为数组；不转换 dtype，未采集时不猜测为 float64。' : '兼容合并展示：字段提取、asarray 和 astype；没有矩阵乘法。'}`;
  if (node.type === 'NumpyAstype') return `out = values.astype(np.${p.dtype})
# 转换 NumPy 数组元素的 dtype，shape 和数据来源保持不变。
# 这仍然是 ndarray，不是 torch.Tensor；不显示 Tensor 梯度状态。
# astype 默认 copy=True；画布只推导规格，不执行转换或创建真实数组。`;
  if (node.type === 'GetDtype') return `dtype = x.dtype
# 只读取类型属性，不读取张量元素，不继承其数据信息来源。`;
  if (node.type === 'ArrayToTensor') return p.mode === 'new_tensor'
    ? `out = reference.new_tensor(values)
# 等价的创建规格：dtype=reference.dtype, device=reference.device。
# 图中通过 GetDtype/GetDevice 单独传入这两个属性，不使用 reference 的数值。
# 创建普通 Tensor，默认 requires_grad=False；并非 nn.Parameter。`
    : `out = torch.tensor(values, device=device)
# dtype 由 NumPy 数组推断；device 来自独立的 GetDevice。
# 复制数据创建普通 Tensor，默认 requires_grad=False；不执行 Python 或实际复制内存。`;
}

export const arrayConversionModules: ModuleDefinition[] = [
  {
    type: 'BatchMetaArray', label: '批量字段转 NumPy', category: '组合计算',
    description: '从 list[dict] 提取字段并执行 asarray。可单独显示数组整理，或兼容旧图的合并 astype；不生成 Tensor。',
    fields: arrayFields,
    getFields: p => arrayFields.filter(field => {
      if (p.operation === undefined && ['operation', 'source_dtype'].includes(field.key)) return false;
      if (p.operation === 'asarray' && p.source_dtype === undefined && field.key === 'source_dtype') return false;
      return field.key !== (p.operation === 'asarray' ? 'dtype' : 'source_dtype');
    }),
    inputs: [{ id: 'img_metas', label: 'img_metas（数据）' }], outputs: output,
    getPorts: p => ({ inputs: [{ id: 'img_metas', label: 'img_metas（数据）' },
      ...viewPorts(p).inputs.filter(port => port.id !== 'in')], outputs: output }),
    inferValue: (values, p) => {
      const metas = values.img_metas;
      requireThat(metas && !isTensorSpec(metas) && 'text' in metas &&
        ['forward', 'list'].includes(metas.kind), '需要不透明 list[dict] 元数据。');
      requireThat(typeof p.field === 'string' && /^[A-Za-z_]\w{0,63}$/.test(p.field), '字段名无效。');
      const operation = p.operation ?? 'asarray_astype';
      requireThat(arrayOperations.includes(String(operation)), '未知 NumPy 数组处理步骤。');
      const dtype = operation === 'asarray' ? String(p.source_dtype ?? 'unknown') : String(p.dtype);
      requireThat(dtypes.includes(dtype) || (operation === 'asarray' && dtype === 'unknown'), '不支持该 NumPy dtype。');
      const { shape, axes } = factoryShape(values, p, 'BatchMetaArray');
      return { outputs: { out: { kind: 'ndarray', dims: shape, axes, dtype } },
        notes: ['数组 shape 是字段契约，不是实际采集。数据配置变化时需重新核对声明。',
          dtype === 'unknown' ? 'asarray 的原始 dtype 未采集；不能假定为 float32 或 float64。'
            : operation === 'asarray' ? 'dtype 仅为原始数组的声明，此节点不执行类型转换。' : `已合并 astype(np.${dtype})。`,
          '输出为 NumPy ndarray，没有 requires_grad；不是 grad:off 的 PyTorch Tensor。'] };
    },
  },
  {
    type: 'NumpyAstype', label: 'astype', category: '维度操作',
    description: 'NumPy ndarray.astype(dtype)：独立的数组类型转换，保留 shape，不创建 Tensor。',
    fields: [select('dtype', '目标 NumPy dtype', 'float32', dtypes)],
    inputs: input, outputs: output,
    inferValue: ({ in: array }, p) => {
      requireThat(array && isNumpyArraySpec(array), 'astype 需要 NumPy ndarray，而不是 Tensor。');
      requireThat(dtypes.includes(array.dtype) || array.dtype === 'unknown', '不支持该原始 NumPy dtype。');
      requireThat(dtypes.includes(String(p.dtype)), '不支持该目标 NumPy dtype。');
      requireThat(array.axes.length === array.dims.length, '数组轴标签与维数不一致。');
      requireThat(array.dims.every(d => typeof d === 'number'
        ? Number.isSafeInteger(d) && d >= 0 : typeof d === 'string' && d.trim().length > 0),
      '数组尺寸需要非负安全整数或符号。');
      return { outputs: { out: { kind: 'ndarray', dims: [...array.dims], axes: [...array.axes], dtype: String(p.dtype) } },
        notes: [`astype 将数组元素转换为 ${p.dtype}；保留 shape，但可能有精度变化。`,
          '输出仍为 NumPy 数组，没有 requires_grad；不执行真实转换，不推断之前的未知 dtype。'] };
    },
  },
  {
    type: 'GetDtype', label: '读取类型 GetDtype', category: '形状查询',
    description: '读取 Tensor.dtype，返回独立 dtype 元数据，不传递 Tensor 元素。',
    fields: [], inputs: input, outputs: [{ id: 'out', label: 'dtype（类型元数据）' }],
    inferValue: ({ in: value }) => ({ outputs: { out: {
      kind: 'dtype', value: tensorValue(value).dtype,
    } }, notes: ['只读取 dtype；不参与梯度计算，不代表数据融合。'] }),
  },
  {
    type: 'ArrayToTensor', label: 'NumPy 转 Tensor', category: '维度操作',
    description: '区分 torch.tensor 的数组 dtype 推断与 new_tensor 的 dtype/device 继承。两者默认不求梯度。',
    fields: [select('mode', '创建方式', 'tensor', modes)],
    ...conversionPorts({ mode: 'tensor' }), getPorts: conversionPorts,
    inferValue: (v, p) => {
      requireThat(modes.includes(String(p.mode)), '未知 Tensor 创建方式。');
      const array = v.in;
      requireThat(array && isNumpyArraySpec(array), '需要 NumPy ndarray，不能把 Tensor 当作数组。');
      requireThat(dtypes.includes(array.dtype), '不支持该 NumPy dtype。');
      requireThat(array.axes.length === array.dims.length, '数组轴标签与维数不一致。');
      requireThat(array.dims.every(d => typeof d === 'number'
        ? Number.isSafeInteger(d) && d >= 0 : typeof d === 'string' && d.trim().length > 0),
      '数组尺寸需要非负安全整数或符号。');
      deviceValue(v.device);
      let dtype = array.dtype;
      if (p.mode === 'new_tensor') {
        requireThat(v.dtype && isDtypeSpec(v.dtype), 'new_tensor 需要 GetDtype 的元数据，不接受参考 Tensor。');
        dtype = v.dtype.value;
        requireThat([...dtypes, 'bfloat16'].includes(dtype), '不支持该输出 dtype。');
      }
      return { outputs: { out: { ...tensor([...array.dims], [...array.axes], dtype),
        autograd: { requiresGrad: false, source: 'inferred' } } },
        notes: ['创建普通 Tensor，默认 requires_grad=False；数值来源只继承数组。',
          p.mode === 'new_tensor' ? 'dtype/device 由独立元数据端口提供，不继承参考 Tensor 的数值或梯度。'
            : 'dtype 保留自 NumPy 数组，device 来自设备端口；不猜测实际 CUDA/CPU 放置。'] };
    },
  },
];
