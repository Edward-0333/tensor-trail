import type { ModelNode, ModuleDefinition, TensorSpec } from '../core/types';
import { axisIndex, requireThat, tensor } from './math';
import { bool, input, num, output } from './primitives';

const indexedOutputs = [
  { id: 'values', label: 'values（数值）' },
  { id: 'indices', label: 'indices（int64 索引，无梯度）' },
];

export function requireRealTensor(x: TensorSpec) {
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
    '该操作支持实数数值 Tensor，不支持 bool 或复数。',
  );
}

export function sortingLabel(node: ModelNode): string | undefined {
  if (node.type === 'MaxWithIndices') return `max(${node.params.dim})`;
  if (node.type === 'Sort') return node.params.descending ? 'sort(desc)' : 'sort(asc)';
  if (node.type === 'RowIndex') return 'x[indices]';
}

export function sortingCode(node: ModelNode): string | undefined {
  if (node.type === 'MaxWithIndices')
    return `values, indices = x.max(dim=${node.params.dim}, keepdim=${node.params.keepdim ? 'True' : 'False'})\n# values 是沿指定轴的最大值；indices 是该轴上的位置，不是原始 query 编号。\n# 并列最大值返回第一个位置；indices 为 int64，不可求导。`;
  if (node.type === 'Sort')
    return `values, indices = torch.sort(x, dim=${node.params.dim}, descending=${node.params.descending ? 'True' : 'False'}, stable=${node.params.stable ? 'True' : 'False'})\n# 返回二元组：第 [0] 项是排序后的值，第 [1] 项是原输入中对应的位置。\n# indices 为 int64，不可求导；stable=False 时不保证并列元素的相对顺序。\n# 静态图只推导规格，不生成真实排序索引。`;
  if (node.type === 'RowIndex')
    return 'out = x[indices]\n# 一维整数高级索引，选择并重排第 0 轴；其余维度完整保留。\n# 索引可以重复，长度不要求等于 x 的行数；不是 torch.gather。\n# 数值可对 x 求导，整数索引不传递梯度；实际索引越界须在运行时检查。';
}

export const sortingModules: ModuleDefinition[] = [
  {
    type: 'MaxWithIndices',
    label: '最大值及索引',
    category: '逻辑与归约',
    description: '对应 x.max(dim)，同时输出最大值和其在归约轴上的索引。',
    fields: [num('dim', '归约轴', -1, -8), bool('keepdim', '保留归约轴', false)],
    inputs: input,
    outputs: indexedOutputs,
    infer: ({ in: x }, p) => {
      requireRealTensor(x);
      requireThat(typeof p.dim === 'number', 'dim 必须是整数。');
      requireThat(typeof p.keepdim === 'boolean', 'keepdim 必须是布尔值。');
      const dim = axisIndex(p.dim, x.shape.length);
      requireThat(x.shape[dim] !== 0, 'max 不能归约长度为 0 的轴。');
      const shape = [...x.shape],
        axes = [...x.axes];
      if (p.keepdim) shape[dim] = 1;
      else {
        shape.splice(dim, 1);
        axes.splice(dim, 1);
      }
      return {
        outputs: {
          values: tensor(shape, axes, x.dtype),
          indices: tensor([...shape], [...axes], 'int64'),
        },
        warnings: typeof x.shape[dim] === 'number' ? [] : ['运行时归约轴长度必须大于 0。'],
        notes: ['values 可参与数据梯度传播；indices 是不可求导的位置索引。'],
      };
    },
  },
  {
    type: 'Sort',
    label: '排序值及索引',
    category: '索引与切片',
    description: '对应 torch.sort，分别输出排序后的值和原始位置索引，形状均与输入相同。',
    fields: [
      num('dim', '排序轴', -1, -8),
      bool('descending', '降序', false),
      bool('stable', '稳定排序', false),
    ],
    inputs: input,
    outputs: indexedOutputs,
    infer: ({ in: x }, p) => {
      requireRealTensor(x);
      requireThat(typeof p.dim === 'number', 'dim 必须是整数。');
      axisIndex(p.dim, x.shape.length);
      requireThat(
        typeof p.descending === 'boolean' && typeof p.stable === 'boolean',
        '排序标志必须是布尔值。',
      );
      return {
        outputs: {
          values: tensor([...x.shape], [...x.axes], x.dtype),
          indices: tensor([...x.shape], [...x.axes], 'int64'),
        },
        notes: ['仅验证规格，不执行数值排序；未指定 stable=True 时不保证并列顺序。'],
      };
    },
  },
  {
    type: 'RowIndex',
    label: '整数索引取行',
    category: '索引与切片',
    description: '对应 x[indices]，沿第 0 轴按一维整数索引取数据，保留其余维度。',
    fields: [],
    inputs: [
      { id: 'in', label: 'x（原始完整数据）' },
      { id: 'indices', label: 'indices（一维整数索引）' },
    ],
    outputs: output,
    infer: ({ in: x, indices }) => {
      requireThat(x.shape.length >= 1, '按行索引的输入至少需要一维。');
      requireThat(
        indices.shape.length === 1 && ['int32', 'int64'].includes(indices.dtype),
        '需要一维 int32/int64 索引。',
      );
      requireThat(
        x.shape[0] !== 0 || typeof indices.shape[0] !== 'number' || indices.shape[0] === 0,
        '空输入不能使用非空索引。',
      );
      return {
        outputs: { out: tensor([indices.shape[0], ...x.shape.slice(1)], [...x.axes], x.dtype) },
        notes: ['仅验证索引规格；负索引、重复索引及具体边界由运行时数值决定。'],
      };
    },
  },
];
