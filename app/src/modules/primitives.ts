import type { Field, ModuleDefinition, Params, TensorSpec } from '../core/types';
import {
  axisIndex,
  broadcast,
  dimensions,
  floating,
  numericList,
  product,
  rank,
  requireThat,
  same,
  tensor,
} from './math';

export const num = (key: string, label: string, value: number, min = 1): Field => ({
  key,
  label,
  default: value,
  kind: 'number',
  min,
});
export const text = (key: string, label: string, value: string): Field => ({
  key,
  label,
  default: value,
  kind: 'text',
});
export const select = (key: string, label: string, value: string, options: string[]): Field => ({
  key,
  label,
  default: value,
  kind: 'select',
  options,
});
export const bool = (key: string, label: string, value: boolean): Field => ({
  key,
  label,
  default: value,
  kind: 'boolean',
});
export const input = [{ id: 'in', label: '输入' }],
  output = [{ id: 'out', label: '输出' }];
const n = (p: Params, key: string) => Number(p[key]);
const pass = (x: TensorSpec, note: string) => ({ outputs: { out: x }, notes: [note] });
const base = (
  type: string,
  label: string,
  category: string,
  description: string,
  fields: Field[],
  infer: ModuleDefinition['infer'],
  inputs = input,
  outputs = output,
): ModuleDefinition => ({ type, label, category, description, fields, infer, inputs, outputs });

export const primitives: ModuleDefinition[] = [
  base(
    'Input',
    '输入 Tensor',
    '输入输出',
    '定义尺寸、轴含义与数据类型。',
    [
      text('shape', '形状', '8, 100, 64'),
      text('axes', '轴含义', 'batch, sequence, feature'),
      select('dtype', '数据类型', 'float32', ['float32', 'float16', 'float64', 'int64']),
    ],
    (_, p) => {
      const shape = dimensions(p.shape);
      const axes = String(p.axes).trim()
        ? String(p.axes)
            .split(',')
            .map((s) => s.trim())
        : shape.map((_, i) => `axis${i}`);
      requireThat(
        axes.length === shape.length && axes.every(Boolean),
        '轴名称数量需与维度一致；留空则自动命名。',
      );
      return {
        outputs: { out: tensor(shape, axes, String(p.dtype)) },
        notes: ['这是网络的数据起点。符号尺寸可填写 B、T 等名称。'],
      };
    },
    [],
  ),
  base(
    'Output',
    '输出 Tensor',
    '输入输出',
    '查看网络最终输出。',
    [],
    ({ in: x }) => pass(x, '当前路径的最终 Tensor。'),
    input,
    [],
  ),
  base(
    'Inlet',
    '模块输入',
    '内部',
    '从父模块接收 Tensor。',
    [text('port', '端口标识', 'in')],
    undefined,
    [],
  ),
  base(
    'Outlet',
    '模块输出',
    '内部',
    '将 Tensor 返回父模块。',
    [text('port', '端口标识', 'out')],
    ({ in: x }) => pass(x, '将结果返回上一级模块。'),
    input,
    [],
  ),
  base(
    'Linear',
    'Linear',
    '基础层',
    '沿最后一维进行线性映射。',
    [num('in_features', '输入特征', 64), num('out_features', '输出特征', 64)],
    ({ in: x }, p) => {
      floating(x);
      const warnings: string[] = [];
      same(x.shape.at(-1)!, n(p, 'in_features'), '最后一维必须等于输入特征数', warnings);
      return {
        outputs: {
          out: tensor(
            [...x.shape.slice(0, -1), n(p, 'out_features')],
            [...x.axes.slice(0, -1), 'feature'],
            x.dtype,
          ),
        },
        warnings,
        notes: [`仅改变最后一维：${p.in_features} → ${p.out_features}。`],
      };
    },
  ),
  ...[1, 2].map((d) =>
    base(
      `Conv${d}d`,
      `Conv${d}d`,
      '卷积',
      '按 channels-first 约定进行卷积。',
      [
        num('in_channels', '输入通道', d === 1 ? 64 : 3),
        num('out_channels', '输出通道', 128),
        num('kernel_size', '卷积核', 3),
        num('stride', '步长', 1),
        num('padding', '填充', 1, 0),
        num('dilation', '膨胀率', 1),
        num('groups', '分组', 1),
      ],
      ({ in: x }, p) => {
        rank(x, d + 2);
        floating(x);
        const warnings: string[] = [];
        same(x.shape[1], n(p, 'in_channels'), '第 1 轴必须等于输入通道数', warnings);
        requireThat(
          n(p, 'in_channels') % n(p, 'groups') === 0 && n(p, 'out_channels') % n(p, 'groups') === 0,
          '分组数必须同时整除输入和输出通道数。',
        );
        const sizes = x.shape.slice(2).map((v) => {
          if (typeof v === 'string')
            return `floor((${v}+${2 * n(p, 'padding')}-${n(p, 'dilation') * (n(p, 'kernel_size') - 1)}-1)/${p.stride}+1)`;
          const value = Math.floor(
            (v + 2 * n(p, 'padding') - n(p, 'dilation') * (n(p, 'kernel_size') - 1) - 1) /
              n(p, 'stride') +
              1,
          );
          requireThat(
            value > 0 && Number.isSafeInteger(value),
            '卷积输出尺寸 ≤ 0 或超出范围，请检查卷积核、填充和输入长度。',
          );
          return value;
        });
        return {
          outputs: {
            out: tensor(
              [x.shape[0], n(p, 'out_channels'), ...sizes],
              ['batch', 'channel', ...(d === 1 ? ['length'] : ['height', 'width'])],
              x.dtype,
            ),
          },
          warnings,
          notes: ['输出长度 = floor((输入 + 2×填充 − 膨胀率×(核−1) − 1) / 步长 + 1)。'],
        };
      },
    ),
  ),
  ...['LSTM', 'GRU', 'RNN'].map((type) =>
    base(
      type,
      type,
      '循环网络',
      '查看序列与独立隐藏状态。',
      [
        num('input_size', '输入特征', 64),
        num('hidden_size', '隐藏特征', 32),
        num('num_layers', '层数', 1),
        bool('bidirectional', '双向', false),
        bool('batch_first', 'Batch 在第 0 轴', true),
      ],
      ({ in: x }, p) => {
        rank(x, 3);
        floating(x);
        const warnings: string[] = [];
        const dirs = p.bidirectional ? 2 : 1;
        same(x.shape[2], n(p, 'input_size'), '输入最后一维必须等于 input_size', warnings);
        const axes = p.batch_first
          ? ['batch', 'sequence', 'feature']
          : ['sequence', 'batch', 'feature'];
        const hidden = tensor(
          [n(p, 'num_layers') * dirs, x.shape[p.batch_first ? 0 : 1], n(p, 'hidden_size')],
          ['layer×direction', 'batch', 'hidden'],
          x.dtype,
        );
        return {
          outputs: {
            out: tensor([x.shape[0], x.shape[1], n(p, 'hidden_size') * dirs], axes, x.dtype),
            hidden,
            ...(type === 'LSTM' ? { cell: hidden } : {}),
          },
          warnings,
          notes: ['序列输出与最终隐藏状态是不同端口；hidden/cell 始终以层数为第 0 轴。'],
        };
      },
      input,
      [
        ...output,
        { id: 'hidden', label: '隐藏状态' },
        ...(type === 'LSTM' ? [{ id: 'cell', label: '细胞状态' }] : []),
      ],
    ),
  ),
  base(
    'Permute',
    'Permute',
    '维度操作',
    '重新排列维度，同时保留轴含义。',
    [text('dims', '轴顺序', '0, 2, 1')],
    ({ in: x }, p) => {
      const dims = numericList(p.dims);
      requireThat(
        dims.length === x.shape.length &&
          new Set(dims).size === dims.length &&
          dims.every((i) => i >= 0 && i < dims.length),
        '轴顺序必须是全部轴索引的一次排列。',
      );
      return {
        outputs: {
          out: tensor(
            dims.map((i) => x.shape[i]),
            dims.map((i) => x.axes[i]),
            x.dtype,
          ),
        },
        notes: [`轴按 ${dims.join(' → ')} 重排，元素数量不变。`],
      };
    },
  ),
  base(
    'Reshape',
    'Reshape',
    '维度操作',
    '改变形状，保持元素总数；允许一个 -1。',
    [text('shape', '目标形状', '8, -1, 64')],
    ({ in: x }, p) => {
      const target = numericList(p.shape);
      const warnings: string[] = [];
      requireThat(
        target.length <= 8 &&
          target.every((v) => v > 0 || v === -1) &&
          target.filter((v) => v === -1).length <= 1,
        '目标维度必须为正整数，最多包含一个 -1。',
      );
      const total = product(x.shape),
        known = product(target.filter((v) => v !== -1));
      const index = target.indexOf(-1);
      const shape: (number | string)[] = [...target];
      if (index >= 0) {
        if (typeof total === 'number') {
          requireThat(total % (known as number) === 0, 'Reshape 的总元素数无法整除已指定的维度。');
          shape[index] = total / (known as number);
        } else {
          shape[index] = `(${total})/${known}`;
          warnings.push('符号维度下的 Reshape 整除约束尚待具体尺寸验证。');
        }
      } else same(total, known, 'Reshape 前后元素总数必须相等', warnings);
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, i) => `axis${i}`),
            x.dtype,
          ),
        },
        warnings,
        notes: ['Reshape 不改变元素数；新轴含义需要按模型解释。'],
      };
    },
  ),
  base(
    'Flatten',
    'Flatten',
    '维度操作',
    '将连续的多个轴合并。',
    [num('start_dim', '开始轴', 1, 0), num('end_dim', '结束轴', -1, -8)],
    ({ in: x }, p) => {
      const start = axisIndex(n(p, 'start_dim'), x.shape.length),
        end = axisIndex(n(p, 'end_dim'), x.shape.length);
      requireThat(start <= end, '开始轴不能位于结束轴之后。');
      return {
        outputs: {
          out: tensor(
            [
              ...x.shape.slice(0, start),
              product(x.shape.slice(start, end + 1)),
              ...x.shape.slice(end + 1),
            ],
            [
              ...x.axes.slice(0, start),
              x.axes.slice(start, end + 1).join('×'),
              ...x.axes.slice(end + 1),
            ],
            x.dtype,
          ),
        },
        notes: ['所选轴相乘，其余轴保留。'],
      };
    },
  ),
  ...['Add', 'Mul'].map((type) =>
    base(
      type,
      type,
      '分支运算',
      type === 'Add' ? '逐元素相加，支持广播与残差。' : '逐元素相乘，支持广播与门控。',
      [],
      ({ a, b }) => {
        const warnings: string[] = [];
        return {
          outputs: { out: broadcast(a, b, warnings) },
          warnings,
          notes: ['从最后一维开始对齐；对应尺寸相等或其中一个为 1。'],
        };
      },
      [
        { id: 'a', label: '输入 A' },
        { id: 'b', label: '输入 B' },
      ],
    ),
  ),
  base(
    'Concat',
    'Concat',
    '分支运算',
    '沿一个轴拼接两个 Tensor。',
    [num('dim', '拼接轴', -1, -8)],
    ({ a, b }, p) => {
      requireThat(
        a.shape.length === b.shape.length && a.dtype === b.dtype,
        'Concat 需要相同维数和 dtype。',
      );
      const axis = axisIndex(n(p, 'dim'), a.shape.length),
        warnings: string[] = [];
      a.shape.forEach((d, i) => {
        if (i !== axis) same(d, b.shape[i], '非拼接轴必须相等', warnings);
      });
      const shape = [...a.shape],
        x = shape[axis],
        y = b.shape[axis];
      shape[axis] = typeof x === 'number' && typeof y === 'number' ? x + y : `(${x}+${y})`;
      return {
        outputs: { out: tensor(shape, [...a.axes], a.dtype) },
        warnings,
        notes: [`沿第 ${axis} 轴拼接，其余维度不变。`],
      };
    },
    [
      { id: 'a', label: '输入 A' },
      { id: 'b', label: '输入 B' },
    ],
  ),
  base(
    'MatMul',
    'MatMul',
    '注意力算子',
    '最后两维做矩阵乘法，前面的维度按广播规则匹配。',
    [],
    ({ a, b }) => {
      requireThat(
        a.shape.length >= 2 && b.shape.length >= 2,
        '第一版 MatMul 接受至少二维的矩阵或批量矩阵。',
      );
      const warnings: string[] = [];
      same(a.shape.at(-1)!, b.shape.at(-2)!, '矩阵乘法的收缩维必须相等', warnings);
      const batch = broadcast(
        tensor(a.shape.slice(0, -2), a.axes.slice(0, -2), a.dtype),
        tensor(b.shape.slice(0, -2), b.axes.slice(0, -2), b.dtype),
        warnings,
      );
      return {
        outputs: {
          out: tensor(
            [...batch.shape, a.shape.at(-2)!, b.shape.at(-1)!],
            [...batch.axes, a.axes.at(-2)!, b.axes.at(-1)!],
            a.dtype,
          ),
        },
        warnings,
        notes: ['[..., M, K] × [..., K, N] → [..., M, N]。'],
      };
    },
    [
      { id: 'a', label: '左矩阵' },
      { id: 'b', label: '右矩阵' },
    ],
  ),
  ...['LayerNorm', 'RMSNorm'].map((type) =>
    base(
      type,
      type,
      '归一化',
      '在最后一个特征轴上归一化。',
      [num('features', '特征维度', 64)],
      ({ in: x }, p) => {
        floating(x);
        const warnings: string[] = [];
        same(x.shape.at(-1)!, n(p, 'features'), '归一化特征数必须匹配最后一维', warnings);
        return { ...pass(x, '归一化改变数值分布，保持形状。'), warnings };
      },
    ),
  ),
  ...['GELU', 'ReLU', 'SiLU', 'Identity'].map((type) =>
    base(type, type, '基础层', '逐元素操作，保持 Tensor 形状。', [], ({ in: x }) =>
      pass(x, `${type} 保持各轴尺寸。`),
    ),
  ),
  base(
    'SplitHeads',
    'SplitHeads',
    '注意力算子',
    '将 [B,T,D] 拆为 [B,H,T,D/H]。',
    [num('heads', '注意力头数', 8)],
    ({ in: x }, p) => {
      rank(x, 3);
      const d = x.shape[2],
        heads = n(p, 'heads'),
        warnings: string[] = [];
      if (typeof d === 'number')
        requireThat(d % heads === 0, `特征维度 ${d} 无法被 ${heads} 个注意力头整除。`);
      else warnings.push(`需要 ${d} 能被 ${heads} 整除，待具体尺寸验证。`);
      return {
        outputs: {
          out: tensor(
            [x.shape[0], heads, x.shape[1], typeof d === 'number' ? d / heads : `(${d})/${heads}`],
            ['batch', 'head', 'sequence', 'head_feature'],
            x.dtype,
          ),
        },
        warnings,
        notes: ['先拆分特征轴，再把 head 轴移到 sequence 前。'],
      };
    },
  ),
  base(
    'MergeHeads',
    'MergeHeads',
    '注意力算子',
    '将 [B,H,T,Dh] 合并为 [B,T,H×Dh]。',
    [],
    ({ in: x }) => {
      rank(x, 4);
      return {
        outputs: {
          out: tensor(
            [x.shape[0], x.shape[2], product([x.shape[1], x.shape[3]])],
            ['batch', 'sequence', 'feature'],
            x.dtype,
          ),
        },
        notes: ['交换 head 与 sequence 轴后合并各头的特征。'],
      };
    },
  ),
  base(
    'Scale',
    'Scale',
    '注意力算子',
    '将分数除以 sqrt(head_dim)，保持形状。',
    [num('head_dim', '每头特征数', 8)],
    ({ in: x }, p) => pass(x, `分数除以 √${p.head_dim}；形状不变。`),
  ),
  base(
    'Softmax',
    'Softmax',
    '注意力算子',
    '沿指定轴归一化分数，保持形状。',
    [num('dim', '归一化轴', -1, -8)],
    ({ in: x }, p) => {
      floating(x);
      axisIndex(n(p, 'dim'), x.shape.length);
      return pass(x, '每个查询对 key 轴上的分数做 Softmax。形状模式不生成权重值。');
    },
  ),
  base(
    'CausalMask',
    'CausalMask',
    '注意力算子',
    '标记因果遮挡，保持分数矩阵形状。',
    [],
    ({ in: x }) => {
      rank(x, 4);
      return pass(x, '屏蔽未来位置。此处只描述形状和遮挡语义，不产生实际注意力数值。');
    },
  ),
];
