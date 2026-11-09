import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { clone } from '../core/types';
import { floating, rank, requireThat, same, tensor } from './math';
import { maskedSelection } from './masking';
import { input, output, num, select, text } from './primitives';

export function probability(value: unknown): number {
  requireThat(typeof value === 'string' && value.trim() !== '', '概率需要填写 0–1 之间的数。');
  const p = Number(value);
  requireThat(Number.isFinite(p) && p >= 0 && p < 1, '概率需要满足 0 ≤ p < 1。');
  return p;
}
const probabilityField = (key: string, label: string, value = '0.0') => ({
  ...text(key, label, value),
  help: '填写 0 ≤ p < 1 的小数。',
});
const codeFields = [
  select('operation', '函数实现', 'NeighborhoodAttention1D', [
    'NeighborhoodAttention1D',
    'DropPath',
    'Interpolate1D',
  ]),
  num('dim', '特征维度', 32),
  num('num_heads', '注意力头数', 2),
  num('kernel_size', '邻域大小', 3),
  probabilityField('attn_drop', '注意力 dropout'),
  probabilityField('proj_drop', '投影 dropout'),
  probabilityField('p', 'DropPath 概率'),
];
const codePorts = (params: Params) => ({
  inputs:
    params.operation === 'Interpolate1D'
      ? [...input, { id: 'like', label: '目标长度参考' }]
      : input,
  outputs: output,
});

/** Fixed, typed function templates. Python is shown for reading, never eval'd. */
export function sequenceCode(node: ModelNode): string | undefined {
  const p = node.params;
  if (node.type === 'Contiguous') return 'def forward(self, x):\n    return x.contiguous()';
  if (node.type === 'BooleanIndex')
    return 'def forward(self, x, mask):\n    return x[mask]  # bool 掩码匹配前若干轴，输出 [N_true, ...剩余轴]';
  if (node.type === 'Dropout')
    return `# 初始化\nself.drop = nn.Dropout(p=${p.p})\n\ndef forward(self, x):\n    return self.drop(x)`;
  if (node.type !== 'CodeFunction') return;
  if (p.operation === 'NeighborhoodAttention1D')
    return `# 初始化：使用 PLUTO 依赖的 natten.NeighborhoodAttention1D\nself.attn = NeighborhoodAttention1D(\n    dim=${p.dim}, num_heads=${p.num_heads}, kernel_size=${p.kernel_size},\n    dilation=None, qkv_bias=True, qk_scale=None,\n    attn_drop=${p.attn_drop}, proj_drop=${p.proj_drop},\n)\n\ndef forward(self, x):\n    # x: [N_valid, L, C]\n    # 内部：Q/K/V 投影 → 局部 QK + 相对位置偏置\n    #      → softmax → attention dropout → 邻域加权 V → 输出投影\n    # 短序列由 NATTEN 补齐到窗口长度，输出裁回原长度。\n    return self.attn(x)`;
  if (p.operation === 'DropPath')
    return `# 初始化：与 PLUTO 的训练/推理开关一致\nself.drop_path = DropPath(${p.p}) if ${p.p} > 0.0 else nn.Identity()\n\ndef forward(self, x):\n    # 训练时按样本随机丢弃整条残差分支；eval 时保持输入。\n    return self.drop_path(x)`;
  if (p.operation === 'Interpolate1D')
    return 'def forward(self, x, like):\n    # 与 NATSequenceEncoder 的 FPN 对齐操作一致\n    return F.interpolate(\n        x, scale_factor=like.shape[-1] / x.shape[-1],\n        mode="linear", align_corners=False,\n    )';
}

export const sequenceOps: ModuleDefinition[] = [
  {
    type: 'BooleanIndex',
    label: '布尔索引',
    category: '索引与切片',
    description:
      'bool 掩码匹配 Tensor 的前若干轴，输出 [N_true, ...剩余轴]。未知 True 数量保留符号。',
    fields: [text('count_symbol', '有效数量符号', 'N_valid')],
    inputs: [...input, { id: 'mask', label: '布尔掩码' }],
    outputs: output,
    infer: ({ in: x, mask }, p) => {
      const warnings: string[] = [];
      const selection = maskedSelection(x, mask, p.count_symbol, warnings);
      return {
        outputs: { out: tensor(selection.shape, selection.axes, x.dtype) },
        warnings,
        notes: [
          `${p.count_symbol} = count_nonzero(mask)，范围为 0…${selection.capacity}；未提供掩码数值，不推测具体数量。`,
        ],
      };
    },
  },
  {
    type: 'Contiguous',
    label: 'Contiguous',
    category: '维度操作',
    description:
      '对应 x.contiguous()。保证连续内存布局，保留形状与 dtype；当前画布只记录语义，不模拟内存复制。',
    fields: [],
    inputs: input,
    outputs: output,
    infer: ({ in: x }) => ({
      outputs: { out: clone(x) },
      notes: ['shape 与 dtype 不变；实际 PyTorch 按需要返回原张量或创建连续副本。'],
    }),
  },
  {
    type: 'Dropout',
    label: 'Dropout',
    category: '基础层',
    description: '训练时逐元素随机丢弃，推理时保持输入；不改变 shape。',
    fields: [probabilityField('p', '丢弃概率')],
    inputs: input,
    outputs: output,
    infer: ({ in: x }, p) => {
      floating(x);
      probability(p.p);
      return {
        outputs: { out: clone(x) },
        notes: ['形状模式不执行随机采样；实际由模块的 training 状态决定。'],
      };
    },
  },
  {
    type: 'CodeFunction',
    label: '函数代码块',
    category: '组合计算',
    description:
      '为邻域注意力、DropPath 和一维线性插值提供有明确接口的函数模板。代码在右侧阅读，画布只显示名称；不执行 Python 或 CUDA。',
    fields: codeFields,
    getFields: (p) =>
      codeFields.filter(
        (f) =>
          f.key === 'operation' ||
          (p.operation === 'DropPath'
            ? f.key === 'p'
            : p.operation === 'NeighborhoodAttention1D'
              ? f.key !== 'p'
              : false),
      ),
    inputs: input,
    outputs: output,
    getPorts: codePorts,
    infer: ({ in: x, like }, p) => {
      floating(x);
      if (p.operation === 'DropPath') {
        probability(p.p);
        requireThat(x.shape.length >= 1, 'DropPath 需要 batch 轴。');
        return {
          outputs: { out: clone(x) },
          notes: ['保留 shape；实际训练时按 batch 样本随机丢弃残差分支，eval 时为恒等映射。'],
        };
      }
      rank(x, 3);
      const warnings: string[] = [];
      if (p.operation === 'NeighborhoodAttention1D') {
        const dim = Number(p.dim),
          heads = Number(p.num_heads),
          kernel = Number(p.kernel_size);
        requireThat(
          Number.isSafeInteger(dim) &&
            dim > 0 &&
            Number.isSafeInteger(heads) &&
            heads > 0 &&
            dim % heads === 0,
          '特征维度必须能被注意力头数整除。',
        );
        requireThat(
          Number.isSafeInteger(kernel) && kernel > 1 && kernel % 2 === 1,
          '邻域大小必须为大于 1 的奇数。',
        );
        probability(p.attn_drop);
        probability(p.proj_drop);
        same(x.shape[2], dim, '邻域注意力输入最后一维必须等于 dim', warnings);
        requireThat(
          typeof x.shape[1] !== 'number' || x.shape[1] > 0,
          '邻域注意力序列长度必须大于 0。',
        );
        return {
          outputs: { out: clone(x) },
          warnings,
          notes: [
            '局部邻域注意力保留 [N,L,C]；包含可训练的 QKV、输出投影和相对位置偏置。',
            '对应本地 NATTEN 0.14.6 接口；短序列会补齐窗口后裁回原长度。函数块不模拟数值计算。',
          ],
        };
      }
      requireThat(p.operation === 'Interpolate1D', '不支持的函数代码块类型。');
      rank(like, 3);
      requireThat(typeof x.shape[2] !== 'number' || x.shape[2] > 0, '插值输入序列长度必须大于 0。');
      requireThat(
        typeof like.shape[2] !== 'number' || like.shape[2] > 0,
        '插值目标序列长度必须大于 0。',
      );
      return {
        outputs: {
          out: tensor(
            [x.shape[0], x.shape[1], like.shape[2]],
            [x.axes[0], x.axes[1], like.axes[2]],
            x.dtype,
          ),
        },
        notes: [
          '只从 like 读取末轴长度，保持 x 的 batch、通道及 dtype。mode=linear，align_corners=False。',
        ],
      };
    },
  },
];
