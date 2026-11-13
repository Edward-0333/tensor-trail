import type { ModelNode, ModuleDefinition } from '../core/types';
import { tensorValue } from '../core/types';
import { axisIndex, floating, product, rank, requireThat, same, tensor } from './math';
import { bool, input, num, output, text } from './primitives';

export function attentionMathCode(node: ModelNode): string | undefined {
  if (node.type === 'Exp') return 'out = torch.exp(x)\n# 逐元素指数，保持 shape 和 dtype；不裁剪或假设数值有限。';
  if (node.type === 'Softplus') return `out = F.softplus(x, beta=${node.params.beta}, threshold=${node.params.threshold})\n# 默认 log(1 + exp(x))；超过阈值使用线性分支，不是 Softmax，不沿任何轴归一化。`;
  if (node.type === 'L2Norm') return `out = torch.norm(x, p=2, dim=${node.params.dim}, keepdim=${node.params.keepdim ? 'True' : 'False'})\n# 沿指定轴 sqrt(sum(x**2))；这里仅推导规格，不计算真实距离。`;
  if (node.type === 'NoGradResult') return '# @torch.no_grad() 上下文中计算结果的边界标记。\n# 标记已执行运算的结果 requires_grad=False；不是源码中的额外 detach 调用。\n# 不适用于直接返回的输入、Parameter 或显式 requires_grad=True 的工厂例外。\n# 仅声明上下文效果，不执行 Python，也不改变数值或数据来源。';
  if (node.type === 'TorchMultiheadAttention') return `# torch.nn.MultiheadAttention，内部保留 Q/K/V 投影与 out_proj。\nself.attn = nn.MultiheadAttention(${node.params.embed_dims}, ${node.params.num_heads}, dropout=${node.params.dropout}, batch_first=${node.params.batch_first ? 'True' : 'False'})\nout, weights = self.attn(query, key, value, attn_mask=attn_mask, need_weights=True, average_attn_weights=${node.params.average_attn_weights ? 'True' : 'False'})\n# 每头 scores = Q @ K.T / sqrt(head_dim)。\n# 浮点 mask 直接相加到 scores；bool mask 的 True 屏蔽该位置，二者不同。\n# softmax -> attention dropout -> 与 V 加权 -> 合并 heads -> out_proj。\n# weights 是 attention dropout 后的权重，默认沿 head 轴平均。\n# 本接口不含外部位置编码、identity 残差、输出 dropout 或 key_padding_mask。\n# 这里只声明标准层接口，不分配 Tensor、不执行注意力。`;
}

export const attentionMath: ModuleDefinition[] = [
  {
    type: 'Exp', label: '指数 Exp', category: '数学运算',
    description: 'torch.exp 的逐元素指数；保持 shape/dtype，不保证有限值。',
    fields: [], inputs: input, outputs: output,
    infer: ({ in: x }) => { floating(x); return { outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) } }; },
  },
  {
    type: 'Softplus', label: 'Softplus', category: '激活函数',
    description: 'F.softplus：带数值稳定阈值的平滑非负激活，不是 Softmax。',
    fields: [text('beta', 'beta', '1.0'), text('threshold', '线性分支阈值', '20.0')],
    inputs: input, outputs: output,
    infer: ({ in: x }, p) => {
      floating(x);
      requireThat(typeof p.beta === 'string' && p.beta.trim() !== '' && Number.isFinite(Number(p.beta)) && Number(p.beta) > 0, 'beta 需要有限正数。');
      requireThat(typeof p.threshold === 'string' && p.threshold.trim() !== '' && Number.isFinite(Number(p.threshold)), 'threshold 需要有限数值。');
      return { outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) } };
    },
  },
  {
    type: 'L2Norm', label: 'L2 范数', category: '逻辑与归约',
    description: 'torch.norm(x, p=2, dim=...)，沿单轴求欧氏范数，不是 LayerNorm。',
    fields: [num('dim', '归约轴', -1, -8), bool('keepdim', '保留维度', false)], inputs: input, outputs: output,
    infer: ({ in: x }, p) => {
      floating(x);
      requireThat(typeof p.dim === 'number' && Number.isSafeInteger(p.dim), 'dim 需要整数。');
      requireThat(typeof p.keepdim === 'boolean', 'keepdim 需要布尔值。');
      const d = axisIndex(p.dim, x.shape.length), shape = [...x.shape], axes = [...x.axes];
      if (p.keepdim) shape[d] = 1;
      else { shape.splice(d, 1); axes.splice(d, 1); }
      return { outputs: { out: tensor(shape, axes, x.dtype) } };
    },
  },
  {
    type: 'NoGradResult', label: 'no_grad 结果', category: '维度操作',
    description: '标注 no_grad 中已计算结果的梯度边界，不是额外的 detach 运算；数值来源仍保留。',
    fields: [], inputs: input, outputs: output,
    inferValue: ({ in: value }) => {
      const x = tensorValue(value);
      return { outputs: { out: { ...tensor([...x.shape], [...x.axes], x.dtype),
        autograd: { requiresGrad: false, source: 'inferred' } } },
        notes: ['上下文结果标记，不是运行时采集；仅用于 no_grad 内运算的输出，不适用于直接透传输入或工厂例外。'] };
    },
  },
  {
    type: 'TorchMultiheadAttention', label: 'nn.MultiheadAttention', category: 'Transformer',
    description: '标准 PyTorch 多头注意力接口，包含 Q/K/V 与输出投影、加性/布尔掩码及 attention dropout；不含外部位置编码和残差。',
    fields: [num('embed_dims', '特征维数', 256), num('num_heads', '注意力头数', 8),
      text('dropout', 'attention dropout', '0.1'), bool('batch_first', 'batch_first', false),
      bool('average_attn_weights', '平均各头的返回权重', true)],
    inputs: ['query', 'key', 'value', 'attn_mask'].map(id => ({ id, label: id })),
    outputs: [{ id: 'out', label: 'out' }, { id: 'weights', label: 'weights' }],
    inferValue: (values, p) => {
      const D = p.embed_dims, H = p.num_heads;
      requireThat(typeof D === 'number' && Number.isSafeInteger(D) && D > 0 &&
        typeof H === 'number' && Number.isSafeInteger(H) && H > 0 && D % H === 0, '特征维数和头数需为正整数且整除。');
      requireThat(typeof p.dropout === 'string' && p.dropout.trim() !== '' && Number.isFinite(Number(p.dropout)) && Number(p.dropout) >= 0 && Number(p.dropout) <= 1, 'Dropout 概率需要在 [0,1]。');
      requireThat(typeof p.batch_first === 'boolean' && typeof p.average_attn_weights === 'boolean', '布局和平均权重选项需为布尔值。');
      const q = tensorValue(values.query), k = tensorValue(values.key), v = tensorValue(values.value), mask = tensorValue(values.attn_mask);
      [q, k, v].forEach(x => { rank(x, 3); floating(x); });
      requireThat(q.dtype === k.dtype && q.dtype === v.dtype, 'Q/K/V 的 dtype 必须相同。');
      requireThat(mask.dtype === 'bool' || mask.dtype === q.dtype, 'mask 需要 bool 或与 Q 相同的浮点 dtype。');
      const warnings: string[] = [], b = p.batch_first ? 0 : 1, s = 1 - b;
      const B = q.shape[b], Q = q.shape[s], K = k.shape[s];
      for (const x of [q, k, v]) {
        same(x.shape[b], B, 'batch 必须一致', warnings);
        same(x.shape[2], D, '特征维数必须一致', warnings);
      }
      same(v.shape[s], K, 'Key/Value 长度必须一致', warnings);
      requireThat(mask.shape.length === 2 || mask.shape.length === 3, 'attn_mask 需要 [Q,K] 或 [B*heads,Q,K]。');
      const expected = mask.shape.length === 2 ? [Q, K] : [product([B, H]), Q, K];
      expected.forEach((d, i) => same(mask.shape[i], d, 'attn_mask 尺寸', warnings));
      const weightShape = p.average_attn_weights ? [B, Q, K] : [B, H, Q, K];
      return { outputs: {
        out: tensor([...q.shape], [...q.axes], q.dtype),
        weights: tensor(weightShape, p.average_attn_weights ? ['B', 'Q', 'K'] : ['B', 'heads', 'Q', 'K'], q.dtype),
      }, warnings, notes: [
        '此处是标准 nn.MultiheadAttention，不含调用方的位置编码或 identity 残差。',
        mask.dtype === 'bool' ? 'bool mask 的 True 表示禁止注意该位置。' : '浮点 mask 加到 softmax 之前的 attention logits，不是布尔屏蔽或乘法门控。',
        '返回权重来自 attention dropout 之后，batch_first 不改变权重布局；静态图不执行网络、不声明实际 requires_grad。',
      ] };
    },
  },
];
