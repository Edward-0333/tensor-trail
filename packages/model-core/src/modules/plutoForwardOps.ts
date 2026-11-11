import type {
  ModuleDefinition,
  ModelNode,
  Params,
  TensorSpec,
  DictionarySpec,
} from '../core/types';
import { clone, isTensorSpec, isIntegerSpec, isDictionarySpec, tensorValue } from '../core/types';
import { checkDictionaryKey } from '../core/dictionary';
import { requireThat, rank, same, floating, tensor, axisIndex } from './math';
import { input, output, text, num, bool, select } from './primitives';
import { probability } from './sequenceOps';

const codeOperations = [
  'ReferenceFeatures',
  'ReferencePosition',
  'R2RAttention',
  'M2MAttention',
  'CatEgo',
  'CheckFinite',
] as const;
export const plutoCodeTypes = new Set([
  'PlutoCode',
  'MaskedMultiheadAttention',
  'PlutoModeParameter',
]);
const named = (...ids: string[]) => ids.map((id) => ({ id, label: id }));
function codePorts(p: Params) {
  const inputs =
    p.operation === 'ReferenceFeatures'
      ? named('position', 'vector', 'orientation')
      : p.operation === 'ReferencePosition'
        ? named('position', 'orientation')
        : p.operation === 'R2RAttention'
          ? named('tgt', 'mask')
          : p.operation === 'M2MAttention'
            ? named('tgt', 'mask', 'm_pos')
            : p.operation === 'CatEgo'
              ? named('memory', 'like')
              : input;
  return { inputs, outputs: output };
}
function dictionaryKeys(p: Params): string[] {
  requireThat(typeof p.keys === 'string' && p.keys.length <= 2000, '字段名需要短文本。');
  const keys = p.keys.split(',').map((k) => k.trim());
  requireThat(
    keys.length >= 1 && keys.length <= 32 && new Set(keys).size === keys.length,
    '需要 1–32 个不重复的字段名。',
  );
  keys.forEach((key) => checkDictionaryKey(key));
  requireThat(
    keys.every((k) => /^[A-Za-z][A-Za-z0-9_]*$/.test(k)),
    '字段名需要字母开头，由字母、数字和下划线组成。',
  );
  return keys;
}
export function plutoOperationLabel(n: ModelNode): string | undefined {
  if (n.type === 'DictionaryPack') return '{ … }';
  if (n.type === 'IntegerGreater') return `> ${n.params.threshold}`;
  if (n.type === 'Squeeze') return 'squeeze()';
}
export function plutoForwardCode(n: ModelNode): string | undefined {
  const p = n.params;
  if (n.type === 'DictionaryPack')
    return `out = {\n${dictionaryKeys(p)
      .map((k) => `    "${k}": ${k},`)
      .join('\n')}\n}`;
  if (n.type === 'IntegerGreater')
    return `available = size > ${p.threshold}  # Python bool，非 Tensor`;
  if (n.type === 'Squeeze') return `x.squeeze(${p.dim})`;
  if (n.type === 'ReferencePlanningBranch')
    return `# 当前画布记录 ref_line_available=${p.available ? 'True' : 'False'} 的分支。\n# 修改输入使 R 从 0 变为非 0（或相反）时，切换本节点分支配置。\nif ref_line_available:\n    trajectory, probability = self.planning_decoder(data, enc_data)\nelse:\n    trajectory, probability = None, None\n# 条件由 reference_line.position.shape[1] > 0 决定，不是有效点掩码。`;
  if (n.type === 'PlutoModeParameter')
    return `self.parameter = nn.Parameter(torch.Tensor(${p.shape}))\nnn.init.normal_(self.parameter, mean=0.0, std=0.01)\n# m_emb 和 m_pos 是两份独立可训练参数。`;
  if (n.type === 'MaskedMultiheadAttention')
    return `self.attn = nn.MultiheadAttention(${p.dim}, ${p.num_heads}, dropout=${p.dropout},\n                                   bias=True, add_bias_kv=False, batch_first=True)\ndef forward(self, query, key, value, mask):\n    return self.attn(query, key, value, key_padding_mask=mask)[0]\n# mask=True 的 key 被屏蔽；这里只接出 attn_output。`;
  if (n.type !== 'PlutoCode') return;
  if (p.operation === 'ReferenceFeatures')
    return 'r_feature = torch.cat([\n    r_position - r_position[..., 0:1, :2],\n    r_vector,\n    torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),\n], dim=-1)';
  if (p.operation === 'ReferencePosition')
    return 'r_pos = torch.cat([\n    r_position[:, :, 0],\n    r_orientation[:, :, 0, None],\n], dim=-1)';
  if (p.operation === 'R2RAttention')
    return `self.norm1 = nn.LayerNorm(${p.dim})\nself.r2r_attn = nn.MultiheadAttention(${p.dim}, ${p.num_heads}, dropout=${p.dropout}, batch_first=True)\nself.dropout1 = nn.Dropout(${p.dropout})\n\nbs, R, M, D = tgt.shape\ntgt = tgt.transpose(1, 2).reshape(bs * M, R, D)\ntgt2 = self.norm1(tgt)\ntgt2 = self.r2r_attn(tgt2, tgt2, tgt2,\n    key_padding_mask=mask.repeat(M, 1))[0]\ntgt = tgt + self.dropout1(tgt2)\n# 按源码使用 repeat(M, 1)；不替换为 repeat_interleave。\n# 后续 M2M 前的恢复尺寸合并在本块输出中：\ntgt = tgt.reshape(bs, M, R, D).transpose(1, 2)`;
  if (p.operation === 'M2MAttention')
    return `self.norm2 = nn.LayerNorm(${p.dim})\nself.m2m_attn = nn.MultiheadAttention(${p.dim}, ${p.num_heads}, dropout=${p.dropout}, batch_first=True)\nself.dropout2 = nn.Dropout(${p.dropout})\n\nbs, R, M, D = tgt.shape\ntgt_tmp = tgt.reshape(bs * R, M, D)\ntgt_valid_mask = ~mask.reshape(-1)\ntgt_valid = tgt_tmp[tgt_valid_mask]\ntgt2_valid = self.norm2(tgt_valid)\ntgt2_valid, _ = self.m2m_attn(\n    tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid)\ntgt_valid = tgt_valid + self.dropout2(tgt2_valid)\ntgt = torch.zeros_like(tgt_tmp)\ntgt[tgt_valid_mask] = tgt_valid\ntgt = tgt.reshape(bs, R, M, D)\n# 后续 flatten/view 与 cross-attention 单独绘制。`;
  if (p.operation === 'CatEgo')
    return 'bs, R, M, D = like.shape\nx = memory[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, M, 1)';
  if (p.operation === 'CheckFinite')
    return 'assert torch.isfinite(q).all()\n# 画布仅传播形状；有限值断言需要在 PyTorch 运行时检查。';
}
const match = (a: TensorSpec, b: TensorSpec, w: string[], message: string) => {
  requireThat(a.shape.length === b.shape.length && a.dtype === b.dtype, message);
  a.shape.forEach((d, i) => same(d, b.shape[i], message, w));
};
export const plutoForwardOps: ModuleDefinition[] = [
  {
    type: 'ValueVariable',
    label: '值变量',
    category: '输入输出',
    description: '命名并传递 Tensor、模块引用、tuple/list、容器说明、整数、torch.Size、Python bool 或 None；保留实际类型。',
    fields: [],
    inputs: input,
    outputs: output,
    inferValue: ({ in: v }) => ({ outputs: { out: clone(v) } }),
  },
  {
    type: 'DictionaryPack',
    label: '字典组装',
    category: '组合计算',
    description: '按字段名连接输入并组装字典；支持 Tensor、子字典和 None。',
    fields: [text('keys', '字段名（逗号分隔）', 'trajectory,probability,prediction')],
    inputs: [],
    outputs: output,
    getPorts: (p) => ({ inputs: named(...dictionaryKeys(p)), outputs: output }),
    inferValue: (v, p) => {
      const fields: DictionarySpec['fields'] = {};
      for (const key of dictionaryKeys(p)) {
        const value = v[key];
        requireThat(
          value && (isTensorSpec(value) || isDictionarySpec(value) || value.kind === 'none'),
          `${key} 需要 Tensor、字典或 None。`,
        );
        if (isTensorSpec(value) || isDictionarySpec(value)) fields[key] = clone(value);
        else fields[key] = { kind: 'none' };
      }
      return {
        outputs: { out: { kind: 'dict', fields } },
        notes: ['按名称组装字典；不堆叠 Tensor，不改变各字段形状。'],
      };
    },
  },
  {
    type: 'IntegerGreater',
    label: '整数大于',
    category: '数学运算',
    description: '局部整数与常量比较，输出 Python bool。',
    fields: [num('threshold', '比较常量', 0, 0)],
    inputs: input,
    outputs: output,
    inferValue: ({ in: v }, p) => {
      requireThat(
        v && isIntegerSpec(v) && typeof v.value === 'number',
        '条件判断需要已采集的具体整数尺寸。',
      );
      return { outputs: { out: { kind: 'bool', value: v.value > Number(p.threshold) } } };
    },
  },
  {
    type: 'ReferenceGate',
    label: '参考线分支校验',
    category: '内部',
    description: '检查所绘制分支与实际输入尺寸相符，避免静默使用过期分支。',
    fields: [bool('available', '参考线存在', true)],
    inputs: named('data', 'available'),
    outputs: output,
    inferValue: ({ data, available }, p) => {
      requireThat(
        available && !isTensorSpec(available) && available.kind === 'bool',
        '分支条件必须是 Python bool。',
      );
      requireThat(isDictionarySpec(data), '需要 data 字典。');
      const ref = data.fields.reference_line;
      requireThat(ref && isDictionarySpec(ref), '缺少 reference_line 字典。');
      const position = tensorValue(ref.fields.position);
      rank(position, 4);
      requireThat(
        typeof position.shape[1] === 'number' && position.shape[1] > 0 === available.value,
        '参考线尺寸与条件不一致。',
      );
      requireThat(
        available.value === p.available,
        '输入的参考线可用性已变化，请切换分支节点的“参考线存在”配置。',
      );
      return { outputs: { out: clone(data) } };
    },
  },
  {
    type: 'NoReferenceOutputs',
    label: 'trajectory, probability = None, None',
    category: '内部',
    description: '没有参考线时，不调用 PlanningDecoder。',
    fields: [],
    inputs: input,
    outputs: named('trajectory', 'probability'),
    inferValue: () => ({
      outputs: { trajectory: { kind: 'none' }, probability: { kind: 'none' } },
    }),
  },
  {
    type: 'Squeeze',
    label: '压缩维度 Squeeze',
    category: '维度操作',
    description: '移除指定轴上长度为 1 的维度，其余情况保持形状。',
    fields: [num('dim', '轴', -1, -8)],
    inputs: input,
    outputs: output,
    infer: ({ in: x }, p) => {
      const axis = axisIndex(Number(p.dim), x.shape.length);
      requireThat(typeof x.shape[axis] === 'number', 'Squeeze 需要已知轴长度。');
      return {
        outputs: {
          out:
            x.shape[axis] === 1
              ? tensor(
                  x.shape.filter((_, i) => i !== axis),
                  x.axes.filter((_, i) => i !== axis),
                  x.dtype,
                )
              : clone(x),
        },
      };
    },
  },
  {
    type: 'PlutoModeParameter',
    label: 'PLUTO 模式参数',
    category: '内部',
    description: '独立可训练参数，正态初始化 std=0.01。',
    fields: [text('shape', '形状', '1,1,12,128')],
    inputs: [],
    outputs: output,
    infer: (_, p) => {
      const shape = String(p.shape).split(',').map(Number);
      requireThat(
        shape.length > 0 &&
          shape.length <= 8 &&
          shape.every((x) => Number.isSafeInteger(x) && x > 0),
        '参数形状需要正整数。',
      );
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, i) => `axis${i}`),
            'float32',
          ),
        },
      };
    },
  },
  {
    type: 'MaskedMultiheadAttention',
    label: '掩码多头注意力',
    category: '注意力算子',
    description: 'batch_first=True，显式接入 query/key/value 和 key_padding_mask。',
    fields: [
      num('dim', '特征维度', 128),
      num('num_heads', '注意力头数', 4),
      text('dropout', '注意力 dropout', '0'),
    ],
    inputs: named('query', 'key', 'value', 'mask'),
    outputs: output,
    infer: ({ query: q, key: k, value: v, mask }, p) => {
      const w: string[] = [];
      [q, k, v].forEach((x) => {
        rank(x, 3);
        floating(x);
        same(x.shape[2], Number(p.dim), '注意力特征数不匹配', w);
      });
      probability(p.dropout);
      requireThat(Number(p.dim) % Number(p.num_heads) === 0, 'dim 必须能被头数整除。');
      match(k, v, w, 'key/value 形状与类型需一致');
      same(q.shape[0], k.shape[0], '注意力 batch 不匹配', w);
      requireThat(q.dtype === k.dtype, '注意力 dtype 不匹配。');
      rank(mask, 2);
      requireThat(mask.dtype === 'bool', 'key_padding_mask 必须是 bool。');
      mask.shape.forEach((d, i) => same(d, k.shape[i], 'mask 需匹配 key 的 batch/长度', w));
      return {
        outputs: { out: clone(q) },
        warnings: w,
        notes: ['True 屏蔽 key；输出保留 query 长度。全遮挡导致的 NaN 需运行时核对。'],
      };
    },
  },
  {
    type: 'PlutoCode',
    label: 'PLUTO 函数代码块',
    category: '组合计算',
    description: '固定的 PLUTO 计算模板，公式和代码在右侧展示；不执行 Python。',
    fields: [
      select('operation', '函数', codeOperations[0], [...codeOperations]),
      num('dim', '特征维度', 128),
      num('num_heads', '注意力头数', 4),
      text('dropout', 'dropout', '0.1'),
    ],
    getFields: (p) =>
      p.operation === 'R2RAttention' || p.operation === 'M2MAttention'
        ? [
            select('operation', '函数', codeOperations[0], [...codeOperations]),
            num('dim', '特征维度', 128),
            num('num_heads', '注意力头数', 4),
            text('dropout', 'dropout', '0.1'),
          ]
        : [select('operation', '函数', codeOperations[0], [...codeOperations])],
    ...codePorts({ operation: codeOperations[0] }),
    getPorts: codePorts,
    infer: (v, p) => {
      const w: string[] = [];
      if (p.operation === 'ReferenceFeatures' || p.operation === 'ReferencePosition') {
        const x = v.position,
          o = v.orientation;
        rank(x, 4);
        rank(o, 3);
        floating(x);
        floating(o);
        requireThat(x.dtype === o.dtype, '位置/朝向类型需一致。');
        same(x.shape[3], 2, '参考线位置最后维需要 XY', w);
        o.shape.forEach((d, i) => same(d, x.shape[i], '参考线位置/朝向前缀不匹配', w));
        requireThat(typeof x.shape[2] !== 'number' || x.shape[2] > 0, '需要至少一个参考线采样点。');
        if (p.operation === 'ReferenceFeatures') {
          match(x, v.vector, w, 'reference vector 与 position 需一致');
          return {
            outputs: { out: tensor([...o.shape, 6], [...o.axes, 'feature'], x.dtype) },
            warnings: w,
          };
        }
        return {
          outputs: {
            out: tensor([...x.shape.slice(0, 2), 3], [...x.axes.slice(0, 2), 'pose'], x.dtype),
          },
          warnings: w,
        };
      }
      if (p.operation === 'CatEgo') {
        const x = v.memory,
          q = v.like;
        rank(x, 3);
        rank(q, 4);
        requireThat(x.shape[1] !== 0, '场景必须含 ego。');
        same(x.shape[0], q.shape[0], 'batch 不一致', w);
        same(x.shape[2], q.shape[3], 'dim 不一致', w);
        requireThat(x.dtype === q.dtype, 'dtype 不一致。');
        return { outputs: { out: clone(q) }, warnings: w };
      }
      if (p.operation === 'CheckFinite')
        return {
          outputs: { out: clone(v.in) },
          notes: ['仅保留源码断言的位置；需要 PyTorch 检查实际有限值。'],
        };
      const q = v.tgt,
        mask = v.mask;
      rank(q, 4);
      rank(mask, 2);
      floating(q);
      requireThat(mask.dtype === 'bool', '参考线掩码必须 bool。');
      mask.shape.forEach((d, i) => same(d, q.shape[i], '参考线 mask 必须匹配 [B,R]', w));
      same(q.shape[3], Number(p.dim), '特征维度不一致', w);
      requireThat(Number(p.dim) % Number(p.num_heads) === 0, 'dim 必须能被头数整除。');
      probability(p.dropout);
      if (p.operation === 'M2MAttention') {
        const pos = v.m_pos;
        rank(pos, 3);
        same(pos.shape[0], 1, 'm_pos 首维需为 1', w);
        same(pos.shape[1], q.shape[2], '模式数不一致', w);
        same(pos.shape[2], q.shape[3], 'm_pos dim 不一致', w);
        requireThat(pos.dtype === q.dtype, 'm_pos dtype 不一致。');
      }
      return {
        outputs: { out: clone(q) },
        warnings: w,
        notes: [
          p.operation === 'R2RAttention'
            ? '沿 R 轴注意力；保留源码 mask.repeat(M,1)，输出恢复 [B,R,M,D]。'
            : '筛选有效参考线，沿 M 轴注意力，再零填充回原位置。',
        ],
      };
    },
  },
];
