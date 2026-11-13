import type { ModelNode, ModuleDefinition, Params, ValueSpec } from '../core/types';
import { isIntegerSpec, isTensorSequenceSpec, isTensorSpec, tensorValue } from '../core/types';
import { axisIndex, floating, product, rank, requireThat, same, tensor } from './math';
import { bool, input, num, output, select, text } from './primitives';

const named = (...ids: string[]) => ids.map(id => ({ id, label: id }));
const positive = (p: Params, key: string) => {
  const n = p[key];
  requireThat(typeof n === 'number' && Number.isSafeInteger(n) && n > 0, `${key} 需要正整数。`);
  return n;
};
const integer = (v: ValueSpec | undefined, name: string) => {
  requireThat(v && isIntegerSpec(v) && typeof v.value === 'number' &&
    Number.isSafeInteger(v.value) && v.value > 0, `${name} 需要已知正整数规格，不是 Tensor。`);
  return v.value;
};
function feature(v: ValueSpec | undefined, D: number, warnings: string[]) {
  const x = tensorValue(v);
  rank(x, 3); floating(x);
  same(x.shape[2], D, '查询特征维数', warnings);
  return x;
}
function pointsAndFeature(v: Record<string, ValueSpec>, D: number, warnings: string[]) {
  const feat = feature(v.query_feat, D, warnings), points = tensorValue(v.query_points);
  rank(points, 4); floating(points);
  requireThat(points.dtype === feat.dtype, '查询坐标与特征 dtype 不一致。');
  same(points.shape[0], feat.shape[0], '查询 batch', warnings);
  same(points.shape[1], feat.shape[1], '查询数量', warnings);
  same(points.shape[3], 3, '坐标 xyz 维度', warnings);
  requireThat(typeof points.shape[2] !== 'number' || points.shape[2] > 0, '查询点集合不能为空。');
  return { points, feat };
}

export function decoderLayerOpCode(n: ModelNode): string | undefined {
  const codes: Record<string, string> = {
    LinearByDimensions: '# in_features/out_features 是当前层的构造规格，不是额外的 forward 数据。\nout = selected_linear(x)\n# 对应 nn.Linear(in_features, out_features)；不创建或共享真实权重，不运行 PyTorch。',
    OPUSSamplingContract: 'sampled_feat = self.sampling(query_points, query_feat, mlvl_feats, occ2img, img_metas)\n# 自适应采样偏移/尺度权重，投影到多帧多相机特征图并采样。\n# 输出 [B,Q,G,num_frames*num_points,embed_dims/G]。\n# 这里是该子模块的接口契约；不执行采样，不伪造特征值。',
    AdaptiveMixingContract: 'query_feat = self.mixing(sampled_feat, query_feat)\n# query 生成通道/采样点混合矩阵；混合后 out_proj，再与 query 残差相加。\n# [B,Q,G,FP,C] -> [B,Q,D]；不是平均池化。内部后续展开。',
    OPUSSelfAttentionContract: 'query_feat, _ = self.self_attn(query_points, query_feat)\n# 基于三维点距离和 query 生成的 tau 构造注意力偏置，再执行多头自注意力。\n# 不是无位置约束的通用 MHA；可包含 ind_mask；这里不展开其内部。\n# 仅画被使用的特征返回值，忽略的第二返回值不接入主数据流。',
    OPUSCrossAttentionContract: 'fused_ego_feat, _ = self.ego_cross_attn(query_points, query_feat, key_points, key_feat)\n# OPUSCrossAttention：距离偏置、多头交叉注意力、位置编码及 query 残差。\n# query_feat 为自车特征；key_feat 同时提供 Key/Value，key_points 为每个场景 query 的点集合。\n# 仅展示被使用的第一个返回值；第二个注意力权重返回值被 _ 忽略。\n# 内部暂不展开，不运行注意力，也不据 detach 后的 Key/Value 断言输出无梯度。',
    MMCVFFNContract: 'out = self.ffn(x)\n# Linear(D, hidden) -> ReLU -> Dropout(p) -> Linear(hidden, D) -> Dropout(p)\n# 默认 add_identity=True：out = x + layers(x)。\n# 不等同于没有残差的两层 MLP；仅声明接口，不执行网络。',
    PointRangeTransform: n.params.mode === 'decode'
      ? 'out = points * (pc_max - pc_min) + pc_min\n# decode_points：归一化坐标 -> 米制坐标，逐 xyz 轴处理，不改变 shape。'
      : 'out = (points - pc_min) / (pc_max - pc_min)\n# encode_points：米制坐标 -> 归一化坐标；没有 clamp/sigmoid。',
    ReduceMean: `out = x.mean(dim=${n.params.dim}, keepdim=${n.params.keepdim ? 'True' : 'False'})\n# 沿该轴求均值；只推导形状，不计算真实数值。`,
  };
  return codes[n.type];
}

export const decoderLayerOps: ModuleDefinition[] = [
  {
    type: 'ReduceMean', label: '均值 Mean', category: '逻辑与归约',
    description: '沿单个轴求浮点均值，可保留该轴；空轴的实际值可能为 NaN，这里不计算数值。',
    fields: [num('dim', '归约轴', 2, -8), bool('keepdim', '保留轴', true)],
    inputs: input, outputs: output,
    infer: ({ in: x }, p) => {
      floating(x);
      const axis = axisIndex(Number(p.dim), x.shape.length), shape = [...x.shape], axes = [...x.axes];
      if (p.keepdim) shape[axis] = 1;
      else { shape.splice(axis, 1); axes.splice(axis, 1); }
      return { outputs: { out: tensor(shape, axes, x.dtype) } };
    },
  },
  {
    type: 'LinearByDimensions', label: 'Linear（动态规格）', category: '基础层',
    description: '当前层的 Linear，特征维数来自局部整数配置。各轮选择不同层实例，不是修改一份权重的形状。',
    fields: [], inputs: named('in', 'in_features', 'out_features'), outputs: output,
    inferValue: v => {
      const x = tensorValue(v.in), I = integer(v.in_features, 'in_features'), O = integer(v.out_features, 'out_features');
      floating(x); requireThat(x.shape.length > 0, 'Linear 不接受零维 Tensor。');
      const warnings: string[] = [];
      same(x.shape.at(-1)!, I, 'Linear 输入特征数', warnings);
      return { outputs: { out: tensor([...x.shape.slice(0, -1), O], [...x.axes.slice(0, -1), 'feature'], x.dtype) }, warnings,
        notes: [`nn.Linear(${I}, ${O})；整数端口只描述构造规格，不作为融合的数据来源。`] };
    },
  },
  {
    type: 'OPUSSamplingContract', label: 'OPUSSampling', category: 'Transformer',
    description: '自适应时空采样的接口规格；内部暂不展开，输出分组采样特征。',
    fields: [num('embed_dims', '特征维数', 256), num('num_frames', '帧数', 5), num('num_views', '相机数', 6),
      num('num_groups', '分组数', 4), num('num_points', '每组每帧采样点', 4), num('num_levels', '特征尺度数', 4),
      select('layout', '采样特征布局', 'channel_last', ['channel_last', 'channel_first'])],
    inputs: named('query_points', 'query_feat', 'mlvl_feats', 'occ2img', 'img_metas'), outputs: output,
    inferValue: (v, p) => {
      const D = positive(p, 'embed_dims'), T = positive(p, 'num_frames'), N = positive(p, 'num_views'),
        G = positive(p, 'num_groups'), P = positive(p, 'num_points'), L = positive(p, 'num_levels');
      requireThat(D % G === 0, 'embed_dims 必须能被 num_groups 整除。');
      requireThat(['channel_first', 'channel_last'].includes(String(p.layout)), '未知采样布局。');
      const warnings: string[] = [], { feat } = pointsAndFeature(v, D, warnings), [B, Q] = feat.shape;
      const occ = tensorValue(v.occ2img); rank(occ, 4); floating(occ);
      const check = (actual: typeof feat.shape, expected: typeof feat.shape, label: string) =>
        actual.forEach((d, i) => same(d, expected[i], label, warnings));
      check(occ.shape, [B, T * N, 4, 4], 'occ2img 尺寸');
      const levels = v.mlvl_feats;
      requireThat(levels && isTensorSequenceSpec(levels) && levels.items.length === L, '采样需要指定数量的多尺度 Tensor 列表。');
      levels.items.forEach(x => {
        rank(x, 5); floating(x); requireThat(x.dtype === feat.dtype, '图像特征 dtype 不一致。');
        check(x.shape, p.layout === 'channel_last'
          ? [product([B, T, G]), N, x.shape[2], x.shape[3], D / G]
          : [product([B, T, G]), D / G, N, x.shape[3], x.shape[4]], '采样图像布局');
      });
      const meta = v.img_metas;
      requireThat(meta && !isTensorSpec(meta) && 'text' in meta && ['forward', 'list'].includes(meta.kind), 'img_metas 需要 list[dict] 元数据。');
      return { outputs: { out: tensor([B, Q, G, T * P, D / G], ['B', 'Q', 'G', 'FP', 'C'], feat.dtype) }, warnings,
        notes: ['接口契约，不运行采样。图像特征、点坐标、query、投影与图像元数据参与输出。',
          'num_points 是采样点数，与每层预测点数 num_refines 不同。'] };
    },
  },
  {
    type: 'AdaptiveMixingContract', label: 'AdaptiveMixing', category: 'Transformer',
    description: '分组通道混合、采样点混合、输出投影及 query 残差，内部暂不展开。',
    fields: [num('embed_dims', '特征维数', 256), num('num_groups', '分组数', 4), num('in_points', '每组输入点数 FP', 20), num('out_points', '内部混合点数', 32)],
    inputs: named('sampled_feat', 'query_feat'), outputs: output,
    inferValue: (v, p) => {
      const D = positive(p, 'embed_dims'), G = positive(p, 'num_groups'), P = positive(p, 'in_points');
      positive(p, 'out_points'); requireThat(D % G === 0, '混合特征维数必须被分组数整除。');
      const warnings: string[] = [], feat = feature(v.query_feat, D, warnings), x = tensorValue(v.sampled_feat);
      rank(x, 5); floating(x); requireThat(x.dtype === feat.dtype, '混合输入 dtype 不一致。');
      const expected = [feat.shape[0], feat.shape[1], G, P, D / G];
      x.shape.forEach((d, i) => same(d, expected[i], 'AdaptiveMixing 输入尺寸', warnings));
      return { outputs: { out: tensor([...feat.shape], [...feat.axes], feat.dtype) }, warnings,
        notes: ['out_proj 后加上原 query；内部归一化和 ReLU 保留在此封装中。'] };
    },
  },
  {
    type: 'OPUSSelfAttentionContract', label: 'OPUSSelfAttention', category: 'Transformer',
    description: '带距离偏置的多头自注意力接口；不是通用无偏置 MHA。这里只输出实际使用的 query_feat。',
    fields: [num('embed_dims', '特征维数', 256), num('num_heads', '注意力头数', 8)],
    inputs: named('query_points', 'query_feat'), outputs: output,
    inferValue: (v, p) => {
      const D = positive(p, 'embed_dims'), H = positive(p, 'num_heads');
      requireThat(D % H === 0, '注意力维数必须能被头数整除。');
      const warnings: string[] = [], { feat } = pointsAndFeature(v, D, warnings);
      return { outputs: { out: tensor([...feat.shape], [...feat.axes], feat.dtype) }, warnings,
        notes: ['距离计算 no_grad 不等于注意力输出不参与梯度；不推测实际 grad mode。',
          '包括内部注意力残差；外层 norm2 单独绘制。'] };
    },
  },
  {
    type: 'OPUSCrossAttentionContract', label: 'OPUSCrossAttention', category: 'Transformer',
    description: '带点距离偏置与位置编码的多头交叉注意力接口；Query 和 Key 数量可以不同，返回融合后的 Query 特征。',
    fields: [num('embed_dims', '特征维数', 256), num('num_heads', '注意力头数', 8)],
    inputs: named('query_points', 'query_feat', 'key_points', 'key_feat'), outputs: output,
    inferValue: (v, p) => {
      const D = positive(p, 'embed_dims'), H = positive(p, 'num_heads');
      requireThat(D % H === 0, '注意力维数必须能被头数整除。');
      const warnings: string[] = [], query = feature(v.query_feat, D, warnings);
      const points = tensorValue(v.query_points);
      rank(points, 3); floating(points);
      const { feat: key } = pointsAndFeature({ query_feat: v.key_feat, query_points: v.key_points }, D, warnings);
      requireThat(points.dtype === query.dtype && key.dtype === query.dtype, '交叉注意力坐标与特征 dtype 不一致。');
      same(points.shape[0], query.shape[0], 'Query batch', warnings);
      same(points.shape[1], query.shape[1], 'Query 数量', warnings);
      same(points.shape[2], 3, 'Query 坐标 xyz', warnings);
      same(key.shape[0], query.shape[0], 'Key batch', warnings);
      requireThat(typeof key.shape[1] !== 'number' || key.shape[1] > 0, 'Key 查询集合不能为空。');
      return { outputs: { out: tensor([...query.shape], [...query.axes], query.dtype) }, warnings,
        notes: ['Query/Key 数量不必相同：输出与 query_feat 同形；Key/Value 来自 key_feat。',
          '内部包含距离偏置、位置编码、多头注意力和 Query 残差；不是简单拼接或平均池化。',
          '第二返回值 attention weights 被 _ 忽略；不绘制其后续消费。',
          '静态接口契约，不执行网络；输出梯度状态需结合真实 grad mode 与参数状态，不直接继承被 detach 的 Key/Value。'] };
    },
  },
  {
    type: 'MMCVFFNContract', label: 'MMCV FFN（含残差）', category: 'Transformer',
    description: 'MMCV FFN 默认两层 Linear、ReLU、Dropout 以及输入残差的接口规格。',
    fields: [num('embed_dims', '特征维数', 256), num('hidden_dims', '隐藏维数', 512), text('dropout', 'Dropout 概率', '0.1')],
    inputs: input, outputs: output,
    inferValue: (v, p) => {
      const D = positive(p, 'embed_dims'); positive(p, 'hidden_dims');
      const drop = Number(p.dropout);
      requireThat(String(p.dropout).trim() !== '' && Number.isFinite(drop) && drop >= 0 && drop <= 1, 'Dropout 概率需在 [0,1]。');
      const warnings: string[] = [], x = feature(v.in, D, warnings);
      return { outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) }, warnings,
        notes: [`Linear(${D},${p.hidden_dims}) / ReLU / Dropout(${drop}) / Linear / Dropout / Add identity。`,
          '训练时有随机 Dropout，推理时关闭；图中不执行网络或声明实际训练状态。'] };
    },
  },
  {
    type: 'PointRangeTransform', label: '坐标范围变换', category: '维度操作',
    description: '按 xyz 的物理范围进行仿射编码/解码；shape 不变，没有 clamp 或 sigmoid。',
    fields: [select('mode', '方向', 'decode', ['decode', 'encode']), text('pc_range', 'xyz 最小值及最大值', '-40,-40,-1,40,40,5.4')],
    inputs: input, outputs: output,
    inferValue: (v, p) => {
      const x = tensorValue(v.in); floating(x);
      requireThat(x.shape.at(-1) === 3, '坐标最后一维需要 xyz=3。');
      const range = String(p.pc_range).split(',').map(s => Number(s.trim()));
      requireThat(range.length === 6 && String(p.pc_range).split(',').every(s => s.trim() !== '') &&
        range.every(Number.isFinite) && range.slice(0, 3).every((n, i) => n < range[i + 3]), 'pc_range 需要有效的三个递增坐标范围。');
      requireThat(['decode', 'encode'].includes(String(p.mode)), '坐标变换方向无效。');
      return { outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) },
        notes: [p.mode === 'decode' ? '归一化坐标映射到米制坐标。' : '米制坐标映射到归一化坐标；不裁剪范围。'] };
    },
  },
];
