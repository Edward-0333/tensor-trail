import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { isIntegerSpec, isTensorSequenceSpec, isTensorSpec, tensorValue } from '../core/types';
import { deviceValue } from './device';
import { factoryShape } from './factoryShape';
import { viewPorts } from './featureMath';
import { floating, product, rank, requireThat, same, tensor } from './math';
import { input, num, output, select, text } from './primitives';

const noGrad = { requiresGrad: false, source: 'inferred' as const };
const named = (...ids: string[]) => ids.map(id => ({ id, label: id }));
const floats = ['float16', 'bfloat16', 'float32', 'float64'];
function positive(p: Params, key: string) {
  const value = p[key];
  requireThat(typeof value === 'number' && Number.isSafeInteger(value) && value > 0,
    `${key} 必须为正整数。`);
  return value;
}

export const decoderCodeTypes = new Set(['BatchMetaTensor', 'OPUSDecoderLayerContract', 'OPUSDecoderLayerAt']);
export function decoderSupportCode(node: ModelNode): string | undefined {
  const p = node.params;
  if (node.type === 'OPUSDecoderLayerAt') return `# index 仅用于选择模块实例和对应的静态规格，不是 forward 的额外 Tensor 参数。
decoder_layer = self.decoder_layers[index]
query_feat, cls_score, refine_pts = decoder_layer(
    query_points, query_feat, mlvl_feats, occ2img, img_metas)
# 各层独立权重，结构相似；不是反复调用同一份权重。
# num_refines = [${p.refines}][index]
# last_refines = 1 if index == 0 else [${p.refines}][index - 1]
# 这里只声明接口契约，未运行神经网络。`;
  if (node.type === 'Detach') return `out = x.detach()
# 数值和 shape 不变，切断这条边的梯度；不是复制数值，也不删除信息来源标记。
# 后续带参数的计算仍可产生新的可求导输出。`;
  if (node.type === 'ExpandAs') return `out = x.expand_as(like)
# like 只提供目标 shape；不融合 like 的数值，也不复制出多份参数。
# 仅大小为 1 的轴可以扩展，可添加前导轴。`;
  if (node.type === 'BatchMetaTensor') return `values = np.asarray([m[${JSON.stringify(p.field)}] for m in img_metas]).astype(np.float32)
out = torch.tensor(values, dtype=torch.${p.dtype}, device=device)
# 字段及输出 shape=[${p.shape}] 是显式契约，不是从不透明 list[dict] 中实际读取的结果。
# device 来自独立 GetDevice，尺寸端口只供静态校验，不读取参考张量元素。
# new_tensor(values) 的 dtype/device 来自接收张量：画此调用时须核对 dtype 设置。
# 本节点只描述从非 Tensor 数据构造张量，requires_grad=False；不执行 NumPy/PyTorch。`;
  if (node.type === 'OPUSDecoderLayerContract') return `# SparseWorld: opus_transformer.py, OPUSTransformerDecoderLayer.forward
# 仅封装的输入/输出契约；后续可用真实算子子图替换此节点。
query_feat, cls_score, refine_pts = decoder_layer(
    query_points, query_feat, mlvl_feats, occ2img, img_metas)
# 输入 points [B,Q,${p.last_refines},3]；特征 [B,Q,${p.embed_dims}]
# 输出 query_feat [B,Q,${p.embed_dims}]
# 输出 cls_score [B,Q,${p.num_refines},${p.num_classes}]，尚未 sigmoid
# 输出 refine_pts [B,Q,${p.num_refines},3]，归一化坐标
# 内部含位置编码、图像采样、AdaptiveMixing、自注意力、FFN、分类及坐标回归。
# 各层为独立实例；此处不运行网络、不读取权重、不推测训练/推理梯度模式。`;
}

export const decoderSupportModules: ModuleDefinition[] = [
  {
    type: 'Detach', label: 'Detach', category: '维度操作',
    description: 'x.detach()：保留值、shape、dtype，切断该分支梯度；不是清除数据信息来源。',
    fields: [], inputs: input, outputs: output,
    infer: ({ in: x }) => ({
      outputs: { out: { ...tensor([...x.shape], [...x.axes], x.dtype), autograd: noGrad } },
      notes: ['与输入共享存储但不再需要梯度；不模拟真实内存别名，不继承 Parameter/Buffer 注册身份。'],
    }),
  },
  {
    type: 'ExpandAs', label: 'ExpandAs', category: '维度操作',
    description: '将单例轴广播为参考 Tensor 的 shape；参考只提供尺寸，不融合数值。',
    fields: [], inputs: [...input, { id: 'like', label: '目标 shape 参考' }], outputs: output,
    infer: ({ in: x, like }) => {
      requireThat(x.shape.length <= like.shape.length, 'expand_as 不能减少维数。');
      const warnings: string[] = [];
      const pad = like.shape.length - x.shape.length;
      x.shape.forEach((size, axis) => {
        if (size !== 1) same(size, like.shape[axis + pad], 'expand_as 只能扩展单例轴', warnings);
      });
      return { outputs: { out: tensor([...like.shape], [...like.axes], x.dtype) }, warnings,
        notes: ['like 只提供尺寸，不传播其 dtype、梯度或信息来源；扩展视图不等于 repeat 复制。'] };
    },
  },
  {
    type: 'BatchMetaTensor', label: '批量元数据转 Tensor', category: '组合计算',
    description: '收集 list[dict] 指定字段并转成 Tensor。形状和 dtype 为显式契约，不伪造不透明元数据的实际内容。',
    fields: [text('field', '字段名', 'lidar2img'), text('shape', '声明形状', 'd0, d1, 4, 4'),
      text('axes', '轴含义', ''), select('dtype', '构造 dtype（须核对调用）', 'float32', floats)],
    inputs: [], outputs: output,
    getPorts: p => ({ inputs: [
      { id: 'img_metas', label: 'img_metas（数据）' },
      { id: 'device', label: 'device（仅设备元数据）' },
      ...viewPorts(p).inputs.filter(port => port.id !== 'in'),
    ], outputs: output }),
    inferValue: (values, p) => {
      const metas = values.img_metas;
      requireThat(metas && !isTensorSpec(metas) && 'text' in metas &&
        ['forward', 'list'].includes(metas.kind), '需要不透明 list[dict] 元数据输入。');
      requireThat(typeof p.field === 'string' && /^[A-Za-z_]\w{0,63}$/.test(p.field), '字段名无效。');
      requireThat(floats.includes(String(p.dtype)), '需要浮点 dtype。');
      deviceValue(values.device);
      const { shape, axes } = factoryShape(values, p, 'BatchMetaTensor');
      return { outputs: { out: { ...tensor(shape, axes, String(p.dtype)), autograd: noGrad } },
        notes: ['输出 shape/dtype 是配置契约，未采集矩阵数值或验证真实字段长度。',
          '从 NumPy/列表构造新 Tensor，默认不求梯度；数据来源保留自 img_metas，而非尺寸/设备参考。'] };
    },
  },
  {
    type: 'OPUSDecoderLayerContract', label: 'OPUSTransformerDecoderLayer（接口契约）', category: '组合计算',
    description: 'OPUS 解码层的静态输入/输出规格，内部暂不展开。不运行 Python、不制造特征值或运行时梯度。',
    fields: [num('embed_dims', '特征维数', 256), num('num_refines', '本层点数', 1),
      num('last_refines', '输入点数', 1), num('num_classes', '类别数', 17),
      num('num_frames', '帧数', 5), num('num_views', '相机数', 6),
      num('num_groups', '分组数', 4), num('num_levels', '图像尺度数', 4),
      select('layout', '图像布局', 'channel_last', ['channel_last', 'channel_first'])],
    inputs: named('query_points', 'query_feat', 'mlvl_feats', 'occ2img', 'img_metas'),
    outputs: named('query_feat', 'cls_score', 'refine_pts'),
    inferValue: (v, p) => {
      const points = tensorValue(v.query_points), feat = tensorValue(v.query_feat),
        occ = tensorValue(v.occ2img), levels = v.mlvl_feats;
      const D = positive(p, 'embed_dims'), P = positive(p, 'num_refines'),
        previous = positive(p, 'last_refines'), classes = positive(p, 'num_classes'),
        T = positive(p, 'num_frames'), N = positive(p, 'num_views'),
        G = positive(p, 'num_groups'), L = positive(p, 'num_levels');
      requireThat(D % G === 0, '特征维数必须能被分组数整除。');
      requireThat(['channel_last', 'channel_first'].includes(String(p.layout)), '未知图像布局。');
      rank(points, 4); rank(feat, 3); rank(occ, 4);
      [points, feat, occ].forEach(floating);
      requireThat(points.dtype === feat.dtype, '当前接口要求 points/feat dtype 一致。');
      const warnings: string[] = [];
      const check = (actual: typeof points.shape, expected: typeof points.shape, label: string) =>
        actual.forEach((d, i) => same(d, expected[i], label, warnings));
      const [B, Q] = feat.shape;
      check(points.shape, [B, Q, previous, 3], 'query_points 尺寸');
      check(feat.shape, [B, Q, D], 'query_feat 尺寸');
      check(occ.shape, [B, T * N, 4, 4], 'occ2img 尺寸');
      requireThat(levels && isTensorSequenceSpec(levels) && levels.items.length === L,
        'mlvl_feats 必须为指定尺度数的 Tensor list/tuple。');
      for (const level of levels.items) {
        rank(level, 5); floating(level);
        requireThat(level.dtype === feat.dtype, '图像特征与 query_feat dtype 需要一致。');
        const channelsLast = p.layout === 'channel_last';
        check(level.shape, channelsLast
          ? [product([B, T, G]), N, level.shape[2], level.shape[3], D / G]
          : [product([B, T, G]), D / G, N, level.shape[3], level.shape[4]], '采样特征布局');
      }
      const metas = v.img_metas;
      requireThat(metas && !isTensorSpec(metas) && 'text' in metas &&
        ['forward', 'list'].includes(metas.kind), 'img_metas 需要 list[dict] 元数据。');
      return { outputs: {
        query_feat: tensor([B, Q, D], ['B', 'Q', 'D'], feat.dtype),
        cls_score: tensor([B, Q, P, classes], ['B', 'Q', 'P', 'classes'], feat.dtype),
        refine_pts: tensor([B, Q, P, 3], ['B', 'Q', 'P', 'xyz'], points.dtype),
      }, warnings, notes: ['这是未展开网络层的静态契约，不是网络执行或采集结果。',
        '图像、位置、查询特征、投影和图像元数据参与此层；不将来源编号等同于梯度或特征质量。',
        '输出可因可训练参数重新进入计算图，不能由输入 points.detach() 推断输出也无梯度。'] };
    },
  },
];

const layerContract = decoderSupportModules.find(def => def.type === 'OPUSDecoderLayerContract')!;
decoderSupportModules.push({
  ...layerContract,
  type: 'OPUSDecoderLayerAt', label: 'decoder_layers[index]',
  description: '按循环索引选择独立 OPUS 解码层的接口规格，不表示共享权重；内部计算可后续展开。',
  fields: [...layerContract.fields.filter(f => !['num_refines', 'last_refines'].includes(f.key)),
    text('refines', '各层点数（依序）', '1,4,16,24,32,48')],
  inputs: [...layerContract.inputs, { id: 'index', label: 'index（选择层实例，非 forward 参数）' }],
  inferValue: (v, p, context) => {
    const refines = String(p.refines).split(',').map(s => Number(s.trim()));
    requireThat(refines.length > 0 && refines.length <= 16 &&
      refines.every(n => Number.isSafeInteger(n) && n > 0), '各层点数需要 1-16 个正整数。');
    const index = v.index;
    requireThat(index && isIntegerSpec(index) && typeof index.value === 'number' &&
      Number.isSafeInteger(index.value) && index.value >= 0 && index.value < refines.length,
    '层索引需要在各层点数列表范围内的已知整数。');
    const i = index.value;
    const result = layerContract.inferValue!(v, {
      ...p, num_refines: refines[i], last_refines: i === 0 ? 1 : refines[i - 1],
    }, context);
    return { ...result, notes: [...(result.notes ?? []),
      `self.decoder_layers[${i}]：独立参数实例；输入点数 ${i === 0 ? 1 : refines[i - 1]}，输出点数 ${refines[i]}。`] };
  },
});
