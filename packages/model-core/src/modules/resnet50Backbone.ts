import type { Dim, Inference, TensorSpec } from '../core/types';
import { floating, rank, requireThat, same, tensor } from './math';

const stages = [
  { id: 'c2', channels: 256, stride: 4 },
  { id: 'c3', channels: 512, stride: 8 },
  { id: 'c4', channels: 1024, stride: 16 },
  { id: 'c5', channels: 2048, stride: 32 },
];

export const resnet50Ports = stages.map((stage, index) => ({
  id: stage.id,
  label: `img_feats[${index}] / ${stage.id.toUpperCase()}`,
}));

export const resnet50Code = `# 固定接口模板：SparseWorld 当前配置的 MMDetection ResNet-50。
# 只画模块名称和四级输出，不展开卷积、残差块，不执行 Python。
img_backbone = dict(
    type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
    frozen_stages=1, norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True, style='pytorch', with_cp=True,
)

# OPUS.extract_img_feat, opus.py:58-64
# 当前 use_grid_mask=False，跳过 img = self.grid_mask(img)。
img_feats = self.img_backbone(img)
# 实际 img_feats 是 tuple(Tensor, Tensor, Tensor, Tensor)，不是 dict。
# 因此 isinstance(img_feats, dict) 为 False，不执行 list(img_feats.values())。
# 图中 c2/c3/c4/c5 是这个 tuple 的四个元素，不是四次 backbone 调用。
# 此处尚未经过 FPN，也没有恢复 [B, N, C, H, W]。
# stem 与 layer1 的参数冻结；layer2-4 可训练；norm_eval 固定 BN 统计量。
# 是否记录梯度还取决于 no_grad、输入及运行上下文；不伪造输出梯度状态。`;

function downsample(size: Dim, stride: number): Dim {
  return typeof size === 'number' ? Math.ceil(size / stride) : `ceil((${size})/${stride})`;
}

export function inferResNet50Backbone(x: TensorSpec): Inference {
  floating(x);
  rank(x, 4);
  const warnings: string[] = [];
  same(x.shape[1], 3, 'ResNet-50 模板需要 3 通道 NCHW 图像', warnings);
  for (const dim of x.shape)
    requireThat(
      typeof dim !== 'number' || (Number.isSafeInteger(dim) && dim > 0),
      'ResNet-50 输入尺寸需要为正整数或未定符号。',
    );
  return {
    outputs: Object.fromEntries(
      stages.map((stage) => [
        stage.id,
        tensor(
          [
            x.shape[0],
            stage.channels,
            downsample(x.shape[2], stage.stride),
            downsample(x.shape[3], stage.stride),
          ],
          [x.axes[0] ?? 'batch_view', 'channel', 'height', 'width'],
          x.dtype,
        ),
      ]),
    ),
    warnings,
    notes: [
      '固定契约：MMDetection ResNet-50，3 通道输入、常规 stem、strides=(1,2,2,2)、dilations=(1,1,1,1)，四个 stage 全部输出。',
      'img_feats 是包含 4 个 Tensor 的 tuple；画布端口 c2/c3/c4/c5 分别展示 img_feats[0]/[1]/[2]/[3]，不堆叠成一个 Tensor。',
      '通道分别为 256/512/1024/2048；总步长分别为 4/8/16/32。奇数尺寸按 ceil(H/stride)、ceil(W/stride) 推导，首维保持不变。',
      '固定训练配置 frozen_stages=1、norm_eval=True、with_cp=True；参数可训练性与实际 Tensor.requires_grad 是不同概念，输出梯度状态保留未知。',
      '这是 backbone 输出，尚未经过 FPN 或外层 batch/view 恢复。仅推导 shape/dtype，不执行网络或声称采集到实际特征。',
    ],
  };
}
