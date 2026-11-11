import type { ModuleDefinition, Params } from '../core/types';
import { resultDtype } from './elementwise';
import { floating, rank, requireThat, same, tensor } from './math';
import { bool, text } from './primitives';

export const imageNormalizationFields = [
  {
    ...text('mean', '通道均值 mean', '[123.675, 116.280, 103.530]'),
    maxLength: 256,
    help: '三个浮点常量的 JSON 数组，顺序对应重排后的通道；不是当前 batch 的统计值。',
  },
  {
    ...text('std', '通道标准差 std', '[58.395, 57.120, 57.375]'),
    maxLength: 256,
    help: '三个正浮点常量的 JSON 数组，顺序与 mean 一致。',
  },
  bool('to_rgb', 'BGR 转 RGB', true),
];

function vector(value: unknown, name: string): number[] {
  requireThat(typeof value === 'string' && value.length <= 256, `${name} 需要三个数的 JSON 数组。`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error(`${name} 需要有效的 JSON 数组，例如 [1.0, 2.0, 3.0]。`);
  }
  requireThat(
    Array.isArray(parsed) &&
      parsed.length === 3 &&
      parsed.every(
        (v) => typeof v === 'number' && Number.isFinite(v) && Number.isFinite(Math.fround(v)),
      ),
    `${name} 需要三个可表示为有限 float32 的数。`,
  );
  if (name === 'std')
    requireThat(
      parsed.every((v) => Math.fround(v) > 0),
      'std 的三个值必须大于 0，且转成 float32 后不能为 0。',
    );
  return parsed;
}

function config(p: Params) {
  requireThat(typeof p.to_rgb === 'boolean', 'to_rgb 必须是布尔值。');
  return { mean: vector(p.mean, 'mean'), std: vector(p.std, 'std'), toRgb: p.to_rgb };
}

// Keep float literals even when a user enters whole numbers, matching this template's float statistics.
const pythonVector = (values: number[]) =>
  `[${values.map((v) => (/[.e]/i.test(String(v)) ? String(v) : `${v}.0`)).join(', ')}]`;

export function imageNormalizationCode(p: Params): string {
  try {
    const c = config(p);
    return `# SparseWorld OPUS.extract_feat: opus.py:86-96
# 当前配置中的浮点常量；不是 BatchNorm，不从当前图片计算 mean/std。
# 本模板假设 torch 默认浮点类型为 float32，与当前调试环境一致。
self.data_aug['img_norm_cfg'] = dict(
    mean=${pythonVector(c.mean)},
    std=${pythonVector(c.std)},
    to_rgb=${c.toRgb ? 'True' : 'False'},
)

# 前提：self.data_aug 非 None；图中表示存在 img_norm_cfg 的路径。
# 训练和推理均执行；这里只展示代码，不在 Tensor Trail 执行 Python。
if 'img_norm_cfg' in self.data_aug:
    img_norm_cfg = self.data_aug['img_norm_cfg']
    norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
    norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

    if img_norm_cfg['to_rgb']:
        img = img[:, [2, 1, 0], :, :]  # BGR to RGB

    img = img - norm_mean.reshape(1, 3, 1, 1)
    img = img / norm_std.reshape(1, 3, 1, 1)

# norm_mean / norm_std: [3] -> [1, 3, 1, 1]
# 广播到每张图像、每个空间位置；图像保持 [B*N, 3, H, W]。
# ${c.toRgb ? '当前 to_rgb=True，先将 BGR 重排为 RGB。' : '当前 to_rgb=False，跳过通道重排，沿用输入通道顺序。'}
# 输入 img.float() 为 float32，当前输出也是 float32。
# 没有可训练参数，不除以 255，不裁剪到 [0,1]，不伪造运行快照。`;
  } catch (error) {
    return `# 图像标准化参数无效：${(error as Error).message}\n# 请修正右侧 mean / std / to_rgb；不执行 Python。`;
  }
}

export const inferImageNormalization: NonNullable<ModuleDefinition['infer']> = ({ in: x }, p) => {
  const c = config(p),
    warnings: string[] = [];
  floating(x);
  rank(x, 4);
  same(x.shape[1], 3, '图像标准化需要 NCHW 布局的 3 通道图像', warnings);
  const stats = tensor([1, 3, 1, 1], ['batch', 'channel', 'height', 'width'], 'float32');
  return {
    outputs: { out: tensor([...x.shape], [...x.axes], resultDtype(x, stats)) },
    warnings,
    notes: [
      '对应 opus.py:86-96；假设 data_aug 非 None 且包含 img_norm_cfg，训练和推理均执行。函数块不模拟条件分支。',
      `mean=${JSON.stringify(c.mean)}，std=${JSON.stringify(c.std)}；固定浮点常量，不是当前 batch 的统计量，也不是可训练参数。`,
      '按当前 torch 默认浮点类型 float32 建立 [3] 张量，与 img 同设备，reshape 为 [1,3,1,1]，广播到全部图像和像素。',
      c.toRgb
        ? 'to_rgb=True：img[:, [2,1,0], :, :]，将 BGR 重排为 RGB 后减均值、除标准差。'
        : 'to_rgb=False：保留输入通道顺序，再减均值、除标准差。',
      '形状不变；按 Tensor 类型提升规则推导 dtype。不执行像素运算，不约束输出到 [0,1]，不模拟设备或自动求导。',
    ],
  };
};
