import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { isSizeSpec, isTensorSpec, shapeText, tensorValue } from '../core/types';
import { requireThat } from './math';
import { input, output, text } from './primitives';

export const imageMetadataCodeTypes = new Set(['TensorShapeSlice', 'UpdateImageMetaInputShape']);

function sliceBound(value: unknown): number | undefined {
  requireThat(typeof value === 'string' && value.length <= 32, '切片边界需要整数文本或留空。');
  const bound = value.trim();
  if (!bound) return undefined;
  requireThat(
    /^-?\d+$/.test(bound) && Number.isSafeInteger(Number(bound)),
    '切片边界需要整数或留空。',
  );
  return Number(bound);
}

function shapeSlice(p: Params) {
  return { start: sliceBound(p.start), end: sliceBound(p.end) };
}

export function imageMetadataCode(node: ModelNode): string | undefined {
  if (node.type === 'TensorShapeSlice') {
    try {
      const { start, end } = shapeSlice(node.params);
      return `input_shape = img.shape[${start ?? ''}:${end ?? ''}]\n# torch.Size 是尺寸元数据，不是 Tensor，不读取像素或参与梯度计算。`;
    } catch {
      return '# 请先修正形状切片边界。';
    }
  }
  if (node.type === 'UpdateImageMetaInputShape')
    return `# SparseWorld: OPUS.extract_feat, opus.py:109-110
for img_meta in img_metas:
    img_meta.update(input_shape=input_shape)
# 原地更新原列表中的每个字典，其他字段保持不变；不修改 img。
# 每个样本写入一个 torch.Size([H, W])，不是为每张相机图像创建一份列表。
# 图中输出表示更新后的状态，不代表 Python 创建了新列表。
# 静态画布只记录更新说明，不模拟未知字典字段或运行时对象引用。`;
}

export const imageMetadataModules: ModuleDefinition[] = [
  {
    type: 'TensorShapeSlice',
    label: '读取形状切片',
    category: '形状查询',
    description: '对应 img.shape[start:end]，输出 tuple-like torch.Size；不是 Tensor 或像素切片。',
    fields: [
      { ...text('start', '开始轴（可留空）', '-2'), maxLength: 32 },
      { ...text('end', '结束轴（不含，可留空）', ''), maxLength: 32 },
    ],
    inputs: input,
    outputs: output,
    inferValue: ({ in: value }, p) => {
      const img = tensorValue(value);
      const { start, end } = shapeSlice(p);
      const dims = img.shape.slice(start, end);
      return {
        outputs: { out: { kind: 'size', dims } },
        notes: [
          `读取轴长度：torch.Size([${dims.join(', ')}])；保留符号尺寸，不读取像素，不修改图像。`,
          '输出是 torch.Size 元数据，不是 Tensor，不参与梯度计算。',
        ],
      };
    },
  },
  {
    type: 'UpdateImageMetaInputShape',
    label: '更新 img_metas.input_shape',
    category: '组合计算',
    description:
      '将 (H,W) 写入 img_metas 每个样本字典的 input_shape。接受 list[dict] 的不透明输入占位，只输出更新说明，不声称采集完整元数据。',
    fields: [],
    inputs: [
      { id: 'img_metas', label: 'img_metas (list[dict])' },
      { id: 'input_shape', label: 'input_shape (torch.Size)' },
    ],
    outputs: output,
    inferValue: ({ img_metas: metas, input_shape: size }) => {
      requireThat(
        metas &&
          !isTensorSpec(metas) &&
          'text' in metas &&
          (metas.kind === 'forward' || metas.kind === 'list'),
        'img_metas 需要 list[dict] 的网络输入或列表占位，不接受 Tensor 或单个字典。',
      );
      requireThat(
        size && isSizeSpec(size) && size.dims.length === 2,
        'input_shape 需要含 H、W 两个尺寸的 torch.Size。',
      );
      const update = `每个 img_meta['input_shape'] = ${shapeText(size)}。`;
      return {
        outputs: {
          out: {
            kind: 'list',
            text: `list[dict] 更新后状态（静态说明，非完整运行时数据）\n${update}\n其他字段保持不变；原地更新，不创建新列表。\n\n输入说明：\n${metas.text}`,
          },
        },
        notes: [
          update,
          '按 img_metas 的样本数遍历，不按展平图像的 B*N 遍历；不推测未采集的列表长度。',
          '输入仅是不透明占位，无法验证真实 list[dict] 内容；这里记录更新契约，未执行 Python。',
          '不改变图像像素、形状或梯度；图中输出是原地修改后的状态版本。',
        ],
      };
    },
  },
];
