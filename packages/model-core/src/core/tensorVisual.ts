import { isCodeRegion } from './ifRegions';
import { isLayoutRegion } from './layoutRegions';
import { clone, type ModelNode, type NodeTensorVisual } from './types';

export const tensorVisualModes = ['off', 'block'] as const;
export const tensorDepthStyles = ['auto', 'solid', 'slices'] as const;
export const tensorVisualTones = [
  'auto',
  'blue',
  'violet',
  'teal',
  'amber',
  'rose',
  'slate',
] as const;
export const minTensorVisualWidth = 140;
export const maxTensorVisualWidth = 420;
export const minTensorBoxWidth = 16,
  maxTensorBoxWidth = 160;
export const minTensorBoxHeight = 12,
  maxTensorBoxHeight = 120;
export const minTensorBoxDepth = 6,
  maxTensorBoxDepth = 72;
export const hexColorPattern = '^#[0-9a-fA-F]{6}$';
export const maxAxisLabels = 8,
  maxAxisLabelLength = 32;
const hexColor = new RegExp(hexColorPattern);
const dimensions = {
  width: [minTensorVisualWidth, maxTensorVisualWidth],
  boxWidth: [minTensorBoxWidth, maxTensorBoxWidth],
  boxHeight: [minTensorBoxHeight, maxTensorBoxHeight],
  boxDepth: [minTensorBoxDepth, maxTensorBoxDepth],
};
const visualKeys = new Set([
  'mode',
  'tone',
  'caption',
  'color',
  'depthStyle',
  'axisLabels',
  ...Object.keys(dimensions),
]);
export const supportsTensorVisual = (node: Pick<ModelNode, 'type'>) =>
  !isCodeRegion(node) && !isLayoutRegion(node);

function requireSupport(node: Pick<ModelNode, 'type'>) {
  if (!supportsTensorVisual(node)) throw Error('范围框与布局分区不支持张量积木。');
}

export function validateTensorVisual(
  node: Pick<ModelNode, 'type'>,
  visual: unknown,
): asserts visual is NodeTensorVisual {
  requireSupport(node);
  if (
    !visual ||
    typeof visual !== 'object' ||
    Array.isArray(visual) ||
    !Object.keys(visual).every((key) => visualKeys.has(key))
  )
    throw Error('张量积木配置无效。');
  const fields = visual as Record<string, unknown>;
  const { mode, tone, caption, color, depthStyle, axisLabels } = fields;
  if (!tensorVisualModes.some((value) => value === mode)) throw Error('张量积木显示模式无效。');
  if (tone !== undefined && !tensorVisualTones.some((value) => value === tone))
    throw Error('张量积木配色无效。');
  if (color !== undefined && (typeof color !== 'string' || !hexColor.test(color)))
    throw Error('自定义积木颜色需要 #RRGGBB。');
  if (depthStyle !== undefined && !tensorDepthStyles.some((style) => style === depthStyle))
    throw Error('积木深度显示方式无效。');
  if (
    axisLabels !== undefined &&
    (!Array.isArray(axisLabels) ||
      axisLabels.length > maxAxisLabels ||
      Array.from(axisLabels).some(
        (label) =>
          typeof label !== 'string' ||
          !label.trim() ||
          Array.from(label).length > maxAxisLabelLength,
      ))
  )
    throw Error(
      `最多可设置 ${maxAxisLabels} 个轴名，每个轴名需为 1–${maxAxisLabelLength} 字的非空文本。`,
    );
  if (
    caption !== undefined &&
    (typeof caption !== 'string' ||
      caption.length > 240 ||
      (caption.length > 120 && Array.from(caption).length > 120))
  )
    throw Error('张量积木说明最多 120 字。');
  for (const [key, [min, max]] of Object.entries(dimensions)) {
    const value = fields[key];
    if (
      value !== undefined &&
      (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max)
    )
      throw Error(`张量积木 ${key} 需要 ${min}–${max} 的有限数值。`);
  }
}

export function setNodeTensorVisual(node: ModelNode, visual: unknown) {
  if (visual === null) {
    requireSupport(node);
    delete node.tensorVisual;
    return;
  }
  validateTensorVisual(node, visual);
  node.tensorVisual = clone(visual);
}
