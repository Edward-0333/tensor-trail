import { isCodeRegion } from './ifRegions';
import { compactLayerTypes } from './nodePresentation';
import { isLayoutRegion } from './layoutRegions';
import { tensorVisualTones } from './tensorVisual';
import type { ModelNode, NodeSubmoduleVisual, SubmoduleIcon, TensorVisualTone } from './types';

export const submoduleIcons = [
  'auto',
  'backbone',
  'attention',
  'cross_attention',
  'mlp',
  'projection',
  'encoder',
  'decoder',
  'head',
  'module',
] as const satisfies readonly SubmoduleIcon[];
export const maxSubmoduleSummaryLength = 120;

export function validateSubmoduleSummary(summary: unknown): asserts summary is string {
  if (
    typeof summary !== 'string' ||
    summary.length > 2 * maxSubmoduleSummaryLength ||
    [...summary].length > maxSubmoduleSummaryLength
  )
    throw Error(`封装模块功能摘要最多 ${maxSubmoduleSummaryLength} 个 Unicode 码点。`);
}

export function validateSubmoduleVisual(
  node: Pick<ModelNode, 'type'>,
  visual: unknown,
): asserts visual is NodeSubmoduleVisual {
  if (node.type !== 'Submodule') throw Error('仅 Submodule 支持封装模块样式。');
  if (
    !visual ||
    typeof visual !== 'object' ||
    Object.prototype.toString.call(visual) !== '[object Object]' ||
    !Object.keys(visual).every((key) => ['icon', 'tone', 'summary'].includes(key))
  )
    throw Error('封装模块样式需要仅含 icon、tone、summary 的对象。');
  const { icon, tone, summary } = visual as Record<string, unknown>;
  if (icon !== undefined && !submoduleIcons.some((value) => value === icon))
    throw Error('封装模块图标原型无效。');
  if (tone !== undefined && !tensorVisualTones.some((value) => value === tone))
    throw Error('封装模块主题色标无效。');
  if (summary !== undefined) validateSubmoduleSummary(summary);
}

/** Recipe and project fields require an object. Only the editing API accepts null. */
export function readNodeSubmoduleVisual(node: ModelNode, value: unknown) {
  validateSubmoduleVisual(node, value);
  const summary = value.summary?.trim();
  if ((value.icon ?? 'auto') === 'auto' && (value.tone ?? 'auto') === 'auto' && !summary) {
    delete node.submoduleVisual;
    return;
  }
  node.submoduleVisual = { ...value, ...(summary !== undefined && { summary }) };
}

export function setNodeSubmoduleVisual(node: ModelNode, visual: unknown) {
  readNodeSubmoduleVisual(node, visual === null ? {} : visual);
}

type Icon = Exclude<SubmoduleIcon, 'auto'>;
const iconTones: Record<Icon, Exclude<TensorVisualTone, 'auto'>> = {
  backbone: 'blue',
  attention: 'violet',
  cross_attention: 'teal',
  mlp: 'amber',
  projection: 'blue',
  encoder: 'teal',
  decoder: 'violet',
  head: 'rose',
  module: 'slate',
};

const nameRules: [RegExp, Icon][] = [
  [/cross[\s_-]*(?:attention|attn)|交叉注意力/i, 'cross_attention'],
  [/attention|注意力/i, 'attention'],
  [/backbone|(?:^|[^a-z])vit(?:$|[^a-z])|v[\s_-]*jepa|resnet|骨干/i, 'backbone'],
  [/mlp|ffn|前馈/i, 'mlp'],
  [/投影|projection|lidar2img|sampling|采样/i, 'projection'],
  [/decoder|解码器/i, 'decoder'],
  [/encoder|编码器/i, 'encoder'],
  [/head|scorer|planner|预测头/i, 'head'],
];
function namedIcon(name: string) {
  return nameRules.find(([pattern]) => pattern.test(name))?.[1];
}

const boundaryOrValue = new Set([
  'Inlet', 'Outlet', 'Input', 'ForwardInput', 'DictionaryInput', 'Output',
  'TensorVariable', 'ValueVariable', 'DeviceVariable', 'LocalParameter',
]);
function inferLayerIcon(steps: ModelNode[]): Icon {
  const types = steps.map((step) => step.type).join(' ');
  const named = namedIcon(types);
  if (named) return named;
  if (/\bConv[123]d\b/i.test(types)) return 'backbone';
  if (/\bLinear\b/.test(types) && /\b(ReLU|GELU|SiLU)\b/i.test(types)) return 'mlp';
  if (/\b(Linear|LinearByDimensions|GridSample)\b/.test(types)) return 'projection';
  return 'module';
}

export function resolveSubmoduleVisual(node: ModelNode) {
  const nodes = node.subgraph?.nodes ?? [];
  const steps = nodes.filter(
    (step) =>
      !step.valueSource &&
      !boundaryOrValue.has(step.type) &&
      !isLayoutRegion(step) &&
      !isCodeRegion(step),
  );
  const visual = node.submoduleVisual;
  const icon =
    visual?.icon && visual.icon !== 'auto'
      ? visual.icon
      : (namedIcon(node.label) ?? namedIcon(node.id) ?? inferLayerIcon(steps));
  const tone = visual?.tone && visual.tone !== 'auto' ? visual.tone : iconTones[icon];
  const semantic = steps.length === 1 && steps[0].type === 'SemanticOp' ? steps[0] : undefined;
  const description =
    typeof semantic?.params.summary === 'string' ? semantic.params.summary.trim() : '';
  const contract = semantic && [semantic.label, description].some((text) =>
    /契约|待展开|未展开|\b(?:placeholder|unexpanded|pending expansion)\b/i.test(
      text.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' '),
    ),
  );
  const status = !nodes.some((n) => n.type === 'Outlet')
    ? 'pending'
    : contract
      ? 'contract'
      : semantic
        ? 'semantic'
        : 'expanded';
  const statusLabel = {
    pending: '待定义输出',
    contract: '待展开 · 契约',
    semantic: '1 个语义步骤',
    expanded: `已展开 · ${steps.length} 步`,
  }[status];
  const layerTypes = [...new Set(steps.map((step) => step.type))];
  const keyLayers = layerTypes.filter((type) => compactLayerTypes.has(type) || namedIcon(type));
  const preview = keyLayers.length && layerTypes.length > 3 ? keyLayers : layerTypes;
  const layers = preview.slice(0, 3).join(' · ') + (preview.length > 3 ? ' 等' : '');
  const summary = visual?.summary?.trim() || description || semantic?.label || layers;
  return { icon, tone, summary, status, statusLabel, steps: steps.length };
}
