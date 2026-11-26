import { shortText } from './limits';
import {
  clone,
  isIdentifier,
  type Graph,
  type ModelEdge,
  type ModelNode,
  type NodeVisualMedia,
} from './types';
import { isCodeRegion } from './ifRegions';
import { isLayoutRegion } from './layoutRegions';
import { readNodeSubmoduleVisual } from './submoduleVisual';

export const imageSourcePattern = /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/;
export const maxImageSourceLength = 512_000;
export const maxMediaItems = 12;
export const mediaLayouts = ['auto', 'row', 'grid', 'column'] as const;
export const imageEmphases = ['normal', 'highlight', 'dim'] as const;
export const supportsVisualMedia = (node: Pick<ModelNode, 'type'>) =>
  !isCodeRegion(node) && !isLayoutRegion(node);

function objectWithKeys(value: unknown, keys: string[]): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.keys(value).every((key) => keys.includes(key))
  );
}

function requireMediaSupport(node: Pick<ModelNode, 'type'>) {
  if (!supportsVisualMedia(node)) throw Error('范围框与布局分区不支持节点插图。');
}

export function validateVisualMedia(
  node: Pick<ModelNode, 'type'>,
  media: unknown,
): asserts media is NodeVisualMedia {
  requireMediaSupport(node);
  if (
    !objectWithKeys(media, ['items', 'layout', 'tileWidth']) ||
    !Array.isArray(media.items) ||
    media.items.length < 1 ||
    media.items.length > maxMediaItems
  )
    throw Error('节点插图需要 1–12 张图片。');
  if (media.layout !== undefined && !mediaLayouts.some((layout) => layout === media.layout))
    throw Error('节点插图布局无效。');
  if (
    media.tileWidth !== undefined &&
    (typeof media.tileWidth !== 'number' ||
      !Number.isFinite(media.tileWidth) ||
      media.tileWidth < 80 ||
      media.tileWidth > 280)
  )
    throw Error('插图宽度需要 80–280 的有限数值。');
  const ids = new Set<string>();
  for (const item of media.items) {
    if (
      !objectWithKeys(item, ['id', 'src', 'caption', 'badge', 'emphasis']) ||
      !isIdentifier(item.id) ||
      ids.has(item.id)
    )
      throw Error('插图需要同节点内唯一的有效 ID。');
    ids.add(item.id);
    if (
      typeof item.src !== 'string' ||
      item.src.length > maxImageSourceLength ||
      !imageSourcePattern.test(item.src)
    )
      throw Error('插图仅支持 PNG、JPEG、WebP 的 Base64 Data URL，每张最多 512,000 字符。');
    if (item.caption !== undefined && !shortText(item.caption, 120))
      throw Error('插图文字说明最多 120 字。');
    if (item.badge !== undefined && !shortText(item.badge, 32)) throw Error('插图角标最多 32 字。');
    if (item.emphasis !== undefined && !imageEmphases.some((style) => style === item.emphasis))
      throw Error('插图样式无效。');
  }
}

export function setNodeMedia(node: ModelNode, media: unknown) {
  if (media === null) {
    requireMediaSupport(node);
    delete node.visualMedia;
    return;
  }
  validateVisualMedia(node, media);
  node.visualMedia = clone(media);
}

export type EdgePresentationPatch = {
  annotation?: string | null;
  presentation?: ModelEdge['presentation'] | null;
};

export function validateEdgePresentation(annotation: unknown, presentation: unknown) {
  if (annotation !== undefined && !shortText(annotation, 160))
    throw Error('箭头说明文字最多 160 字。');
  if (presentation !== undefined && presentation !== 'default' && presentation !== 'slide' && presentation !== 'anchor')
    throw Error('箭头样式需要 default、slide 或 anchor。');
}

export function applyEdgePresentation(edge: ModelEdge, annotation: unknown, presentation: unknown) {
  validateEdgePresentation(annotation ?? undefined, presentation ?? undefined);
  if (annotation === null || annotation === '') delete edge.annotation;
  else if (typeof annotation === 'string') edge.annotation = annotation;
  if (presentation === null) delete edge.presentation;
  else if (presentation === 'slide' || presentation === 'default' || presentation === 'anchor') edge.presentation = presentation;
}

/** A generated module can rebuild after a parameter edit. Keep artwork only on surviving identities. */
export function preserveGeneratedMedia(before: Graph, after: Graph) {
  const nodes = new Map(before.nodes.map((node) => [node.id, node]));
  for (const node of after.nodes) {
    const previous = nodes.get(node.id);
    if (previous?.type !== node.type) continue;
    if (previous.presentation) node.presentation = previous.presentation;
    if (node.type === 'Submodule' && previous.submoduleVisual !== undefined)
      readNodeSubmoduleVisual(node, previous.submoduleVisual);
    if (previous.tensorVisual) node.tensorVisual = clone(previous.tensorVisual);
    if (previous.visualMedia) node.visualMedia = clone(previous.visualMedia);
    if (previous.subgraph && node.subgraph)
      preserveGeneratedMedia(previous.subgraph, node.subgraph);
  }
  const endpoints = (edge: ModelEdge) =>
    `${edge.source}.${edge.sourcePort}>${edge.target}.${edge.targetPort}`;
  const decorated = new Map(
    before.edges
      .filter((e) => e.annotation || e.presentation)
      .map((edge) => [endpoints(edge), edge]),
  );
  for (const edge of after.edges) {
    const previous = decorated.get(endpoints(edge));
    if (previous?.annotation) edge.annotation = previous.annotation;
    if (previous?.presentation) edge.presentation = previous.presentation;
  }
}
