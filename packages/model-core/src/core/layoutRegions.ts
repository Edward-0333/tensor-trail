import { shortText } from './limits';
import type { Graph, ModelNode } from './types';
import { ifRegionMembers, isCodeRegion } from './ifRegions';
import { isValueNode } from './valueLinks';

export type NodeSizes = Record<string, { width: number; height: number }>;
export type NodeMove = { id: string; position: { x: number; y: number } };
export type RegionRect = { x: number; y: number; width: number; height: number };
export const REGION_PADDING = 24;
export const REGION_HEADER = 60;
export const isLayoutRegion = (node: Pick<ModelNode, 'type'>) => node.type === 'LayoutRegion';
export const canJoinLayoutRegion = (node: ModelNode) => !isLayoutRegion(node) && !isCodeRegion(node);
export const layoutRegionMembers = ifRegionMembers;
export const REGION_IMPORTANCE_LABELS = ['未评级', '辅助信息', '数据整理', '关键支撑', '重要计算', '核心计算'] as const;
export const layoutRegionImportance = (node: ModelNode): number => Number(node.params.importance ?? 0);
export const layoutRegionOutputs = (node: ModelNode): string[] =>
  ifRegionMembers({ ...node, params: { members: node.params.outputs ?? '' } });
export const layoutRegionOf = (graph: Graph, id: string) =>
  graph.nodes.find(node => isLayoutRegion(node) && layoutRegionMembers(node).includes(id));

export function layoutRegionOutputOwners(graph: Graph): Map<string, ModelNode> {
  return new Map(graph.nodes.filter(isLayoutRegion).flatMap(region =>
    layoutRegionOutputs(region).map(id => [id, region] as const)));
}

/** Display roles include the linked owner's role, without changing membership. */
export function layoutRegionOutputRegions(graph: Graph): Map<string, ModelNode[]> {
  const explicit = layoutRegionOutputOwners(graph);
  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  const result = new Map<string, ModelNode[]>();
  for (const node of graph.nodes) {
    const local = explicit.get(node.id), regions = local ? [local] : [];
    const source = node.valueSource === undefined ? undefined : nodes.get(node.valueSource);
    if (source && source.id !== node.id && source.valueSource === undefined
      && isValueNode(node.type) && source.type === node.type) {
      const inherited = explicit.get(source.id);
      if (inherited && inherited.id !== local?.id) regions.push(inherited);
    }
    if (regions.length) result.set(node.id, regions);
  }
  return result;
}

function pruneRegionOutputs(region: ModelNode): void {
  if (region.params.outputs === undefined) return;
  const members = new Set(layoutRegionMembers(region));
  region.params.outputs = layoutRegionOutputs(region).filter(id => members.has(id)).join(',');
}

export function checkLayoutRegion(node: ModelNode): void {
  if (!['auto', 'manual'].includes(String(node.params.sizing))) throw Error('分区尺寸模式需要 auto 或 manual。');
  for (const key of ['width', 'height']) {
    const value = node.params[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < (key === 'width' ? 360 : 200) || value > 1000000)
      throw Error('分区宽度至少 360，高度至少 200，最大为 1000000。');
  }
  if (!/^#[0-9a-f]{6}$/i.test(String(node.params.color))) throw Error('分区颜色需要 #RRGGBB。');
  if (node.params.note !== undefined && !shortText(node.params.note, 240))
    throw Error('分区注释需要不超过 240 字符的文本。');
  if (node.params.outputs !== undefined && typeof node.params.outputs !== 'string')
    throw Error('分区输出需要以逗号分隔的节点 ID 文本。');
  if (node.params.importance !== undefined && (typeof node.params.importance !== 'number'
    || !Number.isInteger(node.params.importance) || node.params.importance < 0 || node.params.importance > 5))
    throw Error('分区重要性需要 1-5 的整数；0 表示未评级，仅在当前层级比较。');
  if (node.params.importance_note !== undefined && !shortText(node.params.importance_note, 240))
    throw Error('评级依据需要不超过 240 字符的文本。');
  layoutRegionMembers(node);
  layoutRegionOutputs(node);
}

export function layoutRegionIssues(graph: Graph): string[] {
  const issues: string[] = [], used = new Set<string>();
  for (const region of graph.nodes.filter(isLayoutRegion)) {
    try {
      checkLayoutRegion(region);
      for (const id of layoutRegionMembers(region)) {
        const node = graph.nodes.find(n => n.id === id);
        if (!node || !canJoinLayoutRegion(node)) issues.push(`${region.label}：分区只接收同层普通节点，不嵌套分区或 if/for 范围框。`);
        if (used.has(id)) issues.push(`${id} 只能属于一个布局分区。`);
        used.add(id);
      }
      const members = new Set(layoutRegionMembers(region));
      if (layoutRegionOutputs(region).some(id => !members.has(id)))
        issues.push(`${region.label}：分区输出必须是本分区的成员。`);
    } catch (error) { issues.push((error as Error).message); }
  }
  return issues;
}

export function pruneLayoutRegions(graph: Graph): void {
  const ids = new Set(graph.nodes.filter(canJoinLayoutRegion).map(n => n.id));
  for (const region of graph.nodes.filter(isLayoutRegion)) {
    region.params.members = layoutRegionMembers(region).filter(id => ids.has(id)).join(',');
    pruneRegionOutputs(region);
  }
}

const nodeSize = (id: string, sizes: NodeSizes) => {
  const size = sizes[id];
  return size && Number.isFinite(size.width) && size.width > 0 && Number.isFinite(size.height) && size.height > 0
    ? size : { width: 250, height: 160 };
};

/** Auto regions grow around members in all directions without moving the members. */
export function layoutRegionRect(graph: Graph, region: ModelNode, sizes: NodeSizes = {}): RegionRect {
  const ids = new Set(layoutRegionMembers(region));
  const members = graph.nodes.filter(n => ids.has(n.id));
  const auto = region.params.sizing === 'auto';
  const x = auto ? Math.min(region.position.x, ...members.map(n => n.position.x - REGION_PADDING)) : region.position.x;
  const y = auto ? Math.min(region.position.y, ...members.map(n => n.position.y - REGION_HEADER)) : region.position.y;
  return { x, y,
    width: Math.max(360, region.position.x + Number(region.params.width) - x,
      ...members.map(n => n.position.x - x + nodeSize(n.id, sizes).width + REGION_PADDING)),
    height: Math.max(200, region.position.y + Number(region.params.height) - y,
      ...members.map(n => n.position.y - y + nodeSize(n.id, sizes).height + REGION_PADDING)),
  };
}

/** Manual regions constrain members; auto regions expand while retaining membership. */
export function constrainLayoutMoves(graph: Graph, moves: NodeMove[], sizes: NodeSizes = {}): NodeMove[] {
  const proposed = new Map(moves.map(move => [move.id, { ...move.position }]));
  const owners = new Map<string, ModelNode>();
  for (const region of graph.nodes.filter(isLayoutRegion)) {
    const target = proposed.get(region.id);
    for (const id of layoutRegionMembers(region)) {
      owners.set(id, region);
      if (target) {
        const member = graph.nodes.find(n => n.id === id);
        if (member) proposed.set(id, { x: member.position.x + target.x - region.position.x,
          y: member.position.y + target.y - region.position.y });
      }
    }
  }
  for (const [id, position] of proposed) {
    const region = owners.get(id);
    if (!region || region.params.sizing === 'auto' || proposed.has(region.id)) continue;
    const rect = layoutRegionRect(graph, region, sizes), size = nodeSize(id, sizes);
    position.x = Math.max(rect.x + REGION_PADDING, Math.min(position.x, rect.x + rect.width - REGION_PADDING - size.width));
    position.y = Math.max(rect.y + REGION_HEADER, Math.min(position.y, rect.y + rect.height - REGION_PADDING - size.height));
  }
  return [...proposed].map(([id, position]) => ({ id, position }));
}

/** Preserve internal hand routes on rigid moves; invalidate only routes with a moved endpoint. */
function applyMoves(graph: Graph, moves: NodeMove[]): void {
  const deltas = new Map<string, { x: number; y: number }>();
  for (const move of moves) {
    if (!Number.isFinite(move.position.x) || !Number.isFinite(move.position.y)) throw Error('节点坐标必须是有限数值。');
    const node = graph.nodes.find(n => n.id === move.id);
    if (!node) throw Error(`节点 ${move.id} 不存在。`);
    const delta = { x: move.position.x - node.position.x, y: move.position.y - node.position.y };
    if (delta.x || delta.y) deltas.set(move.id, delta);
    node.position = { ...move.position };
  }
  for (const edge of graph.edges) {
    const a = deltas.get(edge.source), b = deltas.get(edge.target);
    if (!a && !b) continue;
    // Subtracting translated fractional positions can differ by floating-point roundoff.
    if (a && b && Math.abs(a.x - b.x) < 1e-7 && Math.abs(a.y - b.y) < 1e-7) {
      if (edge.waypoints) edge.waypoints = edge.waypoints.map(p => ({ x: p.x + a.x, y: p.y + a.y }));
    } else { delete edge.waypoints; delete edge.labelPosition; }
  }
}

export function moveLayoutNodes(graph: Graph, moves: NodeMove[], sizes: NodeSizes = {}): void {
  const constrained = constrainLayoutMoves(graph, moves, sizes);
  // Keep legacy ungrouped drag routing untouched.
  const grouped = moves.some(m => isLayoutRegion(graph.nodes.find(n => n.id === m.id) ?? { type: '' }) || layoutRegionOf(graph, m.id));
  if (grouped) applyMoves(graph, constrained);
  else for (const move of constrained) {
    const node = graph.nodes.find(n => n.id === move.id);
    if (node) node.position = { ...move.position };
  }
  const moved = new Set(constrained.map(move => move.id));
  for (const region of graph.nodes.filter(isLayoutRegion)) {
    if (region.params.sizing !== 'auto' || !(moved.has(region.id) || layoutRegionMembers(region).some(id => moved.has(id)))) continue;
    const rect = layoutRegionRect(graph, region, sizes);
    // Growing the frame is not a group translation: other members stay in place.
    region.position = { x: rect.x, y: rect.y };
    region.params.width = rect.width; region.params.height = rect.height;
  }
}

export function assignLayoutRegion(graph: Graph, ids: string[], regionId: string | null, sizes: NodeSizes = {}): void {
  const selected = [...new Set(ids)];
  if (!selected.length) return;
  const members = selected.map(id => {
    const node = graph.nodes.find(n => n.id === id);
    if (!node || !canJoinLayoutRegion(node)) throw Error('请选择同层普通模块；布局分区与 if/for 范围框不能作为分区成员。');
    return node;
  });
  const region = regionId === null ? undefined : graph.nodes.find(n => n.id === regionId && isLayoutRegion(n));
  if (regionId !== null && !region) throw Error('目标布局分区不存在。');
  const existing = new Set(region ? layoutRegionMembers(region) : []);
  for (const other of graph.nodes.filter(isLayoutRegion)) {
    if (other.id !== regionId) {
      other.params.members = layoutRegionMembers(other).filter(id => !selected.includes(id)).join(',');
      pruneRegionOutputs(other);
    }
  }
  if (!region) return;
  const additions = members.filter(n => !existing.has(n.id));
  const rect = layoutRegionRect(graph, region, sizes);
  const current = graph.nodes.filter(n => existing.has(n.id));
  let x = rect.x + REGION_PADDING, y = Math.max(rect.y + REGION_HEADER,
    ...current.map(n => n.position.y + nodeSize(n.id, sizes).height + 32));
  let rowHeight = 0;
  const moves: NodeMove[] = [];
  for (const member of additions) {
    const size = nodeSize(member.id, sizes);
    if (x > rect.x + REGION_PADDING && x + size.width > rect.x + rect.width - REGION_PADDING) {
      x = rect.x + REGION_PADDING; y += rowHeight + 32; rowHeight = 0;
    }
    moves.push({ id: member.id, position: { x, y } });
    x += size.width + 40; rowHeight = Math.max(rowHeight, size.height);
    existing.add(member.id);
  }
  region.params.members = [...existing].join(',');
  applyMoves(graph, moves);
  const fitted = layoutRegionRect(graph, region, sizes);
  region.params.width = fitted.width; region.params.height = fitted.height;
}

export function resizeLayoutRegion(graph: Graph, id: string, requested: RegionRect, sizes: NodeSizes = {}): void {
  const region = graph.nodes.find(n => n.id === id && isLayoutRegion(n));
  if (!region) throw Error('布局分区不存在。');
  if (Object.values(requested).some(v => !Number.isFinite(v))) throw Error('分区尺寸与坐标必须是有限数值。');
  const members = new Set(layoutRegionMembers(region));
  const nodes = graph.nodes.filter(n => members.has(n.id));
  // Resizing an edge never pushes or clips a member, including at the top/left edges.
  const x = Math.min(requested.x, ...nodes.map(n => n.position.x - REGION_PADDING));
  const y = Math.min(requested.y, ...nodes.map(n => n.position.y - REGION_HEADER));
  region.position = { x, y };
  region.params.width = Math.max(360, requested.x + requested.width - x);
  region.params.height = Math.max(200, requested.y + requested.height - y);
  region.params.sizing = 'manual';
  const safe = layoutRegionRect(graph, region, sizes);
  region.params.width = safe.width; region.params.height = safe.height;
  checkLayoutRegion(region);
}

/** Tight auto bounds for either the canonical members or a visible canvas projection. */
export function fittedLayoutRegionRect(graph: Graph, region: ModelNode, sizes: NodeSizes = {}): RegionRect {
  const ids = new Set(layoutRegionMembers(region));
  const members = graph.nodes.filter(n => ids.has(n.id));
  if (!members.length) return layoutRegionRect(graph, region, sizes);
  const position = {
    x: Math.min(...members.map(n => n.position.x)) - REGION_PADDING,
    y: Math.min(...members.map(n => n.position.y)) - REGION_HEADER,
  };
  return layoutRegionRect(graph, { ...region, position,
    params: { ...region.params, sizing: 'auto', width: 360, height: 200 } }, sizes);
}

export function fitLayoutRegion(graph: Graph, id: string, sizes: NodeSizes = {}): void {
  const region = graph.nodes.find(n => n.id === id && isLayoutRegion(n));
  if (!region) throw Error('布局分区不存在。');
  const { x, y, width, height } = fittedLayoutRegionRect(graph, region, sizes);
  region.position = { x, y };
  Object.assign(region.params, { sizing: 'auto', width, height });
}
