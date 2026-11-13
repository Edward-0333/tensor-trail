import type { Graph, ModelNode } from './types';

export const isCodeRegion = (node: Pick<ModelNode, 'type'>): boolean =>
  node.type === 'IfRegion' || node.type === 'LoopRegion';

/** A visual code scope, not a Python evaluator or a dataflow operator. */
export function ifRegionMembers(node: ModelNode): string[] {
  const text = String(node.params.members ?? '').trim();
  if (!text) return [];
  const ids = text.split(',').map((id) => id.trim());
  if (
    ids.length > 500 ||
    new Set(ids).size !== ids.length ||
    ids.some(
      (id) =>
        !/^[A-Za-z0-9_-]{1,100}$/.test(id) ||
        ['__proto__', 'constructor', 'prototype'].includes(id),
    )
  )
    throw new Error('范围框成员需要唯一的同层节点 ID，以逗号分隔，最多 500 个。');
  return ids;
}

export function ifRegionIssues(graph: Graph): string[] {
  const issues: string[] = [];
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const owners = new Map<string, string>();
  for (const region of graph.nodes.filter(isCodeRegion)) {
    try {
      for (const id of ifRegionMembers(region)) {
        if (!nodes.has(id)) {
          issues.push(`${region.label}：成员 ${id} 不存在于同一画布中。`);
          continue;
        }
        if (id === region.id) {
          issues.push(`${region.label}：范围框不能包含自身。`);
          continue;
        }
        if (owners.has(id)) issues.push(`${id} 只能有一个直接父范围框，不能重复归属。`);
        else owners.set(id, region.id);
      }
    } catch (error) {
      issues.push((error as Error).message);
    }
  }
  const checked = new Set<string>();
  for (const region of graph.nodes.filter(isCodeRegion)) {
    const path = new Set<string>();
    let id: string | undefined = region.id;
    while (id !== undefined && !checked.has(id)) {
      if (path.has(id)) {
        issues.push(`${id}：范围框嵌套形成了循环。`);
        break;
      }
      path.add(id);
      id = owners.get(id);
    }
    for (const visited of path) checked.add(visited);
  }
  return issues;
}

/** Removing a frame lifts surviving code into its parent; removing code prunes membership. */
export function pruneIfRegions(graph: Graph, removed: ModelNode[] = []): void {
  const ids = new Set(graph.nodes.map((n) => n.id));
  const frames = new Map(removed.filter(isCodeRegion).map((n) => [n.id, ifRegionMembers(n)]));
  for (const region of graph.nodes.filter(isCodeRegion)) {
    const pending = ifRegionMembers(region).reverse();
    const visited = new Set<string>();
    const members: string[] = [];
    while (pending.length) {
      const id = pending.pop()!;
      if (visited.has(id)) continue;
      visited.add(id);
      if (ids.has(id)) members.push(id);
      else pending.push(...(frames.get(id) ?? []).slice().reverse());
    }
    region.params.members = members.join(',');
  }
}
