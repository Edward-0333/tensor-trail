import type { Graph, ModelNode } from './types';

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
    throw new Error('条件框成员需要唯一的同层节点 ID，以逗号分隔，最多 500 个。');
  return ids;
}

export function ifRegionIssues(graph: Graph): string[] {
  const issues: string[] = [];
  const owners = new Map<string, string>();
  for (const region of graph.nodes.filter((n) => n.type === 'IfRegion')) {
    try {
      for (const id of ifRegionMembers(region)) {
        const member = graph.nodes.find((n) => n.id === id);
        if (!member || member.type === 'IfRegion')
          issues.push(`${region.label}：成员 ${id} 不存在或是条件框；嵌套条件请放在函数子图中。`);
        if (owners.has(id)) issues.push(`${id} 不能同时位于两个同层条件框中。`);
        owners.set(id, region.id);
      }
    } catch (error) {
      issues.push((error as Error).message);
    }
  }
  return issues;
}

/** Removing a frame keeps its code; removing code also removes its frame membership. */
export function pruneIfRegions(graph: Graph): void {
  const ids = new Set(graph.nodes.map((n) => n.id));
  for (const region of graph.nodes.filter((n) => n.type === 'IfRegion'))
    region.params.members = ifRegionMembers(region)
      .filter((id) => ids.has(id))
      .join(',');
}
