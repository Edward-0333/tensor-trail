import type { Graph, ModelNode, Port, Registry } from './types';

export function ports(node: ModelNode, registry: Registry): { inputs: Port[]; outputs: Port[] } {
  if (node.subgraph) {
    const boundary = (type: string) =>
      node
        .subgraph!.nodes.filter((n) => n.type === type)
        .map((n) => ({ id: String(n.params.port), label: n.label }));
    return { inputs: boundary('Inlet'), outputs: boundary('Outlet') };
  }
  const def = registry.get(node.type);
  return { inputs: def?.inputs ?? [], outputs: def?.outputs ?? [] };
}

export function orderGraph(graph: Graph): string[] {
  const degree = new Map(graph.nodes.map((n) => [n.id, 0]));
  const next = new Map<string, string[]>();
  for (const e of graph.edges) {
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
    next.set(e.source, [...(next.get(e.source) ?? []), e.target]);
  }
  const ready = graph.nodes.filter((n) => degree.get(n.id) === 0).map((n) => n.id);
  const order: string[] = [];
  for (let i = 0; i < ready.length; i++) {
    const id = ready[i];
    order.push(id);
    for (const target of next.get(id) ?? []) {
      const count = degree.get(target)! - 1;
      degree.set(target, count);
      if (count === 0) ready.push(target);
    }
  }
  if (order.length !== graph.nodes.length)
    throw new Error('连接形成了环路。当前画布使用有向无环图。');
  return order;
}

export function graphIssues(graph: Graph, registry: Registry): string[] {
  const issues: string[] = [];
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  if (nodes.size !== graph.nodes.length) issues.push('节点 ID 重复。');
  const occupied = new Set<string>();
  const edgeIds = new Set<string>();
  for (const e of graph.edges) {
    const from = nodes.get(e.source),
      to = nodes.get(e.target);
    if (edgeIds.has(e.id)) issues.push('连接 ID 重复。');
    edgeIds.add(e.id);
    if (!from || !to) {
      issues.push('连接引用了不存在的节点。');
      continue;
    }
    if (!ports(from, registry).outputs.some((p) => p.id === e.sourcePort))
      issues.push(`${from.label} 没有输出端口 ${e.sourcePort}。`);
    if (!ports(to, registry).inputs.some((p) => p.id === e.targetPort))
      issues.push(`${to.label} 没有输入端口 ${e.targetPort}。`);
    const key = `${e.target}:${e.targetPort}`;
    if (occupied.has(key)) issues.push(`${to.label} 的 ${e.targetPort} 已有输入，请先删除原连接。`);
    occupied.add(key);
  }
  try {
    orderGraph(graph);
  } catch (e) {
    issues.push((e as Error).message);
  }
  return issues;
}

export function graphAt(root: Graph, path: string[]): Graph {
  let graph = root;
  for (const id of path) {
    const child = graph.nodes.find((n) => n.id === id)?.subgraph;
    if (!child) throw new Error('子图路径已失效。');
    graph = child;
  }
  return graph;
}

export function markAncestors(root: Graph, path: string[]) {
  let graph = root;
  for (const id of path) {
    const node = graph.nodes.find((n) => n.id === id)!;
    node.customized = true;
    graph = node.subgraph!;
  }
}
