import type { Graph, ModelEdge, ModelNode, Registry } from './types';
import { clone, uid } from './types';
import { orderGraph, ports } from './graph';
import { isCalculation } from '../modules/libraryCategories';

const boundary = (node: ModelNode) => ['Inlet', 'Outlet'].includes(node.type);
const key = (node: string, port: string) => JSON.stringify([node, port]);

/** A visual contraction must not introduce a cycle through an outside branch. */
export function canMergeCalculations(
  graph: Graph,
  a: string,
  b: string,
  registry: Registry,
): boolean {
  if (a === b) return false;
  const members = new Set([a, b]);
  if (
    [a, b].some(
      (id) => !isCalculation(registry.get(graph.nodes.find((n) => n.id === id)?.type ?? '')),
    )
  )
    return false;
  if (!graph.edges.some((e) => members.has(e.source) && members.has(e.target))) return false;
  try {
    orderGraph({
      nodes: graph.nodes.filter((n) => n.id !== b),
      edges: graph.edges
        .filter((e) => !(members.has(e.source) && members.has(e.target)))
        .map((e) => ({
          ...e,
          source: e.source === b ? a : e.source,
          target: e.target === b ? a : e.target,
        })),
    });
    return true;
  } catch {
    return false;
  }
}

export interface NodeBounds {
  id: string;
  position: { x: number; y: number };
  width: number;
  height: number;
}

/** Flow-space distance (independent of zoom); only a deliberate single-node drag merges. */
export function nearbyCalculation(
  graph: Graph,
  moved: NodeBounds,
  others: NodeBounds[],
  registry: Registry,
): string | undefined {
  const adjacent = new Set(
    graph.edges.flatMap((edge) =>
      edge.source === moved.id ? [edge.target] : edge.target === moved.id ? [edge.source] : [],
    ),
  );
  return others
    .filter((n) => adjacent.has(n.id) && canMergeCalculations(graph, moved.id, n.id, registry))
    .map((n) => ({
      id: n.id,
      distance: Math.hypot(
        Math.max(
          0,
          n.position.x - moved.position.x - moved.width,
          moved.position.x - n.position.x - n.width,
        ),
        Math.max(
          0,
          n.position.y - moved.position.y - moved.height,
          moved.position.y - n.position.y - n.height,
        ),
      ),
    }))
    .filter((n) => n.distance <= 24)
    .sort((a, b) => a.distance - b.distance || a.id.localeCompare(b.id))[0]?.id;
}

/** Expand the exact saved child graph, including fan-out and multiple inputs. */
export function splitCalculationGroup(graph: Graph, id: string): string[] {
  const group = graph.nodes.find((n) => n.id === id);
  if (group?.type !== 'CalculationGroup' || !group.subgraph) throw new Error('请选择计算组合。');
  const child = group.subgraph;
  const operations = child.nodes.filter((n) => !boundary(n));
  const ids = new Map(operations.map((n) => [n.id, uid()]));
  const minX = Math.min(...operations.map((n) => n.position.x), 0);
  const minY = Math.min(...operations.map((n) => n.position.y), 0);
  const nodes = operations.map((n) => ({
    ...clone(n),
    id: ids.get(n.id)!,
    position: {
      x: group.position.x + n.position.x - minX,
      y: group.position.y + n.position.y - minY,
    },
  }));
  const upstream = (
    nodeId: string,
    port: string,
  ): Pick<ModelEdge, 'source' | 'sourcePort'> | undefined => {
    const source = child.nodes.find((n) => n.id === nodeId);
    if (source?.type === 'Inlet') {
      const incoming = graph.edges.find(
        (e) => e.target === id && e.targetPort === source.params.port,
      );
      return incoming && { source: incoming.source, sourcePort: incoming.sourcePort };
    }
    const mapped = ids.get(nodeId);
    return mapped ? { source: mapped, sourcePort: port } : undefined;
  };
  const edges = graph.edges.filter((e) => e.source !== id && e.target !== id);
  for (const edge of child.edges) {
    const target = ids.get(edge.target),
      source = upstream(edge.source, edge.sourcePort);
    if (target && source) edges.push({ id: uid(), ...source, target, targetPort: edge.targetPort });
  }
  for (const edge of graph.edges.filter((e) => e.source === id)) {
    const outlet = child.nodes.find(
      (n) => n.type === 'Outlet' && n.params.port === edge.sourcePort,
    );
    const inner = child.edges.find((e) => e.target === outlet?.id);
    const source = inner && upstream(inner.source, inner.sourcePort);
    if (source)
      edges.push({ id: edge.id, ...source, target: edge.target, targetPort: edge.targetPort });
  }
  graph.nodes = [...graph.nodes.filter((n) => n.id !== id), ...nodes];
  graph.edges = edges;
  return nodes.map((n) => n.id);
}

/** Group graph operations, never replace them with text or discard branch connections. */
export function mergeCalculations(
  graph: Graph,
  a: string,
  b: string,
  registry: Registry,
  position?: ModelNode['position'],
): string {
  if (!canMergeCalculations(graph, a, b, registry))
    throw new Error('只能合并相邻且已连接的计算节点；合并不能跨越外部计算形成环路。');
  const anchor = position ?? graph.nodes.find((n) => n.id === b)!.position;
  // Flatten previous groups before extending, so repeated merging does not add nesting levels.
  const selected = new Set(
    [a, b].flatMap((id) =>
      graph.nodes.find((n) => n.id === id)?.type === 'CalculationGroup'
        ? splitCalculationGroup(graph, id)
        : [id],
    ),
  );
  const order = orderGraph(graph).filter((id) => selected.has(id));
  const members = order.map((id) => graph.nodes.find((n) => n.id === id)!);
  const subgraph: Graph = {
    nodes: members.map((n, i) => ({ ...clone(n), position: { x: i * 280, y: 100 } })),
    edges: [],
  };
  const internal = graph.edges.filter((e) => selected.has(e.source) && selected.has(e.target));
  subgraph.edges = internal.map((e) => ({
    id: uid(),
    source: e.source,
    sourcePort: e.sourcePort,
    target: e.target,
    targetPort: e.targetPort,
  }));
  const inputs = new Map<string, string>(),
    outputs = new Map<string, string>();
  for (const node of members) {
    for (const port of ports(node, registry).inputs) {
      if (internal.some((e) => e.target === node.id && e.targetPort === port.id)) continue;
      const name = `in${inputs.size + 1}`,
        id = uid();
      inputs.set(key(node.id, port.id), name);
      subgraph.nodes.push({
        id,
        type: 'Inlet',
        definitionVersion: 1,
        label: `${node.label} · ${port.label}`.slice(0, 200),
        params: { port: name },
        position: { x: -240, y: inputs.size * 120 },
      });
      subgraph.edges.push({
        id: uid(),
        source: id,
        sourcePort: 'out',
        target: node.id,
        targetPort: port.id,
      });
    }
    for (const port of ports(node, registry).outputs) {
      const outgoing = graph.edges.filter((e) => e.source === node.id && e.sourcePort === port.id);
      if (outgoing.length && outgoing.every((e) => selected.has(e.target))) continue;
      const name = `out${outputs.size + 1}`,
        id = uid();
      outputs.set(key(node.id, port.id), name);
      subgraph.nodes.push({
        id,
        type: 'Outlet',
        definitionVersion: 1,
        label: `${node.label} · ${port.label}`.slice(0, 200),
        params: { port: name },
        position: { x: members.length * 280, y: outputs.size * 120 },
      });
      subgraph.edges.push({
        id: uid(),
        source: node.id,
        sourcePort: port.id,
        target: id,
        targetPort: 'in',
      });
    }
  }
  const id = uid();
  graph.edges = graph.edges
    .filter((e) => !internal.includes(e))
    .map((e) => {
      if (!selected.has(e.source) && !selected.has(e.target)) return e;
      return {
        id: e.id,
        source: selected.has(e.source) ? id : e.source,
        sourcePort: selected.has(e.source)
          ? outputs.get(key(e.source, e.sourcePort))!
          : e.sourcePort,
        target: selected.has(e.target) ? id : e.target,
        targetPort: selected.has(e.target)
          ? inputs.get(key(e.target, e.targetPort))!
          : e.targetPort,
      };
    });
  graph.nodes = [
    ...graph.nodes.filter((n) => !selected.has(n.id)),
    {
      id,
      type: 'CalculationGroup',
      definitionVersion: 1,
      label: '计算组合',
      params: {},
      position: { ...anchor },
      subgraph,
      customized: true,
    },
  ];
  return id;
}
