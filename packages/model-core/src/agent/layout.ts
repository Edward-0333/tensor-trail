import type { Graph } from '../core/types';
import { clone } from '../core/types';
import { orderGraph } from '../core/graph';
import { layoutRegionOf, moveLayoutNodes } from '../core/layoutRegions';

/** Place only unspecified new nodes to the right of existing geometry. */
export function placeNewNodes(graph: Graph, unplaced: Set<string>) {
  const fixed = graph.nodes.filter((n) => !unplaced.has(n.id));
  const baseX = fixed.length ? Math.max(...fixed.map((n) => n.position.x)) + 310 : 0;
  for (const id of orderGraph(graph)) {
    if (!unplaced.has(id)) continue;
    if (layoutRegionOf(graph, id)) continue;
    const node = graph.nodes.find((n) => n.id === id)!;
    const sources = graph.edges
      .filter((e) => e.target === id)
      .map((e) => graph.nodes.find((n) => n.id === e.source)!)
      .filter(Boolean);
    const x = Math.max(baseX, ...sources.map((n) => n.position.x + 310));
    let y = sources[0]?.position.y ?? 0;
    while (
      fixed.some((n) => Math.abs(n.position.x - x) < 300 && Math.abs(n.position.y - y) < 220)
    ) {
      const next = y + 230;
      if (!Number.isFinite(next) || next === y)
        throw new Error('自动放置坐标超出可用范围，请显式指定新节点 position。');
      y = next;
    }
    moveLayoutNodes(graph, [{ id, position: { x, y } }]);
    fixed.push(node);
  }
}

/** Generator parameter changes retain geometry for surviving node/edge identities. */
export function preserveGeneratedLayout(before: Graph, after: Graph) {
  const unplaced = new Set<string>();
  for (const node of after.nodes) {
    const previous = before.nodes.find((n) => n.id === node.id);
    if (!previous) {
      unplaced.add(node.id);
      continue;
    }
    node.position = clone(previous.position);
    if (previous.portLayout) node.portLayout = clone(previous.portLayout);
    if (node.subgraph && previous.subgraph)
      preserveGeneratedLayout(previous.subgraph, node.subgraph);
  }
  for (const edge of after.edges) {
    const previous = before.edges.find(
      (e) =>
        e.source === edge.source &&
        e.sourcePort === edge.sourcePort &&
        e.target === edge.target &&
        e.targetPort === edge.targetPort,
    );
    if (previous?.waypoints) edge.waypoints = clone(previous.waypoints);
    if (previous?.labelPosition !== undefined) edge.labelPosition = previous.labelPosition;
  }
  placeNewNodes(after, unplaced);
}
