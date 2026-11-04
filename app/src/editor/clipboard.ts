import type { Graph } from '../core/types';
import { clone, uid } from '../core/types';
import { parseProject } from '../core/serialization';
import { registry } from '../modules';

const prefix = 'tensor-trail-fragment-v1\n';
export const clipboardText = (fragment: Graph) => prefix + JSON.stringify(fragment);
export function readClipboard(text: string): Graph | null {
  if (!text.startsWith(prefix)) return null;
  if (text.length > 3_000_000) throw new Error('剪贴板内容超过 3 MB。');
  const graph: unknown = JSON.parse(text.slice(prefix.length));
  return parseProject(
    JSON.stringify({ schemaVersion: 1, name: '剪贴板', graph, variants: [] }),
    registry,
  ).graph;
}

export function copySelection(graph: Graph, ids: string[]): Graph {
  const chosen = new Set(ids);
  const nodes = graph.nodes.filter(
    (n) => chosen.has(n.id) && !['Inlet', 'Outlet'].includes(n.type),
  );
  const kept = new Set(nodes.map((n) => n.id));
  return clone({
    nodes,
    edges: graph.edges.filter((e) => kept.has(e.source) && kept.has(e.target)),
  });
}

export function duplicateSelection(fragment: Graph, offset: number): Graph {
  const copy = clone(fragment),
    ids = new Map(copy.nodes.map((n) => [n.id, uid()]));
  for (const node of copy.nodes) {
    node.id = ids.get(node.id)!;
    node.position = { x: node.position.x + offset, y: node.position.y + offset };
  }
  for (const edge of copy.edges) {
    edge.id = uid();
    edge.source = ids.get(edge.source)!;
    edge.target = ids.get(edge.target)!;
    if (edge.waypoints)
      edge.waypoints = edge.waypoints.map((p) => ({ x: p.x + offset, y: p.y + offset }));
  }
  return copy;
}
