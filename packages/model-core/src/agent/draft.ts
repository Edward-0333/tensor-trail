import type { Graph, GraphResult } from '../core/types';
import { ports } from '../core/graph';
import { registry } from '../modules/index';

/** Only unconnected inputs and their consequences are editable incompleteness, not shape errors. */
export function incompleteNodes(graph: Graph, result: GraphResult): Set<string> {
  const memo = new Map<string, boolean>();
  const check = (id: string): boolean => {
    if (memo.has(id)) return memo.get(id)!;
    const node = graph.nodes.find((n) => n.id === id),
      r = result.nodes[id];
    if (!node || !r || !['error', 'blocked'].includes(r.status)) return false;
    memo.set(id, false);
    let incomplete = false;
    if (node.valueSource) incomplete = check(node.valueSource);
    else if (r.iterations && node.subgraph) {
      const failures = r.iterations.flatMap(child =>
        Object.entries(child.nodes).filter(([, v]) => ['error', 'blocked'].includes(v.status)));
      incomplete = failures.length > 0 && r.iterations.every(child => {
        if (child.issues.length) return false;
        const pending = incompleteNodes(node.subgraph!, child);
        return Object.entries(child.nodes).filter(([, v]) => ['error', 'blocked'].includes(v.status))
          .every(([key]) => pending.has(key));
      });
    } else if (r.child && node.subgraph && !r.child.issues.length) {
      const pending = incompleteNodes(node.subgraph, r.child);
      const failures = Object.entries(r.child.nodes).filter(([, child]) =>
        ['error', 'blocked'].includes(child.status),
      );
      incomplete = failures.length > 0 && failures.every(([key]) => pending.has(key));
    } else if (r.status === 'blocked') {
      const incoming = graph.edges.filter((e) => e.target === id);
      const unavailable = incoming.filter((e) => !result.nodes[e.source]?.outputs[e.sourcePort]);
      const unconnected = ports(node, registry).inputs.some(
        (p) => !incoming.some((e) => e.targetPort === p.id),
      );
      incomplete =
        (unconnected || unavailable.length > 0) && unavailable.every((e) => check(e.source));
    }
    memo.set(id, incomplete);
    return incomplete;
  };
  return new Set(graph.nodes.filter((n) => check(n.id)).map((n) => n.id));
}
