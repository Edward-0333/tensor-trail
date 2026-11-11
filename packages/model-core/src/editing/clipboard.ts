import type { Graph } from '../core/types';
import { clone, uid } from '../core/types';
import { parseProject } from '../core/serialization';
import { registry } from '../modules/index';
import { effectiveValueNode } from '../core/valueLinks';
import { ifRegionMembers, pruneIfRegions } from '../core/ifRegions';

const prefix = 'tensor-trail-fragment-v1\n';
export const clipboardText = (fragment: Graph) => prefix + JSON.stringify(fragment);
export function readClipboard(text: string): Graph | null {
  if (!text.startsWith(prefix)) return null;
  if (text.length > 3_000_000) throw new Error('剪贴板内容超过 3 MB。');
  const graph: unknown = JSON.parse(text.slice(prefix.length));
  const record = (value: unknown): value is Record<string, unknown> =>
    value !== null && typeof value === 'object' && !Array.isArray(value);
  if (
    record(graph) &&
    Array.isArray(graph.nodes) &&
    graph.nodes.some((node) => record(node) && ['Inlet', 'Outlet'].includes(String(node.type)))
  ) {
    // Clipboard boundary fragments live in a child scope. Keep root project validation strict.
    const nodes = graph.nodes;
    let dummyId: string | undefined;
    if (
      !nodes.some((node) => record(node) && node.type === 'Inlet' && node.valueSource === undefined)
    ) {
      // Do not let a temporary node accidentally satisfy a dangling reference in malformed data.
      const reserved = new Set<unknown>(
        nodes.flatMap((node) => (record(node) ? [node.id, node.valueSource] : [])),
      );
      if (Array.isArray(graph.edges))
        for (const edge of graph.edges)
          if (record(edge)) {
            reserved.add(edge.source);
            reserved.add(edge.target);
          }
      dummyId = 'clipboard_input';
      let suffix = 2;
      while (reserved.has(dummyId)) dummyId = `clipboard_input_${suffix++}`;
      nodes.push(registry.create('Inlet', { port: 'clipboard_input' }, dummyId));
    }
    const wrapper = registry.create('Submodule', {}, 'clipboard_fragment');
    const wrapped = {
      schemaVersion: 1,
      name: '剪贴板',
      graph: { nodes: [{ ...wrapper, subgraph: graph }], edges: [] },
      variants: [],
    };
    const checked = parseProject(JSON.stringify(wrapped), registry).graph.nodes[0].subgraph!;
    if (dummyId) checked.nodes = checked.nodes.filter((node) => node.id !== dummyId);
    return checked;
  }
  return parseProject(
    JSON.stringify({ schemaVersion: 1, name: '剪贴板', graph, variants: [] }),
    registry,
  ).graph;
}

export function copySelection(graph: Graph, ids: string[]): Graph {
  const chosen = new Set(ids);
  for (const node of graph.nodes)
    if (chosen.has(node.id) && node.type === 'IfRegion')
      for (const member of ifRegionMembers(node)) chosen.add(member);
  const nodes = graph.nodes
    .filter((node) => chosen.has(node.id))
    .map((node) => {
      const copied = clone(effectiveValueNode(graph, node));
      if (copied.valueSource !== undefined && !chosen.has(copied.valueSource))
        delete copied.valueSource;
      return copied;
    });
  // Two selected aliases without their source become independent boundaries in a portable copy.
  // Preserve existing distinct port names when assigning names to the materialized duplicates.
  for (const type of ['Inlet', 'Outlet']) {
    const boundaries = nodes.filter((node) => node.type === type && node.valueSource === undefined);
    const reserved = new Set(boundaries.map((node) => String(node.params.port)));
    const used = new Set<string>();
    for (const node of boundaries) {
      const base = String(node.params.port);
      let port = base;
      let suffix = 2;
      while (used.has(port)) {
        port = `${base.slice(0, 90)}_${suffix++}`;
        while (reserved.has(port)) port = `${base.slice(0, 90)}_${suffix++}`;
      }
      node.params.port = port;
      used.add(port);
    }
  }
  const kept = new Set(nodes.map((n) => n.id));
  pruneIfRegions({ nodes, edges: [] });
  return clone({
    nodes,
    edges: graph.edges.filter((e) => kept.has(e.source) && kept.has(e.target)),
  });
}

export function duplicateSelection(fragment: Graph, offset: number): Graph {
  const copy = clone(fragment),
    ids = new Map(copy.nodes.map((n) => [n.id, uid()]));
  for (const node of copy.nodes) {
    if (node.type === 'IfRegion')
      node.params.members = ifRegionMembers(node).map(id => ids.get(id) ?? id).join(',');
    node.id = ids.get(node.id)!;
    if (node.valueSource !== undefined && ids.has(node.valueSource))
      node.valueSource = ids.get(node.valueSource)!;
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
