import { graphAt } from '../core/graph';
import { clone, uid, type Graph, type ModelEdge, type ModelNode } from '../core/types';
import { isValueNode, linkedValueNodes, valueOwner } from '../core/valueLinks';
import { duplicateSelection } from './clipboard';
import { pruneIfRegions } from '../core/ifRegions';
import { pruneLayoutRegions } from '../core/layoutRegions';

const isBoundary = (node: ModelNode) => ['Inlet', 'Outlet'].includes(node.type);

function uniquePort(graph: Graph, node: ModelNode, base: string): string {
  const used = new Set(
    graph.nodes
      .filter((n) => n.id !== node.id && n.type === node.type && n.valueSource === undefined)
      .map((n) => String(n.params.port)),
  );
  let port = base,
    suffix = 2;
  while (used.has(port)) port = `${base.slice(0, 90)}${suffix++}`;
  return port;
}

function reroute(edge: ModelEdge): ModelEdge {
  const copy = clone(edge);
  delete copy.waypoints;
  delete copy.labelPosition;
  return copy;
}

/** Only local parameters auto-link. Other live-value copies require an explicit request. */
export function createLinkedPaste(
  graph: Graph,
  clipboard: Graph,
  offset: number,
  linkToCurrent: boolean,
  mode: 'local' | 'requested' = 'local',
): Graph {
  const fragment = duplicateSelection(clipboard, offset);
  if (linkToCurrent) {
    fragment.nodes.forEach((node, index) => {
      const original = graph.nodes.find((candidate) => candidate.id === clipboard.nodes[index].id);
      if (!original || original.type !== node.type || !isValueNode(node.type)) return;
      if (mode !== 'requested' && node.type !== 'LocalParameter') return;
      const owner = valueOwner(graph, original);
      node.valueSource = owner.id;
      node.params = clone(owner.params);
      node.label = owner.label;
    });
  }
  // An internally copied reference can now point to a new alias. Flatten it.
  for (const node of fragment.nodes) {
    const source = fragment.nodes.find((candidate) => candidate.id === node.valueSource);
    if (source?.valueSource !== undefined) node.valueSource = source.valueSource;
  }
  const aliases = new Set(
    fragment.nodes.filter((node) => node.valueSource !== undefined).map((node) => node.id),
  );
  fragment.edges = fragment.edges.filter((edge) => !aliases.has(edge.target));
  const combined: Graph = { nodes: [...graph.nodes], edges: [] };
  for (const node of fragment.nodes) {
    if (isBoundary(node) && node.valueSource === undefined)
      node.params.port = uniquePort(combined, node, String(node.params.port));
    combined.nodes.push(node);
  }
  return fragment;
}

function copyInputs(graph: Graph, from: string, to: string): void {
  graph.edges.push(
    ...graph.edges
      .filter((edge) => edge.target === from)
      .map((edge) => ({ ...reroute(edge), id: uid(), target: to })),
  );
}

/** Reuse an existing display without deleting its identity, geometry or outgoing wires. */
export function linkValueInGraph(graph: Graph, id: string, source: string): void {
  const node = graph.nodes.find(n => n.id === id);
  if (!node || !isValueNode(node.type)) throw new Error('请选择已有值节点建立关联。');
  const owner = valueOwner(graph, source);
  if (node.type !== owner.type || !isValueNode(owner.type)) throw new Error('值关联两端必须具有相同类型。');
  if (node.id === owner.id) throw new Error('共享值不能引用自身。');
  if (isBoundary(node)) throw new Error('不能用值关联合并已有子图接口；请显式处理父图端口。');
  if (graph.edges.some(edge => edge.target === id)) throw new Error('目标节点有独立输入连线，请先显式断开再建立值关联。');
  if (graph.nodes.some(n => n.valueSource === id)) throw new Error('目标节点已是其他副本的主节点，不能创建串联值关联。');
  node.valueSource = owner.id;
}

/** Detach a view without losing its current input wiring. */
export function unlinkValueInGraph(root: Graph, path: string[], id: string): void {
  const graph = graphAt(root, path);
  const node = graph.nodes.find((n) => n.id === id);
  if (!node) throw new Error('模块已不存在。');
  const owner = valueOwner(graph, node),
    group = linkedValueNodes(graph, id);
  if (group.length < 2) return;
  node.params = clone(owner.params);
  node.label = owner.label;
  const oldPort = String(owner.params.port);
  if (node.id !== owner.id) {
    delete node.valueSource;
    copyInputs(graph, owner.id, node.id);
  } else {
    const promoted = group.find((member) => member.id !== owner.id)!;
    promoted.params = clone(owner.params);
    promoted.label = owner.label;
    delete promoted.valueSource;
    for (const member of group)
      if (member.id !== owner.id && member.id !== promoted.id) member.valueSource = promoted.id;
    copyInputs(graph, owner.id, promoted.id);
  }
  if (isBoundary(node)) {
    if (!path.length) throw new Error('模块边界只能位于子图中。');
    node.params.port = uniquePort(graph, node, oldPort);
    // A detached Inlet becomes a real new interface, initially fed by the same input.
    if (node.type === 'Inlet') {
      const parent = graphAt(root, path.slice(0, -1)),
        moduleId = path.at(-1)!;
      parent.edges.push(
        ...parent.edges
          .filter((edge) => edge.target === moduleId && edge.targetPort === oldPort)
          .map((edge) => ({ ...reroute(edge), id: uid(), targetPort: String(node.params.port) })),
      );
    }
  }
}

/** Removing a display must not strand the other views of the shared value. */
export function removeValueNodes(root: Graph, path: string[], ids: string[]): void {
  const graph = graphAt(root, path);
  const removed = new Set(
    graph.nodes
      .filter(
        (node) => ids.includes(node.id) && (!isBoundary(node) || node.valueSource !== undefined),
      )
      .map((node) => node.id),
  );
  for (const owner of graph.nodes) {
    if (!removed.has(owner.id) || owner.valueSource !== undefined || !isValueNode(owner.type))
      continue;
    const survivors = graph.nodes.filter(
      (node) => node.valueSource === owner.id && !removed.has(node.id),
    );
    if (!survivors.length) continue;
    const promoted = survivors[0];
    promoted.params = clone(owner.params);
    promoted.label = owner.label;
    delete promoted.valueSource;
    for (const alias of survivors.slice(1)) alias.valueSource = promoted.id;
    graph.edges = graph.edges.map((edge) => {
      if (edge.source !== owner.id && edge.target !== owner.id) return edge;
      return {
        ...reroute(edge),
        source: edge.source === owner.id ? promoted.id : edge.source,
        target: edge.target === owner.id ? promoted.id : edge.target,
      };
    });
  }
  const removedNodes = graph.nodes.filter((node) => removed.has(node.id));
  graph.nodes = graph.nodes.filter((node) => !removed.has(node.id));
  pruneIfRegions(graph, removedNodes);
  pruneLayoutRegions(graph);
  graph.edges = graph.edges.filter(
    (edge) => !removed.has(edge.source) && !removed.has(edge.target),
  );
}
