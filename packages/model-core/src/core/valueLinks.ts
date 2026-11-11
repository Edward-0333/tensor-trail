import { clone, type Graph, type ModelNode } from './types';

const valueTypes = new Set([
  'Input',
  'Output',
  'ForwardInput',
  'DictionaryInput',
  'LocalParameter',
  'TensorVariable',
  'ValueVariable',
  'DeviceVariable',
  'BufferSource',
  'IntegerParameter',
  'Inlet',
  'Outlet',
]);

export const isValueNode = (type: string): boolean => valueTypes.has(type);

/** Resolve only direct links. Chains hide scope and are deliberately invalid. */
export function valueOwner(graph: Graph, nodeOrId: ModelNode | string): ModelNode {
  const id = typeof nodeOrId === 'string' ? nodeOrId : nodeOrId.id;
  const node = graph.nodes.find((candidate) => candidate.id === id);
  if (!node) throw new Error('共享值引用的节点不存在。');
  if (node.valueSource === undefined) return node;
  if (!isValueNode(node.type)) throw new Error('只有输入输出类节点可以共享值。');
  const source = graph.nodes.find((candidate) => candidate.id === node.valueSource);
  if (!source) throw new Error(`${node.label} 的共享值来源不存在于当前画布。`);
  if (source.id === node.id) throw new Error('共享值不能引用自身。');
  if (source.valueSource !== undefined)
    throw new Error('共享值必须直接引用原始节点，不能串联引用。');
  if (source.type !== node.type) throw new Error('共享值节点与来源的类型必须一致。');
  return source;
}

export function linkedValueNodes(graph: Graph, id: string): ModelNode[] {
  const owner = valueOwner(graph, id);
  return graph.nodes.filter((node) => node.id === owner.id || node.valueSource === owner.id);
}

/** Semantics are shared; identity, position, color and port placement remain local. */
export function effectiveValueNode(graph: Graph, node: ModelNode): ModelNode {
  if (node.valueSource === undefined) return node;
  const owner = valueOwner(graph, node);
  const effective = { ...node, params: owner.params, label: owner.label };
  if (owner.tensorAutograd) effective.tensorAutograd = owner.tensorAutograd;
  else delete effective.tensorAutograd;
  return effective;
}

/** Keep serialized alias metadata useful without making it a second source of truth. */
export function syncValueLinks(graph: Graph): void {
  for (const node of graph.nodes) {
    if (node.valueSource !== undefined) {
      const owner = valueOwner(graph, node);
      node.params = clone(owner.params);
      node.label = owner.label;
      if (owner.tensorAutograd) node.tensorAutograd = clone(owner.tensorAutograd);
      else delete node.tensorAutograd;
    }
    if (node.subgraph) syncValueLinks(node.subgraph);
  }
}

export function valueLinkIssues(graph: Graph): string[] {
  const issues: string[] = [];
  for (const node of graph.nodes) {
    if (node.valueSource === undefined) continue;
    try {
      valueOwner(graph, node);
    } catch (error) {
      issues.push((error as Error).message);
    }
    if (graph.edges.some((edge) => edge.target === node.id))
      issues.push(`${node.label} 是共享值副本，不能拥有独立输入连线；请先解除链接。`);
  }
  return issues;
}
