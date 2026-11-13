import type { Graph, RootInputSource } from './types';
import { isForwardInput } from './inputTypes';

/** Only true root arguments create sources. Aliases, parameters and nested Inlets do not. */
export function rootInputSources(graph: Graph): RootInputSource[] {
  return graph.nodes
    .filter(
      (node) =>
        node.valueSource === undefined && (node.type === 'Input' || isForwardInput(node.type)),
    )
    .map((node, i) => ({ id: node.id, index: i + 1, label: node.label }));
}
