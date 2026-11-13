import type { Graph } from '@tensor-trail/model-core/core/types';
import { rootInputSources } from '@tensor-trail/model-core/core/rootInputs';

export interface LevelInput {
  id: string;
  label: string;
  port?: string;
  index: number;
  total: number;
}

/** Count boundary values, not their fields, aliases or unconnected internal sources. */
export function levelInputs(graph: Graph, root: boolean): LevelInput[] {
  if (root) {
    const inputs = rootInputSources(graph);
    return inputs.map(input => ({ ...input, total: inputs.length }));
  }
  const inputs = graph.nodes.filter(node => node.valueSource === undefined && node.type === 'Inlet');
  return inputs.map((node, i) => ({
    id: node.id,
    label: node.label,
    ...(node.type === 'Inlet' ? { port: String(node.params.port) } : {}),
    index: i + 1,
    total: inputs.length,
  }));
}
