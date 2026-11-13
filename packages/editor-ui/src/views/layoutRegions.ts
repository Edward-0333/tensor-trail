import {
  isLayoutRegion, layoutRegionRect, type NodeSizes,
} from '@tensor-trail/model-core/core/layoutRegions';
import type { Graph } from '@tensor-trail/model-core/core/types';
import type { TrailNode } from './ModuleNode';

export const measuredNodeSizes = (nodes: TrailNode[]): NodeSizes => Object.fromEntries(nodes
  .filter(node => !isLayoutRegion(node.data.model) && node.measured?.width && node.measured?.height)
  .map(node => [node.id, { width: node.measured!.width!, height: node.measured!.height! }]));

export function layoutRegions(nodes: TrailNode[]): TrailNode[] {
  const graph: Graph = { nodes: nodes.map(n => ({ ...n.data.model, position: n.position })), edges: [] };
  const sizes = measuredNodeSizes(nodes);
  return nodes.map(node => {
    if (!isLayoutRegion(node.data.model)) return node;
    const rect = layoutRegionRect(graph, graph.nodes.find(n => n.id === node.id)!, sizes);
    return {
      ...node,
      style: { ...node.style,
        pointerEvents: 'none',
        width: node.resizing ? node.measured?.width ?? rect.width : rect.width,
        height: node.resizing ? node.measured?.height ?? rect.height : rect.height,
      },
      zIndex: -10000, draggable: true, dragHandle: '.layout-region-heading',
    };
  });
}
