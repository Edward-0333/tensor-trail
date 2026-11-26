import {
  fittedLayoutRegionRect, isLayoutRegion, layoutRegionMembers, layoutRegionRect, type NodeMove, type NodeSizes,
} from '@tensor-trail/model-core/core/layoutRegions';
import type { Graph, ModelNode } from '@tensor-trail/model-core/core/types';
import type { TrailNode } from './ModuleNode';

export const measuredNodeSizes = (nodes: TrailNode[]): NodeSizes => Object.fromEntries(nodes
  .filter(node => !isLayoutRegion(node.data.model) && node.measured?.width && node.measured?.height)
  .map(node => [node.id, { width: node.measured!.width!, height: node.measured!.height! }]));

function hasHiddenMembers(region: ModelNode, hidden?: ReadonlySet<string>) {
  return !!hidden?.size && isLayoutRegion(region) && region.params.sizing === 'auto' &&
    layoutRegionMembers(region).some(id => hidden.has(id));
}

export function visibleRegionRect(graph: Graph, region: ModelNode, sizes: NodeSizes, hidden?: ReadonlySet<string>) {
  return hasHiddenMembers(region, hidden) ? fittedLayoutRegionRect(graph, region, sizes) : layoutRegionRect(graph, region, sizes);
}

/** The displayed heading can shift when a distant member collapses. Apply the drag delta
 * to the stored origin so every canonical member (including hidden ones) moves together. */
export function regionDragPositions(graph: Graph, moves: NodeMove[], initial: NodeMove[], hidden: ReadonlySet<string>) {
  if (!hidden.size) return moves;
  return moves.map(move => {
    const region = graph.nodes.find(n => n.id === move.id);
    if (!region || !hasHiddenMembers(region, hidden)) return move;
    const start = initial.find(n => n.id === move.id)?.position;
    if (!start) return move;
    return { id: move.id, position: {
      x: region.position.x + move.position.x - start.x,
      y: region.position.y + move.position.y - start.y,
    } };
  });
}

export function layoutRegions(nodes: TrailNode[], hidden?: ReadonlySet<string>): TrailNode[] {
  const graph: Graph = { nodes: nodes.map(n => ({ ...n.data.model, position: n.position })), edges: [] };
  const sizes = measuredNodeSizes(nodes);
  return nodes.map(node => {
    if (!isLayoutRegion(node.data.model)) return node;
    const rect = visibleRegionRect(graph, graph.nodes.find(n => n.id === node.id)!, sizes, hidden);
    return {
      ...node,
      position: node.resizing ? node.position : { x: rect.x, y: rect.y },
      style: { ...node.style,
        pointerEvents: 'none',
        width: node.resizing ? node.measured?.width ?? rect.width : rect.width,
        height: node.resizing ? node.measured?.height ?? rect.height : rect.height,
      },
      zIndex: -10000, draggable: true, dragHandle: '.layout-region-heading',
    };
  });
}
