import type { Graph } from '@tensor-trail/model-core/core/types';
import { canJoinLayoutRegion, layoutRegionOf } from '@tensor-trail/model-core/core/layoutRegions';

export interface CanvasMenuState {
  x: number;
  y: number;
  position: { x: number; y: number };
  nodeIds: string[];
  edgeIds: string[];
}

/** A right click inside a multi-selection acts on the whole selection. */
export function contextNodeSelection(graph: Graph, selected: string[], clicked?: string): string[] {
  const valid = new Set(graph.nodes.map(n => n.id));
  const ids = [...new Set(selected)].filter(id => valid.has(id));
  return clicked && valid.has(clicked) && !ids.includes(clicked) ? [clicked] : ids;
}

export function contextTargets(graph: Graph, ids: string[]) {
  const selected = new Set(ids), nodes = graph.nodes.filter(n => selected.has(n.id));
  const regionMembers = nodes.filter(canJoinLayoutRegion).map(n => n.id);
  return {
    nodes, regionMembers,
    assigned: regionMembers.filter(id => layoutRegionOf(graph, id)),
    deletable: nodes.filter(n => n.valueSource !== undefined || !['Inlet', 'Outlet'].includes(n.type)).map(n => n.id),
  };
}

export function menuPosition(point: { x: number; y: number }, size: { width: number; height: number },
  viewport: { width: number; height: number }) {
  return { x: Math.max(8, Math.min(point.x, viewport.width - size.width - 8)),
    y: Math.max(8, Math.min(point.y, viewport.height - size.height - 8)) };
}
