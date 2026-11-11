import { ifRegionMembers } from '@tensor-trail/model-core/core/ifRegions';
import type { TrailNode } from './ModuleNode';

/** Bounds use live node measurements; membership does not reparent or move the code. */
export function layoutIfRegions(nodes: TrailNode[]): TrailNode[] {
  return nodes.map((node) => {
    if (node.data.model.type !== 'IfRegion') return node;
    const ids = new Set(ifRegionMembers(node.data.model));
    const members = nodes.filter((n) => ids.has(n.id));
    const left = members.length
      ? Math.min(...members.map((n) => n.position.x)) - 40
      : node.position.x;
    const top = members.length
      ? Math.min(...members.map((n) => n.position.y)) - 94
      : node.position.y;
    const right = members.length
      ? Math.max(...members.map((n) => n.position.x + (n.measured?.width ?? 250))) + 40
      : left + 440;
    const bottom = members.length
      ? Math.max(...members.map((n) => n.position.y + (n.measured?.height ?? 160))) + 40
      : top + 220;
    return {
      ...node,
      position: { x: left, y: top },
      style: { ...node.style, width: Math.max(440, right - left), height: bottom - top },
      zIndex: -1,
      draggable: members.length === 0,
      dragHandle: '.if-region-heading',
    };
  });
}
