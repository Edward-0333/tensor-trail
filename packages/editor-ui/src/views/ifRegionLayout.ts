import { ifRegionMembers, isCodeRegion } from '@tensor-trail/model-core/core/ifRegions';
import type { TrailNode } from './ModuleNode';

/** Bounds use live node measurements; membership does not reparent or move the code. */
export function layoutIfRegions(nodes: TrailNode[]): TrailNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const laidOut = new Map<string, TrailNode>();
  const active = new Set<string>();
  const pending = nodes.map((node) => ({ node, expanded: false }));
  // Postorder includes each child's computed frame (and heading), not stale measurements.
  // An explicit stack and active-path guard also tolerate deep or cyclic draft scopes.
  while (pending.length) {
    const entry = pending[pending.length - 1];
    const { node } = entry;
    if (laidOut.has(node.id)) {
      pending.pop();
      continue;
    }
    if (!isCodeRegion(node.data.model)) {
      laidOut.set(node.id, node);
      pending.pop();
      continue;
    }
    const ids = ifRegionMembers(node.data.model);
    if (!entry.expanded) {
      entry.expanded = true;
      active.add(node.id);
      for (const id of ids) {
        const member = byId.get(id);
        if (member && !active.has(id) && !laidOut.has(id))
          pending.push({ node: member, expanded: false });
      }
      continue;
    }
    const members = ids.flatMap((id) => {
      const member = laidOut.get(id);
      return member ? [member] : [];
    });
    const left = members.length
      ? Math.min(...members.map((n) => n.position.x)) - 40
      : node.position.x;
    const top = members.length
      ? Math.min(...members.map((n) => n.position.y)) - 94
      : node.position.y;
    const right = members.length
      ? Math.max(
          ...members.map(
            (n) =>
              n.position.x +
              (isCodeRegion(n.data.model) ? Number(n.style!.width) : (n.measured?.width ?? 250)),
          ),
        ) + 40
      : left + 440;
    const bottom = members.length
      ? Math.max(
          ...members.map(
            (n) =>
              n.position.y +
              (isCodeRegion(n.data.model) ? Number(n.style!.height) : (n.measured?.height ?? 160)),
          ),
        ) + 40
      : top + 220;
    laidOut.set(node.id, {
      ...node,
      position: { x: left, y: top },
      style: { ...node.style, width: Math.max(440, right - left), height: bottom - top },
      zIndex: Math.min(
        -1,
        ...members.filter((n) => isCodeRegion(n.data.model)).map((n) => n.zIndex! - 1),
      ),
      draggable: members.length === 0,
      dragHandle: '.if-region-heading',
    });
    active.delete(node.id);
    pending.pop();
  }
  return nodes.map((node) => laidOut.get(node.id)!);
}
