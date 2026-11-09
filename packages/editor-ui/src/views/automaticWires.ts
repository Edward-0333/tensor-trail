import { simplifyRoute, type Point, type RoutedEdge } from './edgeRouting';
import { normalizeWireRoutes } from './routeNormalization';

export interface WireGeometry extends RoutedEdge {
  automatic: boolean;
}

/** Updating one output does not normalize every other signal. Unchanged route
 * arrays retain their identity for individual edge subscribers.
 */
export function createWirePlanner() {
  const cache = new Map<string, { inputs: WireGeometry[]; paths: Map<string, Point[]> }>();
  return (edges: WireGeometry[]) => {
    const groups = new Map<string, WireGeometry[]>();
    for (const edge of edges) {
      const key = JSON.stringify([edge.source, edge.sourcePort]);
      const group = groups.get(key) ?? [];
      group.push(edge);
      groups.set(key, group);
    }
    const result = new Map<string, Point[]>();
    for (const [key, inputs] of groups) {
      let previous = cache.get(key);
      if (
        !previous ||
        previous.inputs.length !== inputs.length ||
        inputs.some((e, i) => e !== previous!.inputs[i])
      ) {
        const paths = normalizeWireRoutes(inputs);
        const clean = inputs.map((e) => ({ ...e, points: paths.get(e.id)! }));
        for (const [id, points] of automaticWireRoutes(clean)) paths.set(id, points);
        previous = { inputs, paths };
        cache.set(key, previous);
      }
      for (const [id, points] of previous.paths) result.set(id, points);
    }
    for (const key of cache.keys()) if (!groups.has(key)) cache.delete(key);
    return result;
  };
}

/** Plan the fan-out together instead of choosing a different midpoint for each edge.
 * Pinned routes participate in the endpoint envelope, but their geometry is never rewritten.
 * Only forward, opposing ports share this bus; other orientations retain their router.
 */
export function automaticWireRoutes(edges: WireGeometry[]): Map<string, Point[]> {
  const result = new Map<string, Point[]>();
  const groups = new Map<string, WireGeometry[]>();
  const opposite = { right: 'left', left: 'right', top: 'bottom', bottom: 'top' };
  for (const edge of edges) {
    const side = edge.sourceSide;
    if (!side || edge.targetSide !== opposite[side] || edge.points.length < 2) continue;
    const axis = side === 'left' || side === 'right' ? 'x' : 'y';
    const sign = side === 'right' || side === 'bottom' ? 1 : -1;
    if ((edge.points.at(-1)![axis] - edge.points[0][axis]) * sign < 48) continue;
    const key = JSON.stringify([edge.source, edge.sourcePort, side, edge.points[0]]);
    const group = groups.get(key) ?? [];
    group.push(edge);
    groups.set(key, group);
  }
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const side = group[0].sourceSide!;
    const axis = side === 'left' || side === 'right' ? 'x' : 'y';
    const cross = axis === 'x' ? 'y' : 'x';
    const source = group[0].points[0];
    const sign = side === 'right' || side === 'bottom' ? 1 : -1;
    const gap = Math.min(...group.map((e) => (e.points.at(-1)![axis] - source[axis]) * sign));
    const bus = source[axis] + (sign * gap) / 2;
    for (const edge of group) {
      if (!edge.automatic) continue;
      const target = edge.points.at(-1)!;
      result.set(
        edge.id,
        simplifyRoute([
          source,
          { ...source, [axis]: bus },
          { ...target, [axis]: bus, [cross]: target[cross] },
          target,
        ]),
      );
    }
  }
  return result;
}
