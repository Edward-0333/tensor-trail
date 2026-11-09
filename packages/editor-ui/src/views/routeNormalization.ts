import { simplifyRoute, type Point, type RoutedEdge } from './edgeRouting';

// At the maximum 2x canvas zoom this is 0.04 screen pixels. It is numerical
// cleanup, deliberately much smaller than the 12px gesture snapping distance.
export const ROUTE_NOISE = 0.02;

/** Canonicalize subpixel residue within one physical output's fan-out.
 * Coordinate-wise replacement keeps every segment orthogonal. Actual port
 * coordinates are protected; old JSON/pins are never modified by this view.
 * Bounded clusters avoid chaining many small gaps into a meaningful displacement.
 */
export function normalizeWireRoutes(edges: RoutedEdge[]): Map<string, Point[]> {
  const groups = new Map<string, RoutedEdge[]>();
  for (const edge of edges) {
    const key = JSON.stringify([edge.source, edge.sourcePort]);
    const group = groups.get(key) ?? [];
    group.push(edge);
    groups.set(key, group);
  }
  const result = new Map<string, Point[]>();
  for (const group of groups.values()) {
    const axes = {} as Record<'x' | 'y', Map<number, number>>;
    for (const axis of ['x', 'y'] as const) {
      const protectedValues = new Set(
        group.flatMap((e) => (e.points.length ? [e.points[0][axis], e.points.at(-1)![axis]] : [])),
      );
      const values = [...new Set(group.flatMap((e) => e.points.map((p) => p[axis])))].sort(
        (a, b) => a - b,
      );
      const map = new Map<number, number>();
      for (let start = 0; start < values.length; ) {
        let end = start + 1;
        while (end < values.length && values[end] - values[start] <= ROUTE_NOISE) end++;
        const cluster = values.slice(start, end);
        const anchors = cluster.filter((v) => protectedValues.has(v));
        for (const value of cluster) {
          const canonical = protectedValues.has(value)
            ? value
            : anchors.length
              ? anchors.reduce((best, v) =>
                  Math.abs(v - value) < Math.abs(best - value) ? v : best,
                )
              : cluster[0];
          map.set(value, canonical);
        }
        start = end;
      }
      axes[axis] = map;
    }
    for (const edge of group)
      result.set(
        edge.id,
        simplifyRoute(
          edge.points.map((p) => {
            const x = axes.x.get(p.x)!,
              y = axes.y.get(p.y)!;
            return { x: x === p.x ? p.x : x, y: y === p.y ? p.y : y };
          }),
        ),
      );
  }
  return result;
}
