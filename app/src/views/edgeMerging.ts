import type { PortSide } from '../core/types';
import { manualRoute, simplifyRoute, type Point, type RoutedEdge } from './edgeRouting';

const EPS = 1e-6;
const same = (a: Point, b: Point) => Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS;
const axisOf = (a: Point, b: Point) =>
  Math.abs(a.y - b.y) < EPS ? ('x' as const) : ('y' as const);

/** Keep the last/first 24 units attached to the ports, even when snapping end sections. */
function span(points: Point[], index: number) {
  const a = points[index],
    b = points[index + 1];
  if (!a || !b || same(a, b)) return;
  const axis = axisOf(a, b),
    cross = axis === 'x' ? 'y' : 'x';
  if (Math.abs(a[cross] - b[cross]) > EPS) return;
  const sign = Math.sign(b[axis] - a[axis]);
  const start = a[axis] + (index === 0 ? sign * 24 : 0);
  const end = b[axis] - (index === points.length - 2 ? sign * 24 : 0);
  if ((end - start) * sign <= EPS) return;
  return {
    a,
    b,
    axis,
    cross,
    sign,
    start,
    end,
    low: Math.min(start, end),
    high: Math.max(start, end),
  } as const;
}

/** Reject routes that fold back or reconnect to their own earlier path. */
function hasLoop(points: Point[]) {
  for (let i = 0; i < points.length - 2; i++) {
    const a = points[i],
      b = points[i + 1],
      c = points[i + 2];
    if (
      axisOf(a, b) === axisOf(b, c) &&
      (b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) < -EPS
    )
      return true;
  }
  for (let i = 0; i < points.length - 1; i++)
    for (let j = i + 2; j < points.length - 1; j++) {
      const a = points[i],
        b = points[i + 1],
        c = points[j],
        d = points[j + 1];
      if (
        Math.max(Math.min(a.x, b.x), Math.min(c.x, d.x)) <=
          Math.min(Math.max(a.x, b.x), Math.max(c.x, d.x)) + EPS &&
        Math.max(Math.min(a.y, b.y), Math.min(c.y, d.y)) <=
          Math.min(Math.max(a.y, b.y), Math.max(c.y, d.y)) + EPS
      )
        return true;
    }
  return false;
}

/** Find the part actually moved by the user's perpendicular segment drag. */
export function movedSection(
  route: Point[],
  original: Point[],
  index: number,
  delta: Point,
): number[] {
  const a = original[index],
    b = original[index + 1];
  if (!a || !b) return [];
  const axis = axisOf(a, b),
    cross = axis === 'x' ? 'y' : 'x';
  const coordinate = a[cross] + delta[cross];
  const candidates: number[] = [];
  for (let i = 0; i < route.length - 1; i++) {
    const c = route[i],
      d = route[i + 1];
    if (axisOf(c, d) !== axis || Math.abs(c[cross] - coordinate) > EPS) continue;
    const overlap =
      Math.min(Math.max(a[axis], b[axis]), Math.max(c[axis], d[axis])) -
      Math.max(Math.min(a[axis], b[axis]), Math.min(c[axis], d[axis]));
    if (overlap > EPS) candidates.push(i);
  }
  return candidates;
}

export interface WireMerge {
  targetEdgeId: string;
  waypoints: Point[];
  route: Point[];
  distance: number;
  overlap: number;
}

/** Snap to a same-output parallel segment, replacing the prefix with its shared trunk.
 * Geometry only: logical edges and every destination remain independent and intact.
 * Threshold and minimum overlap are in flow units (the UI converts screen pixels).
 */
export function mergeNearbyWire(
  moving: RoutedEdge,
  activeSegments: number[],
  peers: RoutedEdge[],
  sourceSide: PortSide,
  targetSide: PortSide,
  threshold = 12,
  minimumOverlap = 8,
): WireMerge | undefined {
  let best: WireMerge | undefined;
  for (const peer of peers) {
    if (
      peer.id === moving.id ||
      peer.source !== moving.source ||
      peer.sourcePort !== moving.sourcePort
    )
      continue;
    const other = simplifyRoute(peer.points),
      own = simplifyRoute(moving.points);
    if (!other.length || !own.length || !same(other[0], own[0])) continue;
    for (const i of activeSegments) {
      const segment = span(own, i);
      if (!segment) continue;
      for (let j = 0; j < other.length - 1; j++) {
        const candidate = span(other, j);
        if (!candidate || candidate.axis !== segment.axis || candidate.sign !== segment.sign)
          continue;
        const { axis, cross, sign } = segment;
        const low = Math.max(segment.low, candidate.low),
          high = Math.min(segment.high, candidate.high);
        const overlap = high - low,
          distance = Math.abs(segment.a[cross] - candidate.a[cross]);
        if (overlap < minimumOverlap || distance > threshold) continue;
        if (
          best &&
          (distance > best.distance + EPS ||
            (Math.abs(distance - best.distance) < EPS && overlap <= best.overlap))
        )
          continue;
        const join: Point = { ...candidate.a, [axis]: sign > 0 ? high : low };
        const last = i === own.length - 2;
        const end: Point = { ...segment.b, [axis]: segment.end, [cross]: candidate.a[cross] };
        const tail = last ? [{ ...segment.b, [axis]: segment.end }, own.at(-1)!] : own.slice(i + 2);
        const joined = simplifyRoute([...other.slice(0, j + 1), join, end, ...tail]);
        const waypoints = joined.slice(1, -1);
        if (waypoints.length > 100) continue;
        // Check the exact persisted representation, including the fixed port exits.
        const route = simplifyRoute(
          manualRoute(own[0], own.at(-1)!, sourceSide, targetSide, waypoints),
        );
        if (hasLoop(route)) continue;
        best = { targetEdgeId: peer.id, waypoints, route, distance, overlap };
      }
    }
  }
  return best;
}
