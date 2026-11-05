import type { PortSide } from '../core/types';
import {
  moveRouteSegment,
  routePath,
  simplifyRoute,
  type Point,
  type RoutedEdge,
} from './edgeRouting';
const EPS = 1e-6;
const same = (a: Point, b: Point) => Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS;
const length = (a: Point, b: Point) => Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
const sameOutput = (a: RoutedEdge, b: RoutedEdge) =>
  a.source === b.source && a.sourcePort === b.sourcePort;

/** A shared wire must be reachable by the same path from the output.
 * Later crossings after a fork are not electrical-style connections.
 */
export function sharedPrefixLength(a: Point[], b: Point[]): number {
  if (!a.length || !b.length || !same(a[0], b[0])) return 0;
  let i = 1,
    j = 1,
    cursor = a[0],
    shared = 0;
  const sign = (v: number) => (Math.abs(v) < EPS ? 0 : Math.sign(v));
  while (i < a.length && j < b.length) {
    const da = { x: sign(a[i].x - cursor.x), y: sign(a[i].y - cursor.y) };
    const db = { x: sign(b[j].x - cursor.x), y: sign(b[j].y - cursor.y) };
    if (!same(da, db)) break;
    const step = Math.min(length(cursor, a[i]), length(cursor, b[j]));
    if (step > EPS) {
      cursor = { x: cursor.x + da.x * step, y: cursor.y + da.y * step };
      shared += step;
    }
    let advanced = false;
    if (same(cursor, a[i])) {
      i++;
      advanced = true;
    }
    if (same(cursor, b[j])) {
      j++;
      advanced = true;
    }
    if (!advanced && step <= EPS) break;
  }
  return shared;
}
export function prioritizeWires(edges: RoutedEdge[]): RoutedEdge[] {
  return [...edges].sort(
    (a, b) =>
      Number(!!b.labelDragging) - Number(!!a.labelDragging) ||
      Number(!!b.selected) - Number(!!a.selected) ||
      a.id.localeCompare(b.id),
  );
}
const cumulative = (points: Point[]) => {
  const distances = [0];
  for (let i = 1; i < points.length; i++)
    distances.push(distances[i - 1] + length(points[i - 1], points[i]));
  return distances;
};
/** Retain collinear split points: these are genuine branch boundaries for interaction. */
function splitAt(points: Point[], cuts: number[]): Point[] {
  const source = simplifyRoute(points);
  if (!source.length) return [];
  const marks = [...new Set(cuts.filter(Number.isFinite))].sort((a, b) => a - b),
    out = [source[0]];
  let travelled = 0;
  for (let i = 1; i < source.length; i++) {
    const a = source[i - 1],
      b = source[i],
      span = length(a, b);
    for (const d of marks)
      if (d > travelled + EPS && d < travelled + span - EPS) {
        const ratio = (d - travelled) / span;
        out.push({ x: a.x + (b.x - a.x) * ratio, y: a.y + (b.y - a.y) * ratio });
      }
    out.push(b);
    travelled += span;
  }
  return out;
}
export function suffix(points: Point[], cut: number): Point[] {
  if (cut < EPS) return points;
  const divided = splitAt(points, [cut]),
    distances = cumulative(divided);
  const index = distances.findIndex((d) => d >= cut - EPS);
  return index < 0 ? [] : divided.slice(index);
}
export function distanceAt(points: Point[], point: Point): number | undefined {
  let travelled = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1],
      b = points[i];
    if (Math.abs(length(a, point) + length(point, b) - length(a, b)) < EPS)
      return travelled + length(a, point);
    travelled += length(a, b);
  }
}
export function wireDrawing(edge: RoutedEdge, edges: RoutedEdge[]) {
  const ordered = prioritizeWires(
    [...edges.filter((p) => p.id !== edge.id), edge].filter((p) => sameOutput(edge, p)),
  );
  let cut = 0,
    showLabel = true;
  for (const peer of ordered) {
    if (peer.id === edge.id) break;
    const shared = sharedPrefixLength(edge.points, peer.points);
    cut = Math.max(cut, shared);
    if (
      edge.label &&
      peer.label &&
      edge.label.text === peer.label.text &&
      edge.label.horizontal === peer.label.horizontal &&
      same(edge.label, peer.label)
    ) {
      const a = distanceAt(edge.points, edge.label),
        b = distanceAt(peer.points, peer.label);
      if (
        shared > EPS &&
        a !== undefined &&
        b !== undefined &&
        a <= shared + EPS &&
        b <= shared + EPS
      )
        showLabel = false;
    }
  }
  const visible = suffix(edge.points, cut);
  return { path: visible.length > 1 ? routePath(visible) : '', showLabel, sharedLength: cut };
}
export function sharedSections(edge: RoutedEdge, peers: RoutedEdge[]): Point[] {
  return splitAt(
    edge.points,
    peers
      .filter((p) => p.id !== edge.id && sameOutput(edge, p))
      .map((p) => sharedPrefixLength(edge.points, p.points)),
  );
}
export interface WireDragTarget {
  id: string;
  route: Point[];
  index: number;
  sourceSide: PortSide;
  targetSide: PortSide;
}
export interface WireRouteChange {
  id: string;
  waypoints: Point[];
}

/** Capture the routes once on pointer-down, so every preview derives from the same geometry. */
export function sharedDragTargets(
  edge: RoutedEdge,
  sections: Point[],
  index: number,
  peers: RoutedEdge[],
  independent = false,
): WireDragTarget[] {
  const distances = cumulative(sections),
    start = distances[index],
    end = distances[index + 1];
  if (end === undefined || end - start < EPS) return [];
  const candidates = [
    edge,
    ...peers.filter(
      (p) =>
        p.id !== edge.id &&
        !independent &&
        sameOutput(edge, p) &&
        sharedPrefixLength(edge.points, p.points) >= end - EPS,
    ),
  ];
  return candidates
    .map((peer) => {
      const route = splitAt(peer.points, [start, end]);
      return {
        id: peer.id,
        route,
        index: cumulative(route).findIndex((d) => Math.abs(d - start) < EPS),
        sourceSide: peer.sourceSide ?? 'right',
        targetSide: peer.targetSide ?? 'left',
      };
    })
    .filter((p) => p.index >= 0 && p.index < p.route.length - 1);
}
export function moveSharedTargets(
  targets: WireDragTarget[],
  delta: Point,
  coordinate?: number,
): WireRouteChange[] {
  return targets.map((target) => ({
    id: target.id,
    waypoints: moveRouteSegment(
      target.route,
      target.index,
      delta,
      target.sourceSide,
      target.targetSide,
      coordinate,
    ),
  }));
}
