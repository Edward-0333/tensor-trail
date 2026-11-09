import type { PortSide } from '@tensor-trail/model-core/core/types';
import { getSmoothStepPath, Position } from '@xyflow/react';

export interface Point {
  x: number;
  y: number;
}
export interface RoutePoint extends Point {
  insertIndex: number;
}
const direction: Record<PortSide, Point> = {
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
  top: { x: 0, y: -1 },
  bottom: { x: 0, y: 1 },
};

const equal = (a: Point, b: Point) => Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6;

/** Remove zero-length and forward-collinear sections, but keep U-turns. */
export function simplifyRoute(points: Point[]): Point[] {
  const result: Point[] = [];
  for (const point of points) {
    if (result.length && equal(point, result.at(-1)!)) continue;
    while (result.length >= 2) {
      const a = result.at(-2)!,
        b = result.at(-1)!;
      const vertical = a.x === b.x && b.x === point.x && (b.y - a.y) * (point.y - b.y) >= 0;
      const horizontal = a.y === b.y && b.y === point.y && (b.x - a.x) * (point.x - b.x) >= 0;
      if (!vertical && !horizontal) break;
      result.pop();
    }
    result.push({ x: point.x, y: point.y });
  }
  return result;
}

/** Reuse React Flow's automatic routing, with zero-radius bends to recover its polyline. */
export function automaticRoute(
  source: Point,
  target: Point,
  sourceSide: PortSide,
  targetSide: PortSide,
): Point[] {
  const [path] = getSmoothStepPath({
    sourceX: source.x,
    sourceY: source.y,
    targetX: target.x,
    targetY: target.y,
    sourcePosition: sourceSide as Position,
    targetPosition: targetSide as Position,
    offset: 24,
    borderRadius: 0,
  });
  // At radius zero, Q control/end points coincide with the corner. Only M/L/Q
  // occur in this public helper's output; keep this adapter covered for every port side.
  const coordinates = (path.match(/[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  const points: Point[] = [];
  for (let i = 0; i < coordinates.length; i += 2)
    points.push({ x: coordinates[i], y: coordinates[i + 1] });
  return simplifyRoute(points);
}

/** Snap a dragged section to its adjacent parallel sections, in flow coordinates.
 * Use an absolute cross-coordinate to keep fractional routes exactly orthogonal.
 * A zero threshold disables snapping (Alt in the editor).
 */
export function snapSegmentMovement(route: Point[], index: number, delta: Point, threshold = 12) {
  const a = route[index],
    b = route[index + 1];
  if (!a || !b || equal(a, b) || (a.x !== b.x && a.y !== b.y))
    throw new Error('只能拖动水平或竖直线段。');
  const cross = a.y === b.y ? 'y' : 'x';
  const desired = a[cross] + delta[cross];
  let coordinate = desired,
    distance = Infinity,
    snapped = false;
  if (threshold > 0) {
    for (const neighbor of [index - 2, index + 2]) {
      const c = route[neighbor],
        d = route[neighbor + 1];
      if (!c || !d || equal(c, d) || c[cross] !== d[cross]) continue;
      const gap = Math.abs(c[cross] - desired);
      if (gap <= threshold && gap < distance) {
        coordinate = c[cross];
        distance = gap;
        snapped = true;
      }
    }
  }
  return { coordinate, delta: { ...delta, [cross]: coordinate - a[cross] }, snapped };
}

/** Translate a visible straight section perpendicular to itself. Persist regular pins. */
export function moveRouteSegment(
  route: Point[],
  index: number,
  delta: Point,
  sourceSide: PortSide,
  targetSide: PortSide,
  coordinate?: number,
): Point[] {
  const points = route.map((p) => ({ ...p }));
  const a = points[index],
    b = points[index + 1];
  if (!a || !b || equal(a, b)) return points.slice(1, -1);
  const horizontal = a.y === b.y;
  if (!horizontal && a.x !== b.x) throw new Error('只能拖动水平或竖直线段。');
  const shift = (p: Point): Point =>
    horizontal
      ? { x: p.x, y: coordinate ?? p.y + delta.y }
      : { x: coordinate ?? p.x + delta.x, y: p.y };
  const source = points[0],
    target = points.at(-1)!;
  points[index] = shift(a);
  points[index + 1] = shift(b);
  // End sections get a new elbow after the fixed port exit, never move the port.
  // Delete only the two corners of an elbow collapsed by this drag. This also
  // removes local backtracking when opposite-facing sections meet, while keeping
  // unrelated U-turns elsewhere and the fixed source/target exits intact.
  const removed = new Set<number>();
  const coincides = (i: number, j: number) =>
    points[i].x === points[j].x && points[i].y === points[j].y;
  if (index >= 2 && coincides(index - 1, index)) {
    removed.add(index - 1);
    removed.add(index);
  }
  if (index + 3 < points.length && coincides(index + 1, index + 2)) {
    removed.add(index + 1);
    removed.add(index + 2);
  }
  const pins = points.filter((_, i) => i > 0 && i < points.length - 1 && !removed.has(i));
  if (index === 0)
    pins.unshift(
      shift({
        x: source.x + direction[sourceSide].x * 24,
        y: source.y + direction[sourceSide].y * 24,
      }),
    );
  if (index + 1 === points.length - 1)
    pins.push(
      shift({
        x: target.x + direction[targetSide].x * 24,
        y: target.y + direction[targetSide].y * 24,
      }),
    );
  return simplifyRoute(pins);
}

export interface RoutedEdge {
  id: string;
  source: string;
  sourcePort: string;
  points: Point[];
  valueKey?: string;
  color?: string;
  sourceSide?: PortSide;
  targetSide?: PortSide;
  selected?: boolean;
  labelDragging?: boolean;
  label?: { text: string; x: number; y: number; horizontal: boolean };
}
export interface Junction extends Point {
  key: string;
  source: string;
  sourcePort: string;
  color?: string;
}

/** Only a shared prefix is connected: crossings after divergence are not junctions. */
function forkPoint(a: Point[], b: Point[]): Point | undefined {
  if (!a.length || !b.length || !equal(a[0], b[0])) return;
  let i = 1,
    j = 1,
    cursor = { ...a[0] },
    shared = 0;
  while (i < a.length && j < b.length) {
    // Fractional node coordinates can leave a tiny residue at a shared corner.
    // Do not mistake it for movement on the other axis and walk off the route.
    const sign = (value: number) => (Math.abs(value) < 1e-6 ? 0 : Math.sign(value));
    const da = { x: sign(a[i].x - cursor.x), y: sign(a[i].y - cursor.y) };
    const db = { x: sign(b[j].x - cursor.x), y: sign(b[j].y - cursor.y) };
    if (!equal(da, db)) return shared > 1e-6 ? cursor : undefined;
    const lengthA = Math.abs(a[i].x - cursor.x) + Math.abs(a[i].y - cursor.y);
    const lengthB = Math.abs(b[j].x - cursor.x) + Math.abs(b[j].y - cursor.y);
    const length = Math.min(lengthA, lengthB);
    cursor = { x: cursor.x + da.x * length, y: cursor.y + da.y * length };
    shared += length;
    if (equal(cursor, a[i])) i++;
    if (equal(cursor, b[j])) j++;
  }
}

export function branchJunctions(edges: RoutedEdge[]): Junction[] {
  const groups = new Map<string, RoutedEdge[]>();
  for (const edge of edges) {
    const key = JSON.stringify([edge.source, edge.sourcePort]);
    const group = groups.get(key) ?? [];
    group.push(edge);
    groups.set(key, group);
  }
  const junctions = new Map<string, Junction>();
  for (const group of groups.values()) {
    const routes = group.map((edge) => simplifyRoute(edge.points));
    for (let i = 0; i < group.length; i++)
      for (let j = i + 1; j < group.length; j++) {
        const point = forkPoint(routes[i], routes[j]);
        if (!point) continue;
        const { source, sourcePort, color } = group[i];
        const key = JSON.stringify([source, sourcePort, +point.x.toFixed(5), +point.y.toFixed(5)]);
        junctions.set(key, { ...point, key, source, sourcePort, color });
      }
  }
  return [...junctions.values()];
}

/** Keep short exits aligned with the ports, then pass through the user's pins. */
export function manualRoute(
  source: Point,
  target: Point,
  sourceSide: PortSide,
  targetSide: PortSide,
  pins: Point[],
): RoutePoint[] {
  const points: RoutePoint[] = [];
  const append = (point: Point, insertIndex: number) => {
    const last = points.at(-1);
    if (!last || last.x !== point.x || last.y !== point.y) points.push({ ...point, insertIndex });
  };
  const a = {
    x: source.x + direction[sourceSide].x * 24,
    y: source.y + direction[sourceSide].y * 24,
  };
  const b = {
    x: target.x + direction[targetSide].x * 24,
    y: target.y + direction[targetSide].y * 24,
  };
  append(source, 0);
  append(a, 0);
  pins.forEach((pin, i) => {
    const last = points.at(-1)!;
    append(
      i === 0 && direction[sourceSide].x !== 0 ? { x: last.x, y: pin.y } : { x: pin.x, y: last.y },
      i,
    );
    append(pin, i);
  });
  const last = points.at(-1)!;
  append(
    direction[targetSide].x !== 0 ? { x: b.x, y: last.y } : { x: last.x, y: b.y },
    pins.length,
  );
  append(b, pins.length);
  append(target, pins.length);
  return points;
}

export function routePath(points: Point[]): string {
  return points.map((point, i) => `${i ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
}

/** Default to the longest section; manual positions stay on the current route as it changes. */
export function routeLabelAnchor(
  points: Point[],
  position?: number,
): Point & { horizontal: boolean; position: number } {
  let anchor = { ...(points[0] ?? { x: 0, y: 0 }), horizontal: true, position: 0.5 };
  const lengths = points.slice(1).map((p, i) => Math.hypot(p.x - points[i].x, p.y - points[i].y));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  if (total === 0) return anchor;
  const manual = position !== undefined && Number.isFinite(position);
  const fraction = manual ? Math.max(0, Math.min(1, position)) : 0.5;
  const distance = fraction * total;
  let longest = 0;
  let travelled = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1],
      b = points[i];
    const length = lengths[i - 1];
    if (length > 0 && manual && distance <= travelled + length) {
      const t = Math.max(0, Math.min(1, (distance - travelled) / length));
      return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        horizontal: a.y === b.y,
        position: fraction,
      };
    }
    if (!manual && length > longest) {
      longest = length;
      anchor = {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        horizontal: a.y === b.y,
        position: (travelled + length / 2) / total,
      };
    }
    travelled += length;
  }
  return anchor;
}

/** Project a dragged anchor onto its own line, preferring continuity at equal-distance crossings. */
export function projectLabelPosition(points: Point[], point: Point, preferred = 0.5): number {
  const lengths = points.slice(1).map((p, i) => Math.hypot(p.x - points[i].x, p.y - points[i].y));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  if (total === 0) return 0.5;
  let travelled = 0,
    bestDistance = Infinity,
    position = preferred;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1],
      b = points[i],
      length = lengths[i - 1];
    if (!length) continue;
    const dx = b.x - a.x,
      dy = b.y - a.y;
    const t = Math.max(
      0,
      Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (length * length)),
    );
    const distance = Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
    const candidate = (travelled + t * length) / total;
    if (
      distance < bestDistance - 1e-6 ||
      (Math.abs(distance - bestDistance) <= 1e-6 &&
        Math.abs(candidate - preferred) < Math.abs(position - preferred))
    ) {
      bestDistance = distance;
      position = candidate;
    }
    travelled += length;
  }
  return Math.max(0, Math.min(1, position));
}

export function routeMidpoint(points: Point[]): Point {
  const lengths = points.slice(1).map((p, i) => Math.hypot(p.x - points[i].x, p.y - points[i].y));
  let remaining = lengths.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < lengths.length; i++) {
    if (lengths[i] && remaining <= lengths[i]) {
      const t = remaining / lengths[i];
      return {
        x: points[i].x + (points[i + 1].x - points[i].x) * t,
        y: points[i].y + (points[i + 1].y - points[i].y) * t,
      };
    }
    remaining -= lengths[i];
  }
  return points[0] ?? { x: 0, y: 0 };
}

/** Insert next to the segment that was clicked, even on a route with many pins. */
export function routeInsertIndex(points: RoutePoint[], point: Point): number {
  let closest = Infinity,
    index = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1],
      b = points[i];
    const dx = b.x - a.x,
      dy = b.y - a.y;
    const t = Math.max(
      0,
      Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy || 1)),
    );
    const distance = Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
    if (distance < closest) {
      closest = distance;
      index = b.insertIndex;
    }
  }
  return index;
}
