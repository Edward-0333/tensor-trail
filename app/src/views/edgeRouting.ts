import type { PortSide } from '../core/types';

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
  if (!points.length) return '';
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const previous = points[i - 1],
      current = points[i],
      next = points[i + 1];
    const incoming = Math.hypot(current.x - previous.x, current.y - previous.y);
    const outgoing = Math.hypot(next.x - current.x, next.y - current.y);
    const turns =
      (current.x - previous.x) * (next.y - current.y) !==
      (current.y - previous.y) * (next.x - current.x);
    const radius = turns ? Math.min(6, incoming / 2, outgoing / 2) : 0;
    if (!radius) path += ` L ${current.x} ${current.y}`;
    else {
      const enter = {
        x: current.x + ((previous.x - current.x) * radius) / incoming,
        y: current.y + ((previous.y - current.y) * radius) / incoming,
      };
      const exit = {
        x: current.x + ((next.x - current.x) * radius) / outgoing,
        y: current.y + ((next.y - current.y) * radius) / outgoing,
      };
      path += ` L ${enter.x} ${enter.y} Q ${current.x} ${current.y} ${exit.x} ${exit.y}`;
    }
  }
  if (points.length > 1) path += ` L ${points.at(-1)!.x} ${points.at(-1)!.y}`;
  return path;
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
