import type { PortSide } from '@tensor-trail/model-core/core/types';
import { automaticRoute, simplifyRoute, type Point } from './edgeRouting';
import { ROUTE_NOISE } from './routeNormalization';

/** Reconnect the last approach to a moved port. Redundant collinear pins are
 * not anchors: keeping them beyond the new port would draw through the module
 * before turning back. Only the terminal elbow and its neighbor may move;
 * the rest of a manually routed path stays where the user placed it.
 */
function reconnectEnd(points: Point[], port: Point, side: PortSide, otherSide: PortSide): Point[] {
  if (points.length < 2) return points;
  const axis = side === 'left' || side === 'right' ? 'x' : 'y';
  const cross = axis === 'x' ? 'y' : 'x';
  const sign = side === 'left' || side === 'top' ? -1 : 1;
  const old = points.at(-1)!;
  let corner = points.length - 1;
  while (corner > 0 && Math.abs(points[corner - 1][cross] - old[cross]) <= ROUTE_NOISE) corner--;
  // A pinned straight line has no meaningful interior elbow to preserve.
  if (corner === 0) return automaticRoute(points[0], port, otherSide, side);
  const prefix = points.slice(0, corner + 1).map((p) => ({ ...p }));
  const bend = prefix[corner];
  bend[cross] = port[cross];
  if ((bend[axis] - port[axis]) * sign < 24) {
    if (corner < 2) return automaticRoute(points[0], port, otherSide, side);
    bend[axis] = port[axis] + sign * 24;
    // Move the perpendicular approach as a whole, keeping the previous leg orthogonal.
    prefix[corner - 1][axis] = bend[axis];
  }
  return simplifyRoute([...prefix, port]);
}

export function reconnectMovedPorts(
  route: Point[],
  sourceSide: PortSide,
  targetSide: PortSide,
  sourceDelta?: Point,
  targetDelta?: Point,
): Point[] {
  let points = route;
  const moved = (d?: Point) => d && (Math.abs(d.x) > 1e-6 || Math.abs(d.y) > 1e-6);
  if (moved(sourceDelta)) {
    const p = route[0];
    points = reconnectEnd(
      [...points].reverse(),
      { x: p.x + sourceDelta!.x, y: p.y + sourceDelta!.y },
      sourceSide,
      targetSide,
    ).reverse();
  }
  if (moved(targetDelta)) {
    const p = route.at(-1)!;
    points = reconnectEnd(
      points,
      { x: p.x + targetDelta!.x, y: p.y + targetDelta!.y },
      targetSide,
      sourceSide,
    );
  }
  return points;
}
