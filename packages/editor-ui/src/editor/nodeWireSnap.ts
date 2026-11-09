import type { Graph, PortSide } from '@tensor-trail/model-core/core/types';
import {
  automaticRoute,
  manualRoute,
  simplifyRoute,
  type Point,
  type RoutedEdge,
} from '../views/edgeRouting';
import { automaticWireRoutes } from '../views/automaticWires';
import { mergeNearbyWire } from '../views/edgeMerging';
import type { WireRouteChange } from '../views/sharedWires';
import { reconnectMovedPorts } from '../views/endpointRouting';

type Axis = 'x' | 'y';
export interface NodePosition {
  id: string;
  position: Point;
}
export interface WireSnapGuide {
  key: string;
  axis: Axis;
  coordinate: number;
  from: Point;
  to: Point;
}
export interface NodeWireSnap {
  positions: NodePosition[];
  guides: WireSnapGuide[];
}

/** Use the drag-start geometry, never yesterday's snapped preview, as the reference.
 * Every selected node receives the same correction. An internal edge in a moving
 * selection is not a stationary anchor. Thresholds are expressed in screen pixels.
 */
export function snapNodesToWires(
  graph: Graph,
  routes: RoutedEdge[],
  initial: NodePosition[],
  moving: NodePosition[],
  zoom = 1,
  previous: WireSnapGuide[] = [],
  disabled = false,
): NodeWireSnap {
  if (disabled || !moving.length) return { positions: moving, guides: [] };
  const movingById = new Map(moving.map((n) => [n.id, n.position]));
  const initialById = new Map(initial.map((n) => [n.id, n.position]));
  const candidates: (WireSnapGuide & { correction: number; distance: number })[] = [];
  const byId = new Map(routes.map((r) => [r.id, r]));
  const models = new Map(graph.edges.map((e) => [e.id, e]));
  const parallel = (side: PortSide) => (side === 'left' || side === 'right' ? 'y' : 'x');
  for (const edge of graph.edges) {
    const sourceMoving = movingById.has(edge.source),
      targetMoving = movingById.has(edge.target);
    if (sourceMoving === targetMoving) continue;
    const route = byId.get(edge.id);
    if (!route?.points.length) continue;
    const id = sourceMoving ? edge.source : edge.target;
    const start = initialById.get(id),
      current = movingById.get(id);
    const side = sourceMoving ? route.sourceSide : route.targetSide;
    if (!start || !current || !side) continue;
    const axis = parallel(side),
      along = axis === 'x' ? 'y' : 'x';
    const originalPort = sourceMoving ? route.points[0] : route.points.at(-1)!;
    const port = {
      x: originalPort.x + current.x - start.x,
      y: originalPort.y + current.y - start.y,
    };
    const add = (coordinate: number, reference: Point, key: string) => {
      const correction = coordinate - port[axis];
      const distance = Math.abs(correction) * zoom;
      const sticky = previous.some((g) => g.key === key && g.axis === axis);
      if (distance > (sticky ? 18 : 12)) return;
      const from = { ...reference, [axis]: coordinate };
      const sign = side === 'left' || side === 'top' ? -1 : 1;
      // A guide to an old pin must not extend back through the moving module.
      if ((from[along] - port[along]) * sign < 0) from[along] = port[along] + (sign * 80) / zoom;
      candidates.push({
        key,
        axis,
        coordinate,
        from,
        to: { ...port, [axis]: coordinate },
        correction,
        distance,
      });
    };
    const pin = sourceMoving ? edge.waypoints?.[0] : edge.waypoints?.at(-1);
    if (pin) add(pin[axis], pin, `${edge.id}:pin`);
    const otherSide = sourceMoving ? route.targetSide : route.sourceSide;
    const opposite = sourceMoving ? route.points.at(-1)! : route.points[0];
    if (otherSide && parallel(otherSide) === axis) add(opposite[axis], opposite, `${edge.id}:port`);
    // A peer must carry the same output. Merely crossing another signal is not a connection.
    for (const peer of routes) {
      const model = models.get(peer.id);
      if (
        !model ||
        peer.id === edge.id ||
        movingById.has(model.source) ||
        movingById.has(model.target) ||
        peer.source !== edge.source ||
        peer.sourcePort !== edge.sourcePort
      )
        continue;
      for (let i = 1; i < peer.points.length; i++) {
        const a = peer.points[i - 1],
          b = peer.points[i];
        if (a[axis] !== b[axis] || Math.abs(b[along] - a[along]) * zoom < 16) continue;
        const low = Math.min(a[along], b[along]),
          high = Math.max(a[along], b[along]);
        if (port[along] < low - 80 / zoom || port[along] > high + 80 / zoom) continue;
        add(
          a[axis],
          { ...a, [along]: Math.max(low, Math.min(high, port[along])) },
          `${edge.id}:${peer.id}:${i}`,
        );
      }
    }
  }
  const guides: WireSnapGuide[] = [],
    correction = { x: 0, y: 0 };
  for (const axis of ['x', 'y'] as const) {
    const choices = candidates
      .filter((c) => c.axis === axis)
      .sort((a, b) => a.distance - b.distance || a.key.localeCompare(b.key));
    const best =
      choices.find((c) => previous.some((g) => g.key === c.key && g.axis === axis)) ?? choices[0];
    if (!best) continue;
    correction[axis] = best.correction;
    guides.push({ key: best.key, axis, coordinate: best.coordinate, from: best.from, to: best.to });
  }
  return {
    positions: moving.map((n) => ({
      id: n.id,
      position: { x: n.position.x + correction.x, y: n.position.y + correction.y },
    })),
    guides,
  };
}

/** Commit layout together with node positions, never create undo entries while dragging.
 * Internal pinned wires travel with a selected group. External routes reconnect
 * their endpoint approaches; remote bends stay fixed, and nearby parallel sections
 * may adopt a same-output trunk on release.
 */
export function nodeDragRoutes(
  graph: Graph,
  routes: RoutedEdge[],
  initial: NodePosition[],
  moving: NodePosition[],
  zoom = 1,
  disabled = false,
): WireRouteChange[] {
  const starts = new Map(initial.map((n) => [n.id, n.position]));
  const deltas = new Map(
    moving.flatMap((n) => {
      const old = starts.get(n.id);
      return old ? [[n.id, { x: n.position.x - old.x, y: n.position.y - old.y }] as const] : [];
    }),
  );
  const modelEdges = new Map(graph.edges.map((e) => [e.id, e]));
  const changedPins = new Map(
    nodeWirePreview(graph, routes, initial, moving).map((c) => [c.id, c.waypoints]),
  );
  const shifted = routes.map((route) => {
    const edge = modelEdges.get(route.id);
    if (!edge || !route.sourceSide || !route.targetSide || route.points.length < 2)
      return { ...route, automatic: false };
    const sourceDelta = deltas.get(edge.source),
      targetDelta = deltas.get(edge.target);
    const shift = (p: Point, delta?: Point) => (delta ? { x: p.x + delta.x, y: p.y + delta.y } : p);
    const pins = changedPins.get(edge.id) ?? edge.waypoints;
    const source = shift(route.points[0], sourceDelta),
      target = shift(route.points.at(-1)!, targetDelta);
    const points = pins?.length
      ? simplifyRoute(manualRoute(source, target, route.sourceSide, route.targetSide, pins))
      : automaticRoute(source, target, route.sourceSide, route.targetSide);
    return { ...route, points, automatic: !pins?.length };
  });
  const planned = automaticWireRoutes(shifted);
  const peers = shifted.map((r) => ({ ...r, points: planned.get(r.id) ?? r.points }));
  if (!disabled)
    for (const peer of peers) {
      const edge = modelEdges.get(peer.id);
      if (
        !edge?.waypoints?.length ||
        !peer.sourceSide ||
        !peer.targetSide ||
        deltas.has(edge.source) === deltas.has(edge.target)
      )
        continue;
      const result = mergeNearbyWire(
        peer,
        peer.points.slice(1).map((_, i) => i),
        peers,
        peer.sourceSide,
        peer.targetSide,
        12 / zoom,
        8 / zoom,
      );
      if (result) {
        changedPins.set(peer.id, result.waypoints);
        peer.points = result.route;
      }
    }
  return [...changedPins]
    .filter(([id, pins]) => JSON.stringify(pins) !== JSON.stringify(modelEdges.get(id)?.waypoints))
    .map(([id, waypoints]) => ({ id, waypoints }));
}

export function groupWirePreview(
  graph: Graph,
  initial: NodePosition[],
  moving: NodePosition[],
): WireRouteChange[] {
  const starts = new Map(initial.map((n) => [n.id, n.position]));
  const deltas = new Map(
    moving.flatMap((n) => {
      const old = starts.get(n.id);
      return old ? [[n.id, { x: n.position.x - old.x, y: n.position.y - old.y }] as const] : [];
    }),
  );
  return graph.edges.flatMap((e) => {
    const a = deltas.get(e.source),
      b = deltas.get(e.target);
    return e.waypoints?.length && a && b && Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6
      ? [{ id: e.id, waypoints: e.waypoints.map((p) => ({ x: p.x + a.x, y: p.y + a.y })) }]
      : [];
  });
}

/** One layout preview is reused by the final commit, so pointer-up cannot restore
 * the old endpoint pins. No project state or undo history is changed here.
 */
export function nodeWirePreview(
  graph: Graph,
  routes: RoutedEdge[],
  initial: NodePosition[],
  moving: NodePosition[],
): WireRouteChange[] {
  const changes = groupWirePreview(graph, initial, moving);
  const handled = new Set(changes.map((c) => c.id));
  const starts = new Map(initial.map((n) => [n.id, n.position]));
  const deltas = new Map(
    moving.flatMap((n) => {
      const old = starts.get(n.id);
      return old ? [[n.id, { x: n.position.x - old.x, y: n.position.y - old.y }] as const] : [];
    }),
  );
  const rendered = new Map(routes.map((r) => [r.id, r]));
  for (const edge of graph.edges) {
    if (handled.has(edge.id) || !edge.waypoints?.length) continue;
    const a = deltas.get(edge.source),
      b = deltas.get(edge.target);
    if ((!a || Math.hypot(a.x, a.y) < 1e-6) && (!b || Math.hypot(b.x, b.y) < 1e-6)) continue;
    const route = rendered.get(edge.id);
    if (!route?.sourceSide || !route.targetSide || route.points.length < 2) continue;
    const next = reconnectMovedPorts(route.points, route.sourceSide, route.targetSide, a, b);
    changes.push({ id: edge.id, waypoints: next.slice(1, -1) });
  }
  return changes.filter(
    (c) =>
      JSON.stringify(c.waypoints) !==
      JSON.stringify(graph.edges.find((e) => e.id === c.id)?.waypoints),
  );
}
