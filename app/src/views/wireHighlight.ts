import { routePath, type Point, type RoutedEdge, type Junction } from './edgeRouting';
import { distanceAt, sharedPrefixLength, sharedSections, suffix } from './sharedWires';

const EPS = 1e-6;
const length = (a: Point, b: Point) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const sameOutput = (a: RoutedEdge, b: RoutedEdge) =>
  a.source === b.source && a.sourcePort === b.sourcePort;
const valueKey = (edge: RoutedEdge) =>
  edge.valueKey ?? JSON.stringify([edge.source, edge.sourcePort]);

/** A visual selection, independent of React Flow's editable/deletable edge selection.
 * Keep branch membership rather than screen coordinates so moving a junction keeps
 * the same subtree highlighted. Its entry is the last fork from excluded branches.
 */
export interface WireSelection {
  edgeId: string;
  members: string[];
}

export function nearestSection(points: Point[], point: Point): number {
  let nearest = 0,
    best = Infinity;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i],
      b = points[i + 1];
    const dx = b.x - a.x,
      dy = b.y - a.y;
    const squared = dx * dx + dy * dy;
    if (squared < EPS) continue;
    const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / squared));
    const distance = Math.hypot(point.x - a.x - dx * t, point.y - a.y - dy * t);
    // At a fork/corner prefer the downstream section.
    if (distance <= best + EPS) {
      nearest = i;
      best = distance;
    }
  }
  return nearest;
}

export function selectWireSection(
  edge: RoutedEdge,
  edges: RoutedEdge[],
  index: number,
): WireSelection {
  const sections = sharedSections(edge, edges);
  const end = sections.slice(1, index + 2).reduce((d, p, i) => d + length(sections[i], p), 0);
  const members = edges
    .filter(
      (peer) =>
        peer.id === edge.id ||
        (index === 0
          ? valueKey(peer) === valueKey(edge)
          : sameOutput(edge, peer) && sharedPrefixLength(edge.points, peer.points) >= end - EPS),
    )
    .map((peer) => peer.id);
  return { edgeId: edge.id, members: [...new Set([edge.id, ...members])] };
}

/** Distance from each physical source where downstream highlighting begins. */
export function wireHighlightStarts(
  edges: RoutedEdge[],
  selection: WireSelection | null,
): Map<string, number> {
  const starts = new Map<string, number>();
  const owner = edges.find((edge) => edge.id === selection?.edgeId && edge.selected);
  const members = new Set(owner ? selection!.members : []);
  for (const edge of edges) {
    if (members.has(edge.id)) {
      const excluded = edges.filter((peer) => sameOutput(edge, peer) && !members.has(peer.id));
      starts.set(
        edge.id,
        Math.max(0, ...excluded.map((peer) => sharedPrefixLength(edge.points, peer.points))),
      );
    } else if (edge.selected && edge.id !== owner?.id) {
      // Keyboard and box selections retain their ordinary whole-edge highlighting.
      starts.set(edge.id, 0);
    }
  }
  return starts;
}

/** Split only the stroke owned by this edge; never draw the common trunk twice. */
export function highlightedDrawing(points: Point[], sharedLength: number, start?: number) {
  const visible = suffix(points, sharedLength);
  const path = (p: Point[]) => (p.length > 1 ? routePath(p) : '');
  if (start === undefined) return { normal: path(visible), highlighted: '' };
  const cut = Math.max(sharedLength, start);
  const highlight = suffix(points, cut);
  if (start <= sharedLength + EPS) return { normal: '', highlighted: path(visible) };
  const prefix = [] as Point[];
  let travelled = sharedLength;
  for (let i = 0; i < visible.length; i++) {
    if (i) travelled += length(visible[i - 1], visible[i]);
    if (travelled >= cut - EPS) break;
    prefix.push(visible[i]);
  }
  if (highlight.length) prefix.push(highlight[0]);
  return { normal: path(prefix), highlighted: path(highlight) };
}

export function pointHighlighted(
  edge: RoutedEdge,
  point: Point,
  starts: Map<string, number>,
): boolean {
  const start = starts.get(edge.id),
    distance = distanceAt(edge.points, point);
  return start !== undefined && distance !== undefined && distance >= start - EPS;
}

export function junctionHighlighted(
  point: Junction,
  edges: RoutedEdge[],
  starts: Map<string, number>,
): boolean {
  return edges.some(
    (edge) =>
      edge.source === point.source &&
      edge.sourcePort === point.sourcePort &&
      pointHighlighted(edge, point, starts),
  );
}
