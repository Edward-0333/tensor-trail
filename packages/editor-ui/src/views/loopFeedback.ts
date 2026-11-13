import type { Graph, ModelNode, LoopFeedbackLayout } from '@tensor-trail/model-core/core/types';
import { forLoopStates } from '@tensor-trail/model-core/core/forLoop';
import { manualRoute, routeLabelAnchor, simplifyRoute, type Point } from './edgeRouting';

export interface LoopFeedback {
  state: string;
  source: string;
  target: string;
}

/** Match declared state ports, not labels, shapes, or tensor identity. */
export function loopFeedback(graph: Graph, owner?: ModelNode): LoopFeedback[] {
  if (owner?.type !== 'ForLoop') return [];
  let states: string[];
  try { states = forLoopStates(owner); } catch { return []; }
  return states.flatMap(state => {
    const boundary = (type: string) => graph.nodes.filter(node =>
      node.type === type && node.valueSource === undefined && node.params.port === state);
    const inputs = boundary('Inlet'), outputs = boundary('Outlet');
    // An unfinished or ambiguous interface must not imply a valid feedback binding.
    return inputs.length === 1 && outputs.length === 1
      ? [{ state, source: outputs[0].id, target: inputs[0].id }] : [];
  });
}

interface FeedbackNode {
  id: string;
  position: Point;
  measured?: { width?: number; height?: number };
}

export interface FeedbackRoute extends LoopFeedback {
  points: Point[];
  targetSide: LoopFeedbackLayout['targetSide'];
  label: ReturnType<typeof routeLabelAnchor>;
}

/** Presentation-only return lanes; never insert these paths into Graph.edges. */
export function layoutLoopFeedback(bindings: LoopFeedback[], nodes: FeedbackNode[],
  layouts?: Record<string, LoopFeedbackLayout>) {
  const visible = nodes.filter(node => node.measured?.width && node.measured?.height);
  if (!bindings.length || !visible.length) return { routes: [] as FeedbackRoute[] };
  const left = Math.min(...visible.map(n => n.position.x));
  const top = Math.min(...visible.map(n => n.position.y));
  const right = Math.max(...visible.map(n => n.position.x + n.measured!.width!));
  const bottom = Math.max(...visible.map(n => n.position.y + n.measured!.height!));
  const byId = new Map(visible.map(n => [n.id, n]));
  const score = (points: Point[]) => points.slice(1).reduce((total, b, i) => {
    const a = points[i];
    const hits = visible.filter(n => {
      const x = n.position.x, y = n.position.y;
      const r = x + n.measured!.width!, d = y + n.measured!.height!;
      return a.y === b.y
        ? a.y > y && a.y < d && Math.max(a.x, b.x) > x && Math.min(a.x, b.x) < r
        : a.x > x && a.x < r && Math.max(a.y, b.y) > y && Math.min(a.y, b.y) < d;
    }).length;
    return total + hits * 1e6 + Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  }, 0);
  const routes = bindings.flatMap<FeedbackRoute>((binding, index) => {
    const source = byId.get(binding.source), target = byId.get(binding.target);
    if (!source || !target) return [];
    const margin = 48 + index * 36;
    const start = { x: source.position.x + source.measured!.width!, y: source.position.y + source.measured!.height! / 2 };
    const end = { x: target.position.x, y: target.position.y + target.measured!.height! / 2 };
    const candidates = [bottom + margin, top - margin].map(y => [start,
      { x: right + margin, y: start.y }, { x: right + margin, y },
      { x: left - margin, y }, { x: left - margin, y: end.y }, end]);
    // Local state inputs can sit near the outputs. A short right-hand return
    // avoids drawing their feedback across the entire body and unrelated nodes.
    candidates.push([start, { x: right + margin, y: start.y },
      { x: right + margin, y: end.y }, { x: end.x + target.measured!.width!, y: end.y }]);
    const automatic = simplifyRoute(candidates.sort((a, b) => score(a) - score(b))[0]);
    const stored = layouts?.[binding.state];
    const targetSide = stored?.targetSide ?? (automatic.at(-1)!.x === end.x ? 'left' : 'right');
    const endpoint = { x: end.x + (targetSide === 'right' ? target.measured!.width! : 0), y: end.y };
    const points = stored ? simplifyRoute(manualRoute(start, endpoint, 'right', targetSide, stored.waypoints)) : automatic;
    const label = routeLabelAnchor(points, stored?.labelPosition);
    return [{ ...binding, points, label, targetSide }];
  });
  if (!routes.length) return { routes };
  const all = routes.flatMap(r => r.points);
  const x = Math.min(left, ...all.map(p => p.x)) - 16;
  const y = Math.min(top, ...all.map(p => p.y)) - 32;
  return { routes, bounds: {
    x, y, width: Math.max(right, ...all.map(p => p.x)) + 16 - x,
    height: Math.max(bottom, ...all.map(p => p.y)) + 16 - y,
  } };
}
