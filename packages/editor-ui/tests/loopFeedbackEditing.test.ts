import { describe, expect, it } from 'vitest';
import type { LoopFeedbackLayout } from '@tensor-trail/model-core/core/types';
import { buildModel } from '@tensor-trail/model-core/agent';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject, validateProjectObject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules';
import { decoderLoopRecipe } from '../../model-core/tests/fixtures/decoderLoop';
import { useEditor } from '../src/editor/store';
import { layoutLoopFeedback } from '../src/views/loopFeedback';
import { manualRoute, moveRouteSegment, routeLabelAnchor, simplifyRoute } from '../src/views/edgeRouting';

const layout: LoopFeedbackLayout = { targetSide: 'left', waypoints: [
  { x: 1400, y: 100 }, { x: 1400, y: 900 }, { x: -100, y: 900 }, { x: -100, y: 100 },
], labelPosition: 0.6 };
function setup() {
  const p = buildModel(decoderLoopRecipe()).project!;
  useEditor.getState().load(p);
  useEditor.setState({ past: [], future: [] });
  useEditor.getState().navigate(['loop']);
  return useEditor.getState().project;
}
const owner = () => useEditor.getState().project.graph.nodes.find(n => n.id === 'loop')!;

describe('editable loop feedback layouts', () => {
  it('saves only presentation metadata and supports undo, redo, and reopening', () => {
    const before = setup();
    const result = inferProject(before, registry), nodes = owner().subgraph;
    expect(useEditor.getState().setLoopFeedbackLayout('query_points', layout)).toBe(true);
    const after = useEditor.getState().project;
    expect(owner().loopFeedbackRoutes).toEqual({ query_points: layout });
    expect(owner().subgraph).toEqual(nodes);
    expect(owner().customized).toBe(before.graph.nodes.find(n => n.id === 'loop')!.customized);
    expect(inferProject(after, registry)).toEqual(result);
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
    layout.waypoints[0].x += 1;
    expect(owner().loopFeedbackRoutes!.query_points.waypoints[0].x).not.toBe(layout.waypoints[0].x);
    layout.waypoints[0].x -= 1;
  });

  it('shares layout across iterations and resets one line without deleting state or other paths', () => {
    setup();
    useEditor.getState().setLoopFeedbackLayout('query_points', layout);
    useEditor.getState().setLoopFeedbackLayout('query_feat', layout);
    const before = useEditor.getState().project;
    useEditor.getState().selectLoopIteration(['loop'], 5);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().setLoopFeedbackLayout('query_points')).toBe(true);
    expect(owner().loopFeedbackRoutes).toEqual({ query_feat: layout });
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().navigate(['loop']);
    useEditor.getState().setLoopFeedbackLayout('query_points');
    useEditor.getState().setLoopFeedbackLayout('query_feat');
    expect(owner().loopFeedbackRoutes).toBeUndefined();
    expect(owner().params.states).toContain('query_points');
  });

  it('rejects unknown state, fixed inputs and edits outside a loop body', () => {
    const before = setup();
    for (const state of ['unknown', 'index', 'mlvl_feats']) {
      expect(useEditor.getState().setLoopFeedbackLayout(state, layout)).toBe(false);
      expect(useEditor.getState().project).toBe(before);
    }
    useEditor.getState().navigate([]);
    expect(useEditor.getState().setLoopFeedbackLayout('query_points', layout)).toBe(false);
    expect(useEditor.getState().project).toBe(before);
  });

  it.each([
    { ...layout, targetSide: 'top' },
    { ...layout, waypoints: [{ x: Infinity, y: 0 }] },
    { ...layout, waypoints: [{ x: 1, y: NaN }] },
    { ...layout, waypoints: [{ x: '1', y: 0 }] },
    { ...layout, waypoints: Array.from({ length: 101 }, () => ({ x: 0, y: 0 })) },
    { ...layout, labelPosition: -0.1 },
    { ...layout, labelPosition: 1.1 },
    { ...layout, labelPosition: null },
  ])('rejects invalid saved geometry without replacing the project (%j)', invalid => {
    const before = setup();
    expect(useEditor.getState().setLoopFeedbackLayout('query_points', invalid as LoopFeedbackLayout)).toBe(false);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
  });

  it('validates files with excessive routes or metadata attached to a non-loop', () => {
    const p = structuredClone(setup());
    p.graph.nodes[0].loopFeedbackRoutes = { x: layout };
    expect(() => validateProjectObject(p, registry)).toThrow('循环回传布局');
    const q = structuredClone(setup());
    q.graph.nodes.find(n => n.id === 'loop')!.loopFeedbackRoutes = Object.fromEntries(
      Array.from({ length: 9 }, (_, i) => [`v${i}`, layout]));
    expect(() => parseProject(JSON.stringify(q), registry)).toThrow('最多 8 条');
  });

  it('uses saved waypoints and label positions, with endpoints following moved nodes', () => {
    const binding = [{ state: 'x', source: 'out', target: 'in' }];
    const nodes = [
      { id: 'out', position: { x: 900, y: 50 }, measured: { width: 200, height: 100 } },
      { id: 'in', position: { x: 0, y: 50 }, measured: { width: 200, height: 100 } },
    ];
    const saved = structuredClone(layout), result = layoutLoopFeedback(binding, nodes, { x: saved });
    const route = result.routes[0];
    expect(route.points).toEqual(simplifyRoute(manualRoute({ x: 1100, y: 100 }, { x: 0, y: 100 },
      'right', 'left', saved.waypoints)));
    expect(route.label).toEqual(routeLabelAnchor(route.points, 0.6));
    nodes[1].position = { x: -300, y: 120 };
    const updated = layoutLoopFeedback(binding, nodes, { x: saved }).routes[0];
    expect(updated.points.at(-1)).toEqual({ x: -300, y: 170 });
    expect(saved).toEqual(layout);
    for (const p of route.points) {
      expect(p.x).toBeGreaterThanOrEqual(result.bounds!.x);
      expect(p.y).toBeLessThanOrEqual(result.bounds!.y + result.bounds!.height);
    }
  });

  it('pins the side of a short right-hand return and allows dragging terminal segments', () => {
    const result = layoutLoopFeedback([{ state: 'x', source: 'out', target: 'in' }], [
      { id: 'out', position: { x: 900, y: 300 }, measured: { width: 200, height: 100 } },
      { id: 'in', position: { x: 700, y: 0 }, measured: { width: 200, height: 100 } },
    ], { x: { targetSide: 'right', waypoints: [{ x: 1200, y: 350 }, { x: 1200, y: 50 }] } });
    const route = result.routes[0];
    expect(route.points.at(-1)).toEqual({ x: 900, y: 50 });
    const pins = moveRouteSegment(route.points, 0, { x: 0, y: 60 }, 'right', route.targetSide);
    const moved = simplifyRoute(manualRoute(route.points[0], route.points.at(-1)!, 'right', route.targetSide, pins));
    expect(moved[0]).toEqual(route.points[0]);
    expect(moved.at(-1)).toEqual(route.points.at(-1));
    expect(moved).not.toEqual(route.points);
  });
});
