import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { registry } from '@tensor-trail/model-core/modules';
import { decoderLoopRecipe, decoderStates } from '../../model-core/tests/fixtures/decoderLoop';
import { layoutLoopFeedback, loopFeedback } from '../src/views/loopFeedback';
import { LoopFeedbackOverlay } from '../src/views/LoopFeedbackOverlay';

function fixture() {
  const project = buildModel(decoderLoopRecipe()).project!;
  const owner = project.graph.nodes.find(n => n.id === 'loop')!;
  return { project, owner, graph: owner.subgraph! };
}

describe('loop state feedback presentation', () => {
  it('links only declared states, from Outlet to Inlet, not captures or index', () => {
    const { graph, owner } = fixture();
    const bindings = loopFeedback(graph, owner);
    expect(bindings).toEqual(decoderStates.map(state => ({ state, source: `next_${state}`, target: state })));
    expect(bindings.some(b => ['index', 'mlvl_feats', 'occ2img', 'img_metas'].includes(b.state))).toBe(false);
    graph.nodes.find(n => n.id === 'query_feat')!.label = '任意显示名称';
    expect(loopFeedback(graph, owner)).toEqual(bindings);
  });

  it('does not invent recurrence for root, normal modules, or independent loops', () => {
    const { graph, owner } = fixture();
    expect(loopFeedback(graph)).toEqual([]);
    for (const type of ['Submodule', 'ForEachTensor', 'OPUSDecoderLayer', 'WhileLoop'])
      expect(loopFeedback(graph, { ...owner, type })).toEqual([]);
    expect(loopFeedback(graph, { ...owner, params: { states: 'a,a' } })).toEqual([]);
  });

  it('skips missing/ambiguous interfaces and does not treat linked copies as boundaries', () => {
    const { graph, owner } = fixture();
    const input = graph.nodes.find(n => n.id === 'query_feat')!;
    graph.nodes.push({ ...input, id: 'alias', valueSource: input.id });
    expect(loopFeedback(graph, owner)).toHaveLength(4);
    graph.nodes.push({ ...input, id: 'duplicate' });
    expect(loopFeedback(graph, owner).map(b => b.state)).not.toContain('query_feat');
    graph.nodes = graph.nodes.filter(n => n.id !== 'next_query_points');
    expect(loopFeedback(graph, owner).map(b => b.state)).toEqual(['cls_scores', 'refine_pts']);
  });

  it('shows recurrence even when state shapes and list lengths change, without adding graph cycles', () => {
    const { project, graph, owner } = fixture();
    const before = structuredClone(project), result = inferProject(project, registry).nodes.loop;
    expect(result.inputs.query_points).toMatchObject({ shape: [2, 1040, 1, 3] });
    expect(result.outputs.query_points).toMatchObject({ shape: [2, 1040, 48, 3] });
    expect(loopFeedback(graph, owner)).toHaveLength(4);
    expect(project).toEqual(before);
    expect(inferProject(project, registry).nodes.loop).toEqual(result);
    expect(loopFeedback(graph, { ...owner, params: { ...owner.params, iterations: 0 } })).toHaveLength(4);
  });

  it('routes outside the drawing with separate lanes and fits the entire feedback extent', () => {
    const bindings = [{ state: 'value', source: 'out', target: 'in' },
      { state: 'other', source: 'out2', target: 'in2' }];
    const nodes = [
      { id: 'in', position: { x: -100, y: 0 }, measured: { width: 200, height: 100 } },
      { id: 'out', position: { x: 900, y: 0 }, measured: { width: 200, height: 100 } },
      { id: 'in2', position: { x: 100, y: 300 }, measured: { width: 200, height: 150 } },
      { id: 'out2', position: { x: 900, y: 300 }, measured: { width: 200, height: 150 } },
    ];
    const before = structuredClone(nodes), { routes, bounds } = layoutLoopFeedback(bindings, nodes);
    expect(routes[0].points[0]).toEqual({ x: 1100, y: 50 });
    expect(routes[0].points.at(-1)).toEqual({ x: -100, y: 50 });
    expect(routes[0].label.y < 0 || routes[0].label.y > 450).toBe(true);
    expect(routes[0].label.y).not.toBe(routes[1].label.y);
    for (const { points } of routes) for (const p of points) {
      expect(p.x).toBeGreaterThanOrEqual(bounds!.x);
      expect(p.y).toBeGreaterThanOrEqual(bounds!.y);
      expect(p.x).toBeLessThanOrEqual(bounds!.x + bounds!.width);
      expect(p.y).toBeLessThanOrEqual(bounds!.y + bounds!.height);
    }
    expect(nodes).toEqual(before);
    nodes[0].position.x -= 200;
    expect(layoutLoopFeedback(bindings, nodes).routes[0].points.at(-1)!.x).toBe(-300);
  });

  it('prefers a short return to a nearby state without crossing unrelated nodes', () => {
    const { routes } = layoutLoopFeedback([{ state: 'list', source: 'out', target: 'in' }], [
      { id: 'in', position: { x: 700, y: 0 }, measured: { width: 150, height: 100 } },
      { id: 'out', position: { x: 1000, y: 200 }, measured: { width: 200, height: 100 } },
      { id: 'unrelated', position: { x: 300, y: 0 }, measured: { width: 250, height: 350 } },
    ]);
    expect(routes[0].points.at(-1)).toEqual({ x: 850, y: 50 });
    expect(routes[0].points.every(p => p.x >= 850)).toBe(true);
  });

  it('waits for measured nodes and safely tolerates incomplete graphs', () => {
    expect(layoutLoopFeedback([], []).routes).toEqual([]);
    expect(layoutLoopFeedback([{ state: 'x', source: 'out', target: 'in' }], [
      { id: 'in', position: { x: 0, y: 0 } },
    ]).routes).toEqual([]);
  });

  it('renders dashed directional, labelled cues, not interactive graph edges', () => {
    const callbacks = { onSelect: () => {}, onChange: () => {} };
    const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(LoopFeedbackOverlay, { ...callbacks, routes: [{
      state: 'query_points', source: 'next_query_points', target: 'query_points',
      targetSide: 'left', points: [{ x: 900, y: 100 }, { x: 100, y: 100 }],
      label: { x: 500, y: 100, horizontal: true, position: 0.5 },
    }] })));
    expect(html).toContain('stroke-dasharray="7 6"');
    expect(html).toContain('marker-end="url(#loop-feedback-');
    expect(html).toContain('query_points → 下一轮');
    expect(html).toContain('值和 shape 可改变');
    expect(html).not.toContain('react-flow__edge');
    expect(html).toContain('loop-feedback-hit');
    expect(renderToStaticMarkup(createElement(LoopFeedbackOverlay, { ...callbacks, routes: [] }))).toBe('');
  });
});
