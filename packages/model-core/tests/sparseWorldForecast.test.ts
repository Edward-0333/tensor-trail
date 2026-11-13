import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import type { Recipe, RecipeNode } from '../src/agent/types';
import { inferProject } from '../src/core/engine';
import type { DictionarySpec, ModelNode, TensorSpec, TensorSequenceSpec } from '../src/core/types';
import { registry } from '../src/modules';

const states = [
  'curr_query_feat',
  'curr_query_pos',
  'curr_query_timestamp',
  'forecast_points_list',
  'forecast_semantics_list',
  'pred_trajs_list',
  'forecast_points_mask_list',
];
const asRecipe = (node: ModelNode): RecipeNode => ({
  id: node.id,
  type: node.type,
  label: node.label,
  params: node.params,
  position: node.position,
  ...(node.subgraph
    ? {
        subgraph: {
          nodes: node.subgraph.nodes.map(asRecipe),
          connections: node.subgraph.edges.map((e) => [
            `${e.source}.${e.sourcePort}`,
            `${e.target}.${e.targetPort}`,
          ]),
        },
      }
    : {}),
});
function model(iterations = 1, training = true): Recipe {
  const step = registry.create('SparseWorldForecastStep', {}, 'loop'),
    graph = step.subgraph!;
  graph.nodes.find((n) => n.id === 'interval')!.params.port = 'index';
  const nodes: RecipeNode[] = [];
  const connections: Recipe['connections'] = [];
  const inputs: Record<string, number[]> = {
    ego_feat: [2, 1, 256],
    curr_query_feat: [2, 720, 256],
    curr_query_pos: [2, 720, 48, 3],
    curr_query_timestamp: [2, 720, 48, 1],
    query_feat: [2, 1040, 256],
    query_pos: [2, 1040, 48, 3],
    ind_stamps_all: [1040],
    temporal_trajs: [2, 6, 2],
  };
  for (const [id, shape] of Object.entries(inputs)) {
    nodes.push({
      id,
      type: 'Input',
      params: { shape: shape.join(','), axes: '', dtype: 'float32' },
    });
    connections.push([`${id}.out`, `loop.${id}`]);
  }
  nodes.push(
    { id: 'training', type: 'BooleanParameter', params: { value: training } },
    { id: 'num_fu_query', type: 'IntegerList', params: { values: '60,60,60,60,40,40' } },
    { id: 'img_metas', type: 'ForwardInput' },
  );
  for (const id of ['training', 'num_fu_query', 'img_metas'])
    connections.push([`${id}.out`, `loop.${id}`]);
  for (const id of states.filter((id) => id.endsWith('list'))) {
    nodes.push({ id, type: 'TensorSequencePack', params: { kind: 'list', input_count: 0 } });
    connections.push([`${id}.out`, `loop.${id}`]);
  }
  const body = asRecipe(step).subgraph;
  nodes.push({
    id: 'loop',
    type: 'ForLoop',
    params: { states: states.join(','), iterations },
    subgraph: body,
  });
  states.forEach((id) => {
    nodes.push({ id: `out_${id}`, type: 'Output' });
    connections.push([`loop.${id}`, `out_${id}.in`]);
  });
  return { name: 'forecast recurrent shape verification', nodes, connections };
}

describe('SparseWorld forecast detail', () => {
  it.each([1, 2, 3, 4, 5, 6])(
    'carries seven states for %i iterations without stacking unequal query counts',
    (count) => {
      const built = buildModel(model(count));
      expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
      const result = inferProject(built.project!, registry).nodes.loop;
      const sizes = [780, 840, 900, 960, 1000, 1040].slice(0, count);
      expect(
        (result.outputs.forecast_points_list as TensorSequenceSpec).items.map((t) => t.shape),
      ).toEqual(sizes.map((q) => [2, q, 48, 3]));
      expect(
        (result.outputs.forecast_semantics_list as TensorSequenceSpec).items.map((t) => t.shape),
      ).toEqual(sizes.map((q) => [2, q, 48, 17]));
      expect(
        (result.outputs.forecast_points_mask_list as TensorSequenceSpec).items.map((t) => t.shape),
      ).toEqual(sizes.map((q) => [2, q, 48]));
      expect(
        (result.outputs.pred_trajs_list as TensorSequenceSpec).items.map((t) => t.shape),
      ).toEqual(sizes.map(() => [2, 1, 2]));
      expect(result.outputs.curr_query_timestamp).toMatchObject({
        shape: [2, sizes.at(-1), 48, 1],
      });
      expect(validateModel(built.project).ok).toBe(true);
    },
  );
  it('keeps eval masks empty while the conditional training region remains only a static display', () => {
    const built = buildModel(model(6, false));
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const result = inferProject(built.project!, registry).nodes.loop;
    expect(result.outputs.forecast_points_mask_list).toMatchObject({ kind: 'list', items: [] });
  });
  it('reflects coordinate transforms, boolean promotion and the first-step tensor sizes', () => {
    const built = buildModel(model());
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const body = inferProject(built.project!, registry).nodes.loop.child!;
    const shapes: Record<string, number[]> = {
      expanded_feat: [2, 780, 256],
      expanded_pos: [2, 780, 48, 3],
      position_input: [2, 780, 192],
      updated_feat: [2, 780, 256],
      reg_offset: [2, 780, 48, 3],
      cls_score: [2, 780, 48, 17],
      vel_offset: [2, 780, 48, 2],
      pred_labels: [2, 780, 48],
      pred_moving_mask: [2, 780, 48, 1],
      moving_velocity: [2, 780, 48, 2],
      reg_offset_flat: [2, 780, 144],
      gt_traj: [2, 1, 2],
      pred_traj_expand: [2, 1, 3],
      flatten_points: [2, 37440, 3],
      gt_points: [2, 780, 48, 3],
      points_mask: [2, 780, 48],
    };
    for (const [id, shape] of Object.entries(shapes))
      expect(body.nodes[id].outputs.out, id).toMatchObject({ shape });
    expect((body.nodes.moving_velocity.outputs.out as TensorSpec).dtype).toBe('float32');
    expect(body.nodes.pred_moving_mask.outputs.out).toMatchObject({ dtype: 'bool' });
    expect(body.nodes.refine_points.child!.nodes.center.outputs.out).toMatchObject({
      shape: [2, 780, 1, 3],
    });
    expect(body.nodes.trans_points.child!.nodes.forward_translation_axis.outputs.out).toMatchObject(
      { shape: [2, 1, 3] },
    );
    expect(body.nodes.trans_points.child!.nodes.rotate_forward.outputs.out).toMatchObject({
      shape: [2, 37440, 3],
    });
  });
  it('does not inherit prediction values through zeros_like, and rejects incomplete single-loop previews', () => {
    const built = buildModel(model());
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    built.project!.lineageEnabled = true;
    const body = inferProject(built.project!, registry).nodes.loop.child!;
    expect(
      (body.nodes.pred_traj_expand.outputs.out as TensorSpec).lineage?.sources.map((s) => s.id),
    ).toEqual(['temporal_trajs']);
    const guard = registry.get('SingleIterationPreview')!.inferValue!;
    expect(guard({ iterations: { kind: 'int', value: 1 } }, {}).outputs.out).toEqual({
      kind: 'int',
      value: 0,
    });
    expect(() => guard({ iterations: { kind: 'int', value: 6 } }, {})).toThrow(/首轮/);
  });
  it.each([
    [true, 1, 1],
    [false, 1, 2],
    [false, 5, 6],
    [false, 6, 6],
  ] as const)(
    'finishes pretrain=%s, count=%i with %i trajectories',
    (pretrain, count, trajectories) => {
      const r = model(count);
      r.nodes.push(
        { id: 'finish', type: 'SparseWorldForecastFinish' },
        { id: 'pretrain', type: 'BooleanParameter', params: { value: pretrain } },
        { id: 'horizon', type: 'IntegerParameter', params: { expression: '6' } },
        {
          id: 'base_outputs',
          type: 'ForwardInput',
          params: {
            schema: {
              kind: 'dict',
              fields: {
                cls_score: {
                  shape: [2, 720, 48, 17],
                  axes: ['B', 'Q', 'P', 'class'],
                  dtype: 'float32',
                },
                refine_pts: {
                  shape: [2, 720, 48, 3],
                  axes: ['B', 'Q', 'P', 'xyz'],
                  dtype: 'float32',
                },
                outs: {
                  kind: 'dict',
                  fields: {
                    query_feat: { shape: [2, 1040, 256], axes: ['B', 'Q', 'C'], dtype: 'float32' },
                  },
                },
              },
            },
          },
        },
        { id: 'outputs', type: 'Output' },
      );
      r.connections.push(
        ['ego_feat.out', 'finish.ego_feat'],
        ['base_outputs.out', 'finish.base_outputs'],
        ['pretrain.out', 'finish.pretrain'],
        ['horizon.out', 'finish.num_fu_frames'],
        ['finish.outputs', 'outputs.in'],
      );
      for (const name of states.filter((s) => s !== 'curr_query_timestamp'))
        r.connections.push([`loop.${name}`, `finish.${name}`]);
      const built = buildModel(r);
      expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
      const graph = inferProject(built.project!, registry),
        fields = (graph.nodes.finish.outputs.outputs as DictionarySpec).fields;
      expect(Object.keys(fields)).toEqual([
        'cls_score',
        'refine_pts',
        'outs',
        'forecast_semantics_list',
        'forecast_points_list',
        'pred_trajs_list',
        'forecast_points_mask_list',
      ]);
      expect(fields.refine_pts).toMatchObject({ shape: [2, 720, 48, 3] });
      expect((fields.pred_trajs_list as TensorSequenceSpec).items).toHaveLength(trajectories);
      expect((fields.forecast_points_list as TensorSequenceSpec).items).toHaveLength(count);
      const child = graph.nodes.finish.child!;
      expect(child.nodes.zero_points.outputs.out).toMatchObject({
        shape: [2, 1, 3],
        autograd: { requiresGrad: false },
      });
      expect(child.nodes.extra_attention.inputs.key_points).toEqual(
        graph.nodes.loop.outputs.curr_query_pos,
      );
    },
  );
});
