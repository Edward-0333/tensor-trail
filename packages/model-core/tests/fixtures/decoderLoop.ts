import type { Recipe, RecipeGraph } from '../../src/agent/types';

export const decoderCounts = [1, 4, 16, 24, 32, 48];
export const decoderStates = ['query_points', 'query_feat', 'cls_scores', 'refine_pts'];

export function decoderBody(): RecipeGraph {
  const captures = ['mlvl_feats', 'occ2img', 'img_metas'];
  return {
    nodes: [
      ...['index', ...decoderStates, ...captures].map(id => ({ id, type: 'Inlet', params: { port: id } })),
      { id: 'detach', type: 'Detach' },
      { id: 'layer', type: 'OPUSDecoderLayerAt' },
      { id: 'append_cls', type: 'TensorListAppend' },
      { id: 'append_pts', type: 'TensorListAppend' },
      ...decoderStates.map(id => ({ id: `next_${id}`, type: 'Outlet', params: { port: id } })),
    ],
    connections: [
      ['query_points.out', 'detach.in'], ['detach.out', 'layer.query_points'],
      ...['query_feat', 'index', ...captures].map((id): [string, string] => [`${id}.out`, `layer.${id}`]),
      ['cls_scores.out', 'append_cls.items'], ['layer.cls_score', 'append_cls.item'],
      ['refine_pts.out', 'append_pts.items'], ['layer.refine_pts', 'append_pts.item'],
      ['layer.refine_pts', 'next_query_points.in'], ['layer.query_feat', 'next_query_feat.in'],
      ['append_cls.out', 'next_cls_scores.in'], ['append_pts.out', 'next_refine_pts.in'],
    ],
  };
}

export function decoderLoopRecipe(iterations = 6): Recipe {
  return {
    name: 'Stateful decoder loop',
    nodes: [
      { id: 'points', type: 'Input', params: { shape: '2,1040,1,3', axes: '' } },
      { id: 'feat', type: 'Input', params: { shape: '2,1040,256', axes: '' } },
      { id: 'occ', type: 'Input', params: { shape: '2,30,4,4', axes: '' } },
      { id: 'metas', type: 'ForwardInput', params: { text: 'list[dict]' } },
      ...[64, 32, 16, 8].map((h, i) => ({ id: `f${i}`, type: 'Input', params: { shape: `40,6,${h},${176 / 2 ** i},64`, axes: '' } })),
      { id: 'features', type: 'TensorSequencePack' },
      { id: 'cls', type: 'TensorSequencePack', params: { kind: 'list', input_count: 0 } },
      { id: 'pts', type: 'TensorSequencePack', params: { kind: 'list', input_count: 0 } },
      { id: 'loop', type: 'ForLoop', params: { iterations, states: decoderStates.join(',') }, subgraph: decoderBody() },
      ...decoderStates.map(id => ({ id: `out_${id}`, type: 'Output' })),
    ],
    connections: [
      ['points.out', 'loop.query_points'], ['feat.out', 'loop.query_feat'],
      ['occ.out', 'loop.occ2img'], ['metas.out', 'loop.img_metas'],
      ['cls.out', 'loop.cls_scores'], ['pts.out', 'loop.refine_pts'], ['features.out', 'loop.mlvl_feats'],
      ...[0, 1, 2, 3].map((i): [string, string] => [`f${i}.out`, `features.item${i}`]),
      ...decoderStates.map((id): [string, string] => [`loop.${id}`, `out_${id}.in`]),
    ],
  };
}
