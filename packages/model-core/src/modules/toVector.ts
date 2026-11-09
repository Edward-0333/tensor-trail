import type { Graph, ModelNode, ModuleDefinition, Params } from '../core/types';
import { tensorValue } from '../core/types';
import { normalizeTensorSelection } from '../core/indexingTypes';
import { requireThat, same } from './math';

function buildToVector(): Graph {
  const graph: Graph = { nodes: [], edges: [] };
  const node = (
    id: string,
    type: string,
    label: string,
    x: number,
    y: number,
    params: Params = {},
  ): ModelNode => {
    const normalized = normalizeTensorSelection(type, params);
    const result: ModelNode = {
      id,
      ...normalized,
      label,
      definitionVersion: 1,
      position: { x, y },
    };
    graph.nodes.push(result);
    return result;
  };
  const edge = (source: string, target: string, targetPort = 'in') => {
    graph.edges.push({
      id: `${source}_${target}_${targetPort}`,
      source,
      sourcePort: 'out',
      target,
      targetPort,
    });
  };
  const slice = (id: string, label: string, dim: string, start: string, stop: string, y: number) =>
    node(id, 'TensorIndex', label, 200, y, { mode: 'slice', dim, start, stop, step: 1 });

  node('feat', 'Inlet', 'feat', 0, 260, { port: 'feat' });
  node('valid_mask', 'Inlet', 'valid_mask', 0, 40, { port: 'valid_mask' });
  slice('mask_previous', 'valid_mask[..., :-1]', '-1', '', '-1', 0);
  slice('mask_next', 'valid_mask[..., 1:]', '-1', '1', '', 100);
  node('mask_pair', 'BitwiseAnd', '相邻帧都有效', 390, 40);
  node('mask_aligned', 'AlignTrailingDims', '按 feat 维数补齐掩码', 590, 40);
  slice('feat_next', 'feat[:, :, 1:, ...]', '2', '1', '', 230);
  slice('feat_previous', 'feat[:, :, :-1, ...]', '2', '', '-1', 350);
  node('difference', 'Subtract', '后一帧 − 前一帧', 450, 230);
  node('zeros', 'ZerosLike', '无效位置填零', 590, 350);
  node('where', 'Where', '有效时取差分，否则为零', 800, 210);
  node('out', 'Outlet', 'vec', 1010, 210, { port: 'out' });

  edge('valid_mask', 'mask_previous');
  edge('valid_mask', 'mask_next');
  edge('mask_previous', 'mask_pair', 'a');
  edge('mask_next', 'mask_pair', 'b');
  edge('mask_pair', 'mask_aligned');
  edge('feat', 'mask_aligned', 'like');
  edge('feat', 'feat_next');
  edge('feat', 'feat_previous');
  edge('feat_next', 'difference', 'a');
  edge('feat_previous', 'difference', 'b');
  edge('feat_next', 'zeros');
  edge('mask_aligned', 'where', 'condition');
  edge('difference', 'where', 'x');
  edge('zeros', 'where', 'y');
  edge('where', 'out');
  // Keep the rank-reference and zero-value branches away from the difference card.
  const rankReference = graph.edges.find((item) => item.id === 'feat_mask_aligned_like')!;
  rankReference.waypoints = [
    { x: 160, y: 282 },
    { x: 160, y: 150 },
    { x: 560, y: 150 },
    { x: 560, y: 68 },
  ];
  rankReference.labelPosition = 0.58;
  const zeroReference = graph.edges.find((item) => item.id === 'feat_next_zeros_in')!;
  zeroReference.waypoints = [
    { x: 380, y: 247 },
    { x: 380, y: 420 },
    { x: 560, y: 420 },
    { x: 560, y: 367 },
  ];
  zeroReference.labelPosition = 0.68;
  return graph;
}

export const toVector: ModuleDefinition = {
  type: 'ToVector',
  label: 'to_vector',
  category: '组合计算',
  description:
    '接收 feat [B,A,T,...] 与布尔 valid_mask [B,A,T]，计算相邻帧差分；两帧均有效时保留差分，否则置零。时间轴变为 max(T−1,0)，不包含可训练参数；双击查看内部函数。',
  fields: [],
  inputs: [
    { id: 'feat', label: 'feat' },
    { id: 'valid_mask', label: 'valid_mask' },
  ],
  outputs: [{ id: 'out', label: 'vec' }],
  composite: true,
  validate: (inputs) => {
    const feat = tensorValue(inputs.feat),
      mask = tensorValue(inputs.valid_mask);
    requireThat(
      ['float16', 'bfloat16', 'float32', 'float64'].includes(feat.dtype),
      'to_vector 的 feat 需要浮点 Tensor。',
    );
    requireThat(feat.shape.length >= 3, 'to_vector 的 feat 至少需要 [B,A,T] 三个轴。');
    requireThat(mask.dtype === 'bool', 'to_vector 的 valid_mask 必须是 bool Tensor。');
    requireThat(mask.shape.length === 3, 'to_vector 的 valid_mask 需要 [B,A,T] 三个轴。');
    const warnings: string[] = [];
    for (let axis = 0; axis < 3; axis++)
      same(
        feat.shape[axis],
        mask.shape[axis],
        `to_vector 的 feat 与 valid_mask 第 ${axis} 轴必须相等`,
        warnings,
      );
    return warnings;
  },
  build: buildToVector,
};
