import type { Graph, ModelNode, Params } from '../core/types';
import type { ModuleRegistry } from './registry';
import { num, text } from './primitives';

function builder(registry: ModuleRegistry) {
  const graph: Graph = { nodes: [], edges: [] };
  const node = (id: string, type: string, label: string, x: number, y: number, params: Params = {}, subgraph?: Graph) => {
    const n = registry.create(type, params, id);
    Object.assign(n, { label, position: { x, y } });
    if (subgraph) { n.subgraph = subgraph; n.customized = true; }
    graph.nodes.push(n);
  };
  const edge = (from: string, to: string) => {
    const [source, sourcePort = 'out'] = from.split('.'), [target, targetPort = 'in'] = to.split('.');
    graph.edges.push({ id: `e${graph.edges.length}`, source, sourcePort, target, targetPort });
  };
  const inlet = (id: string, x: number, y: number, label = id) => node(id, 'Inlet', label, x, y, { port: id });
  const outlet = (id: string, port: string, x: number, y: number, label = port) => node(id, 'Outlet', label, x, y, { port });
  return { graph, node, edge, inlet, outlet };
}

function distances(registry: ModuleRegistry, pcRange: string) {
  const g = builder(registry);
  g.inlet('points1', 40, 80); g.inlet('points2', 40, 400);
  for (const [i, y] of [[1, 80], [2, 400]]) {
    g.node(`decode${i}`, 'PointRangeTransform', `decode_points(points${i})`, 330, y, { mode: 'decode', pc_range: pcRange });
    g.edge(`points${i}.out`, `decode${i}.in`);
    g.node(`metric${i}`, 'TensorVariable', `points${i}（米制）`, 590, y);
    g.edge(`decode${i}.out`, `metric${i}.in`);
  }
  g.node('mean2', 'ReduceMean', 'points2.mean(dim=2)', 900, 400, { dim: 2, keepdim: false });
  g.edge('metric2.out', 'mean2.in');
  g.node('key_centers', 'TensorVariable', 'points2（每个 Key 的点集均值）', 1160, 400);
  g.edge('mean2.out', 'key_centers.in');
  g.node('expand_query', 'Unsqueeze', 'points1.unsqueeze(-2)', 1480, 80, { dim: -2 });
  g.node('expand_key', 'Unsqueeze', 'points2.unsqueeze(-3)', 1480, 400, { dim: -3 });
  g.edge('metric1.out', 'expand_query.in'); g.edge('key_centers.out', 'expand_key.in');
  g.node('delta', 'Subtract', '逐对坐标差（广播）', 1780, 220);
  g.edge('expand_query.out', 'delta.a'); g.edge('expand_key.out', 'delta.b');
  g.node('distance', 'L2Norm', 'torch.norm(..., dim=-1)', 2050, 220, { dim: -1, keepdim: false });
  g.edge('delta.out', 'distance.in');
  g.node('negative_distance', 'Function', '负距离 -dist', 2320, 220, { expression: '-x' });
  g.edge('distance.out', 'negative_distance.in');
  g.node('no_grad_result', 'NoGradResult', '@torch.no_grad() 的返回结果', 2590, 220);
  g.edge('negative_distance.out', 'no_grad_result.in');
  g.outlet('return_dist', 'dist', 2870, 220, 'dist = -||points1 - mean(points2)||');
  g.edge('no_grad_result.out', 'return_dist.in');
  return g.graph;
}

function attentionWrapper(registry: ModuleRegistry, D: number, H: number, drop: number) {
  const g = builder(registry);
  g.inlet('query', 40, 60, 'query（也作为 identity）');
  g.inlet('key', 40, 390, 'key（加位置前也作为 value）');
  g.inlet('query_pos', 40, 210); g.inlet('key_pos', 40, 540); g.inlet('attn_mask', 40, 880);
  g.node('add_query_pos', 'Add', 'query = query + query_pos', 430, 140);
  g.edge('query.out', 'add_query_pos.a'); g.edge('query_pos.out', 'add_query_pos.b');
  g.node('add_key_pos', 'Add', 'key = key + key_pos', 430, 460);
  g.edge('key.out', 'add_key_pos.a'); g.edge('key_pos.out', 'add_key_pos.b');
  for (const [id, source, y] of [
    ['query', 'add_query_pos', 140], ['key', 'add_key_pos', 460], ['value', 'key', 730],
  ] as const) {
    g.node(`transpose_${id}`, 'Permute', `${id}.transpose(0,1)`, 760, y, { dims: '1,0,2' });
    g.edge(`${source}.out`, `transpose_${id}.in`);
    g.node(`${id}_sequence_first`, 'TensorVariable', `${id}（sequence-first）`, 1050, y);
    g.edge(`transpose_${id}.out`, `${id}_sequence_first.in`);
  }
  g.node('torch_attention', 'TorchMultiheadAttention', `self.attn = nn.MultiheadAttention(${D}, ${H})`, 1450, 440,
    { embed_dims: D, num_heads: H, dropout: String(drop), batch_first: false, average_attn_weights: true });
  for (const id of ['query', 'key', 'value']) g.edge(`${id}_sequence_first.out`, `torch_attention.${id}`);
  g.edge('attn_mask.out', 'torch_attention.attn_mask');
  g.node('out_sequence_first', 'TensorVariable', 'out（sequence-first）', 1830, 440);
  g.edge('torch_attention.out', 'out_sequence_first.in');
  g.node('restore_batch_first', 'Permute', 'out.transpose(0,1)', 2130, 440, { dims: '1,0,2' });
  g.edge('out_sequence_first.out', 'restore_batch_first.in');
  g.node('proj_drop', 'Dropout', 'self.proj_drop(p=0)', 2410, 440, { p: '0.0' });
  g.node('dropout_layer', 'Dropout', 'self.dropout_layer(p=0)', 2690, 440, { p: '0.0' });
  g.edge('restore_batch_first.out', 'proj_drop.in'); g.edge('proj_drop.out', 'dropout_layer.in');
  g.node('residual_add', 'Add', 'identity + output（原始 query 残差）', 2980, 440);
  g.edge('dropout_layer.out', 'residual_add.a'); g.edge('query.out', 'residual_add.b');
  g.outlet('return_feat', 'out', 3320, 440, '融合后的 Query 特征'); g.edge('residual_add.out', 'return_feat.in');
  g.node('attention_weights', 'TensorVariable', 'weight（默认各头平均）', 1830, 900);
  g.edge('torch_attention.weights', 'attention_weights.in');
  g.outlet('return_weights', 'weights', 2200, 900); g.edge('attention_weights.out', 'return_weights.in');
  return g.graph;
}

export function opusCrossAttentionCode(node: ModelNode): string | undefined {
  if (node.type !== 'OPUSCrossAttention') return;
  return `# SparseWorld/mmdet3d/models/sparsedetectors/opus_transformer.py: OPUSCrossAttention\n# 本图直接展开 inner_forward，不重复绘制 checkpoint 与普通调用的数据流。\nif self.training and query_feat.requires_grad:\n    return cp(self.inner_forward, query_points, query_feat, key_points, key_feat, use_reentrant=False)\nelse:\n    return self.inner_forward(query_points, query_feat, key_points, key_feat)\n\n# inner_forward:\ndist = torch.exp(self.calc_points_dists(query_points, key_points) / 40)\ntau = F.softplus(self.gen_tau(key_feat)).permute(0, 2, 1)\nattn_mask = (dist[:, None, :, :] * tau[:, :, None, :]).flatten(0, 1)\nquery_pos = self.pos_embed(query_points * 0)\nkey_pos = self.pos_embed(decode_points(key_points, self.pc_range).mean(2))\nreturn self.attention(query_feat, key_feat, attn_mask=attn_mask, query_pos=query_pos, key_pos=key_pos)\n# pos_embed 两次调用共享同一实例。calc_points_dists 单独处于 no_grad，不是整个注意力 no_grad。\n# cp 是激活检查点重计算，不切断梯度；DUMP 文件保存不画入神经网络数据流。`;
}

export function installOPUSCrossAttention(registry: ModuleRegistry) {
  registry.register({
    type: 'OPUSCrossAttention', label: 'OPUSCrossAttention（可展开）', category: 'Transformer',
    description: '展开 SparseWorld 的距离偏置、Key tau、共享位置编码、标准注意力调用及残差；checkpoint 只影响执行/重算策略，见源码说明。',
    fields: [num('embed_dims', '特征维数', 256), num('num_heads', '头数', 8), text('dropout', '注意力 dropout', '0.1'),
      text('pc_range', '占据坐标范围', '-40,-40,-1,40,40,5.4')],
    inputs: ['query_points', 'query_feat', 'key_points', 'key_feat'].map(id => ({ id, label: id })),
    outputs: [{ id: 'fused_ego_feat', label: 'fused_ego_feat' }, { id: 'weights', label: 'weights（调用侧 _ 忽略）' }],
    composite: true,
    validate: (inputs, p) => registry.get('OPUSCrossAttentionContract')!.inferValue!(inputs, p).warnings ?? [],
    build: p => {
      const g = builder(registry), D = Number(p.embed_dims), H = Number(p.num_heads);
      const pc = String(p.pc_range);
      g.inlet('query_points', 40, 40); g.inlet('query_feat', 40, 220);
      g.inlet('key_points', 40, 400); g.inlet('key_feat', 40, 580);
      g.node('calc_points_dists', 'Submodule', 'calc_points_dists · @torch.no_grad()', 500, -80, {}, distances(registry, pc));
      g.edge('query_points.out', 'calc_points_dists.points1'); g.edge('key_points.out', 'calc_points_dists.points2');
      g.node('dist_negative', 'TensorVariable', 'dist（负欧氏距离）', 880, -50); g.edge('calc_points_dists.dist', 'dist_negative.in');
      g.node('divide_40', 'Function', 'dist / 40', 1180, -30, { expression: 'x / 40' }); g.edge('dist_negative.out', 'divide_40.in');
      g.node('exp_distance', 'Exp', 'torch.exp(dist / 40)', 1430, -30); g.edge('divide_40.out', 'exp_distance.in');
      g.node('distance_weight', 'TensorVariable', 'dist = exp(-距离/40)', 1650, -50); g.edge('exp_distance.out', 'distance_weight.in');
      g.node('dist_add_head_axis', 'Unsqueeze', 'dist[:, None, :, :]', 1980, -30, { dim: 1 }); g.edge('distance_weight.out', 'dist_add_head_axis.in');
      g.node('gen_tau', 'Linear', `self.gen_tau: Linear(${D}, ${H})`, 500, 430, { in_features: D, out_features: H }); g.edge('key_feat.out', 'gen_tau.in');
      g.node('tau_raw', 'TensorVariable', 'tau（每个 Key、每个 head）', 830, 450); g.edge('gen_tau.out', 'tau_raw.in');
      g.node('softplus_tau', 'Softplus', 'F.softplus(tau)', 1180, 460); g.edge('tau_raw.out', 'softplus_tau.in');
      g.node('permute_tau', 'Permute', 'tau.permute(0,2,1)', 1450, 460, { dims: '0,2,1' }); g.edge('softplus_tau.out', 'permute_tau.in');
      g.node('tau', 'TensorVariable', 'tau（B, heads, K）', 1690, 450); g.edge('permute_tau.out', 'tau.in');
      g.node('tau_add_query_axis', 'Unsqueeze', 'tau[:, :, None, :]', 1980, 470, { dim: 2 }); g.edge('tau.out', 'tau_add_query_axis.in');
      g.node('bias_multiply', 'Mul', 'dist * tau（广播）', 2260, 210);
      g.edge('dist_add_head_axis.out', 'bias_multiply.a'); g.edge('tau_add_query_axis.out', 'bias_multiply.b');
      g.node('bias_per_head', 'TensorVariable', 'attn_mask（B, heads, Q, K）', 2520, 180); g.edge('bias_multiply.out', 'bias_per_head.in');
      g.node('flatten_bias', 'Flatten', 'attn_mask.flatten(0,1)', 2870, 210, { start_dim: 0, end_dim: 1 }); g.edge('bias_per_head.out', 'flatten_bias.in');
      g.node('attn_mask', 'TensorVariable', 'attn_mask（浮点加性偏置）', 3150, 180); g.edge('flatten_bias.out', 'attn_mask.in');
      g.node('query_times_zero', 'Function', 'query_points * 0', 500, 820, { expression: 'x * 0' }); g.edge('query_points.out', 'query_times_zero.in');
      g.node('query_zero', 'TensorVariable', 'query_points * 0（非 new_zeros）', 770, 800); g.edge('query_times_zero.out', 'query_zero.in');
      g.node('query_pos_embed', 'Linear', 'self.pos_embed · Query（共享同一实例）', 1160, 790, { in_features: 3, out_features: D }); g.edge('query_zero.out', 'query_pos_embed.in');
      g.node('query_pos', 'TensorVariable', 'query_pos', 1530, 800); g.edge('query_pos_embed.out', 'query_pos.in');
      g.node('decode_key_points', 'PointRangeTransform', 'decode_points(key_points)', 500, 1180, { mode: 'decode', pc_range: pc }); g.edge('key_points.out', 'decode_key_points.in');
      g.node('key_points_metric', 'TensorVariable', 'key_points（米制）', 820, 1160); g.edge('decode_key_points.out', 'key_points_metric.in');
      g.node('mean_key_points', 'ReduceMean', '.mean(2)', 1160, 1180, { dim: 2, keepdim: false }); g.edge('key_points_metric.out', 'mean_key_points.in');
      g.node('key_points_mean', 'TensorVariable', '每个 Key 的平均位置', 1410, 1160); g.edge('mean_key_points.out', 'key_points_mean.in');
      g.node('key_pos_embed', 'Linear', 'self.pos_embed · Key（共享同一实例）', 1780, 1150, { in_features: 3, out_features: D }); g.edge('key_points_mean.out', 'key_pos_embed.in');
      g.node('key_pos', 'TensorVariable', 'key_pos', 2160, 1160); g.edge('key_pos_embed.out', 'key_pos.in');
      g.node('attention', 'Submodule', 'self.attention（位置编码 + MHA + 残差）', 3550, 780, {}, attentionWrapper(registry, D, H, Number(p.dropout)));
      for (const [from, port] of [['query_feat', 'query'], ['key_feat', 'key'], ['attn_mask', 'attn_mask'], ['query_pos', 'query_pos'], ['key_pos', 'key_pos']])
        g.edge(`${from}.out`, `attention.${port}`);
      g.outlet('fused_output', 'fused_ego_feat', 4100, 820, 'fused_ego_feat（第一返回值）'); g.edge('attention.out', 'fused_output.in');
      g.outlet('weights_output', 'weights', 4100, 1110, 'weights（第二返回值，调用方 _ 忽略）'); g.edge('attention.weights', 'weights_output.in');
      return g.graph;
    },
  });
}
