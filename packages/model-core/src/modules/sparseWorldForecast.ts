import type { Graph, Params } from '../core/types';
import type { ModuleRegistry } from './registry';
import { num, text } from './primitives';

function graphBuilder(registry: ModuleRegistry) {
  const graph: Graph = { nodes: [], edges: [] };
  const n = (
    id: string,
    type: string,
    label: string,
    x: number,
    y: number,
    params: Params = {},
    subgraph?: Graph,
  ) => {
    const node = registry.create(type, params, id);
    Object.assign(node, { label, position: { x, y } });
    if (subgraph) {
      node.subgraph = subgraph;
      node.customized = true;
    }
    graph.nodes.push(node);
    return id;
  };
  const e = (from: string, to: string) => {
    const [source, sourcePort = 'out'] = from.split('.'),
      [target, targetPort = 'in'] = to.split('.');
    graph.edges.push({ id: `e${graph.edges.length}`, source, sourcePort, target, targetPort });
  };
  const input = (id: string, y: number) => n(id, 'Inlet', id, 20, y, { port: id });
  const output = (source: string, port: string, x: number, y: number) => {
    n(`return_${port}`, 'Outlet', port, x, y, { port });
    e(source, `return_${port}.in`);
  };
  const variable = (
    id: string,
    source: string,
    label: string,
    x: number,
    y: number,
    type = 'TensorVariable',
  ) => {
    n(id, type, label, x, y);
    e(source, id);
    return id;
  };
  return { graph, n, e, input, output, variable };
}

function mlp(registry: ModuleRegistry, widths: number[], activation = 'ReLU', normalized = false) {
  const g = graphBuilder(registry);
  g.input('in', 80);
  let last = 'in',
    x = 320;
  widths.slice(1).forEach((width, i) => {
    const id = `linear${i}`;
    g.n(id, 'Linear', `Linear(${widths[i]}, ${width})`, x, 80, {
      in_features: widths[i],
      out_features: width,
    });
    g.e(last, id);
    last = id;
    x += 300;
    if (normalized) {
      g.n(`norm${i}`, 'LayerNorm', `LayerNorm(${width})`, x, 80, { features: width });
      g.e(last, `norm${i}`);
      last = `norm${i}`;
      x += 280;
    }
    if (normalized || i < widths.length - 2) {
      g.n(
        `act${i}`,
        activation,
        activation === 'ReLU' ? 'ReLU(inplace=True)' : 'Softplus()',
        x,
        80,
      );
      g.e(last, `act${i}`);
      last = `act${i}`;
      x += 240;
    }
  });
  g.output(last, 'out', x, 80);
  return g.graph;
}

function refineGraph(registry: ModuleRegistry, P: number, pc: string) {
  const g = graphBuilder(registry);
  g.input('points_proposal', 80);
  g.input('points_delta', 460);
  g.n('shape', 'TensorSize', 'B, Q = points_delta.shape[:2]', 330, 620, {
    axes: '0,1',
    names: 'B,Q',
  });
  g.e('points_delta', 'shape');
  g.n('reshape_delta', 'Reshape', 'points_delta.reshape(B,Q,P,3)', 720, 460, {
    shape: `d0,d1,${P},3`,
  });
  g.e('points_delta', 'reshape_delta');
  g.e('shape.dim0', 'reshape_delta.d0');
  g.e('shape.dim1', 'reshape_delta.d1');
  g.variable('delta', 'reshape_delta', 'points_delta（米制偏移）', 1050, 460);
  g.n('decode', 'PointRangeTransform', 'decode_points(points_proposal)', 330, 80, {
    mode: 'decode',
    pc_range: pc,
  });
  g.e('points_proposal', 'decode');
  g.n('mean', 'ReduceMean', 'mean(dim=2, keepdim=True)', 700, 80, { dim: 2, keepdim: true });
  g.e('decode', 'mean');
  g.variable('center', 'mean', '每个 Query 的点集中心（米）', 1050, 80);
  g.n('add_delta', 'Add', '中心 + 每个点的独立偏移（广播）', 1450, 260);
  g.e('center', 'add_delta.a');
  g.e('delta', 'add_delta.b');
  g.variable('new_points', 'add_delta', 'new_points（米制）', 1800, 260);
  g.n('encode', 'PointRangeTransform', 'encode_points（无 clamp）', 2160, 260, {
    mode: 'encode',
    pc_range: pc,
  });
  g.e('new_points', 'encode');
  g.output('encode', 'out', 2500, 260);
  return g.graph;
}

function transformGraph(registry: ModuleRegistry, pc: string) {
  const g = graphBuilder(registry);
  g.input('points_proposal', 80);
  g.input('points_delta', 420);
  g.input('trans_matrix', 900);
  g.n('to_cpu', 'TensorToDevice', 'trans_matrix.cpu()', 340, 1280, { target: 'cpu' });
  g.e('trans_matrix', 'to_cpu');
  g.n('inverse', 'MatrixInverse', 'torch.linalg.inv', 620, 1280);
  g.e('to_cpu', 'inverse');
  g.n('to_cuda', 'TensorToDevice', '.cuda()', 900, 1280, { target: 'cuda' });
  g.e('inverse', 'to_cuda');
  g.variable('inverse_matrix', 'to_cuda', 'inv_trans_matrix', 1200, 1280);
  // The four-index expressions are decomposed into ordinary axis slicing and unsqueeze.
  for (const [prefix, source, y] of [
    ['forward', 'trans_matrix', 800],
    ['inverse', 'inverse_matrix', 1500],
  ] as const) {
    g.n(`${prefix}_rows`, 'TensorIndex', 'matrix[..., :3, :]', 1540, y, {
      mode: 'slice',
      dim: '1',
      start: '',
      stop: '3',
    });
    g.e(source, `${prefix}_rows`);
    g.n(`${prefix}_rotation`, 'TensorIndex', 'matrix[..., :3, :3]', 1840, y, {
      mode: 'slice',
      dim: '2',
      start: '',
      stop: '3',
    });
    g.e(`${prefix}_rows`, `${prefix}_rotation`);
    g.n(`${prefix}_transpose`, 'Permute', 'rotation.transpose(1,2)', 2160, y, { dims: '0,2,1' });
    g.e(`${prefix}_rotation`, `${prefix}_transpose`);
    g.n(`${prefix}_translation`, 'TensorIndex', 'matrix[..., :3, 3]', 1840, y + 220, {
      mode: 'index',
      dim: '2',
      index: '3',
    });
    g.e(`${prefix}_rows`, `${prefix}_translation`);
    g.n(`${prefix}_translation_axis`, 'Unsqueeze', 'matrix[..., None, :3, 3]', 2160, y + 220, {
      dim: 1,
    });
    g.e(`${prefix}_translation`, `${prefix}_translation_axis`);
  }
  g.n('decode', 'PointRangeTransform', 'decode_points（米制）', 340, 80, {
    mode: 'decode',
    pc_range: pc,
  });
  g.e('points_proposal', 'decode');
  g.n('rotate_forward', 'MatMul', 'points @ R.T', 2530, 80);
  g.e('decode', 'rotate_forward.a');
  g.e('forward_transpose', 'rotate_forward.b');
  g.n('translate_forward', 'Add', '+ t（沿点轴广播）', 2840, 80);
  g.e('rotate_forward', 'translate_forward.a');
  g.e('forward_translation_axis', 'translate_forward.b');
  g.n('add_delta', 'Add', '+ points_delta', 3150, 80);
  g.e('translate_forward', 'add_delta.a');
  g.e('points_delta', 'add_delta.b');
  g.n('rotate_inverse', 'MatMul', 'new_points @ R_inv.T', 3470, 80);
  g.e('add_delta', 'rotate_inverse.a');
  g.e('inverse_transpose', 'rotate_inverse.b');
  g.n('translate_inverse', 'Add', '+ t_inv（返回原坐标系）', 3790, 80);
  g.e('rotate_inverse', 'translate_inverse.a');
  g.e('inverse_translation_axis', 'translate_inverse.b');
  g.n('encode', 'PointRangeTransform', 'encode_points', 4110, 80, { mode: 'encode', pc_range: pc });
  g.e('translate_inverse', 'encode');
  g.output('encode', 'out', 4420, 80);
  return g.graph;
}

/** A reusable single iteration. The project may inline its nodes in a LoopRegion. */
function stepGraph(registry: ModuleRegistry, D: number, P: number, pc: string) {
  const g = graphBuilder(registry);
  const inputs = [
    'interval',
    'ego_feat',
    'curr_query_feat',
    'curr_query_pos',
    'curr_query_timestamp',
    'query_feat',
    'query_pos',
    'ind_stamps_all',
    'num_fu_query',
    'img_metas',
    'temporal_trajs',
    'training',
    'forecast_points_list',
    'forecast_semantics_list',
    'pred_trajs_list',
    'forecast_points_mask_list',
  ];
  inputs.forEach((id, i) => g.input(id, i * 140));
  g.n('batch_size', 'TensorSize', 'B = curr_query_feat.shape[0]', 350, -650, {
    axes: '0',
    names: 'B',
  });
  g.e('curr_query_feat', 'batch_size');
  g.n('ego_feat_dtype', 'GetDtype', 'ego_feat.dtype', 400, -400);
  g.e('ego_feat', 'ego_feat_dtype');
  g.n('ego_feat_device', 'GetDevice', 'ego_feat.device', 400, -300, { tensor_name: 'ego_feat' });
  g.e('ego_feat', 'ego_feat_device');
  g.n('create_ego_ones', 'NewOnes', 'ego_feat.new_ones(B,1,3)', 700, -350, {
    metadata_inputs: true,
    shape: 'd0,1,3',
    axes: 'batch,ego_query,xyz',
    requires_grad: false,
  });
  g.e('batch_size.dim0', 'create_ego_ones.d0');
  g.e('ego_feat_dtype', 'create_ego_ones.dtype');
  g.e('ego_feat_device', 'create_ego_ones.device');
  g.variable('ego_ones', 'create_ego_ones', '全一张量', 1000, -350);
  g.n('multiply_half', 'Function', '乘以 0.5', 1300, -350, { expression: 'x * 0.5' });
  g.e('ego_ones', 'multiply_half');
  g.variable('ego_query_points', 'multiply_half', 'ego_query_points（固定0.5）', 1600, -350);
  g.n('detach_key_points', 'Detach', 'curr_query_pos.detach()', 700, 0);
  g.e('curr_query_pos', 'detach_key_points');
  g.variable('detached_key_points', 'detach_key_points', 'curr_query_pos.detach()', 1050, 0);
  g.n('detach_key_feat', 'Detach', 'curr_query_feat.detach()', 700, 360);
  g.e('curr_query_feat', 'detach_key_feat');
  g.variable('detached_key_feat', 'detach_key_feat', 'curr_query_feat.detach()', 1050, 360);
  g.n('ego_cross_attn', 'OPUSCrossAttention', 'self.ego_cross_attn', 1900, 40, {
    embed_dims: D,
    num_heads: 8,
    pc_range: pc,
  });
  g.e('ego_query_points', 'ego_cross_attn.query_points');
  g.e('ego_feat', 'ego_cross_attn.query_feat');
  g.e('detached_key_points', 'ego_cross_attn.key_points');
  g.e('detached_key_feat', 'ego_cross_attn.key_feat');
  g.variable('fused_ego_feat', 'ego_cross_attn.fused_ego_feat', 'fused_ego_feat', 2320, 80);
  g.n(
    'traj_head',
    'Submodule',
    'self.traj_head（共享实例）',
    2730,
    80,
    {},
    mlp(registry, [D, D * 2, 2], 'Softplus'),
  );
  g.e('fused_ego_feat', 'traj_head.in');
  g.variable('pred_traj', 'traj_head.out', 'pred_traj（XY）', 3100, 80);
  g.n('append_traj', 'TensorListAppend', 'pred_trajs_list.append(pred_traj)', 3460, 80);
  g.e('pred_trajs_list', 'append_traj.items');
  g.e('pred_traj', 'append_traj.item');
  g.variable(
    'trajs_after_append',
    'append_traj',
    'pred_trajs_list（本轮后）',
    3820,
    80,
    'ValueVariable',
  );

  g.n('next_stamp', 'IntegerAddOne', 'interval + 1', 600, 850);
  g.e('interval', 'next_stamp');
  g.n('future_mask', 'TensorEqualInt', 'ind_stamps_all == interval + 1', 1000, 850);
  g.e('ind_stamps_all', 'future_mask.in');
  g.e('next_stamp', 'future_mask.value');
  g.n('future_count', 'IntegerListGet', 'self.num_fu_query[interval]', 1000, 1100);
  g.e('num_fu_query', 'future_count.in');
  g.e('interval', 'future_count.index');
  for (const [kind, y] of [
    ['feat', 750],
    ['pos', 1050],
  ] as const) {
    g.n(
      `select_future_${kind}`,
      'BooleanIndexAxis',
      `query_${kind}[:, ind_stamps_all == interval + 1]`,
      1480,
      y,
      { dim: '1', declared_count: true, count_symbol: 'Q_future' },
    );
    g.e(`query_${kind}`, `select_future_${kind}.in`);
    g.e('future_mask', `select_future_${kind}.mask`);
    g.e('future_count', `select_future_${kind}.count`);
    g.variable(`future_${kind}`, `select_future_${kind}`, `本时间组 query_${kind}`, 1850, y);
    g.n(`concat_${kind}`, 'Concat', `cat([curr_query_${kind}, future_${kind}], dim=1)`, 2260, y, {
      dim: 1,
    });
    g.e(`curr_query_${kind}`, `concat_${kind}.a`);
    g.e(`future_${kind}`, `concat_${kind}.b`);
  }
  g.variable('expanded_feat', 'concat_feat', 'curr_query_feat（追加未来组）', 2630, 750);
  g.n('detach_concat_pos', 'Detach', '拼接后整体 detach()', 2630, 1050);
  g.e('concat_pos', 'detach_concat_pos');
  g.variable('expanded_pos', 'detach_concat_pos', 'curr_query_pos（追加并detach）', 2940, 1050);
  g.n('six', 'IntegerParameter', '源码常量 6', 1550, 1560, { expression: '6' });
  g.n('timestamp_condition', 'IntegerCompare', 'interval < 6', 1850, 1560, { operator: 'lt' });
  g.e('interval', 'timestamp_condition.a');
  g.e('six', 'timestamp_condition.b');
  const timestampStart = g.graph.nodes.length;
  g.n('pos_dtype', 'GetDtype', 'curr_query_pos.dtype', 2280, 1460);
  g.e('expanded_pos', 'pos_dtype');
  g.n('pos_device', 'GetDevice', 'curr_query_pos.device', 2280, 1580, {
    tensor_name: 'curr_query_pos',
  });
  g.e('expanded_pos', 'pos_device');
  g.n(
    'new_timestamp_ones',
    'NewOnes',
    'curr_query_pos.new_ones(B, num_fu_query[interval], P, 1)',
    2610,
    1470,
    {
      metadata_inputs: true,
      shape: `d0,d1,${P},1`,
      axes: 'batch,query,point,time',
      requires_grad: false,
    },
  );
  g.e('batch_size.dim0', 'new_timestamp_ones.d0');
  g.e('future_count', 'new_timestamp_ones.d1');
  g.e('pos_dtype', 'new_timestamp_ones.dtype');
  g.e('pos_device', 'new_timestamp_ones.device');
  g.n('timestamp_half', 'Function', '* 0.5（所有新增组均为0.5）', 2980, 1470, {
    expression: 'x * 0.5',
  });
  g.e('new_timestamp_ones', 'timestamp_half');
  g.n(
    'concat_timestamp',
    'Concat',
    'cat([curr_query_timestamp, new_timestamp], dim=1)',
    3310,
    1470,
    { dim: 1 },
  );
  g.e('curr_query_timestamp', 'concat_timestamp.a');
  g.e('timestamp_half', 'concat_timestamp.b');
  g.n('timestamp_if', 'IfRegion', 'if interval < 6', 2240, 1380, {
    condition: 'interval < 6',
    members: g.graph.nodes
      .slice(timestampStart)
      .map((n) => n.id)
      .join(','),
    note: '当前配置6轮均成立。旧时间戳不累加；原始720组为0，新增组均为0.5。',
  });
  g.n('timestamp_after_if', 'SelectValue', 'curr_query_timestamp（条件后）', 3700, 1470);
  g.e('timestamp_condition', 'timestamp_after_if.condition');
  g.e('concat_timestamp', 'timestamp_after_if.true_value');
  g.e('curr_query_timestamp', 'timestamp_after_if.false_value');
  g.variable(
    'timestamp_updated',
    'timestamp_after_if',
    'curr_query_timestamp（没有整体递增）',
    4100,
    1470,
  );

  g.n(
    'concat_position_time',
    'Concat',
    'cat([curr_query_pos, curr_query_timestamp], dim=-1)',
    4550,
    1060,
    { dim: -1 },
  );
  g.e('expanded_pos', 'concat_position_time.a');
  g.e('timestamp_updated', 'concat_position_time.b');
  g.n('flatten_position_time', 'Flatten', 'flatten(2,3)', 4860, 1060, { start_dim: 2, end_dim: 3 });
  g.e('concat_position_time', 'flatten_position_time');
  g.variable('position_input', 'flatten_position_time', '位置+时间（48×4=192）', 5170, 1060);
  g.n(
    'position_encoder',
    'Submodule',
    'self.position_encoder',
    5540,
    1060,
    {},
    mlp(registry, [P * 4, D, D], 'ReLU', true),
  );
  g.e('position_input', 'position_encoder.in');
  g.variable('pos_embedding', 'position_encoder.out', 'pos_embedding', 5910, 1060);
  g.n('add_fused_ego', 'Add', 'curr_query_feat + fused_ego_feat（广播）', 5570, 670);
  g.e('expanded_feat', 'add_fused_ego.a');
  g.e('fused_ego_feat', 'add_fused_ego.b');
  g.n('add_position', 'Add', '+ pos_embedding', 6280, 690);
  g.e('add_fused_ego', 'add_position.a');
  g.e('pos_embedding', 'add_position.b');
  g.variable('updated_feat', 'add_position', 'curr_query_feat（本轮更新）', 6620, 690);

  for (const [kind, unit, y] of [
    ['reg', 3, 700],
    ['cls', 17, 1110],
    ['vel', 2, 1580],
  ] as const) {
    g.n(
      `${kind}_branch`,
      'Submodule',
      `self.${kind}_branch`,
      7070,
      y,
      {},
      mlp(registry, [D, D, D, P * unit]),
    );
    g.e('updated_feat', `${kind}_branch.in`);
    g.n(`${kind}_unflatten`, 'Unflatten', `unflatten(-1, (-1, ${unit}))`, 7470, y, {
      dim: -1,
      sizes: `-1,${unit}`,
    });
    g.e(`${kind}_branch.out`, `${kind}_unflatten`);
  }
  g.n('reg_half', 'Function', 'reg_offset * 0.5', 7790, 700, { expression: 'x * 0.5' });
  g.e('reg_unflatten', 'reg_half');
  g.variable('reg_offset', 'reg_half', 'reg_offset（米制XYZ）', 8100, 700);
  g.variable('cls_score', 'cls_unflatten', 'cls_score（logits，非概率）', 7790, 1110);
  g.variable('vel_offset', 'vel_unflatten', 'vel_offset（XY，没有乘0.5）', 7790, 1580);
  g.n('argmax_labels', 'Argmax', 'cls_score.argmax(-1)', 8130, 1110);
  g.e('cls_score', 'argmax_labels');
  g.variable('pred_labels', 'argmax_labels', 'pred_labels（类别索引）', 8450, 1110);
  g.n('label_ge2', 'TensorCompareScalar', 'pred_labels >= 2', 8790, 1030, {
    operator: 'ge',
    value: '2',
  });
  g.e('pred_labels', 'label_ge2');
  g.n('label_le10', 'TensorCompareScalar', 'pred_labels <= 10', 8790, 1270, {
    operator: 'le',
    value: '10',
  });
  g.e('pred_labels', 'label_le10');
  g.n('moving_and', 'LogicalAnd', 'torch.logical_and', 9100, 1150);
  g.e('label_ge2', 'moving_and.a');
  g.e('label_le10', 'moving_and.b');
  g.n('moving_unsqueeze', 'Unsqueeze', 'unsqueeze(-1)', 9410, 1150, { dim: -1 });
  g.e('moving_and', 'moving_unsqueeze');
  g.variable('pred_moving_mask', 'moving_unsqueeze', 'pred_moving_mask（bool）', 9700, 1150);
  g.n('reg_xy', 'TensorIndex', 'reg_offset[..., :2]', 8500, 510, {
    mode: 'slice',
    dim: '-1',
    start: '',
    stop: '2',
  });
  g.e('reg_offset', 'reg_xy');
  g.n('reg_z', 'TensorIndex', 'reg_offset[..., 2:]', 8500, 710, {
    mode: 'slice',
    dim: '-1',
    start: '2',
    stop: '',
  });
  g.e('reg_offset', 'reg_z');
  g.n('moving_velocity', 'Mul', 'vel_offset * pred_moving_mask', 10100, 1500);
  g.e('vel_offset', 'moving_velocity.a');
  g.e('pred_moving_mask', 'moving_velocity.b');
  g.n('reg_xy_plus_vel', 'Add', 'XY偏移 + 移动类别速度项', 10460, 510);
  g.e('reg_xy', 'reg_xy_plus_vel.a');
  g.e('moving_velocity', 'reg_xy_plus_vel.b');
  g.n('concat_reg_offset', 'Concat', 'cat([XY + vel*mask, Z], dim=-1)', 10810, 710, { dim: -1 });
  g.e('reg_xy_plus_vel', 'concat_reg_offset.a');
  g.e('reg_z', 'concat_reg_offset.b');
  g.n('flatten_reg', 'Flatten', 'reg_offset.flatten(2,3)', 11180, 710, {
    start_dim: 2,
    end_dim: 3,
  });
  g.e('concat_reg_offset', 'flatten_reg');
  g.variable('reg_offset_flat', 'flatten_reg', 'reg_offset（48×3）', 11500, 710);
  g.n('refine_points', 'SparseWorldRefinePoints', 'self.refine_points', 11860, 710, {
    num_refines: P,
    pc_range: pc,
  });
  g.e('expanded_pos', 'refine_points.points_proposal');
  g.e('reg_offset_flat', 'refine_points.points_delta');
  g.variable('updated_pos', 'refine_points.out', 'curr_query_pos（本轮预测，归一化）', 12280, 710);
  for (const [kind, source, y] of [
    ['points', 'updated_pos', 720],
    ['semantics', 'cls_score', 1160],
  ] as const) {
    g.n(`append_${kind}`, 'TensorListAppend', `forecast_${kind}_list.append(...)`, 12700, y);
    g.e(`forecast_${kind}_list`, `append_${kind}.items`);
    g.e(source, `append_${kind}.item`);
    g.variable(
      `${kind}_after_append`,
      `append_${kind}`,
      `forecast_${kind}_list（本轮后）`,
      13100,
      y,
      'ValueVariable',
    );
  }

  const maskStart = g.graph.nodes.length;
  g.n(
    'stack_ego2lidar',
    'BatchMetaStack',
    "np.stack([meta['ego2lidar'] for meta in img_metas])",
    6900,
    2460,
    { field: 'ego2lidar', shape: 'd0,4,4', axes: 'batch,row,column' },
  );
  g.e('img_metas', 'stack_ego2lidar.img_metas');
  g.e('batch_size.dim0', 'stack_ego2lidar.d0');
  g.n(
    'create_ego2lidar',
    'ArrayToTensorExplicit',
    'torch.tensor(..., device=cuda, dtype=float32)',
    7260,
    2460,
    { dtype: 'float32', device: 'cuda' },
  );
  g.e('stack_ego2lidar', 'create_ego2lidar');
  g.variable('ego2lidar', 'create_ego2lidar', 'ego2lidar', 7640, 2460);
  g.n('slice_gt_traj', 'DynamicSlice', 'temporal_trajs[:, interval:interval+1, :]', 6900, 2050, {
    dim: 1,
  });
  g.e('temporal_trajs', 'slice_gt_traj.in');
  g.e('interval', 'slice_gt_traj.start');
  g.e('next_stamp', 'slice_gt_traj.stop');
  g.variable('gt_traj', 'slice_gt_traj', 'gt_traj（真实轨迹）', 7260, 2050);
  g.n('negative_gt_traj', 'Function', '-gt_traj', 7640, 2050, { expression: '-x' });
  g.e('gt_traj', 'negative_gt_traj');
  g.n('pred_traj_first', 'TensorIndex', 'pred_traj[:, :, :1]', 6900, 2820, {
    mode: 'slice',
    dim: '2',
    start: '',
    stop: '1',
  });
  g.e('pred_traj', 'pred_traj_first');
  g.n('zeros_for_z', 'ZerosLike', 'zeros_like（只取shape/dtype/device）', 7260, 2820);
  g.e('pred_traj_first', 'zeros_for_z');
  g.n('concat_traj_delta', 'Concat', 'cat([-gt_traj, zeros_like(...)], dim=-1)', 8070, 2050, {
    dim: -1,
  });
  g.e('negative_gt_traj', 'concat_traj_delta.a');
  g.e('zeros_for_z', 'concat_traj_delta.b');
  g.variable(
    'pred_traj_expand',
    'concat_traj_delta',
    'pred_traj_expand = [-GT_x,-GT_y,0]',
    8470,
    2050,
  );
  g.n('flatten_points', 'Flatten', 'curr_query_pos.flatten(1,2)', 8470, 2450, {
    start_dim: 1,
    end_dim: 2,
  });
  g.e('updated_pos', 'flatten_points');
  g.n('trans_points', 'SparseWorldTransformPoints', 'self.trans_points', 8910, 2310, {
    pc_range: pc,
  });
  g.e('flatten_points', 'trans_points.points_proposal');
  g.e('pred_traj_expand', 'trans_points.points_delta');
  g.e('ego2lidar', 'trans_points.trans_matrix');
  g.n('restore_points_shape', 'ReshapeLike', 'reshape(curr_query_pos.shape)', 9370, 2310);
  g.e('trans_points.out', 'restore_points_shape.in');
  g.e('updated_pos', 'restore_points_shape.like');
  g.variable(
    'gt_points',
    'restore_points_shape',
    'gt_points（GT平移后的预测点，非真值点集）',
    9740,
    2310,
  );
  g.n('gt_points_x', 'TensorIndex', 'gt_points[..., 0]', 10250, 2310, {
    mode: 'index',
    dim: '-1',
    index: '0',
  });
  g.e('gt_points', 'gt_points_x');
  g.n('mask_x_ge0', 'TensorCompareScalar', '归一化 X >= 0（仅下界）', 10570, 2310, {
    operator: 'ge',
    value: '0',
  });
  g.e('gt_points_x', 'mask_x_ge0');
  g.variable('points_mask', 'mask_x_ge0', 'forecast_points_mask（bool）', 10920, 2310);
  g.n('append_mask', 'TensorListAppend', 'forecast_points_mask_list.append(mask)', 11320, 2310);
  g.e('forecast_points_mask_list', 'append_mask.items');
  g.e('points_mask', 'append_mask.item');
  g.n('training_mask_if', 'IfRegion', 'if self.training', 6860, 1950, {
    condition: 'self.training',
    members: g.graph.nodes
      .slice(maskStart)
      .map((n) => n.id)
      .join(','),
    note: '使用GT轨迹生成单侧X有效掩码；不使用预测轨迹数值，不改写已收集的预测点。eval不执行此区域。',
  });
  g.n('mask_after_if', 'SelectValue', 'forecast_points_mask_list（条件后）', 12100, 2310);
  g.e('training', 'mask_after_if.condition');
  g.e('append_mask', 'mask_after_if.true_value');
  g.e('forecast_points_mask_list', 'mask_after_if.false_value');
  g.variable(
    'masks_after_append',
    'mask_after_if',
    'forecast_points_mask_list（本轮后）',
    12600,
    2310,
    'ValueVariable',
  );
  const states: Record<string, string> = {
    curr_query_feat: 'updated_feat',
    curr_query_pos: 'updated_pos',
    curr_query_timestamp: 'timestamp_updated',
    forecast_points_list: 'points_after_append',
    forecast_semantics_list: 'semantics_after_append',
    pred_trajs_list: 'trajs_after_append',
    forecast_points_mask_list: 'masks_after_append',
  };
  Object.entries(states).forEach(([port, source], i) => g.output(source, port, 13800, i * 300));
  return g.graph;
}

function finishGraph(registry: ModuleRegistry, D: number, pc: string) {
  const g = graphBuilder(registry);
  [
    'base_outputs',
    'ego_feat',
    'curr_query_pos',
    'curr_query_feat',
    'pred_trajs_list',
    'forecast_points_list',
    'forecast_semantics_list',
    'forecast_points_mask_list',
    'pretrain',
    'num_fu_frames',
  ].forEach((id, i) => g.input(id, i * 140));
  g.n('not_pretrain', 'BooleanNot', 'not self.pretrain', 350, -400);
  g.e('pretrain', 'not_pretrain');
  g.n('trajectory_count', 'SequenceLength', 'len(pred_trajs_list)', 350, -180);
  g.e('pred_trajs_list', 'trajectory_count');
  g.n(
    'less_than_horizon',
    'IntegerCompare',
    'len(pred_trajs_list) < self.num_fu_frames',
    700,
    -180,
    { operator: 'lt' },
  );
  g.e('trajectory_count', 'less_than_horizon.a');
  g.e('num_fu_frames', 'less_than_horizon.b');
  g.n('extra_condition', 'BooleanAnd', 'not pretrain and len(trajs) < horizon', 1080, -300);
  g.e('not_pretrain', 'extra_condition.a');
  g.e('less_than_horizon', 'extra_condition.b');
  const start = g.graph.nodes.length;
  g.n('batch_size', 'TensorSize', 'B = ego_feat.shape[0]', 400, 300, { axes: '0', names: 'B' });
  g.e('ego_feat', 'batch_size');
  g.n('ego_dtype', 'GetDtype', 'ego_feat.dtype', 400, 460);
  g.e('ego_feat', 'ego_dtype');
  g.n('ego_device', 'GetDevice', 'ego_feat.device', 400, 580, { tensor_name: 'ego_feat' });
  g.e('ego_feat', 'ego_device');
  g.n('zeros', 'NewZeros', 'ego_feat.new_zeros(B,1,3)', 730, 420, {
    metadata_inputs: true,
    shape: 'd0,1,3',
    axes: 'batch,ego_query,xyz',
    requires_grad: false,
  });
  g.e('batch_size.dim0', 'zeros.d0');
  g.e('ego_dtype', 'zeros.dtype');
  g.e('ego_device', 'zeros.device');
  g.variable('zero_points', 'zeros', '全零坐标（不同于循环内0.5）', 1100, 420);
  g.n(
    'extra_attention',
    'OPUSCrossAttention',
    'self.ego_cross_attn（同一实例；无detach）',
    1510,
    620,
    { embed_dims: D, num_heads: 8, pc_range: pc },
  );
  g.e('zero_points', 'extra_attention.query_points');
  g.e('ego_feat', 'extra_attention.query_feat');
  g.e('curr_query_pos', 'extra_attention.key_points');
  g.e('curr_query_feat', 'extra_attention.key_feat');
  g.variable(
    'extra_fused_feat',
    'extra_attention.fused_ego_feat',
    'fused_ego_feat（额外一步）',
    1910,
    640,
  );
  g.n(
    'extra_traj_head',
    'Submodule',
    'self.traj_head（同一实例）',
    2270,
    640,
    {},
    mlp(registry, [D, D * 2, 2], 'Softplus'),
  );
  g.e('extra_fused_feat', 'extra_traj_head.in');
  g.variable('extra_traj', 'extra_traj_head.out', 'pred_traj（额外一步）', 2650, 640);
  g.n('append_extra_traj', 'TensorListAppend', 'pred_trajs_list.append(pred_traj)', 3000, 640);
  g.e('pred_trajs_list', 'append_extra_traj.items');
  g.e('extra_traj', 'append_extra_traj.item');
  g.n('extra_if', 'IfRegion', '循环后可选额外轨迹', 350, 230, {
    condition: 'not self.pretrain and len(pred_trajs_list) < self.num_fu_frames',
    members: g.graph.nodes
      .slice(start)
      .map((n) => n.id)
      .join(','),
    note: '只追加一次轨迹，不补齐6步；不追加占据点/语义/mask。epoch=0为False；epoch5-9可为True。此区域静态检查不表示执行。',
  });
  g.n('final_trajs', 'SelectValue', 'pred_trajs_list（按条件取更新前/后）', 3420, 620);
  g.e('extra_condition', 'final_trajs.condition');
  g.e('append_extra_traj', 'final_trajs.true_value');
  g.e('pred_trajs_list', 'final_trajs.false_value');
  g.variable('trajs_result', 'final_trajs', 'pred_trajs_list（最终）', 3850, 620, 'ValueVariable');
  g.n('pack_forecasts', 'DictionaryPack', 'dict（四个预测结果列表）', 4320, 1000, {
    keys: 'forecast_semantics_list,forecast_points_list,pred_trajs_list,forecast_points_mask_list',
  });
  g.e('forecast_semantics_list', 'pack_forecasts.forecast_semantics_list');
  g.e('forecast_points_list', 'pack_forecasts.forecast_points_list');
  g.e('trajs_result', 'pack_forecasts.pred_trajs_list');
  g.e('forecast_points_mask_list', 'pack_forecasts.forecast_points_mask_list');
  g.n('update_outputs', 'DictionaryUpdate', 'outputs.update(...)', 4800, 1000);
  g.e('base_outputs', 'update_outputs.base');
  g.e('pack_forecasts', 'update_outputs.updates');
  g.variable(
    'outputs',
    'update_outputs',
    'outputs（当前帧 + 未来列表）',
    5210,
    1000,
    'ValueVariable',
  );
  g.output('outputs', 'outputs', 5670, 1000);
  return g.graph;
}

export function installSparseWorldForecast(registry: ModuleRegistry) {
  const pc = text('pc_range', '占据坐标范围', '-40,-40,-1,40,40,5.4');
  registry.register({
    type: 'SparseWorldRefinePoints',
    label: 'SparseWorld refine_points',
    category: '组合计算',
    composite: true,
    description: '解码点集、求中心、加独立回归偏移、重新编码；不是旧点逐点加偏移。',
    fields: [num('num_refines', '点数', 48), pc],
    inputs: ['points_proposal', 'points_delta'].map((id) => ({ id, label: id })),
    outputs: [{ id: 'out', label: 'refine points' }],
    build: (p) => refineGraph(registry, Number(p.num_refines), String(p.pc_range)),
  });
  registry.register({
    type: 'SparseWorldTransformPoints',
    label: 'SparseWorld trans_points',
    category: '组合计算',
    composite: true,
    description: 'CPU求逆、解码、正向坐标变换、平移、逆变换、编码；输入点需展平为[B,M,3]。',
    fields: [pc],
    inputs: ['points_proposal', 'points_delta', 'trans_matrix'].map((id) => ({ id, label: id })),
    outputs: [{ id: 'out', label: 'transformed points' }],
    build: (p) => transformGraph(registry, String(p.pc_range)),
  });
  registry.register({
    type: 'SparseWorldForecastStep',
    label: 'SparseWorld 未来预测单轮',
    category: '组合计算',
    composite: true,
    description:
      'forward_backbone循环体详细图；可作为ForLoop循环体或在父画布平铺。各轮复用同一份网络参数。',
    fields: [num('embed_dims', '特征维数', 256), num('num_refines', '点数', 48), pc],
    inputs: [],
    outputs: [],
    build: (p) =>
      stepGraph(registry, Number(p.embed_dims), Number(p.num_refines), String(p.pc_range)),
  });
  registry.register({
    type: 'SparseWorldForecastFinish',
    label: 'SparseWorld 预测结果返回',
    category: '组合计算',
    composite: true,
    description:
      '循环后的可选额外轨迹、结果字典update与return；条件区仅静态检查，SelectValue按声明条件选择实际结果规格。',
    fields: [num('embed_dims', '特征维数', 256), pc],
    inputs: [],
    outputs: [],
    build: (p) => finishGraph(registry, Number(p.embed_dims), String(p.pc_range)),
  });
}
