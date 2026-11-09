import type { Graph, ModelNode, Params, Field } from '../core/types';
import { ModuleRegistry } from './registry';
import { input, output, num, bool, select } from './primitives';
import { requireThat } from './math';

export const mapCompositeTypes = new Set([
  'MapEncoder',
  'PointsEncoder',
  'FourierEmbedding',
  'MapMLP',
]);
function integer(p: Params, key: string, max = 4096) {
  const v = Number(p[key]);
  requireThat(Number.isSafeInteger(v) && v > 0 && v <= max, `${key} 需要 1–${max} 的整数。`);
  return v;
}
export function mapEncoderCode(n: ModelNode): string | undefined {
  const p = n.params;
  if (n.type === 'PointsEncoder')
    return `def forward(self, x, mask):\n    bs, n, _ = x.shape\n    x_valid = self.first_mlp(x[mask])\n    x_features = torch.zeros(bs, n, 256, device=x.device)\n    x_features[mask] = x_valid\n    pooled_feature = x_features.max(dim=1)[0]\n    x_features = torch.cat([x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1)\n    x_features_valid = self.second_mlp(x_features[mask])\n    res = torch.zeros(bs, n, self.encoder_channel, device=x.device)\n    res[mask] = x_features_valid\n    return res.max(dim=1)[0]`;
  if (n.type === 'FourierEmbedding')
    return `def forward(self, continuous_inputs):\n    x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi\n    x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)\n    continuous_embs = [self.mlps[i](x[..., i, :]) for i in range(self.input_dim)]\n    x = torch.stack(continuous_embs).sum(dim=0)\n    return self.to_out(x)\n\n# freqs.weight 是可训练频率权重，当前形状 [${p.input_dim}, ${p.num_freq_bands}]。`;
  if (n.type === 'MapMLP')
    return `# nn.Sequential；前后 Linear 保留独立权重和 bias=True。\n# ${p.norm === 'bn' ? 'BatchNorm1d' : 'LayerNorm'}，中间 ReLU。\n# ${p.pre_norm ? '输入先归一化，再 ReLU → Linear。' : 'Linear → 归一化 → ReLU → Linear。'}\ndef forward(self, x):\n    return self.layers(x)`;
  if (n.type === 'MapEncoder')
    return `# 当前配置快照：use_lane_boundary=${p.use_lane_boundary ? 'True' : 'False'}\n# 子图逐项绘制 MapEncoder.forward；修改节点配置重建相应分支。\n# polygon_feature: [B,M,P,${Number(p.polygon_channel) + (p.use_lane_boundary ? 4 : 0)}]\n# polygon_encoder: PointsEncoder\n# speed_limit_emb: FourierEmbedding(1, ${p.dim}, 64)\n# 未知限速直接读取 unknown_speed_emb.weight 并广播赋值。\n# 返回 x_polygon + (x_type + x_on_route + x_tl_status + x_speed_limit)。`;
}

export function installMapEncoder(registry: ModuleRegistry) {
  function builder() {
    const graph: Graph = { nodes: [], edges: [] };
    const node = (
      id: string,
      type: string,
      label: string,
      x: number,
      y: number,
      params: Params = {},
    ) => {
      const n = registry.create(type, params, id);
      n.label = label;
      n.position = { x, y };
      graph.nodes.push(n);
      return n;
    };
    const edge = (
      source: string,
      target: string,
      targetPort = 'in',
      sourcePort = 'out',
      waypoints?: { x: number; y: number }[],
    ) => {
      const e = {
        id: `${source}_${sourcePort}_${target}_${targetPort}`,
        source,
        sourcePort,
        target,
        targetPort,
        ...(waypoints ? { waypoints } : {}),
      };
      graph.edges.push(e);
      return e;
    };
    const inlet = (id: string, x: number, y: number) => node(id, 'Inlet', id, x, y, { port: id });
    const outlet = (source: string, x: number, y: number) => {
      node('out', 'Outlet', 'return', x, y, { port: 'out' });
      edge(source, 'out');
    };
    const value = (id: string, label: string, source: string, x: number, y: number) => {
      node(id, 'TensorVariable', label, x, y);
      edge(source, id);
    };
    const local = (
      id: string,
      label: string,
      source: string,
      port: string,
      x: number,
      y: number,
    ) => {
      node(id, 'LocalParameter', label, x, y);
      edge(source, id, 'in', port);
    };
    const alias = (id: string, label: string, owner: string, x: number, y: number) => {
      const n = node(id, 'LocalParameter', label, x, y);
      n.valueSource = owner;
      n.portLayout = { output: 'right' };
    };
    const size = (
      id: string,
      source: string,
      axes: string,
      names: string,
      x: number,
      y: number,
    ) => {
      node(id, 'TensorSize', `${source}.shape`, x, y, { axes, names });
      edge(source, id);
    };
    return { graph, node, edge, inlet, outlet, value, local, alias, size };
  }
  const register = (
    type: string,
    label: string,
    fields: Field[],
    build: (p: Params) => Graph,
    inputs = input,
  ) =>
    registry.register({
      type,
      label,
      fields,
      build,
      inputs,
      outputs: output,
      composite: true,
      category: type === 'MapMLP' ? '内部' : '网络结构',
      description:
        '按 PLUTO forward 分层绘制，双击进入。参数为创建时配置快照；在本节点修改会重建子图，项目超参数不会自动重建。',
    });
  register(
    'MapMLP',
    'MLP',
    [
      num('in_features', '输入维度', 10),
      num('hidden_features', '中间维度', 128),
      num('out_features', '输出维度', 256),
      select('norm', '归一化', 'bn', ['bn', 'ln']),
      bool('pre_norm', '直接从归一化开始', false),
      bool('training', '训练模式', true),
    ],
    (p) => {
      const inputDim = integer(p, 'in_features'),
        hidden = integer(p, 'hidden_features'),
        out = integer(p, 'out_features'),
        g = builder();
      g.inlet('in', 0, 100);
      let last = 'in';
      if (!p.pre_norm) {
        g.node('linear0', 'StateLinear', '0 · Linear', 250, 100, {
          in_features: inputDim,
          out_features: hidden,
          bias: true,
        });
        g.edge(last, 'linear0');
        last = 'linear0';
      }
      const width = p.pre_norm ? inputDim : hidden;
      g.node(
        'norm',
        p.norm === 'bn' ? 'StateBatchNorm' : 'LayerNorm',
        p.norm === 'bn' ? 'BatchNorm1d' : 'LayerNorm',
        550,
        100,
        { features: width, ...(p.norm === 'bn' ? { training: p.training } : {}) },
      );
      g.edge(last, 'norm');
      g.node('relu', 'ReLU', 'ReLU', 850, 100);
      g.edge('norm', 'relu');
      g.node('linear1', 'StateLinear', 'Linear', 1150, 100, {
        in_features: width,
        out_features: out,
        bias: true,
      });
      g.edge('relu', 'linear1');
      g.outlet('linear1', 1460, 100);
      return g.graph;
    },
  );
  register(
    'PointsEncoder',
    'PointsEncoder',
    [
      num('feat_channel', '点输入维度', 10),
      num('encoder_channel', '输出维度', 128),
      bool('training', '训练模式', true),
    ],
    (p) => {
      const cin = integer(p, 'feat_channel'),
        dim = integer(p, 'encoder_channel'),
        g = builder();
      g.inlet('in', 0, 100);
      g.inlet('mask', 0, 600);
      g.size('size', 'in', '0,1', 'bs,n', 250, -200);
      g.local('bs', 'bs', 'size', 'dim0', 510, -240);
      g.local('n', 'n', 'size', 'dim1', 510, -150);
      g.node('select_valid', 'BooleanIndex', 'x[mask]', 300, 100, { count_symbol: 'N_points' });
      g.edge('in', 'select_valid');
      g.edge('mask', 'select_valid', 'mask');
      g.node('first_mlp', 'MapMLP', 'first_mlp', 600, 80, {
        in_features: cin,
        hidden_features: 128,
        out_features: 256,
        norm: 'bn',
        training: p.training,
      });
      g.edge('select_valid', 'first_mlp');
      g.value('x_valid', 'x_valid', 'first_mlp', 920, 85);
      g.alias('bs_zero1', 'bs', 'bs', 1040, 340);
      g.alias('n_zero1', 'n', 'n', 1040, 420);
      g.node('zeros1', 'Zeros', 'zeros(bs,n,256)', 1270, 360, { shape: 'd0,d1,256' });
      g.edge('bs_zero1', 'zeros1', 'd0');
      g.edge('n_zero1', 'zeros1', 'd1');
      g.edge('in', 'zeros1', 'device_like', 'out', [
        { x: 190, y: 800 },
        { x: 1200, y: 800 },
      ]);
      g.node('fill1', 'BooleanAssign', 'x_features[mask] = x_valid', 1550, 150, {
        count_symbol: 'N_points',
      });
      g.edge('zeros1', 'fill1');
      g.edge('mask', 'fill1', 'mask', 'out', [
        { x: 180, y: 650 },
        { x: 1480, y: 650 },
      ]);
      g.edge('x_valid', 'fill1', 'values');
      g.value('x_features', 'x_features', 'fill1', 1770, 130);
      g.node('pool1', 'ReduceMax', 'max(dim=1).values', 2040, 150, { dim: 1 });
      g.edge('x_features', 'pool1');
      g.value('pooled_feature', 'pooled_feature', 'pool1', 2280, 130);
      g.node('unsqueeze_pool', 'Unsqueeze', 'unsqueeze(1)', 2540, 150, { dim: 1 });
      g.edge('pooled_feature', 'unsqueeze_pool');
      g.alias('n_repeat', 'n', 'n', 2560, 340);
      g.node('repeat_pool', 'Repeat', 'repeat(1,n,1)', 2800, 150, { repeats: '1,d0,1' });
      g.edge('unsqueeze_pool', 'repeat_pool');
      g.edge('n_repeat', 'repeat_pool', 'd0');
      g.node('cat', 'Concat', '局部特征 + pooled_feature', 3100, 350, { dim: -1, input_count: 2 });
      g.edge('x_features', 'cat', 'a', 'out', [
        { x: 1990, y: 700 },
        { x: 3020, y: 700 },
      ]);
      g.edge('repeat_pool', 'cat', 'b');
      g.node('select_second', 'BooleanIndex', 'x_features[mask]', 3390, 350, {
        count_symbol: 'N_points',
      });
      g.edge('cat', 'select_second');
      g.edge('mask', 'select_second', 'mask', 'out', [
        { x: 180, y: 1100 },
        { x: 3310, y: 1100 },
      ]);
      g.node('second_mlp', 'MapMLP', 'second_mlp', 3650, 330, {
        in_features: 512,
        hidden_features: 256,
        out_features: dim,
        norm: 'bn',
        training: p.training,
      });
      g.edge('select_second', 'second_mlp');
      g.value('x_features_valid', 'x_features_valid', 'second_mlp', 3980, 330);
      g.alias('bs_zero2', 'bs', 'bs', 3370, 800);
      g.alias('n_zero2', 'n', 'n', 3370, 880);
      g.node('zeros2', 'Zeros', 'zeros(bs,n,encoder_channel)', 3650, 840, {
        shape: `d0,d1,${dim}`,
      });
      g.edge('bs_zero2', 'zeros2', 'd0');
      g.edge('n_zero2', 'zeros2', 'd1');
      g.edge('in', 'zeros2', 'device_like', 'out', [
        { x: 190, y: 1400 },
        { x: 3570, y: 1400 },
      ]);
      g.node('fill2', 'BooleanAssign', 'res[mask] = x_features_valid', 4300, 550, {
        count_symbol: 'N_points',
      });
      g.edge('zeros2', 'fill2');
      g.edge('mask', 'fill2', 'mask', 'out', [
        { x: 180, y: 1100 },
        { x: 4210, y: 1100 },
      ]);
      g.edge('x_features_valid', 'fill2', 'values');
      g.node('pool2', 'ReduceMax', 'res.max(dim=1).values', 4590, 550, { dim: 1 });
      g.edge('fill2', 'pool2');
      g.outlet('pool2', 4850, 535);
      return g.graph;
    },
    [
      { id: 'in', label: 'x' },
      { id: 'mask', label: 'mask' },
    ],
  );
  register(
    'FourierEmbedding',
    'FourierEmbedding',
    [
      num('input_dim', '连续输入维度', 1),
      num('hidden_dim', '输出维度', 128),
      num('num_freq_bands', '频率数量', 64),
    ],
    (p) => {
      const inputs = integer(p, 'input_dim', 16),
        dim = integer(p, 'hidden_dim'),
        freqs = integer(p, 'num_freq_bands'),
        g = builder();
      g.inlet('in', 0, 100);
      g.node('unsqueeze', 'Unsqueeze', 'continuous_inputs.unsqueeze(-1)', 270, 100, { dim: -1 });
      g.edge('in', 'unsqueeze');
      g.node('freqs_weight', 'LearnedParameter', 'freqs.weight', 270, -220, {
        shape: `${inputs},${freqs}`,
      });
      g.node('frequency_mul', 'Mul', '输入 × 频率权重', 570, 100);
      g.edge('unsqueeze', 'frequency_mul', 'a');
      g.edge('freqs_weight', 'frequency_mul', 'b');
      g.node('scale', 'Function', '× 2π', 880, 100, {
        argument: 'x',
        expression: 'x * 2 * math.pi',
      });
      g.edge('frequency_mul', 'scale');
      g.node('cos', 'Cos', 'cos()', 1150, 0);
      g.node('sin', 'Sin', 'sin()', 1150, 220);
      g.edge('scale', 'cos');
      g.edge('scale', 'sin');
      g.node('cat', 'Concat', 'cos / sin / 原始输入', 1450, 100, { dim: -1, input_count: 3 });
      g.edge('cos', 'cat', 'a');
      g.edge('sin', 'cat', 'b');
      g.edge('unsqueeze', 'cat', 'c', 'out', [
        { x: 490, y: 520 },
        { x: 1360, y: 520 },
      ]);
      for (let i = 0; i < inputs; i++) {
        g.node(`select${i}`, 'TensorIndex', `x[...,${i},:]`, 1760, i * 280, {
          mode: 'index',
          dim: '-2',
          index: String(i),
        });
        g.edge('cat', `select${i}`);
        g.node(`mlp${i}`, 'MapMLP', `mlps[${i}]`, 2030, i * 280 - 15, {
          in_features: freqs * 2 + 1,
          hidden_features: dim,
          out_features: dim,
          norm: 'ln',
        });
        g.edge(`select${i}`, `mlp${i}`);
      }
      const mid = (inputs - 1) * 140;
      g.node('stack', 'Stack', 'stack(continuous_embs)', 2370, mid, {
        dim: 0,
        input_count: inputs,
      });
      for (let i = 0; i < inputs; i++) g.edge(`mlp${i}`, 'stack', String.fromCharCode(97 + i));
      g.node('sum', 'ReduceSum', 'sum(dim=0)', 2680, mid, { dim: 0 });
      g.edge('stack', 'sum');
      g.node('to_out', 'MapMLP', 'to_out', 2950, mid - 15, {
        in_features: dim,
        hidden_features: dim,
        out_features: dim,
        norm: 'ln',
        pre_norm: true,
      });
      g.edge('sum', 'to_out');
      g.outlet('to_out', 3280, mid);
      return g.graph;
    },
  );
  register(
    'MapEncoder',
    'MapEncoder',
    [
      num('polygon_channel', '基础点特征维度', 6),
      num('dim', '输出特征维度', 128),
      bool('use_lane_boundary', '包含左右边界', true),
      bool('training', '训练模式', true),
    ],
    (p) => {
      const dim = integer(p, 'dim'),
        cin = integer(p, 'polygon_channel') + (p.use_lane_boundary ? 4 : 0),
        g = builder();
      g.inlet('data', 0, 900);
      const fields: [string, number, boolean][] = [
        ['polygon_center', 0, false],
        ['point_position', 300, false],
        ['point_vector', 600, false],
        ['point_orientation', 900, false],
        ['valid_mask', 1600, false],
        ['polygon_type', 2100, true],
        ['polygon_on_route', 2400, true],
        ['polygon_tl_status', 2700, true],
        ['polygon_has_speed_limit', 3000, false],
        ['polygon_speed_limit', 3300, false],
      ];
      for (const [name, y, cast] of fields) {
        g.node(`${name}_field`, 'DictionaryGet', `data[map][${name}]`, 260, y, {
          path: `map.${name}`,
        });
        g.edge('data', `${name}_field`);
        if (cast) {
          g.node(`${name}_long`, 'Cast', '.long()', 550, y, { dtype: 'int64' });
          g.edge(`${name}_field`, `${name}_long`);
        }
        g.value(name, name, cast ? `${name}_long` : `${name}_field`, 830, y - 15);
      }
      const pick = (
        id: string,
        source: string,
        axis: number,
        index: number,
        x: number,
        y: number,
      ) => {
        g.node(id, 'TensorIndex', `${source}[轴${axis},${index}]`, x, y, {
          mode: 'index',
          dim: String(axis),
          index: String(index),
        });
        g.edge(source, id);
      };
      g.node('center_xy', 'TensorIndex', 'polygon_center[..., :2]', 1140, 0, {
        mode: 'slice',
        dim: '-1',
        start: '',
        stop: '2',
      });
      g.edge('polygon_center', 'center_xy');
      g.node('center_unsqueeze', 'Unsqueeze', '[..., None, :2]', 1410, 0, { dim: -2 });
      g.edge('center_xy', 'center_unsqueeze');
      pick('centerline_position', 'point_position', 2, 0, 1140, 300);
      pick('centerline_vector', 'point_vector', 2, 0, 1140, 600);
      pick('centerline_heading', 'point_orientation', 2, 0, 1140, 900);
      g.node('relative_position', 'Subtract', '中心线相对 polygon_center', 1710, 300);
      g.edge('centerline_position', 'relative_position', 'a');
      g.edge('center_unsqueeze', 'relative_position', 'b');
      g.node('heading_cos', 'Cos', 'cos()', 1420, 850);
      g.node('heading_sin', 'Sin', 'sin()', 1420, 1050);
      g.edge('centerline_heading', 'heading_cos');
      g.edge('centerline_heading', 'heading_sin');
      g.node('heading_stack', 'Stack', '方向向量', 1710, 950, { dim: -1, input_count: 2 });
      g.edge('heading_cos', 'heading_stack', 'a');
      g.edge('heading_sin', 'heading_stack', 'b');
      if (p.use_lane_boundary)
        for (const [side, index, y] of [
          ['left', 1, 1250],
          ['right', 2, 1450],
        ] as const) {
          pick(`${side}_position`, 'point_position', 2, index, 1140, y);
          g.node(`${side}_offset`, 'Subtract', `${side} - centerline`, 1710, y);
          g.edge(`${side}_position`, `${side}_offset`, 'a');
          g.edge('centerline_position', `${side}_offset`, 'b');
        }
      g.node('feature_cat', 'Concat', 'polygon_feature', 2100, 600, {
        dim: -1,
        input_count: p.use_lane_boundary ? 5 : 3,
      });
      g.edge('relative_position', 'feature_cat', 'a');
      g.edge('centerline_vector', 'feature_cat', 'b');
      g.edge('heading_stack', 'feature_cat', 'c');
      if (p.use_lane_boundary) {
        g.edge('left_offset', 'feature_cat', 'd');
        g.edge('right_offset', 'feature_cat', 'e');
      }
      g.value('polygon_feature', 'polygon_feature', 'feature_cat', 2420, 600);
      g.size('feature_size', 'polygon_feature', '0,1,2,3', 'bs,M,P,C', 2680, 200);
      ['bs', 'M', 'P', 'C'].forEach((name, i) =>
        g.local(name, name, 'feature_size', `dim${i}`, 2920, 50 + i * 90),
      );
      g.alias('bs_product', 'bs', 'bs', 2640, 1000);
      g.alias('M_product', 'M', 'M', 2640, 1080);
      g.node('batch_product', 'IntegerMultiply', 'bs × M', 2880, 1040);
      g.edge('bs_product', 'batch_product', 'a');
      g.edge('M_product', 'batch_product', 'b');
      g.local('BM', 'bs×M', 'batch_product', 'out', 3090, 1040);
      g.node('feature_reshape', 'Reshape', 'polygon_feature.reshape(bs*M,P,C)', 3400, 600, {
        shape: 'd0,d1,d2',
      });
      g.edge('polygon_feature', 'feature_reshape');
      ['BM', 'P', 'C'].forEach((name, i) => {
        g.alias(`${name}_feature`, name === 'BM' ? 'bs×M' : name, name, 3160, 730 + i * 80);
        g.edge(`${name}_feature`, 'feature_reshape', `d${i}`);
      });
      g.node('mask_view', 'View', 'valid_mask.view(bs*M,P)', 3400, 1600, { shape: 'd0,d1' });
      g.edge('valid_mask', 'mask_view');
      ['BM', 'P'].forEach((name, i) => {
        g.alias(`${name}_mask`, name === 'BM' ? 'bs×M' : name, name, 3160, 1710 + i * 80);
        g.edge(`${name}_mask`, 'mask_view', `d${i}`);
      });
      g.node('polygon_encoder', 'PointsEncoder', 'polygon_encoder', 3740, 650, {
        feat_channel: cin,
        encoder_channel: dim,
        training: p.training,
      });
      g.edge('feature_reshape', 'polygon_encoder');
      g.edge('mask_view', 'polygon_encoder', 'mask');
      g.value('encoded_polygons', '编码后的 polygon', 'polygon_encoder', 4060, 650);
      g.node('polygon_view', 'View', 'view(bs,M,-1)', 4370, 650, { shape: 'd0,d1,-1' });
      g.edge('encoded_polygons', 'polygon_view');
      ['bs', 'M'].forEach((name, i) => {
        g.alias(`${name}_polygon_view`, name, name, 4140, 830 + i * 80);
        g.edge(`${name}_polygon_view`, 'polygon_view', `d${i}`);
      });
      g.value('x_polygon', 'x_polygon', 'polygon_view', 4660, 640);
      for (const [name, field, count, y] of [
        ['type', 'polygon_type', 3, 2100],
        ['on_route', 'polygon_on_route', 2, 2400],
        ['tl_status', 'polygon_tl_status', 4, 2700],
      ] as const) {
        const id = name === 'tl_status' ? 'traffic_light_emb' : `${name}_emb`;
        g.node(id, 'Embedding', id, 3740, y, { num_embeddings: count, embedding_dim: dim });
        g.edge(field, id);
        g.value(`x_${name}`, `x_${name}`, id, 4060, y);
      }
      g.node('known_speed', 'BooleanIndex', 'polygon_speed_limit[has_speed_limit]', 1250, 3300, {
        count_symbol: 'N_speed',
      });
      g.edge('polygon_speed_limit', 'known_speed');
      g.edge('polygon_has_speed_limit', 'known_speed', 'mask');
      g.node('speed_unsqueeze', 'Unsqueeze', 'unsqueeze(-1)', 1550, 3300, { dim: -1 });
      g.edge('known_speed', 'speed_unsqueeze');
      g.node('speed_limit_emb', 'FourierEmbedding', 'speed_limit_emb', 1880, 3280, {
        input_dim: 1,
        hidden_dim: dim,
        num_freq_bands: 64,
      });
      g.edge('speed_unsqueeze', 'speed_limit_emb');
      g.value('speed_emb', '已知限速编码', 'speed_limit_emb', 2240, 3280);
      g.alias('bs_speed_zero', 'bs', 'bs', 4350, 2940);
      g.alias('M_speed_zero', 'M', 'M', 4350, 3020);
      g.node('speed_zeros', 'Zeros', 'zeros(bs,M,dim)', 4650, 2960, { shape: `d0,d1,${dim}` });
      g.edge('bs_speed_zero', 'speed_zeros', 'd0');
      g.edge('M_speed_zero', 'speed_zeros', 'd1');
      g.edge('x_polygon', 'speed_zeros', 'device_like');
      g.node('speed_known_fill', 'BooleanAssign', '写入已知限速', 4990, 2950, {
        count_symbol: 'N_speed',
      });
      g.edge('speed_zeros', 'speed_known_fill');
      g.edge('polygon_has_speed_limit', 'speed_known_fill', 'mask', 'out', [
        { x: 1160, y: 3180 },
        { x: 4910, y: 3180 },
      ]);
      g.edge('speed_emb', 'speed_known_fill', 'values');
      g.node('unknown_mask', 'Invert', '~polygon_has_speed_limit', 4800, 3430);
      g.edge('polygon_has_speed_limit', 'unknown_mask');
      g.node('unknown_speed_weight', 'LearnedParameter', 'unknown_speed_emb.weight', 4800, 3650, {
        shape: `1,${dim}`,
      });
      g.node('speed_unknown_fill', 'BooleanAssign', '广播未知限速权重', 5370, 2950, {
        count_symbol: 'N_unknown_speed',
        allow_broadcast: true,
      });
      g.edge('speed_known_fill', 'speed_unknown_fill');
      g.edge('unknown_mask', 'speed_unknown_fill', 'mask');
      g.edge('unknown_speed_weight', 'speed_unknown_fill', 'values');
      g.value('x_speed_limit', 'x_speed_limit', 'speed_unknown_fill', 5710, 2940);
      g.node('add_type_route', 'Add', 'x_type + x_on_route', 4460, 2200);
      g.edge('x_type', 'add_type_route', 'a');
      g.edge('x_on_route', 'add_type_route', 'b');
      g.node('add_light', 'Add', '+ x_tl_status', 4810, 2200);
      g.edge('add_type_route', 'add_light', 'a');
      g.edge('x_tl_status', 'add_light', 'b');
      g.node('add_speed', 'Add', '+ x_speed_limit', 6080, 2200);
      g.edge('add_light', 'add_speed', 'a');
      g.edge('x_speed_limit', 'add_speed', 'b');
      g.node('add_polygon', 'Add', 'x_polygon += 属性编码', 6450, 1100);
      g.edge('x_polygon', 'add_polygon', 'a');
      g.edge('add_speed', 'add_polygon', 'b');
      g.outlet('add_polygon', 6800, 1100);
      return g.graph;
    },
    [{ id: 'data', label: 'data' }],
  );
}
