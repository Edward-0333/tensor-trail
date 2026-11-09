import type { Graph, Params, Field, ModelNode } from '../core/types';
import { ModuleRegistry } from './registry';
import { input, output, num, text, bool } from './primitives';
import { requireThat } from './math';
import { probability } from './sequenceOps';

export const plutoCompositeTypes = new Set([
  'PlutoEncoderLayer',
  'PlutoEncoderMLP',
  'PlutoAgentPredictor',
  'PlutoPlanningDecoder',
  'PlutoDecoderLayer',
  'PlutoDecoderFFN',
  'ReferencePlanningBranch',
]);
export function plutoCompositeCode(n: ModelNode): string | undefined {
  const p = n.params;
  if (n.type === 'PlutoEncoderLayer')
    return `# TransformerEncoderLayer(dim=${p.dim}, num_heads=${p.num_heads}, drop_path=${p.drop_path})\n# 构造时未指定 drop / attn_drop，均为 0。\nsrc2 = self.norm1(src)\nsrc2, attn = self.attn(src2, src2, src2, key_padding_mask=key_padding_mask)\nsrc = src + self.drop_path1(src2)\nsrc = src + self.drop_path2(self.mlp(self.norm2(src)))\nreturn src  # return_attn_weights=False`;
  if (n.type === 'PlutoAgentPredictor')
    return `# 三套独立 MLPLayer(${p.dim}, ${2 * Number(p.dim)}, ${2 * Number(p.future_steps)})\nbs, N, _ = x.shape\nloc = self.loc_predictor(x).view(bs, N, ${p.future_steps}, 2)\nyaw = self.yaw_predictor(x).view(bs, N, ${p.future_steps}, 2)\nvel = self.vel_predictor(x).view(bs, N, ${p.future_steps}, 2)\nreturn torch.cat([loc, yaw, vel], dim=-1)  # [..., 6]`;
  if (n.type === 'PlutoPlanningDecoder')
    return `# 当前配置：num_mode=${p.num_modes}, decoder_depth=${p.decoder_depth}, num_heads=${p.num_heads}\n# cat_x=${p.cat_x ? 'True' : 'False'}；false 时没有自车特征拼接支路。\n# 子画布展示 reference 编码、reference × mode query、解码器循环、四个独立输出头。\n# pi_head 输出原始 logits，不在此处 softmax，也不挑选 top1。\nreturn traj, pi`;
  if (n.type === 'PlutoDecoderLayer')
    return '# DecoderLayer.forward\n# R2R → 有效参考线上的 M2M → 场景 cross-attention → FFN\n# 每一步保留源码 pre-norm、残差和 dropout。\n# R2R/M2M 复杂重排与掩码处理可在各函数块右侧查看。';
}
export function installPlutoForward(registry: ModuleRegistry) {
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
    const edge = (source: string, target: string, targetPort = 'in', sourcePort = 'out') =>
      graph.edges.push({
        id: `${source}_${sourcePort}_${target}_${targetPort}`,
        source,
        sourcePort,
        target,
        targetPort,
      });
    const inlet = (id: string, x: number, y: number) => node(id, 'Inlet', id, x, y, { port: id });
    const outlet = (
      id: string,
      source: string,
      x: number,
      y: number,
      port = 'out',
      sourcePort = 'out',
    ) => {
      node(id, 'Outlet', id, x, y, { port });
      edge(source, id, 'in', sourcePort);
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
    const local = (id: string, source: string, port: string, x: number, y: number) => {
      node(id, 'LocalParameter', id, x, y);
      edge(source, id, 'in', port);
    };
    return { graph, node, edge, inlet, outlet, size, local };
  }
  const positive = (p: Params, k: string, max = 4096) => {
    const n = Number(p[k]);
    requireThat(Number.isSafeInteger(n) && n > 0 && n <= max, `${k} 需要 1–${max} 的整数。`);
    return n;
  };
  const core = [num('dim', '特征维度', 128), num('num_heads', '注意力头数', 4)];
  const decoderFields = [
    ...core,
    num('num_modes', '模式数', 12),
    num('decoder_depth', '解码层数', 4),
    num('future_steps', '未来步数', 80),
    text('dropout', 'dropout', '0.1'),
    bool('cat_x', '拼接自车特征', false),
    bool('training', '训练模式', true),
  ];
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
      category: type === 'PlutoEncoderMLP' || type === 'PlutoDecoderFFN' ? '内部' : '网络结构',
      description: '按 PLUTO 源码绘制，双击展开。参数为当前配置快照；修改本节点配置会重建子图。',
      fields,
      inputs,
      outputs: output,
      composite: true,
      build,
    });
  register('PlutoEncoderMLP', 'Mlp', [num('dim', '特征维度', 128)], (p) => {
    const d = positive(p, 'dim'),
      g = builder();
    g.inlet('in', 0, 100);
    const steps: [string, string, Params][] = [
      ['fc1', 'StateLinear', { in_features: d, out_features: 4 * d, bias: true }],
      ['act', 'GELU', {}],
      ['drop1', 'Dropout', { p: '0' }],
      ['fc2', 'StateLinear', { in_features: 4 * d, out_features: d, bias: true }],
      ['drop2', 'Dropout', { p: '0' }],
    ];
    let prev = 'in';
    steps.forEach(([id, type, params], i) => {
      g.node(id, type, id, 300 + i * 300, 100, params);
      g.edge(prev, id);
      prev = id;
    });
    g.outlet('out', prev, 1850, 100);
    return g.graph;
  });
  register(
    'PlutoEncoderLayer',
    'TransformerEncoderLayer',
    [...core, text('drop_path', 'DropPath 概率', '0')],
    (p) => {
      const d = positive(p, 'dim'),
        heads = positive(p, 'num_heads');
      requireThat(d % heads === 0, 'dim 必须能被头数整除');
      probability(p.drop_path);
      const g = builder();
      g.inlet('src', 0, 0);
      g.inlet('key_padding_mask', 0, 350);
      g.node('norm1', 'LayerNorm', 'norm1', 300, 0, { features: d });
      g.edge('src', 'norm1');
      g.node('attn', 'MaskedMultiheadAttention', 'attn', 600, 0, {
        dim: d,
        num_heads: heads,
        dropout: '0',
      });
      ['query', 'key', 'value'].forEach((port) => g.edge('norm1', 'attn', port));
      g.edge('key_padding_mask', 'attn', 'mask');
      g.node('drop_path1', 'CodeFunction', 'drop_path1', 920, 0, {
        operation: 'DropPath',
        p: p.drop_path,
      });
      g.edge('attn', 'drop_path1');
      g.node('residual1', 'Add', 'src + attention', 1220, 0);
      g.edge('src', 'residual1', 'a');
      g.edge('drop_path1', 'residual1', 'b');
      g.node('norm2', 'LayerNorm', 'norm2', 1520, 0, { features: d });
      g.edge('residual1', 'norm2');
      g.node('mlp', 'PlutoEncoderMLP', 'mlp', 1820, 0, { dim: d });
      g.edge('norm2', 'mlp');
      g.node('drop_path2', 'CodeFunction', 'drop_path2', 2120, 0, {
        operation: 'DropPath',
        p: p.drop_path,
      });
      g.edge('mlp', 'drop_path2');
      g.node('residual2', 'Add', 'src + mlp', 2420, 0);
      g.edge('residual1', 'residual2', 'a');
      g.edge('drop_path2', 'residual2', 'b');
      g.outlet('out', 'residual2', 2720, 0);
      return g.graph;
    },
    [
      { id: 'src', label: 'src' },
      { id: 'key_padding_mask', label: 'key_padding_mask' },
    ],
  );
  register(
    'PlutoAgentPredictor',
    'AgentPredictor',
    [num('dim', '特征维度', 128), num('future_steps', '未来步数', 80)],
    (p) => {
      const d = positive(p, 'dim'),
        t = positive(p, 'future_steps'),
        g = builder();
      g.inlet('in', 0, 320);
      g.size('size', 'in', '0,1', 'bs,N', 260, -300);
      g.local('bs', 'size', 'dim0', 540, -340);
      g.local('N', 'size', 'dim1', 540, -240);
      ['loc', 'yaw', 'vel'].forEach((part, i) => {
        const y = i * 320;
        g.node(`${part}_predictor`, 'MapMLP', `${part}_predictor`, 850, y, {
          in_features: d,
          hidden_features: 2 * d,
          out_features: t * 2,
          norm: 'ln',
        });
        g.edge('in', `${part}_predictor`);
        g.node(part, 'View', part, 1190, y, { shape: `d0,d1,${t},2` });
        g.edge(`${part}_predictor`, part);
        g.edge('bs', part, 'd0');
        g.edge('N', part, 'd1');
      });
      g.node('prediction', 'Concat', 'prediction', 1550, 320, { dim: -1, input_count: 3 });
      ['loc', 'yaw', 'vel'].forEach((name, i) =>
        g.edge(name, 'prediction', String.fromCharCode(97 + i)),
      );
      g.outlet('out', 'prediction', 1850, 320);
      return g.graph;
    },
  );
  register(
    'PlutoDecoderFFN',
    'ffn',
    [num('dim', '特征维度', 128), text('dropout', 'dropout', '0.1')],
    (p) => {
      const d = positive(p, 'dim');
      probability(p.dropout);
      const g = builder();
      g.inlet('in', 0, 0);
      const steps: [string, string, Params][] = [
        ['linear0', 'StateLinear', { in_features: d, out_features: 4 * d, bias: true }],
        ['relu', 'ReLU', {}],
        ['dropout', 'Dropout', { p: p.dropout }],
        ['linear3', 'StateLinear', { in_features: 4 * d, out_features: d, bias: true }],
      ];
      let prev = 'in';
      steps.forEach(([id, type, params], i) => {
        g.node(id, type, id, 300 + i * 300, 0, params);
        g.edge(prev, id);
        prev = id;
      });
      g.outlet('out', prev, 1550, 0);
      return g.graph;
    },
  );
  register(
    'PlutoDecoderLayer',
    'DecoderLayer',
    [...core, text('dropout', 'dropout', '0.1')],
    (p) => {
      const d = positive(p, 'dim'),
        heads = positive(p, 'num_heads');
      requireThat(d % heads === 0, 'dim 必须能被头数整除');
      probability(p.dropout);
      const g = builder();
      ['tgt', 'memory', 'tgt_key_padding_mask', 'memory_key_padding_mask', 'm_pos'].forEach(
        (id, i) => g.inlet(id, 0, i * 240),
      );
      g.node('r2r', 'PlutoCode', 'R2R 注意力', 380, 0, {
        operation: 'R2RAttention',
        dim: d,
        num_heads: heads,
        dropout: p.dropout,
      });
      g.edge('tgt', 'r2r', 'tgt');
      g.edge('tgt_key_padding_mask', 'r2r', 'mask');
      g.node('m2m', 'PlutoCode', 'M2M 注意力', 760, 0, {
        operation: 'M2MAttention',
        dim: d,
        num_heads: heads,
        dropout: p.dropout,
      });
      g.edge('r2r', 'm2m', 'tgt');
      g.edge('tgt_key_padding_mask', 'm2m', 'mask');
      g.edge('m_pos', 'm2m', 'm_pos');
      g.node('flatten', 'Flatten', 'tgt.view(bs, R*M, D)', 1100, 0, { start_dim: 1, end_dim: 2 });
      g.edge('m2m', 'flatten');
      g.node('norm3', 'LayerNorm', 'norm3', 1420, 0, { features: d });
      g.edge('flatten', 'norm3');
      g.node('cross_attn', 'MaskedMultiheadAttention', 'cross_attn', 1740, 0, {
        dim: d,
        num_heads: heads,
        dropout: p.dropout,
      });
      g.edge('norm3', 'cross_attn', 'query');
      g.edge('memory', 'cross_attn', 'key');
      g.edge('memory', 'cross_attn', 'value');
      g.edge('memory_key_padding_mask', 'cross_attn', 'mask');
      g.node('dropout2', 'Dropout', 'dropout2', 2060, 0, { p: p.dropout });
      g.edge('cross_attn', 'dropout2');
      g.node('residual3', 'Add', 'tgt + cross attention', 2380, 0);
      g.edge('flatten', 'residual3', 'a');
      g.edge('dropout2', 'residual3', 'b');
      g.node('norm4', 'LayerNorm', 'norm4', 2700, 0, { features: d });
      g.edge('residual3', 'norm4');
      g.node('ffn', 'PlutoDecoderFFN', 'ffn', 3020, 0, { dim: d, dropout: p.dropout });
      g.edge('norm4', 'ffn');
      g.node('dropout3', 'Dropout', 'dropout3', 3340, 0, { p: p.dropout });
      g.edge('ffn', 'dropout3');
      g.node('residual4', 'Add', 'tgt + ffn', 3660, 0);
      g.edge('residual3', 'residual4', 'a');
      g.edge('dropout3', 'residual4', 'b');
      g.size('size', 'tgt', '0,1,2,3', 'bs,R,M,D', 3340, 380);
      g.node('restore', 'Reshape', 'tgt', 3980, 0, { shape: 'd0,d1,d2,d3' });
      g.edge('residual4', 'restore');
      for (const [i, id] of ['bs', 'R', 'M', 'D'].entries()) {
        g.local(id, 'size', `dim${i}`, 3660, 300 + i * 100);
        g.edge(id, 'restore', `d${i}`);
      }
      g.outlet('out', 'restore', 4300, 0);
      return g.graph;
    },
    ['tgt', 'memory', 'tgt_key_padding_mask', 'memory_key_padding_mask', 'm_pos'].map((id) => ({
      id,
      label: id,
    })),
  );
  register(
    'PlutoPlanningDecoder',
    'PlanningDecoder',
    decoderFields,
    (p) => {
      const d = positive(p, 'dim'),
        heads = positive(p, 'num_heads'),
        m = positive(p, 'num_modes', 64),
        depth = positive(p, 'decoder_depth', 16),
        t = positive(p, 'future_steps');
      probability(p.dropout);
      requireThat(d % heads === 0, 'dim 必须能被头数整除');
      const g = builder();
      g.inlet('data', 0, 500);
      g.inlet('enc_data', 0, 1580);
      ['position', 'vector', 'orientation', 'valid_mask'].forEach((name, i) => {
        g.node(`r_${name}`, 'DictionaryGet', `r_${name}`, 300, i * 260, {
          path: `reference_line.${name}`,
        });
        g.edge('data', `r_${name}`);
      });
      ['enc_emb', 'enc_key_padding_mask'].forEach((name, i) => {
        g.node(name, 'DictionaryGet', name, 300, 1500 + i * 280, { path: name });
        g.edge('enc_data', name);
      });
      g.node('r_any', 'Any', 'any(-1)', 650, 780, { dim: '-1' });
      g.edge('r_valid_mask', 'r_any');
      g.node('r_padding', 'Invert', 'r_key_padding_mask', 950, 780);
      g.edge('r_any', 'r_padding');
      g.node('r_feature', 'PlutoCode', 'r_feature', 700, 140, { operation: 'ReferenceFeatures' });
      ['position', 'vector', 'orientation'].forEach((name) =>
        g.edge(`r_${name}`, 'r_feature', name),
      );
      g.size('size', 'r_feature', '0,1,2,3', 'bs,R,P,C', 1040, -260);
      ['bs', 'R', 'P', 'C'].forEach((id, i) =>
        g.local(id, 'size', `dim${i}`, 1340, -400 + i * 100),
      );
      g.node('bs_R', 'IntegerMultiply', 'bs × R', 1610, -220);
      g.edge('bs', 'bs_R', 'a');
      g.edge('R', 'bs_R', 'b');
      g.node('r_feature_flat', 'Reshape', 'r_feature', 1640, 160, { shape: 'd0,d1,d2' });
      g.edge('r_feature', 'r_feature_flat');
      g.edge('bs_R', 'r_feature_flat', 'd0');
      g.edge('P', 'r_feature_flat', 'd1');
      g.edge('C', 'r_feature_flat', 'd2');
      g.node('r_mask_flat', 'View', 'r_valid_mask', 1640, 500, { shape: 'd0,d1' });
      g.edge('r_valid_mask', 'r_mask_flat');
      g.edge('bs_R', 'r_mask_flat', 'd0');
      g.edge('P', 'r_mask_flat', 'd1');
      g.node('r_encoder', 'PointsEncoder', 'r_encoder', 1970, 150, {
        feat_channel: 6,
        encoder_channel: d,
        training: p.training,
      });
      g.edge('r_feature_flat', 'r_encoder');
      g.edge('r_mask_flat', 'r_encoder', 'mask');
      g.node('r_emb_view', 'View', 'r_emb', 2300, 160, { shape: `d0,d1,${d}` });
      g.edge('r_encoder', 'r_emb_view');
      g.edge('bs', 'r_emb_view', 'd0');
      g.edge('R', 'r_emb_view', 'd1');
      g.node('r_pos', 'PlutoCode', 'r_pos', 1640, 1120, { operation: 'ReferencePosition' });
      g.edge('r_position', 'r_pos', 'position');
      g.edge('r_orientation', 'r_pos', 'orientation');
      g.node('r_pos_emb', 'FourierEmbedding', 'r_pos_emb', 1970, 1110, {
        input_dim: 3,
        hidden_dim: d,
        num_freq_bands: 64,
      });
      g.edge('r_pos', 'r_pos_emb');
      g.node('r_emb_add', 'Add', 'r_emb + position', 2640, 160);
      g.edge('r_emb_view', 'r_emb_add', 'a');
      g.edge('r_pos_emb', 'r_emb_add', 'b');
      g.node('r_unsqueeze', 'Unsqueeze', 'r_emb.unsqueeze(2)', 2940, 160, { dim: 2 });
      g.edge('r_emb_add', 'r_unsqueeze');
      g.node('r_repeat', 'Repeat', 'r_emb.repeat', 3240, 160, { repeats: `1,1,${m},1` });
      g.edge('r_unsqueeze', 'r_repeat');
      g.node('m_emb', 'PlutoModeParameter', 'm_emb', 2940, 600, { shape: `1,1,${m},${d}` });
      g.node('m_repeat', 'Repeat', 'm_emb.repeat(bs,R,1,1)', 3240, 600, { repeats: 'd0,d1,1,1' });
      g.edge('m_emb', 'm_repeat');
      g.edge('bs', 'm_repeat', 'd0');
      g.edge('R', 'm_repeat', 'd1');
      g.node('q_cat', 'Concat', 'reference / mode', 3550, 300, { dim: -1, input_count: 2 });
      g.edge('r_repeat', 'q_cat', 'a');
      g.edge('m_repeat', 'q_cat', 'b');
      g.node('q_proj', 'StateLinear', 'q_proj', 3860, 300, {
        in_features: 2 * d,
        out_features: d,
        bias: true,
      });
      g.edge('q_cat', 'q_proj');
      g.node('m_pos', 'PlutoModeParameter', 'm_pos', 3860, 900, { shape: `1,${m},${d}` });
      let last = 'q_proj';
      for (let i = 0; i < depth; i++) {
        const id = `decoder_block_${i}`,
          x = 4220 + i * 680;
        g.node(id, 'PlutoDecoderLayer', `decoder_blocks[${i}]`, x, 300, {
          dim: d,
          num_heads: heads,
          dropout: p.dropout,
        });
        g.edge(last, id, 'tgt');
        g.edge('enc_emb', id, 'memory');
        g.edge('r_padding', id, 'tgt_key_padding_mask');
        g.edge('enc_key_padding_mask', id, 'memory_key_padding_mask');
        g.edge('m_pos', id, 'm_pos');
        g.node(`finite_${i}`, 'PlutoCode', 'isfinite(q)', x + 340, 300, {
          operation: 'CheckFinite',
        });
        g.edge(id, `finite_${i}`);
        last = `finite_${i}`;
      }
      const end = 4220 + depth * 680;
      if (p.cat_x) {
        g.node('ego_repeat', 'PlutoCode', 'ego → [B,R,M,D]', end, 900, { operation: 'CatEgo' });
        g.edge('enc_emb', 'ego_repeat', 'memory');
        g.edge(last, 'ego_repeat', 'like');
        g.node('ego_cat', 'Concat', 'q / ego', end, 300, { dim: -1, input_count: 2 });
        g.edge(last, 'ego_cat', 'a');
        g.edge('ego_repeat', 'ego_cat', 'b');
        g.node('cat_x_proj', 'StateLinear', 'cat_x_proj', end + 320, 300, {
          in_features: 2 * d,
          out_features: d,
          bias: true,
        });
        g.edge('ego_cat', 'cat_x_proj');
        last = 'cat_x_proj';
      }
      const base = end + (p.cat_x ? 680 : 0);
      for (const [i, owner] of ['bs', 'R'].entries()) {
        const n = g.node(`${owner}_head`, 'LocalParameter', owner, base + 350 + i * 250, -350);
        n.valueSource = owner;
      }
      ['loc', 'yaw', 'vel', 'pi'].forEach((part, i) => {
        const y = i * 320;
        g.node(`${part}_head`, 'MapMLP', `${part}_head`, base + 350, y, {
          in_features: d,
          hidden_features: part === 'pi' ? d : 2 * d,
          out_features: part === 'pi' ? 1 : t * 2,
          norm: 'ln',
        });
        g.edge(last, `${part}_head`);
        if (part === 'pi') {
          g.node('pi', 'Squeeze', 'pi', base + 690, y, { dim: -1 });
          g.edge('pi_head', 'pi');
        } else {
          g.node(part, 'View', part, base + 690, y, { shape: `d0,d1,${m},${t},2` });
          g.edge(`${part}_head`, part);
          g.edge('bs_head', part, 'd0');
          g.edge('R_head', part, 'd1');
        }
      });
      g.node('traj', 'Concat', 'traj', base + 1020, 300, { dim: -1, input_count: 3 });
      ['loc', 'yaw', 'vel'].forEach((name, i) => g.edge(name, 'traj', String.fromCharCode(97 + i)));
      g.outlet('trajectory', 'traj', base + 1350, 300, 'trajectory');
      g.outlet('probability', 'pi', base + 1350, 960, 'probability');
      // Wrap the decoder loop onto the next row so the overview stays navigable.
      for (const n of g.graph.nodes) {
        const block = /^(decoder_block|finite)_(\d+)$/.exec(n.id);
        if (block)
          n.position = {
            x: 2300 + Number(block[2]) * 580 + (block[1] === 'finite' ? 300 : 0),
            y: 2260,
          };
        else if (['ego_repeat', 'ego_cat', 'cat_x_proj'].includes(n.id)) {
          n.position.x -= end - (2300 + depth * 580);
          n.position.y += 1960;
        } else if (
          [
            'bs_head',
            'R_head',
            'loc_head',
            'yaw_head',
            'vel_head',
            'pi_head',
            'loc',
            'yaw',
            'vel',
            'pi',
            'traj',
            'trajectory',
            'probability',
          ].includes(n.id)
        ) {
          n.position.x -= base - (2300 + depth * 580 + (p.cat_x ? 680 : 0));
          n.position.y += 1960;
        }
      }
      return g.graph;
    },
    [
      { id: 'data', label: 'data' },
      { id: 'enc_data', label: 'enc_data' },
    ],
  );
  register(
    'ReferencePlanningBranch',
    'if ref_line_available',
    [bool('available', '参考线存在', true), ...decoderFields],
    (p) => {
      const g = builder();
      g.inlet('data', 0, 0);
      g.inlet('enc_data', 0, 280);
      g.inlet('available', 0, 560);
      g.node('gate', 'ReferenceGate', '检查当前分支', 300, 0, { available: p.available });
      g.edge('data', 'gate', 'data');
      g.edge('available', 'gate', 'available');
      if (p.available) {
        g.node(
          'planning_decoder',
          'PlutoPlanningDecoder',
          'planning_decoder',
          650,
          100,
          Object.fromEntries(Object.entries(p).filter(([key]) => key !== 'available')),
        );
        g.edge('gate', 'planning_decoder', 'data');
        g.edge('enc_data', 'planning_decoder', 'enc_data');
        g.outlet('trajectory', 'planning_decoder', 1000, 0, 'trajectory', 'trajectory');
        g.outlet('probability', 'planning_decoder', 1000, 300, 'probability', 'probability');
      } else {
        g.node('none', 'NoReferenceOutputs', 'None, None', 650, 100);
        g.edge('gate', 'none');
        g.outlet('trajectory', 'none', 1000, 0, 'trajectory', 'trajectory');
        g.outlet('probability', 'none', 1000, 300, 'probability', 'probability');
      }
      return g.graph;
    },
    [
      { id: 'data', label: 'data' },
      { id: 'enc_data', label: 'enc_data' },
      { id: 'available', label: 'available' },
    ],
  );
}
