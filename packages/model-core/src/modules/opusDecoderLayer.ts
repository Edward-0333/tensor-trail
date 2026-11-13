import type { Graph, Params } from '../core/types';
import type { ModuleRegistry } from './registry';
import { num, select, text } from './primitives';
import { requireThat } from './math';

function builder(registry: ModuleRegistry) {
  const graph: Graph = { nodes: [], edges: [] };
  const node = (id: string, type: string, label: string, x: number, y: number, params: Params = {}, subgraph?: Graph) => {
    const n = registry.create(type, params, id);
    Object.assign(n, { label, position: { x, y } });
    if (subgraph) { n.subgraph = subgraph; n.customized = true; }
    graph.nodes.push(n);
    return n;
  };
  const edge = (from: string, to: string) => {
    const [source, sourcePort = 'out'] = from.split('.'), [target, targetPort = 'in'] = to.split('.');
    graph.edges.push({ id: `e${graph.edges.length}`, source, sourcePort, target, targetPort });
  };
  const inlet = (id: string, x: number, y: number, label = id) => node(id, 'Inlet', label, x, y, { port: id });
  const outlet = (id: string, port: string, x: number, y: number, label = port) => node(id, 'Outlet', label, x, y, { port });
  const integer = (id: string, value: number, x: number, y: number, label = id) => node(id, 'IntegerParameter', label, x, y, { expression: String(value) });
  return { graph, node, edge, inlet, outlet, integer };
}

function mlp(registry: ModuleRegistry, D: number, role: 'position' | 'cls' | 'reg') {
  const g = builder(registry);
  g.inlet('in', 60, 100, role === 'position' ? 'flattened_points' : 'query_feat');
  g.inlet('features', 60, 380, role === 'position' ? '3 * last_refines（构造规格）' : '输出特征数（构造规格）');
  g.integer('D', D, 60, 600, 'embed_dims');
  let prev = 'in', x = 500;
  for (let i = 0; i < (role === 'position' ? 2 : 3); i++) {
    const dynamic = role === 'position' ? i === 0 : i === 2;
    const id = `linear${i}`;
    g.node(id, dynamic ? 'LinearByDimensions' : 'Linear', dynamic
      ? role === 'position' ? 'Linear(3*last_refines, D)' : 'Linear(D, output_features)'
      : `Linear(${D}, ${D})`, x, 100, dynamic ? {} : { in_features: D, out_features: D });
    g.edge(`${prev}.out`, `${id}.in`);
    if (dynamic) {
      g.edge(role === 'position' ? 'features.out' : 'D.out', `${id}.in_features`);
      g.edge(role === 'position' ? 'D.out' : 'features.out', `${id}.out_features`);
    }
    prev = id; x += 350;
    if (role === 'position' || i < 2) {
      if (role !== 'reg') {
        g.node(`norm${i}`, 'LayerNorm', `LayerNorm(${D})`, x, 100, { features: D });
        g.edge(`${prev}.out`, `norm${i}.in`); prev = `norm${i}`; x += 350;
      }
      g.node(`relu${i}`, 'ReLU', 'ReLU', x, 100);
      g.edge(`${prev}.out`, `relu${i}.in`); prev = `relu${i}`; x += 300;
    }
  }
  g.outlet('return', 'out', x, 100);
  g.edge(`${prev}.out`, 'return.in');
  return g.graph;
}

function refine(registry: ModuleRegistry, pcRange: string) {
  const g = builder(registry);
  g.inlet('query_points', 60, 100, 'points_proposal（归一化）');
  g.inlet('reg_offset', 60, 500, 'points_delta（已乘 scale）');
  g.inlet('P', 60, 800, 'self.num_refines');
  g.node('size', 'TensorSize', 'B, Q = points_delta.shape[:2]', 450, 800, { axes: '0,1', names: 'B,Q' });
  g.edge('reg_offset.out', 'size.in');
  g.node('view_delta', 'Reshape', 'reshape(B,Q,P,3)', 850, 500, { shape: 'd0,d1,d2,3' });
  g.edge('reg_offset.out', 'view_delta.in');
  g.edge('size.dim0', 'view_delta.d0'); g.edge('size.dim1', 'view_delta.d1'); g.edge('P.out', 'view_delta.d2');
  g.node('decode', 'PointRangeTransform', 'decode_points（转米制坐标）', 450, 100, { mode: 'decode', pc_range: pcRange });
  g.edge('query_points.out', 'decode.in');
  g.node('center', 'ReduceMean', 'mean(dim=2, keepdim=True)', 850, 100, { dim: 2, keepdim: true });
  g.edge('decode.out', 'center.in');
  g.node('add_delta', 'Add', '中心坐标 + 预测位移（广播）', 1250, 100);
  g.edge('center.out', 'add_delta.a'); g.edge('view_delta.out', 'add_delta.b');
  g.node('encode', 'PointRangeTransform', 'encode_points（转回归一化坐标）', 1600, 100, { mode: 'encode', pc_range: pcRange });
  g.edge('add_delta.out', 'encode.in');
  g.outlet('return', 'out', 2000, 100, 'refine_pt'); g.edge('encode.out', 'return.in');
  return g.graph;
}

export function installOPUSDecoderLayer(registry: ModuleRegistry) {
  registry.register({
    type: 'OPUSDecoderLayer', label: 'OPUSTransformerDecoderLayer', category: 'Transformer', composite: true,
    description: 'OPUS 解码层的 forward 数据流：位置编码、时空采样、混合、自注意力、FFN、分类与坐标细化；各轮独立参数。',
    fields: [num('embed_dims', '特征维数', 256), num('num_classes', '类别数', 17), num('num_frames', '帧数', 5),
      num('num_views', '相机数', 6), num('num_groups', '分组数', 4), num('num_points', '采样点数', 4), num('num_levels', '尺度数', 4),
      num('num_heads', '注意力头数', 8), text('refines', '各层预测点数', '1,4,16,24,32,48'),
      text('scale', '回归位移缩放', '0.5'), text('pc_range', 'xyz 范围', '-40,-40,-1,40,40,5.4'),
      select('layout', '图像布局', 'channel_last', ['channel_last', 'channel_first'])],
    inputs: ['query_points', 'query_feat', 'mlvl_feats', 'occ2img', 'img_metas', 'index'].map(id => ({ id, label: id })),
    outputs: ['query_feat', 'cls_score', 'refine_pts'].map(id => ({ id, label: id })),
    build: p => {
      const D = Number(p.embed_dims), classes = Number(p.num_classes), T = Number(p.num_frames), G = Number(p.num_groups);
      const counts = String(p.refines).split(',').map(s => Number(s.trim()));
      requireThat(counts.length > 0 && counts.length <= 16 && counts.every(n => Number.isSafeInteger(n) && n > 0), '各层点数需要 1-16 个正整数。');
      const scale = Number(p.scale); requireThat(String(p.scale).trim() !== '' && Number.isFinite(scale), 'scale 必须为有限数值。');
      const g = builder(registry);
      ['query_points', 'query_feat', 'mlvl_feats', 'occ2img', 'img_metas', 'index'].forEach((id, i) =>
        g.inlet(id, 60, 80 + i * 240, id === 'index' ? 'i（层配置索引，非 forward 数据）' : id));
      g.node('flatten', 'Flatten', 'query_points.flatten(2,3)', 500, 80, { start_dim: 2, end_dim: 3 });
      g.edge('query_points.out', 'flatten.in');
      g.node('position_encoder', 'Submodule', 'self.position_encoder · 两层 MLP', 850, 80, {}, mlp(registry, D, 'position'));
      g.edge('flatten.out', 'position_encoder.in');
      g.node('query_pos', 'TensorVariable', 'query_pos', 1200, 80); g.edge('position_encoder.out', 'query_pos.in');
      g.node('add_pos', 'Add', 'query_feat + query_pos', 1350, 350);
      g.edge('query_feat.out', 'add_pos.a'); g.edge('query_pos.out', 'add_pos.b');
      g.node('positioned_feat', 'TensorVariable', 'query_feat（加入位置编码）', 1650, 350); g.edge('add_pos.out', 'positioned_feat.in');
      g.node('sampling', 'OPUSSamplingContract', 'self.sampling · OPUSSampling', 2050, 350, {
        embed_dims: D, num_frames: T, num_views: p.num_views, num_groups: G,
        num_points: p.num_points, num_levels: p.num_levels, layout: p.layout,
      });
      ['query_points', 'mlvl_feats', 'occ2img', 'img_metas'].forEach(id => g.edge(`${id}.out`, `sampling.${id}`));
      g.edge('positioned_feat.out', 'sampling.query_feat');
      g.node('sampled_feat', 'TensorVariable', 'sampled_feat [B,Q,G,FP,C]', 2400, 350); g.edge('sampling.out', 'sampled_feat.in');
      g.node('mixing', 'AdaptiveMixingContract', 'self.mixing · 含 query 残差', 2800, 350, {
        embed_dims: D, num_groups: G, in_points: T * Number(p.num_points), out_points: 32,
      });
      g.edge('sampled_feat.out', 'mixing.sampled_feat'); g.edge('positioned_feat.out', 'mixing.query_feat');
      g.node('norm1', 'LayerNorm', 'self.norm1', 3150, 350, { features: D }); g.edge('mixing.out', 'norm1.in');
      g.node('mixed_feat', 'TensorVariable', 'query_feat（图像信息融合后）', 3500, 350); g.edge('norm1.out', 'mixed_feat.in');
      g.node('self_attn', 'OPUSSelfAttentionContract', 'self.self_attn · 距离偏置自注意力', 3900, 350, { embed_dims: D, num_heads: p.num_heads });
      g.edge('query_points.out', 'self_attn.query_points'); g.edge('mixed_feat.out', 'self_attn.query_feat');
      g.node('norm2', 'LayerNorm', 'self.norm2', 4250, 350, { features: D }); g.edge('self_attn.out', 'norm2.in');
      g.node('ffn', 'MMCVFFNContract', `self.ffn · ${D}→512→${D} · 含残差`, 4600, 350, { embed_dims: D, hidden_dims: 512, dropout: '0.1' });
      g.edge('norm2.out', 'ffn.in');
      g.node('norm3', 'LayerNorm', 'self.norm3', 4950, 350, { features: D }); g.edge('ffn.out', 'norm3.in');
      g.node('updated_feat', 'TensorVariable', 'query_feat（本层最终特征）', 5300, 350); g.edge('norm3.out', 'updated_feat.in');
      g.outlet('return_feat', 'query_feat', 5750, -170, 'return query_feat'); g.edge('updated_feat.out', 'return_feat.in');

      // Layer configuration is separate from value features and never creates source IDs.
      g.node('refines', 'IntegerList', 'num_refines（各层配置）', 500, 1380, { values: counts.join(',') });
      g.node('previous_refines', 'IntegerList', 'last_refines（各层配置）', 500, 1600, { values: [1, ...counts.slice(0, -1)].join(',') });
      g.node('select_P', 'IntegerListGet', 'num_refines[i]', 850, 1380);
      g.node('select_prev', 'IntegerListGet', 'last_refines[i]', 850, 1600);
      g.edge('refines.out', 'select_P.in'); g.edge('index.out', 'select_P.index');
      g.edge('previous_refines.out', 'select_prev.in'); g.edge('index.out', 'select_prev.index');
      g.node('P', 'LocalParameter', 'self.num_refines', 1150, 1380); g.edge('select_P.out', 'P.in');
      g.integer('xyz', 3, 1150, 1760, 'xyz = 3');
      g.integer('classes', classes, 1600, 1760, 'num_classes');
      g.node('position_dims', 'IntegerMultiply', '3 * last_refines', 1500, 1600);
      g.edge('select_prev.out', 'position_dims.a'); g.edge('xyz.out', 'position_dims.b'); g.edge('position_dims.out', 'position_encoder.features');
      g.node('cls_dims', 'IntegerMultiply', 'P * num_classes', 1900, 1380); g.edge('P.out', 'cls_dims.a'); g.edge('classes.out', 'cls_dims.b');
      g.node('reg_dims', 'IntegerMultiply', 'P * 3', 1900, 1600); g.edge('P.out', 'reg_dims.a'); g.edge('xyz.out', 'reg_dims.b');
      g.node('shape_BQ', 'TensorSize', 'B,Q = query_points.shape[:2]', 450, 950, { axes: '0,1', names: 'B,Q' }); g.edge('query_points.out', 'shape_BQ.in');
      g.node('B', 'LocalParameter', 'B', 800, 950); g.node('Q', 'LocalParameter', 'Q', 800, 1130);
      g.edge('shape_BQ.dim0', 'B.in'); g.edge('shape_BQ.dim1', 'Q.in');

      g.node('cls_branch', 'Submodule', 'self.cls_branch · 分类 MLP', 5750, 100, {}, mlp(registry, D, 'cls'));
      g.edge('updated_feat.out', 'cls_branch.in'); g.edge('cls_dims.out', 'cls_branch.features');
      g.node('cls_raw', 'TensorVariable', 'cls_score [B,Q,P*num_classes]', 6100, 100); g.edge('cls_branch.out', 'cls_raw.in');
      g.node('cls_view', 'View', 'view(B,Q,P,num_classes)', 6500, 100, { shape: 'd0,d1,d2,d3' }); g.edge('cls_raw.out', 'cls_view.in');
      for (const [i, src] of ['B', 'Q', 'P', 'classes'].entries()) {
        const alias = g.node(`${src}_cls`, src === 'classes' ? 'IntegerParameter' : 'LocalParameter', src, 6350, 350 + i * 130, src === 'classes' ? { expression: String(classes) } : {});
        alias.valueSource = src;
        g.edge(`${alias.id}.out`, `cls_view.d${i}`);
      }
      g.outlet('return_cls', 'cls_score', 6900, 100, 'return cls_score（logits）'); g.edge('cls_view.out', 'return_cls.in');
      g.node('reg_branch', 'Submodule', 'self.reg_branch · 位移 MLP', 5750, 1000, {}, mlp(registry, D, 'reg'));
      g.edge('updated_feat.out', 'reg_branch.in'); g.edge('reg_dims.out', 'reg_branch.features');
      g.node('scale_offset', 'Function', `self.scale * offset（${scale}）`, 6100, 1000, { argument: 'x', expression: `${scale} * x` }); g.edge('reg_branch.out', 'scale_offset.in');
      g.node('reg_offset', 'TensorVariable', 'reg_offset [B,Q,P*3]（米制位移）', 6500, 1000); g.edge('scale_offset.out', 'reg_offset.in');
      g.node('refine_points', 'Submodule', 'self.refine_points · 中心 + 位移', 6900, 1000, {}, refine(registry, String(p.pc_range)));
      g.edge('query_points.out', 'refine_points.query_points'); g.edge('reg_offset.out', 'refine_points.reg_offset'); g.edge('P.out', 'refine_points.P');
      g.outlet('return_points', 'refine_pts', 7300, 1000, 'return refine_pt（归一化坐标）'); g.edge('refine_points.out', 'return_points.in');
      return g.graph;
    },
  });
}
