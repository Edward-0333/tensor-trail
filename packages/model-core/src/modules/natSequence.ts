import type { Graph, ModelNode, Params, Field } from '../core/types';
import { ModuleRegistry } from './registry';
import { bool, input, output, num, text } from './primitives';
import { numericList, requireThat } from './math';
import { probability } from './sequenceOps';

export const natCompositeTypes = new Set([
  'NATSequenceEncoder',
  'ConvTokenizer',
  'ConvDownsampler',
  'NATBlock',
  'NATLayer',
  'NATMlp',
]);
const rate = (key: string, label: string, value = '0.0') => text(key, label, value);
const layerFields = [
  num('dim', '特征维度', 32),
  num('num_heads', '注意力头数', 2),
  num('kernel_size', '邻域大小', 3),
  num('mlp_ratio', 'MLP 扩展倍数', 3),
  rate('drop', 'Dropout'),
  rate('attn_drop', 'Attention dropout'),
  rate('drop_path', 'DropPath'),
];
function positive(p: Params, key: string, max = 4096) {
  const n = Number(p[key]);
  requireThat(Number.isSafeInteger(n) && n > 0 && n <= max, `${key} 需要 1–${max} 的整数。`);
  return n;
}
function validateLayer(p: Params) {
  const dim = positive(p, 'dim'),
    heads = positive(p, 'num_heads'),
    kernel = positive(p, 'kernel_size', 31);
  requireThat(dim % heads === 0, 'dim 必须能被 num_heads 整除。');
  requireThat(kernel > 1 && kernel % 2 === 1, 'kernel_size 必须为大于 1 的奇数。');
  positive(p, 'mlp_ratio', 16);
  probability(p.drop);
  probability(p.attn_drop);
  probability(p.drop_path);
}

export function natSource(node: ModelNode): string | undefined {
  if (node.type === 'NATConv1d') {
    const p = node.params;
    return `# 初始化\nself.conv = nn.Conv1d(\n    ${p.in_channels}, ${p.out_channels}, kernel_size=${p.kernel_size},\n    stride=${p.stride}, padding=${p.padding},\n    dilation=${p.dilation}, groups=${p.groups}, bias=${p.bias ? 'True' : 'False'},\n)\n\ndef forward(self, x):\n    return self.conv(x)`;
  }
  if (node.type === 'NATSequenceEncoder')
    return `# 来源：pluto/src/models/pluto/layers/embedding.py\n# 当前展开 out_indices=[0,1,2]；各 level 深度通过节点参数配置。\ndef forward(self, x):\n    x = self.embed(x)\n    out = []\n    for idx, level in enumerate(self.levels):\n        x, xo = level(x)\n        x_out = getattr(self, f"norm{idx}")(xo)\n        out.append(x_out.permute(0, 2, 1).contiguous())\n    laterals = [conv(out[i]) for i, conv in enumerate(self.lateral_convs)]\n    for i in range(len(out) - 1, 0, -1):\n        laterals[i - 1] = laterals[i - 1] + F.interpolate(\n            laterals[i],\n            scale_factor=laterals[i - 1].shape[-1] / laterals[i].shape[-1],\n            mode="linear", align_corners=False,\n        )\n    out = self.fpn_conv(laterals[0])\n    return out[:, :, -1]`;
  if (node.type === 'NATBlock')
    return `def forward(self, x):\n    for blk in self.blocks:\n        x = blk(x)\n    if self.downsample is None:\n        return x, x\n    return self.downsample(x), x`;
  if (node.type === 'NATLayer')
    return `def forward(self, x):\n    shortcut = x\n    x = self.norm1(x)\n    x = self.attn(x)\n    x = shortcut + self.drop_path(x)\n    x = x + self.drop_path(self.mlp(self.norm2(x)))\n    return x`;
  if (node.type === 'ConvTokenizer')
    return `def forward(self, x):\n    # PLUTO 当前构造时 norm_layer=None\n    return self.proj(x).permute(0, 2, 1)`;
  if (node.type === 'ConvDownsampler')
    return `def forward(self, x):\n    x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)\n    return self.norm(x)`;
  if (node.type === 'NATMlp')
    return `def forward(self, x):\n    x = self.fc1(x)\n    x = self.act(x)\n    x = self.drop(x)\n    x = self.fc2(x)\n    return self.drop(x)`;
}

export function installNAT(registry: ModuleRegistry) {
  function graphBuilder() {
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
        target,
        sourcePort,
        targetPort,
      });
    const inlet = () => node('x', 'Inlet', 'x', 0, 120, { port: 'in' });
    const outlet = (source: string, x: number, y: number) => {
      node('out', 'Outlet', 'out', x, y, { port: 'out' });
      edge(source, 'out');
    };
    return { graph, node, edge, inlet, outlet };
  }
  function register(
    type: string,
    label: string,
    fields: Field[],
    build: (p: Params) => Graph,
    outputs = output,
  ) {
    registry.register({
      type,
      label,
      fields,
      inputs: input,
      outputs,
      composite: true,
      category: type === 'NATSequenceEncoder' ? '网络结构' : '内部',
      description:
        '按 PLUTO 的 forward() 保持模块封装；双击逐层查看内部数据流。参数修改会重新生成内部结构；当前是形状与结构展示。',
      build,
    });
  }
  const conv = registry.get('Conv1d')!;
  registry.register({
    ...conv,
    type: 'NATConv1d',
    label: 'Conv1d',
    category: '内部',
    fields: [...conv.fields, bool('bias', '偏置', true)],
    description: 'NAT 内部的一维卷积，显式保留 bias 设置；形状规则与 Conv1d 相同。',
  });

  register(
    'ConvTokenizer',
    'ConvTokenizer',
    [num('in_chans', '输入通道', 9), num('embed_dim', '嵌入维度', 32)],
    (p) => {
      const g = graphBuilder();
      g.inlet();
      g.node('proj', 'NATConv1d', 'proj', 240, 100, {
        in_channels: positive(p, 'in_chans'),
        out_channels: positive(p, 'embed_dim'),
        kernel_size: 3,
        stride: 1,
        padding: 1,
        bias: true,
      });
      g.node('permute', 'Permute', 'permute(0, 2, 1)', 530, 100, { dims: '0,2,1' });
      g.edge('x', 'proj');
      g.edge('proj', 'permute');
      g.outlet('permute', 820, 120);
      return g.graph;
    },
  );
  register('ConvDownsampler', 'ConvDownsampler', [num('dim', '输入特征维度', 32)], (p) => {
    const dim = positive(p, 'dim'),
      g = graphBuilder();
    g.inlet();
    g.node('to_channels', 'Permute', 'permute(0, 2, 1)', 220, 100, { dims: '0,2,1' });
    g.node('reduction', 'NATConv1d', 'reduction', 500, 100, {
      in_channels: dim,
      out_channels: 2 * dim,
      kernel_size: 3,
      stride: 2,
      padding: 1,
      bias: false,
    });
    g.node('to_sequence', 'Permute', 'permute(0, 2, 1)', 780, 100, { dims: '0,2,1' });
    g.node('norm', 'LayerNorm', 'norm', 1060, 100, { features: 2 * dim });
    g.edge('x', 'to_channels');
    g.edge('to_channels', 'reduction');
    g.edge('reduction', 'to_sequence');
    g.edge('to_sequence', 'norm');
    g.outlet('norm', 1360, 120);
    return g.graph;
  });
  register(
    'NATMlp',
    'Mlp',
    [num('dim', '输入输出维度', 32), num('mlp_ratio', '扩展倍数', 3), rate('drop', 'Dropout')],
    (p) => {
      const dim = positive(p, 'dim'),
        ratio = positive(p, 'mlp_ratio', 16),
        g = graphBuilder();
      probability(p.drop);
      g.inlet();
      g.node('fc1', 'Linear', 'fc1', 220, 100, { in_features: dim, out_features: dim * ratio });
      g.node('act', 'GELU', 'act', 500, 100);
      g.node('drop1', 'Dropout', 'drop · 第一次调用', 780, 100, { p: p.drop });
      g.node('fc2', 'Linear', 'fc2', 1060, 100, { in_features: dim * ratio, out_features: dim });
      g.node('drop2', 'Dropout', 'drop · 第二次调用', 1340, 100, { p: p.drop });
      ['x', 'fc1', 'act', 'drop1', 'fc2', 'drop2']
        .slice(1)
        .forEach((id, i, ids) => g.edge(i ? ids[i - 1] : 'x', id));
      g.outlet('drop2', 1630, 120);
      return g.graph;
    },
  );
  register('NATLayer', 'NATLayer', layerFields, (p) => {
    validateLayer(p);
    const g = graphBuilder();
    g.inlet();
    g.node('norm1', 'LayerNorm', 'norm1', 230, 100, { features: p.dim });
    g.node('attn', 'CodeFunction', 'NeighborhoodAttention1D', 520, 110, {
      operation: 'NeighborhoodAttention1D',
      dim: p.dim,
      num_heads: p.num_heads,
      kernel_size: p.kernel_size,
      attn_drop: p.attn_drop,
      proj_drop: p.drop,
    });
    g.node('drop_path1', 'CodeFunction', 'drop_path · 第一次调用', 850, 110, {
      operation: 'DropPath',
      p: p.drop_path,
    });
    g.node('residual1', 'Add', 'shortcut + 分支', 1130, 110);
    g.node('norm2', 'LayerNorm', 'norm2', 1440, 100, { features: p.dim });
    g.node('mlp', 'NATMlp', 'mlp', 1730, 100, { dim: p.dim, mlp_ratio: p.mlp_ratio, drop: p.drop });
    g.node('drop_path2', 'CodeFunction', 'drop_path · 第二次调用', 2030, 110, {
      operation: 'DropPath',
      p: p.drop_path,
    });
    g.node('residual2', 'Add', 'x + 分支', 2310, 110);
    g.edge('x', 'norm1');
    g.edge('norm1', 'attn');
    g.edge('attn', 'drop_path1');
    g.edge('x', 'residual1', 'a');
    g.edge('drop_path1', 'residual1', 'b');
    g.edge('residual1', 'norm2');
    g.edge('norm2', 'mlp');
    g.edge('mlp', 'drop_path2');
    g.edge('residual1', 'residual2', 'a');
    g.edge('drop_path2', 'residual2', 'b');
    g.graph.edges.find((e) => e.source === 'x' && e.target === 'residual1')!.waypoints = [
      { x: 160, y: -80 },
      { x: 1100, y: -80 },
    ];
    g.graph.edges.find((e) => e.source === 'residual1' && e.target === 'residual2')!.waypoints = [
      { x: 1380, y: -80 },
      { x: 2280, y: -80 },
    ];
    g.outlet('residual2', 2610, 120);
    return g.graph;
  });
  register(
    'NATBlock',
    'NATBlock',
    [
      ...layerFields.filter((f) => f.key !== 'drop_path'),
      num('depth', 'NATLayer 数量', 2),
      text('drop_paths', '各层 DropPath', '0.0,0.04'),
      bool('downsample', '下采样', true),
    ],
    (p) => {
      const depth = positive(p, 'depth', 8);
      requireThat(typeof p.drop_paths === 'string', 'drop_paths 需要逗号分隔的概率。');
      const rates = p.drop_paths.split(',').map((v) => v.trim());
      requireThat(rates.length === depth, 'DropPath 概率数量必须等于 depth。');
      rates.forEach(probability);
      const g = graphBuilder();
      g.inlet();
      let last = 'x';
      for (let i = 0; i < depth; i++) {
        const id = `block${i}`;
        g.node(id, 'NATLayer', `blocks[${i}]`, 240 + i * 300, 100, {
          dim: p.dim,
          num_heads: p.num_heads,
          kernel_size: p.kernel_size,
          mlp_ratio: p.mlp_ratio,
          drop: p.drop,
          attn_drop: p.attn_drop,
          drop_path: rates[i],
        });
        g.edge(last, id);
        last = id;
      }
      const right = 260 + depth * 300;
      const beforeDownsample = last;
      if (p.downsample) {
        g.node('downsample', 'ConvDownsampler', 'downsample', right, 100, { dim: p.dim });
        g.edge(last, 'downsample');
        last = 'downsample';
      }
      g.node('next', 'Outlet', 'x · 下一层输入', right + 320, 120, { port: 'next' });
      g.edge(last, 'next');
      g.node('xo', 'Outlet', 'xo · 下采样前', right, 360, { port: 'xo' });
      g.edge(beforeDownsample, 'xo');
      return g.graph;
    },
    [
      { id: 'next', label: 'x' },
      { id: 'xo', label: 'xo' },
    ],
  );

  register(
    'NATSequenceEncoder',
    'NATSequenceEncoder',
    [
      num('in_chans', '输入通道', 9),
      num('embed_dim', '首层维度', 32),
      num('mlp_ratio', 'MLP 扩展倍数', 3),
      text('depths', '三级深度', '2,2,2'),
      text('num_heads', '三级注意力头数', '2,4,8'),
      text('kernel_size', '三级邻域大小', '3,3,5'),
      rate('drop_rate', 'Dropout'),
      rate('attn_drop_rate', 'Attention dropout'),
      rate('drop_path_rate', '最大 DropPath', '0.2'),
    ],
    (p) => {
      const embed = positive(p, 'embed_dim', 1024),
        inChans = positive(p, 'in_chans'),
        ratio = positive(p, 'mlp_ratio', 16);
      const depths = numericList(p.depths),
        heads = numericList(p.num_heads),
        kernels = numericList(p.kernel_size);
      requireThat(
        [depths, heads, kernels].every((xs) => xs.length === 3),
        '当前实现对应 PLUTO 的三级编码器，depths、num_heads、kernel_size 均需三个值。',
      );
      requireThat(
        depths.every((d) => d > 0 && d <= 8),
        '每级深度必须为 1–8。',
      );
      probability(p.drop_rate);
      probability(p.attn_drop_rate);
      const maxDrop = probability(p.drop_path_rate);
      const total = depths.reduce((sum, d) => sum + d, 0),
        g = graphBuilder();
      g.inlet();
      g.node('embed', 'ConvTokenizer', 'embed', 220, 100, { in_chans: inChans, embed_dim: embed });
      g.edge('x', 'embed');
      let cursor = 0;
      for (let i = 0; i < 3; i++) {
        const x = 540 + i * 480,
          dim = embed * 2 ** i;
        const rates = Array.from({ length: depths[i] }, () =>
          String(Number(((maxDrop * cursor++) / (total - 1)).toPrecision(8))),
        );
        g.node(`level${i}`, 'NATBlock', `levels[${i}]`, x, 100, {
          dim,
          depth: depths[i],
          num_heads: heads[i],
          kernel_size: kernels[i],
          mlp_ratio: ratio,
          drop: p.drop_rate,
          attn_drop: p.attn_drop_rate,
          drop_paths: rates.join(','),
          downsample: i < 2,
        });
        g.edge(i ? `level${i - 1}` : 'embed', `level${i}`, 'in', i ? 'next' : 'out');
        g.node(`norm${i}`, 'LayerNorm', `norm${i}`, x, 340, { features: dim });
        g.edge(`level${i}`, `norm${i}`, 'in', 'xo');
        g.node(`permute${i}`, 'Permute', 'permute(0, 2, 1)', x, 540, { dims: '0,2,1' });
        g.edge(`norm${i}`, `permute${i}`);
        g.node(`contiguous${i}`, 'Contiguous', 'contiguous', x, 760);
        g.edge(`permute${i}`, `contiguous${i}`);
        g.node(`lateral${i}`, 'NATConv1d', `lateral_convs[${i}]`, x, 840, {
          in_channels: dim,
          out_channels: embed * 4,
          kernel_size: 3,
          padding: 1,
          stride: 1,
          bias: true,
        });
        g.edge(`contiguous${i}`, `lateral${i}`);
        // Vertical branches keep the main forward's module calls distinct from their internals.
        for (const id of [`norm${i}`, `permute${i}`, `contiguous${i}`, `lateral${i}`])
          g.graph.nodes.find((n) => n.id === id)!.portLayout = { input: 'top', output: 'bottom' };
      }
      g.node('upsample2', 'CodeFunction', '插值：level 2 → level 1', 1500, 1060, {
        operation: 'Interpolate1D',
      });
      g.edge('lateral2', 'upsample2');
      g.edge('lateral1', 'upsample2', 'like');
      g.node('fuse1', 'Add', 'laterals[1] + 上采样结果', 1020, 1240);
      g.edge('lateral1', 'fuse1', 'a');
      g.edge('upsample2', 'fuse1', 'b');
      g.node('upsample1', 'CodeFunction', '插值：融合 level 1 → level 0', 1020, 1470, {
        operation: 'Interpolate1D',
      });
      g.edge('fuse1', 'upsample1');
      g.edge('lateral0', 'upsample1', 'like');
      g.node('fuse0', 'Add', 'laterals[0] + 上采样结果', 540, 1630);
      g.edge('lateral0', 'fuse0', 'a');
      g.edge('upsample1', 'fuse0', 'b');
      g.node('fpn_conv', 'NATConv1d', 'fpn_conv', 900, 1810, {
        in_channels: embed * 4,
        out_channels: embed * 4,
        kernel_size: 3,
        padding: 1,
        bias: true,
      });
      g.edge('fuse0', 'fpn_conv');
      g.node('last_step', 'TensorIndex', 'out[:, :, -1]', 1250, 1810, {
        mode: 'index',
        dim: '2',
        index: '-1',
      });
      g.edge('fpn_conv', 'last_step');
      g.outlet('last_step', 1580, 1810);
      return g.graph;
    },
  );
}
