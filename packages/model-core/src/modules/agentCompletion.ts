import type { Graph, ModelNode, Params, Field } from '../core/types';
import { clone } from '../core/types';
import { ModuleRegistry } from './registry';
import { bool, input, output, num, text, select } from './primitives';
import { dimensions, floating, rank, requireThat, same, tensor, axisIndex } from './math';

export const agentCompositeTypes = new Set([
  'EgoHistoryBranch',
  'StateAttentionEncoder',
  'EgoStateMLP',
]);
export const agentCodeTypes = new Set(['StateAttentionReadout']);
const widthFields = [num('state_channel', '状态通道', 6), num('dim', '特征维度', 128)];
const stateFields = [
  ...widthFields,
  text('state_dropout', '状态遮挡概率', '0.75'),
  bool('training', '训练模式', true),
];
function positive(p: Params, name: string, max = 4096) {
  const v = Number(p[name]);
  requireThat(Number.isSafeInteger(v) && v >= 1 && v <= max, `${name} 需要 1–${max} 的整数。`);
  return v;
}
function dropout(p: Params) {
  requireThat(
    typeof p.state_dropout === 'string' && p.state_dropout.trim() !== '',
    '状态遮挡概率需要 0–1 的数。',
  );
  const v = Number(p.state_dropout);
  requireThat(Number.isFinite(v) && v >= 0 && v <= 1, '状态遮挡概率需要满足 0 ≤ p ≤ 1。');
  return v;
}
function stateSettings(p: Params) {
  const channels = positive(p, 'state_channel', 64),
    dim = positive(p, 'dim'),
    drop = dropout(p);
  requireThat(dim % 4 === 0, 'StateAttentionEncoder 的 dim 必须能被固定的 4 个头整除。');
  requireThat(typeof p.training === 'boolean', 'training 必须是开关值。');
  requireThat(
    !p.training || drop === 0 || channels >= 3,
    '训练状态遮挡需要至少 3 个始终可见的通道。',
  );
  return { channels, dim, drop };
}

export function agentCompletionCode(n: ModelNode): string | undefined {
  const p = n.params;
  if (n.type === 'EgoHistoryBranch')
    return `# 当前配置快照：use_ego_history=${p.use_ego_history ? 'True' : 'False'}\n# 可在本节点切换；切换会重建当前分支，跳过分支不执行 ego_state_emb。\ndef forward(self, data, x_agent):\n    if not self.use_ego_history:\n        ego_feature = data["current_state"][:, : self.state_channel]\n        x_ego = self.ego_state_emb(ego_feature)\n        x_agent[:, 0] = x_ego\n    return x_agent`;
  if (n.type === 'StateAttentionEncoder')
    return 'def forward(self, x):\n    x_embed = []\n    for i, linear in enumerate(self.linears):\n        x_embed.append(linear(x[:, i, None]))\n    x_embed = torch.stack(x_embed, dim=1)\n    pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)\n    x_embed += pos_embed\n\n    if self.training and self.state_dropout > 0:\n        visible_tokens = torch.zeros(\n            (x_embed.shape[0], 3), device=x.device, dtype=torch.bool\n        )\n        dropout_tokens = (\n            torch.rand((x_embed.shape[0], self.state_channel - 3), device=x.device)\n            < self.state_dropout\n        )\n        key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)\n    else:\n        key_padding_mask = None\n\n    query = self.query.repeat(x_embed.shape[0], 1, 1)\n\n    x_state = self.attn(\n        query=query,\n        key=x_embed,\n        value=x_embed,\n        key_padding_mask=key_padding_mask,\n    )[0]\n\n    return x_state[:, 0]';
  if (n.type === 'StateAttentionReadout')
    return `# 初始化\nself.attn = nn.MultiheadAttention(embed_dim=${p.dim}, num_heads=4, batch_first=True)\nself.state_channel = ${p.state_channel}\nself.state_dropout = ${p.state_dropout}\n# 本次画布快照 training=${p.training ? 'True' : 'False'}，对应实际 model.train()/eval()。\n\ndef forward(self, query, x_embed):\n    if self.training and self.state_dropout > 0:\n        visible_tokens = torch.zeros((x_embed.shape[0], 3),\n                                     device=x_embed.device, dtype=torch.bool)\n        dropout_tokens = (torch.rand((x_embed.shape[0], self.state_channel - 3),\n                                     device=x_embed.device) < self.state_dropout)\n        key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)\n    else:\n        key_padding_mask = None\n    return self.attn(query=query, key=x_embed, value=x_embed,\n                     key_padding_mask=key_padding_mask)[0]`;
  if (n.type === 'Embedding')
    return `self.type_emb = nn.Embedding(${p.num_embeddings}, ${p.embedding_dim})\n\ndef forward(self, category):\n    return self.type_emb(category)  # 可训练的类别查表；不将类别数当作输出轴。`;
  if (n.type === 'IndexAssign')
    return `def forward(self, x, values):\n    index = [slice(None)] * x.ndim\n    index[${p.dim}] = ${p.index}\n    x[tuple(index)] = values\n    return x\n\n# x[:, 0] = x_ego：覆盖自车行，其他 agent 保持原值；不与旧自车行相加。`;
  if (n.type === 'LearnedParameter')
    return `self.parameter = nn.Parameter(torch.empty(${p.shape}, dtype=torch.${p.dtype}))\nnn.init.normal_(self.parameter, std=0.02)\n\ndef forward(self):\n    return self.parameter  # 参数值从训练权重获得，画布不填造随机数值。`;
  if (n.type === 'RepeatBatch')
    return 'def forward(self, parameter, like):\n    return parameter.repeat(like.shape[0], 1, 1)\n\n# 参数在 batch 维重复；梯度累积到同一份可训练参数。';
  if (n.type === 'StateLinear')
    return `self.linear = nn.Linear(${p.in_features}, ${p.out_features}, bias=${p.bias ? 'True' : 'False'})\n\ndef forward(self, x):\n    return self.linear(x)`;
  if (n.type === 'StateBatchNorm')
    return `self.norm = nn.BatchNorm1d(${p.features})\n\ndef forward(self, x):\n    return self.norm(x)`;
  if (n.type === 'EgoStateMLP')
    return `# state_attn_encoder=False 时使用的替代编码器\nself.net = nn.Sequential(\n    nn.Linear(${p.state_channel}, ${p.dim}, bias=False),\n    nn.BatchNorm1d(${p.dim}), nn.ReLU(),\n    nn.Linear(${p.dim}, ${p.dim}, bias=True),\n)\n\ndef forward(self, x):\n    return self.net(x)`;
}

export function installAgentCompletion(registry: ModuleRegistry) {
  registry.register({
    type: 'Embedding',
    label: 'Embedding',
    category: '基础层',
    description: '将整数类别索引映射为可训练向量；输出在输入形状末尾增加 embedding_dim。',
    fields: [num('num_embeddings', '类别数量', 4), num('embedding_dim', '嵌入维度', 128)],
    inputs: input,
    outputs: output,
    infer: ({ in: x }, p) => {
      positive(p, 'num_embeddings');
      const dim = positive(p, 'embedding_dim');
      requireThat(['int32', 'int64'].includes(x.dtype), 'Embedding 输入需要 int32/int64 索引。');
      return {
        outputs: { out: tensor([...x.shape, dim], [...x.axes, 'feature'], 'float32') },
        notes: [
          `权重 [${p.num_embeddings}, ${dim}]；当前记录 float32 模型权重。`,
          `运行时索引必须在 0…${Number(p.num_embeddings) - 1}，形状记录无法核查类别数值。`,
        ],
      };
    },
  });
  registry.register({
    type: 'IndexAssign',
    label: '整数索引赋值',
    category: '索引与切片',
    description: '在指定轴的一个整数位置写入 Tensor，其余位置不变；当前支持不广播的完整切面赋值。',
    fields: [num('dim', '操作轴', 1, 0), num('index', '索引', 0, 0)],
    inputs: [...input, { id: 'values', label: '写入值' }],
    outputs: output,
    infer: ({ in: x, values }, p) => {
      const axis = axisIndex(Number(p.dim), x.shape.length),
        index = Number(p.index);
      requireThat(Number.isSafeInteger(index) && index >= 0, '索引需为非负整数。');
      requireThat(
        typeof x.shape[axis] !== 'number' || index < Number(x.shape[axis]),
        '赋值索引越界。',
      );
      requireThat(values.dtype === x.dtype, '写入值和目标 dtype 必须一致。');
      rank(values, x.shape.length - 1);
      const warnings: string[] = [];
      x.shape
        .filter((_, i) => i !== axis)
        .forEach((d, i) => same(d, values.shape[i], '写入切面的形状必须匹配', warnings));
      return {
        outputs: { out: clone(x) },
        warnings,
        notes: ['输出表示赋值后的 Tensor；只覆盖指定切面，其余位置保持原值。'],
      };
    },
  });
  registry.register({
    type: 'LearnedParameter',
    label: '可训练参数',
    category: '基础层',
    description:
      'nn.Parameter：属于模型权重，接收反向梯度；不同于超参数或输入常量。当前使用正态初始化 std=0.02。',
    fields: [
      text('shape', '参数形状', '1, 1, 128'),
      select('dtype', '数据类型', 'float32', ['float16', 'float32', 'float64']),
    ],
    inputs: [],
    outputs: output,
    infer: (_, p) => {
      const shape = dimensions(p.shape);
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, i) => `axis${i}`),
            String(p.dtype),
          ),
        },
        notes: ['可训练模型参数；初始化 std=0.02，真实数值由训练权重决定。'],
      };
    },
  });
  registry.register({
    type: 'RepeatBatch',
    label: '按 batch 重复',
    category: '维度操作',
    description: '将 [1,L,D] 参数按参考 Tensor 的第 0 轴重复成 [B,L,D]。',
    fields: [],
    inputs: [...input, { id: 'like', label: 'batch 参考' }],
    outputs: output,
    infer: ({ in: x, like }) => {
      rank(x, 3);
      requireThat(like.shape.length >= 1, '参考需要 batch 轴。');
      const warnings: string[] = [];
      same(x.shape[0], 1, '待重复参数首轴必须为 1', warnings);
      return {
        outputs: {
          out: tensor([like.shape[0], ...x.shape.slice(1)], ['batch', ...x.axes.slice(1)], x.dtype),
        },
        warnings,
      };
    },
  });
  registry.register({
    type: 'StateAttentionReadout',
    label: '状态遮挡与注意力',
    category: '内部',
    description:
      '训练时前三个状态 token 始终可见，其余随机遮挡；eval 时 mask=None。query 对状态 tokens 做 4 头交叉注意力。',
    fields: stateFields,
    inputs: [
      { id: 'query', label: 'query' },
      { id: 'tokens', label: '状态 tokens' },
    ],
    outputs: output,
    infer: ({ query, tokens }, p) => {
      const { channels, dim, drop } = stateSettings(p);
      rank(query, 3);
      rank(tokens, 3);
      floating(query);
      floating(tokens);
      requireThat(query.dtype === tokens.dtype, 'query/tokens dtype 必须一致。');
      const warnings: string[] = [];
      same(query.shape[0], tokens.shape[0], 'query/tokens batch 必须一致', warnings);
      same(query.shape[1], 1, 'query 数量必须为 1', warnings);
      same(query.shape[2], dim, 'query 特征维度', warnings);
      same(tokens.shape[1], channels, '状态 token 数量', warnings);
      same(tokens.shape[2], dim, '状态特征维度', warnings);
      return {
        outputs: { out: clone(query) },
        warnings,
        notes: [
          p.training && drop > 0
            ? `训练分支：前三个 token mask=False，其余以 ${drop} 的概率遮挡；不在画布生成随机掩码。`
            : '推理或零遮挡概率：key_padding_mask=None。',
          '包括 nn.MultiheadAttention 的可训练 Q/K/V 和输出投影；无 causal mask，attention dropout=0。',
        ],
      };
    },
  });
  const linear = registry.get('Linear')!;
  registry.register({
    ...linear,
    type: 'StateLinear',
    category: '内部',
    fields: [...linear.fields, bool('bias', '偏置', true)],
  });
  registry.register({
    type: 'StateBatchNorm',
    label: 'BatchNorm1d',
    category: '内部',
    description: 'MLP 中的二维 [B,D] BatchNorm1d。',
    fields: [num('features', '特征数量', 128), bool('training', '训练模式', true)],
    inputs: input,
    outputs: output,
    infer: ({ in: x }, p) => {
      rank(x, 2);
      floating(x);
      const warnings: string[] = [];
      same(x.shape[1], positive(p, 'features'), 'BatchNorm 特征数量', warnings);
      requireThat(
        !p.training || typeof x.shape[0] !== 'number' || x.shape[0] !== 1,
        'BatchNorm1d 训练时 batch 不能为 1；空 batch 保留为空。',
      );
      return { outputs: { out: clone(x) }, warnings };
    },
  });
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
    const edge = (source: string, target: string, targetPort = 'in', sourcePort = 'out') => {
      const e = {
        id: `${source}_${sourcePort}_${target}_${targetPort}`,
        source,
        sourcePort,
        target,
        targetPort,
      };
      graph.edges.push(e);
      return e;
    };
    const inlet = (id: string, x: number, y: number) => node(id, 'Inlet', id, x, y, { port: id });
    const outlet = (source: string, x: number, y: number) => {
      node('out', 'Outlet', 'return', x, y, { port: 'out' });
      edge(source, 'out');
    };
    return { graph, node, edge, inlet, outlet };
  }
  const composite = (
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
      category: type === 'EgoStateMLP' ? '内部' : '网络结构',
      description:
        'PLUTO 配置快照对应的结构；双击查看内部。修改节点参数会重新生成内部图，模型超参数不会自动重建此结构。',
    });
  composite('StateAttentionEncoder', 'StateAttentionEncoder', stateFields, (p) => {
    const { channels, dim } = stateSettings(p),
      g = builder();
    g.inlet('in', 0, 200);
    for (let i = 0; i < channels; i++) {
      const y = i * 250;
      g.node(`select${i}`, 'TensorIndex', `x[:, ${i}]`, 260, y, {
        mode: 'index',
        dim: '1',
        index: String(i),
      });
      g.node(`unsqueeze${i}`, 'Unsqueeze', '[:, None]', 480, y, { dim: -1 });
      g.node(`linear${i}`, 'StateLinear', `linears[${i}]`, 700, y - 15, {
        in_features: 1,
        out_features: dim,
        bias: true,
      });
      g.edge('in', `select${i}`);
      g.edge(`select${i}`, `unsqueeze${i}`);
      g.edge(`unsqueeze${i}`, `linear${i}`);
    }
    const middle = (channels - 1) * 125;
    g.node('stack', 'Stack', 'x_embed', 1000, middle, { dim: 1, input_count: channels });
    const stackPorts = registry.get('Stack')!.getPorts!({ dim: 1, input_count: channels }).inputs;
    for (let i = 0; i < channels; i++) g.edge(`linear${i}`, 'stack', stackPorts[i].id);
    g.node('pos_embed', 'LearnedParameter', 'self.pos_embed', 1030, -330, {
      shape: `1, ${channels}, ${dim}`,
    });
    g.node('pos_repeat', 'RepeatBatch', 'pos_embed.repeat(B, 1, 1)', 1310, -110);
    g.edge('pos_embed', 'pos_repeat');
    g.edge('stack', 'pos_repeat', 'like');
    g.node('add_position', 'Add', 'x_embed + pos_embed', 1550, middle);
    g.edge('stack', 'add_position', 'a');
    g.edge('pos_repeat', 'add_position', 'b');
    g.node('query_parameter', 'LearnedParameter', 'self.query', 1540, -330, {
      shape: `1, 1, ${dim}`,
    });
    g.node('query_repeat', 'RepeatBatch', 'query.repeat(B, 1, 1)', 1840, -110);
    g.edge('query_parameter', 'query_repeat');
    g.edge('add_position', 'query_repeat', 'like');
    g.node('attn', 'StateAttentionReadout', '状态遮挡 + attn', 2140, middle, p);
    g.edge('query_repeat', 'attn', 'query');
    g.edge('add_position', 'attn', 'tokens');
    g.node('first_query', 'TensorIndex', 'x_state[:, 0]', 2440, middle, {
      mode: 'index',
      dim: '1',
      index: '0',
    });
    g.edge('attn', 'first_query');
    g.outlet('first_query', 2700, middle - 10);
    return g.graph;
  });
  composite(
    'EgoStateMLP',
    'ego_state_emb · MLP',
    [...widthFields, bool('training', '训练模式', true)],
    (p) => {
      const dim = positive(p, 'dim'),
        channels = positive(p, 'state_channel'),
        g = builder();
      g.inlet('in', 0, 100);
      g.node('linear0', 'StateLinear', '0 · Linear', 250, 100, {
        in_features: channels,
        out_features: dim,
        bias: false,
      });
      g.node('norm', 'StateBatchNorm', '1 · BatchNorm1d', 540, 100, {
        features: dim,
        training: p.training,
      });
      g.node('relu', 'ReLU', '2 · ReLU', 830, 100);
      g.node('linear1', 'StateLinear', '3 · Linear', 1120, 100, {
        in_features: dim,
        out_features: dim,
        bias: true,
      });
      ['linear0', 'norm', 'relu', 'linear1'].forEach((id, i, ids) =>
        g.edge(i ? ids[i - 1] : 'in', id),
      );
      g.outlet('linear1', 1410, 100);
      return g.graph;
    },
  );
  composite(
    'EgoHistoryBranch',
    'if not use_ego_history',
    [
      bool('use_ego_history', '使用自车历史（勾选后跳过替换）', false),
      bool('state_attn_encoder', '使用状态注意力编码器', true),
      ...stateFields,
    ],
    (p) => {
      requireThat(
        typeof p.use_ego_history === 'boolean' && typeof p.state_attn_encoder === 'boolean',
        '分支配置必须是开关值。',
      );
      const g = builder();
      g.inlet('data', 0, 20);
      g.inlet('x_agent', 0, 420);
      if (p.use_ego_history) {
        g.outlet('x_agent', 380, 420);
        return g.graph;
      }
      const channels = positive(p, 'state_channel', 64),
        dim = positive(p, 'dim');
      g.node('current_state', 'DictionaryGet', 'data["current_state"]', 250, 20, {
        path: 'current_state',
      });
      g.edge('data', 'current_state');
      g.node('state_slice', 'TensorIndex', '[:, :self.state_channel]', 540, 20, {
        mode: 'slice',
        dim: '1',
        start: '',
        stop: String(channels),
        step: 1,
      });
      g.edge('current_state', 'state_slice');
      g.node('ego_feature', 'TensorVariable', 'ego_feature', 830, 5);
      g.edge('state_slice', 'ego_feature');
      g.node(
        'ego_state_emb',
        p.state_attn_encoder ? 'StateAttentionEncoder' : 'EgoStateMLP',
        'ego_state_emb',
        1120,
        0,
        { state_channel: channels, dim, state_dropout: p.state_dropout, training: p.training },
      );
      g.edge('ego_feature', 'ego_state_emb');
      g.node('x_ego', 'TensorVariable', 'x_ego', 1450, 5);
      g.edge('ego_state_emb', 'x_ego');
      g.node('assign_ego', 'IndexAssign', 'x_agent[:, 0] = x_ego', 1740, 300, { dim: 1, index: 0 });
      g.edge('x_agent', 'assign_ego');
      g.edge('x_ego', 'assign_ego', 'values');
      g.outlet('assign_ego', 2040, 290);
      return g.graph;
    },
    [
      { id: 'data', label: 'data' },
      { id: 'x_agent', label: 'x_agent' },
    ],
  );
}
