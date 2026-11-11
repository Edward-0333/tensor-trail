import type { Graph, ModelNode, Params } from '../core/types';
import { clone } from '../core/types';
import { ModuleRegistry } from './registry';
import { bool, input, num, output, select, text } from './primitives';
import { floating, rank, requireThat, tensor } from './math';
import { moduleReferenceName } from './moduleReference';

const weightFields = [
  num('num_embeddings', '查询数量', 1040),
  num('embedding_dim', '坐标维数', 3),
  select('dtype', '权重类型', 'float32', ['float16', 'float32', 'float64']),
  bool('requires_grad', '参数 requires_grad', true),
];

function weightSettings(p: Params) {
  for (const name of ['num_embeddings', 'embedding_dim'])
    requireThat(Number.isSafeInteger(p[name]) && Number(p[name]) > 0, `${name} 需要正整数。`);
  requireThat(['float16', 'float32', 'float64'].includes(String(p.dtype)), '权重需要浮点类型。');
  requireThat(typeof p.requires_grad === 'boolean', 'requires_grad 需要布尔值。');
}

function uniformSettings(p: Params) {
  requireThat(Number.isSafeInteger(p.start) && Number(p.start) >= 0, '起始列需要非负整数。');
  requireThat(
    Number.isSafeInteger(p.stop) && Number(p.stop) > Number(p.start),
    '结束列必须大于起始列。',
  );
  for (const name of ['low', 'high'])
    requireThat(
      typeof p[name] === 'string' &&
        String(p[name]).trim() !== '' &&
        Number.isFinite(Number(p[name])),
      `${name} 需要有限数值。`,
    );
  requireThat(Number(p.low) < Number(p.high), '均匀分布需要 low < high。');
}

export function embeddingInitializationCode(node: ModelNode): string | undefined {
  const p = node.params;
  if (node.type === 'EmbeddingXYZModuleInit')
    return `# OPUSHead._init_layers：仅构造阶段执行，不是 forward。\n${p.reference} = nn.Embedding(${p.num_embeddings}, 3, dtype=torch.${p.dtype})\nnn.init.uniform_(${p.reference}.weight[:, 0], 0, 1.1)\nnn.init.uniform_(${p.reference}.weight[:, 1:], 0, 1)${p.requires_grad === false ? `\n${p.reference}.weight.requires_grad_(False)` : ''}\nout = ${p.reference}  # 模块对象，不是 Tensor\n# forward 中 GetParameter 再读取已有 .weight；不调用 Embedding.forward。\n# 图中不生成随机数或重复初始化实际模型，checkpoint 可以覆盖初值。`;
  if (node.type === 'EmbeddingModule')
    return `${p.reference} = nn.Embedding(${p.num_embeddings}, ${p.embedding_dim}, dtype=torch.${p.dtype})${p.requires_grad === false ? `\n${p.reference}.weight.requires_grad_(False)` : ''}\nout = ${p.reference}  # nn.Embedding 模块对象\n# 构造时自动注册 weight (nn.Parameter)，默认 N(0,1) 初始化。\n# GetParameter 只读取已注册参数，不创建它。`;
  if (node.type === 'EmbeddingXYZInit')
    return `# OPUSHead._init_layers：仅在模型构造阶段执行，不是 forward。\nself.init_points = nn.Embedding(${p.num_embeddings}, 3, dtype=torch.${p.dtype})\nnn.init.uniform_(self.init_points.weight[:, 0], 0, 1.1)\nnn.init.uniform_(self.init_points.weight[:, 1:], 0, 1)${p.requires_grad === false ? '\nself.init_points.weight.requires_grad_(False)' : ''}\n\n# 模块对象是 nn.Embedding；out 端口展示它的 .weight (nn.Parameter)。\n# forward 直接读取已有权重，没有调用 self.init_points(indices)。\n# checkpoint 可以覆盖初始化值；图中不生成随机数、不重复初始化实际模型。`;
  if (node.type === 'EmbeddingWeight')
    return `# 构造阶段：创建模块及注册参数，而非查表调用。\nembedding = nn.Embedding(${p.num_embeddings}, ${p.embedding_dim}, dtype=torch.${p.dtype})${p.requires_grad === false ? '\nembedding.weight.requires_grad_(False)' : ''}\nout = embedding.weight  # nn.Parameter [num_embeddings, embedding_dim]\n# 默认权重初始化为 N(0,1)，SparseWorld 随后覆盖所有坐标列。\n# requires_grad 来自构造声明，不代表 .grad 已有数值或已经 backward。`;
  if (node.type === 'UniformInitColumns') {
    const selection =
      Number(p.stop) === Number(p.start) + 1 ? String(p.start) : `${p.start}:${p.stop}`;
    return `# 构造阶段；nn.init 内部禁用梯度记录，原地写入同一参数。\nnn.init.uniform_(weight[:, ${selection}], ${p.low}, ${p.high})\nout = weight\n# 输出表示完整参数的更新状态，不是新参数，也不是选中列的张量。\n# 参数的 requires_grad 不因初始化内部的 no_grad 而变成 False。`;
  }
}

export function installEmbeddingInitialization(registry: ModuleRegistry) {
  registry.register({
    type: 'EmbeddingModule',
    label: '构造 Embedding 模块',
    category: '基础层',
    description:
      '仅构造阶段：创建 nn.Embedding 模块并注册 weight；输出模块引用，不执行索引查表。用 GetParameter 显式读取参数。',
    fields: [text('reference', '模块引用路径', 'self.init_points'), ...weightFields],
    inputs: [],
    outputs: output,
    inferValue: (_, p) => {
      weightSettings(p);
      const reference = moduleReferenceName(p.reference);
      return {
        outputs: {
          out: {
            kind: 'module',
            reference,
            moduleType: 'nn.Embedding',
            parameters: {
              weight: {
                ...tensor(
                  [Number(p.num_embeddings), Number(p.embedding_dim)],
                  ['query', 'coordinate'],
                  String(p.dtype),
                ),
                autograd: { requiresGrad: Boolean(p.requires_grad), source: 'inferred' },
                parameterRef: { module: reference, name: 'weight' },
              },
            },
          },
        },
        notes: [
          '模块对象没有张量形状和梯度标志；参数表是声明的静态规格，不含真实权重数值。',
          '仅构造阶段执行；forward 读取已持有的参数，不重新构造模块。',
        ],
      };
    },
  });
  registry.register({
    type: 'EmbeddingWeight',
    label: '构造 Embedding / 注册 weight',
    category: '基础层',
    description:
      '构造 nn.Embedding，展示其注册的 nn.Parameter；不接收索引，不执行 Embedding.forward 查表。仅描述构造阶段。',
    fields: weightFields,
    inputs: [],
    outputs: output,
    infer: (_, p) => {
      weightSettings(p);
      return {
        outputs: {
          out: {
            ...tensor(
              [Number(p.num_embeddings), Number(p.embedding_dim)],
              ['query', 'coordinate'],
              String(p.dtype),
            ),
            autograd: { requiresGrad: Boolean(p.requires_grad), source: 'inferred' },
          },
        },
        notes: [
          '输出是模块的 .weight 参数，不是 nn.Embedding 模块对象或查表结果。',
          'requires_grad 来自构造声明；真实运行中若冻结参数，需更新此声明或使用运行快照。',
          '默认 N(0,1) 初始化；后续显式初始化或 checkpoint 可以覆盖权重。画布不生成数值。',
        ],
      };
    },
  });
  registry.register({
    type: 'UniformInitColumns',
    label: '坐标列均匀初始化',
    category: '基础层',
    description:
      '仅构造阶段：nn.init.uniform_ 原地初始化二维权重的选中列，输出完整权重状态，保持 requires_grad；不执行训练或生成随机数。',
    fields: [
      num('start', '起始列', 0, 0),
      num('stop', '结束列（不含）', 1),
      text('low', '均匀分布下界', '0'),
      text('high', '均匀分布上界', '1'),
    ],
    inputs: input,
    outputs: output,
    infer: ({ in: weight }, p) => {
      uniformSettings(p);
      rank(weight, 2);
      floating(weight);
      const width = weight.shape[1];
      requireThat(typeof width !== 'number' || Number(p.stop) <= width, '初始化列超出权重宽度。');
      return {
        outputs: { out: clone(weight) },
        warnings: typeof width === 'number' ? [] : ['符号权重宽度尚需确认初始化列范围。'],
        notes: [
          `仅将列 [${p.start}:${p.stop}] 初始化为 U(${p.low}, ${p.high})；其他列保持原值。`,
          '图中返回完整权重的更新状态；实际代码仍是同一个 Parameter，不新增参数对象。',
          '初始化写入不记录计算图，但不改变参数自身的 requires_grad。',
        ],
      };
    },
  });
  registry.register({
    type: 'EmbeddingXYZInit',
    label: 'self.init_points [构造/初始化]',
    category: '网络结构',
    description:
      'SparseWorld OPUSHead._init_layers：构造 Embedding(Q,3)，x 初始化 U(0,1.1)，y/z 初始化 U(0,1)。仅模型构造时执行；输出 .weight 而非模块对象，forward 只读取已存参数。',
    composite: true,
    fields: weightFields.filter((f) => f.key !== 'embedding_dim'),
    inputs: [],
    outputs: output,
    build: (p) => {
      weightSettings({ ...p, embedding_dim: 3 });
      const g: Graph = { nodes: [], edges: [] };
      const add = (id: string, type: string, label: string, x: number, params: Params = {}) => {
        const node: ModelNode = {
          ...registry.create(type, params, id),
          label,
          position: { x, y: 120 },
        };
        if (type === 'TensorVariable' || type === 'Outlet') node.tensorBorderColor = '#60a5fa';
        g.nodes.push(node);
      };
      add('embedding', 'EmbeddingWeight', 'self.init_points = nn.Embedding(Q, 3)', 40, {
        ...p,
        embedding_dim: 3,
      });
      add('registered_weight', 'TensorVariable', 'weight (Parameter)', 390);
      add('init_x', 'UniformInitColumns', 'weight[:, 0] ~ U(0, 1.1)', 720, {
        start: 0,
        stop: 1,
        low: '0',
        high: '1.1',
      });
      add('init_yz', 'UniformInitColumns', 'weight[:, 1:] ~ U(0, 1)', 1060, {
        start: 1,
        stop: 3,
        low: '0',
        high: '1',
      });
      add('initialized_weight', 'TensorVariable', '已初始化 weight (Parameter)', 1400);
      add('weight', 'Outlet', '持有 .weight，供 forward 读取', 1760, { port: 'out' });
      const chain = [
        'embedding',
        'registered_weight',
        'init_x',
        'init_yz',
        'initialized_weight',
        'weight',
      ];
      for (let i = 1; i < chain.length; i++)
        g.edges.push({
          id: `init_${i}`,
          source: chain[i - 1],
          sourcePort: 'out',
          target: chain[i],
          targetPort: 'in',
        });
      return g;
    },
  });
  registry.register({
    type: 'EmbeddingXYZModuleInit',
    label: 'self.init_points [构造/初始化]',
    category: '网络结构',
    description:
      '构造 Embedding(Q,3)，原地初始化 weight 后输出同一个模块引用；forward 通过 GetParameter 读取参数。不是 Embedding.forward。',
    composite: true,
    fields: [
      text('reference', '模块引用路径', 'self.init_points'),
      ...weightFields.filter((f) => f.key !== 'embedding_dim'),
    ],
    inputs: [],
    outputs: output,
    build: (p) => {
      const { reference, ...weightParams } = p;
      moduleReferenceName(reference);
      const g = registry.create('EmbeddingXYZInit', weightParams).subgraph!;
      const embedding = g.nodes.find((n) => n.id === 'embedding')!;
      g.nodes[0] = {
        ...registry.create('EmbeddingModule', { ...p, embedding_dim: 3 }, 'embedding'),
        label: embedding.label,
        position: embedding.position,
      };
      g.nodes.push({
        ...registry.create('GetParameter', { name: 'weight' }, 'get_weight'),
        position: { x: 200, y: -120 },
      });
      g.nodes.push({
        ...registry.create('ModuleParameterState', { name: 'weight' }, 'initialized_module'),
        position: { x: 1400, y: 380 },
      });
      g.nodes.push({
        ...registry.create('ValueVariable', {}, 'module_ref'),
        label: 'self.init_points',
        position: { x: 40, y: 380 },
      });
      const outlet = g.nodes.find((n) => n.id === 'weight')!;
      outlet.label = 'self.init_points（已初始化模块）';
      delete outlet.tensorBorderColor;
      g.edges = g.edges.filter((e) => e.id !== 'init_1' && e.id !== 'init_5');
      const connect = (source: string, target: string, targetPort = 'in') =>
        g.edges.push({
          id: `module_init_${g.edges.length}`,
          source,
          sourcePort: 'out',
          target,
          targetPort,
        });
      connect('embedding', 'get_weight');
      connect('get_weight', 'registered_weight');
      connect('embedding', 'module_ref');
      connect('module_ref', 'initialized_module', 'module');
      connect('initialized_weight', 'initialized_module', 'parameter');
      connect('initialized_module', 'weight');
      return g;
    },
  });
}
