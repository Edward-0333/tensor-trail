import type { Graph, Params, TensorSpec } from '../core/types';
import type { ModuleRegistry } from './registry';
import { bool, input, num, output, select } from './primitives';
import { floating, rank, requireThat, same } from './math';

function builder(registry: ModuleRegistry) {
  const graph: Graph = { nodes: [], edges: [] };
  const node = (id: string, type: string, params: Params = {}, x = 0, y = 0, label?: string) => {
    const value = registry.create(type, params, id);
    value.position = { x, y };
    if (label) value.label = label;
    graph.nodes.push(value);
    return id;
  };
  const edge = (source: string, target: string, targetPort = 'in', sourcePort = 'out') =>
    graph.edges.push({
      id: `${source}_${sourcePort}_${target}_${targetPort}`,
      source,
      target,
      sourcePort,
      targetPort,
    });
  return { graph, node, edge };
}

function validateSequence(inputs: Record<string, TensorSpec>, p: Params): string[] {
  const x = inputs.in;
  rank(x, 3);
  floating(x);
  const warnings: string[] = [];
  same(x.shape[2], Number(p.d_model), '输入特征维度需要匹配 d_model', warnings);
  return warnings;
}
const validateAttention = (inputs: Record<string, TensorSpec>, p: Params) => {
  requireThat(
    Number(p.d_model) % Number(p.heads) === 0,
    `特征维度 ${p.d_model} 无法被 ${p.heads} 个注意力头整除。`,
  );
  return validateSequence(inputs, p);
};

export function installComposites(registry: ModuleRegistry) {
  registry.register({
    type: 'Block',
    label: '自定义模块',
    category: '内部',
    description: '保存的独立子网络。',
    fields: [],
    inputs: input,
    outputs: output,
    composite: true,
  });
  registry.register({
    type: 'SelfAttention',
    label: 'Self-Attention',
    category: 'Transformer',
    description: '多头自注意力，可展开到 Q/K/V 和矩阵乘法。',
    composite: true,
    fields: [
      num('d_model', '特征维度', 64),
      num('heads', '注意力头数', 8),
      bool('causal', '因果遮挡', false),
    ],
    inputs: input,
    outputs: output,
    validate: validateAttention,
    build: (p) => {
      const { graph, node, edge } = builder(registry),
        d = Number(p.d_model),
        heads = Number(p.heads);
      node('input', 'Inlet', {}, 0, 200, '输入 X');
      for (const [i, key] of ['q', 'k', 'v'].entries()) {
        node(
          key,
          'Linear',
          { in_features: d, out_features: d },
          250,
          i * 200,
          `${key.toUpperCase()} 投影`,
        );
        node(`${key}_heads`, 'SplitHeads', { heads }, 500, i * 200, `${key.toUpperCase()} 拆头`);
        edge('input', key);
        edge(key, `${key}_heads`);
      }
      node('transpose_k', 'Permute', { dims: '0, 1, 3, 2' }, 750, 200, 'K 转置');
      edge('k_heads', 'transpose_k');
      node('scores', 'MatMul', {}, 1000, 0, 'Q × Kᵀ');
      edge('q_heads', 'scores', 'a');
      edge('transpose_k', 'scores', 'b');
      node('scale', 'Scale', { head_dim: d / heads }, 1000, 200, '缩放分数');
      edge('scores', 'scale');
      let previous = 'scale';
      if (p.causal) {
        node('mask', 'CausalMask', {}, 1000, 400);
        edge(previous, 'mask');
        previous = 'mask';
      }
      node('weights', 'Softmax', { dim: -1 }, 750, 600, '注意力权重');
      edge(previous, 'weights');
      node('context', 'MatMul', {}, 500, 600, '权重 × V');
      edge('weights', 'context', 'a');
      edge('v_heads', 'context', 'b');
      node('merge', 'MergeHeads', {}, 250, 600);
      edge('context', 'merge');
      node('projection', 'Linear', { in_features: d, out_features: d }, 0, 600, '输出投影');
      edge('merge', 'projection');
      node('output', 'Outlet', {}, 0, 800, '注意力输出');
      edge('projection', 'output');
      return graph;
    },
  });
  registry.register({
    type: 'FFN',
    label: 'Feed Forward',
    category: 'Transformer',
    description: '可替换激活函数或使用 SwiGLU 门控前馈结构。',
    composite: true,
    fields: [
      num('d_model', '输入输出特征', 64),
      num('d_ff', '中间特征', 128),
      select('kind', '前馈结构', 'Standard', ['Standard', 'SwiGLU']),
      select('activation', '激活函数', 'GELU', ['GELU', 'ReLU', 'SiLU']),
    ],
    inputs: input,
    outputs: output,
    validate: validateSequence,
    build: (p) => {
      const { graph, node, edge } = builder(registry),
        d = Number(p.d_model),
        hidden = Number(p.d_ff);
      node('input', 'Inlet', {}, 0, 150);
      node('up', 'Linear', { in_features: d, out_features: hidden }, 250, 50, '升维投影');
      edge('input', 'up');
      node('activation', p.kind === 'SwiGLU' ? 'SiLU' : String(p.activation), {}, 500, 50);
      edge('up', 'activation');
      let previous = 'activation';
      if (p.kind === 'SwiGLU') {
        node('gate', 'Linear', { in_features: d, out_features: hidden }, 250, 300, '门控投影');
        edge('input', 'gate');
        node('gated', 'Mul', {}, 750, 150, '门控乘法');
        edge('activation', 'gated', 'a');
        edge('gate', 'gated', 'b');
        previous = 'gated';
      }
      node('down', 'Linear', { in_features: hidden, out_features: d }, 750, 400, '降维投影');
      edge(previous, 'down');
      node('output', 'Outlet', {}, 500, 400);
      edge('down', 'output');
      return graph;
    },
  });
  registry.register({
    type: 'Transformer',
    label: 'Transformer Block',
    category: 'Transformer',
    description: '可展开的 Transformer，支持归一化位置、RMSNorm 与 SwiGLU 变体。',
    composite: true,
    fields: [
      num('d_model', '特征维度', 64),
      num('heads', '注意力头数', 8),
      num('d_ff', '前馈中间维度', 128),
      select('order', '归一化位置', 'Pre-Norm', ['Pre-Norm', 'Post-Norm']),
      select('norm', '归一化类型', 'LayerNorm', ['LayerNorm', 'RMSNorm']),
      select('ffn', '前馈结构', 'Standard', ['Standard', 'SwiGLU']),
      select('activation', '激活函数', 'GELU', ['GELU', 'ReLU', 'SiLU']),
      bool('causal', '因果遮挡', false),
    ],
    inputs: input,
    outputs: output,
    validate: validateAttention,
    build: (p) => {
      const { graph, node, edge } = builder(registry),
        d = Number(p.d_model);
      node('input', 'Inlet', {}, 0, 100);
      node('norm1', String(p.norm), { features: d }, 250, 100, '归一化 1');
      node(
        'attention',
        'SelfAttention',
        { d_model: d, heads: p.heads, causal: p.causal },
        500,
        100,
      );
      node('residual1', 'Add', {}, 750, 100, '残差连接 1');
      node('norm2', String(p.norm), { features: d }, 750, 380, '归一化 2');
      node(
        'ffn',
        'FFN',
        { d_model: d, d_ff: p.d_ff, kind: p.ffn, activation: p.activation },
        500,
        380,
      );
      node('residual2', 'Add', {}, 250, 380, '残差连接 2');
      node('output', 'Outlet', {}, 0, 380);
      if (p.order === 'Pre-Norm') {
        edge('input', 'norm1');
        edge('norm1', 'attention');
        edge('input', 'residual1', 'a');
        edge('attention', 'residual1', 'b');
        edge('residual1', 'norm2');
        edge('norm2', 'ffn');
        edge('residual1', 'residual2', 'a');
        edge('ffn', 'residual2', 'b');
        edge('residual2', 'output');
      } else {
        edge('input', 'attention');
        edge('input', 'residual1', 'a');
        edge('attention', 'residual1', 'b');
        edge('residual1', 'norm1');
        edge('norm1', 'ffn');
        edge('norm1', 'residual2', 'a');
        edge('ffn', 'residual2', 'b');
        edge('residual2', 'norm2');
        edge('norm2', 'output');
      }
      return graph;
    },
  });
}
