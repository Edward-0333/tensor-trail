import type { ModelNode, ModuleDefinition, Params, TensorSequenceSpec } from '../core/types';
import { clone, isTensorSequenceSpec, tensorValue } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { requireThat } from './math';
import { input, num, output, select, text } from './primitives';

function settings(p: Params): { kind: TensorSequenceSpec['kind']; count: number } {
  requireThat(p.kind === 'tuple' || p.kind === 'list', '序列类型需要为 tuple 或 list。');
  requireThat(
    typeof p.input_count === 'number' &&
      Number.isInteger(p.input_count) &&
      p.input_count >= 0 &&
      p.input_count <= 16,
    '序列元素数量需要为 0-16 的整数。',
  );
  return { kind: p.kind, count: p.input_count };
}

function sequencePorts(p: Params) {
  // Keep the node editable while invalid parameters are being corrected.
  const count =
    typeof p.input_count === 'number' && Number.isFinite(p.input_count)
      ? Math.min(16, Math.max(0, Math.floor(p.input_count)))
      : 0;
  return {
    inputs: Array.from({ length: count }, (_, i) => ({ id: `item${i}`, label: `[${i}]` })),
    outputs: output,
  };
}

export function tensorSequenceCode(node: ModelNode): string | undefined {
  if (node.type === 'TensorListAppend')
    return 'items.append(item)\nout = items\n# 输出表示 append 后的列表状态；保留各元素 shape/梯度，不做 stack/concat。';
  if (node.type === 'TensorSequenceGet')
    return `out = items[${node.params.index}]\n# 读取 list/tuple 中已有的 Tensor，保留其形状、类型和梯度标注。\n# 不是 Tensor 轴索引，不移除 Tensor 的 batch 轴，也不 stack 或复制真实张量。`;
  if (node.type !== 'TensorSequencePack') return;
  try {
    const { kind, count } = settings(node.params);
    const items = Array.from({ length: count }, (_, i) => `item${i}`).join(', ');
    const literal = kind === 'list' ? `[${items}]` : `(${items}${count === 1 ? ',' : ''})`;
    return `out = ${literal}\n# ${count === 0 ? '初始化空容器，不分配 Tensor，也不代表已完成 append 或 reshape。' : '按顺序组成容器，保留每个 Tensor 的独立形状和梯度标注。'}\n# 不是 torch.stack 或 torch.cat，不执行张量计算或复制真实特征。\n# 用于返回已有 tuple 时，只表达元素与容器的关系，不代表源码又创建了一份 Tensor。`;
  } catch {
    return '# 请修正序列类型或元素数量。';
  }
}

export const tensorSequence: ModuleDefinition = {
  type: 'TensorSequencePack',
  label: 'Tensor 序列',
  category: '组合计算',
  description:
    '按顺序组织不同形状的 Tensor 为 tuple/list；数量为 0 时表示空容器。不是堆叠，不进行张量计算。',
  fields: [
    select('kind', '容器类型', 'tuple', ['tuple', 'list']),
    num('input_count', '元素数量', 4, 0),
  ],
  inputs: sequencePorts({ input_count: 4 }).inputs,
  outputs: output,
  getPorts: sequencePorts,
  inferValue: (inputs, p) => {
    const { kind, count } = settings(p);
    const items = Array.from({ length: count }, (_, i) => clone(tensorValue(inputs[`item${i}`])));
    return {
      outputs: { out: { kind, items } },
      notes: [
        `${kind} 容器，共 ${count} 个 Tensor；元素可以有不同的 shape 或 dtype。`,
        count === 0
          ? '这是空容器初始化；没有 Tensor 元素，不是形状为 [0] 的 Tensor。'
          : '只记录元素顺序及元数据；不是 stack/concat，不新增 Tensor 维度。',
        '容器本身没有 requires_grad，元素各自保留原有标注；不模拟 Python 对象身份或实际内存分配。',
      ],
    };
  },
};

export const tensorSequenceGet: ModuleDefinition = {
  type: 'TensorSequenceGet',
  label: 'Tensor 序列取元素',
  category: '索引与切片',
  description: '从 Tensor 的 list/tuple 中取一个元素，保留完整 Tensor 形状；不是 Tensor 轴索引。',
  fields: [{ ...text('index', '元素索引', '0'), maxLength: 256 }],
  inputs: input,
  outputs: output,
  inferValue: ({ in: source }, p, context) => {
    requireThat(source && isTensorSequenceSpec(source), '需要包含 Tensor 的 list 或 tuple。');
    const index = resolveIntegerExpression(p.index, context?.hyperparameters);
    const count = source.items.length;
    requireThat(index >= -count && index < count, `索引 ${index} 超出 ${count} 个元素的范围。`);
    return {
      outputs: { out: clone(tensorValue(source.items[index < 0 ? index + count : index])) },
      notes: [`读取 ${source.kind}[${index}]；不移除 Tensor 轴，不创建新的真实 Tensor。`],
    };
  },
};

export const tensorListAppend: ModuleDefinition = {
  type: 'TensorListAppend', label: 'append', category: '组合计算',
  description: '向 Tensor 列表追加一个元素，输出更新后的列表规格；不堆叠，不复制真实 Tensor。',
  fields: [], inputs: [{ id: 'items', label: 'list' }, { id: 'item', label: 'Tensor' }], outputs: output,
  inferValue: ({ items, item }) => {
    requireThat(items && isTensorSequenceSpec(items) && items.kind === 'list', 'append 需要 Tensor list。');
    requireThat(items.items.length < 256, 'append 结果最多支持 256 个 Tensor。');
    return { outputs: { out: { kind: 'list', items: [...clone(items.items), clone(tensorValue(item))] } },
      notes: ['列表状态更新，保留元素的独立形状、梯度和信息来源；不模拟 Python 对象身份。'] };
  },
};
