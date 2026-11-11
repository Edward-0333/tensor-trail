import type {
  Graph,
  GraphResult,
  ModelNode,
  NodeResult,
  Port,
  TensorSpec,
  ValueSpec,
} from './types';
import { clone, isTensorSequenceSpec, isTensorSpec } from './types';

export const MAX_LOOP_ITEMS = 16;
const boundary = (graph: Graph, type: string) =>
  graph.nodes.filter((n) => n.type === type && n.valueSource === undefined);

export function checkForEachBody(graph: Graph): void {
  const inputs = boundary(graph, 'Inlet').map((n) => String(n.params.port));
  const outputs = boundary(graph, 'Outlet').map((n) => String(n.params.port));
  if (!inputs.includes('item')) throw Error('ForEach 循环体需要 item 输入。');
  if (inputs.some((p) => ['items', 'initial', 'out'].includes(p)))
    throw Error(
      'ForEach 的 items、initial、out 为外层保留端口；循环体使用 item、可选 index 和其他捕获输入。',
    );
  if (outputs.length !== 1 || outputs[0] !== 'out')
    throw Error('ForEach 循环体需要唯一的 out 输出，每轮输出一个 Tensor。');
}

export function forEachPorts(node: ModelNode): { inputs: Port[]; outputs: Port[] } {
  return {
    inputs: [
      { id: 'items', label: 'items (tuple/list)' },
      { id: 'initial', label: 'initial (list)' },
      ...(node.subgraph ? boundary(node.subgraph, 'Inlet') : [])
        .filter(
          (n) => !['item', 'index', 'items', 'initial', 'out'].includes(String(n.params.port)),
        )
        .map((n) => ({ id: String(n.params.port), label: n.label })),
    ],
    outputs: [{ id: 'out', label: 'collected (list)' }],
  };
}

export function inferForEach(
  body: Graph,
  inputs: Record<string, ValueSpec>,
  evaluate: (external: Record<string, ValueSpec>) => GraphResult,
): Omit<NodeResult, 'inputs'> {
  checkForEachBody(body);
  const sequence = inputs.items,
    initial = inputs.initial;
  if (!sequence || !isTensorSequenceSpec(sequence))
    throw Error('ForEach items 需要已知元素的 Tensor tuple/list。');
  if (!initial || !isTensorSequenceSpec(initial) || initial.kind !== 'list')
    throw Error('ForEach initial 需要 Tensor list，通常为 []。');
  if (sequence.items.length > MAX_LOOP_ITEMS)
    throw Error(`ForEach 静态推导最多支持 ${MAX_LOOP_ITEMS} 次迭代。`);
  if (initial.items.length + sequence.items.length > 256)
    throw Error('ForEach 静态收集结果最多支持 256 个 Tensor。');
  const captures = Object.fromEntries(
    Object.entries(inputs).filter(([key]) => !['items', 'initial'].includes(key)),
  );
  const collected: TensorSpec[] = clone(initial.items),
    iterations: GraphResult[] = [];
  const errors: string[] = [],
    warnings: string[] = [];
  for (const [i, item] of sequence.items.entries()) {
    const child = evaluate({ ...captures, item: clone(item), index: { kind: 'int', value: i } });
    iterations.push(child);
    const failures = [
      ...child.issues,
      ...Object.entries(child.nodes)
        .filter(([, r]) => r.status === 'error' || r.status === 'blocked')
        .map(([id, r]) => `${body.nodes.find((n) => n.id === id)?.label}: ${r.messages[0]}`),
    ];
    const value = child.outputs.out;
    if (!failures.length && (!value || !isTensorSpec(value)))
      failures.push('循环体 out 必须为一个 Tensor。');
    if (failures.length) errors.push(`迭代 ${i}: ${failures.slice(0, 3).join('；')}`);
    else collected.push(clone(value as TensorSpec));
    for (const r of Object.values(child.nodes))
      if (r.status === 'warning') warnings.push(...r.messages.map((m) => `迭代 ${i}: ${m}`));
  }
  return {
    status: errors.length ? 'error' : warnings.length ? 'warning' : 'ok',
    outputs: errors.length ? {} : { out: { kind: 'list', items: collected } },
    messages: errors.length ? errors : [...new Set(warnings)],
    notes: [
      `逐元素处理 ${sequence.items.length} 次；初始列表 ${initial.items.length} 个元素，按输入顺序收集。`,
      '每轮使用独立的 item 和相同的捕获输入；不将上一轮特征传给下一轮。',
      'out 边界代表本轮要 append 的 Tensor，列表收集由 ForEach 完成，不进行 stack 或 concat。',
      '迭代显示为静态形状推导，不是 Python 执行或运行时快照；列表输出表示更新后的状态，不模拟对象身份。',
    ],
    iterations,
    ...(iterations.length ? { child: iterations[0] } : {}),
  };
}

export type LoopSelections = Record<string, number>;
export function loopIteration(result: NodeResult | undefined, selected = 0): number {
  const count = result?.iterations?.length ?? 0;
  return count && Number.isInteger(selected) ? Math.max(0, Math.min(count - 1, selected)) : 0;
}
