import type { Graph, GraphResult, ModelNode, NodeResult, ValueSpec } from './types';
import { clone, isIntegerSpec, isTensorSpec } from './types';

export function whileStates(node: ModelNode): string[] {
  const names = String(node.params.states)
    .split(',')
    .map((s) => s.trim());
  if (
    !names.length ||
    names.length > 8 ||
    new Set(names).size !== names.length ||
    names.some(
      (s) =>
        !/^[A-Za-z][A-Za-z0-9_]*$/.test(s) ||
        ['i', 'num_points', 'constructor', 'prototype'].includes(s),
    )
  )
    throw Error('循环状态需要 1-8 个不重复标识符，不能使用保留端口 i / num_points。');
  return names;
}
const boundary = (g: Graph, type: string) =>
  g.nodes.filter((n) => n.type === type && n.valueSource === undefined);
export function checkWhileBody(node: ModelNode) {
  if (!node.subgraph) throw Error('WhileLoop 缺少循环体。');
  const states = whileStates(node);
  const ins = boundary(node.subgraph, 'Inlet').map((n) => String(n.params.port));
  const outs = boundary(node.subgraph, 'Outlet').map((n) => String(n.params.port));
  if (!['i', 'num_points', ...states].every((s) => ins.includes(s)))
    throw Error('循环体需要 i、num_points 和每个状态的 Inlet。');
  if (outs.length !== states.length + 1 || !['i', ...states].every((s) => outs.includes(s)))
    throw Error('循环体 Outlet 必须包含下一轮 i 和全部状态，不能遗漏或增加状态。');
}
export function whilePorts(node: ModelNode) {
  return {
    inputs: boundary(node.subgraph ?? { nodes: [], edges: [] }, 'Inlet').map((n) => ({
      id: String(n.params.port),
      label: n.label,
    })),
    outputs: whileStates(node).map((s) => ({ id: s, label: `${s}（循环结束规格）` })),
  };
}
function sameState(a: ValueSpec, b?: ValueSpec) {
  return (
    b &&
    isTensorSpec(a) &&
    isTensorSpec(b) &&
    a.dtype === b.dtype &&
    JSON.stringify(a.shape) === JSON.stringify(b.shape)
  );
}
/** Infer one symbolic body, never execute or unroll the data-dependent assignment loop. */
export function inferWhile(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  evaluate: (v: Record<string, ValueSpec>) => GraphResult,
): Omit<NodeResult, 'inputs'> {
  checkWhileBody(node);
  const states = whileStates(node),
    start = inputs.i,
    limit = inputs.num_points;
  if (
    !start ||
    !isIntegerSpec(start) ||
    typeof start.value !== 'number' ||
    !Number.isSafeInteger(start.value) ||
    start.value < 0
  )
    throw Error('有界 While 的初始 i 必须是非负整数。');
  if (
    !limit ||
    !isIntegerSpec(limit) ||
    (typeof limit.value === 'number' && (!Number.isSafeInteger(limit.value) || limit.value < 0))
  )
    throw Error('num_points 必须为非负整数或符号整数。');
  for (const s of states)
    if (!inputs[s] || !isTensorSpec(inputs[s]))
      throw Error(`循环状态 ${s} 暂只支持形状不变的 Tensor。`);
  const notes = [
    '有界循环：while i < num_points，每轮 i += 1；不支持 break/continue 或任意条件循环。',
    '只检查一轮符号化循环体及状态形状不变量，不执行 Python、不分配 Tensor、不生成分配结果或分支轨迹。',
    '状态 Outlet 代表下一轮同名变量；由循环语义反馈，不通过普通画布连线制造环路，也不表示新建 Tensor 副本。',
  ];
  if (typeof limit.value === 'number' && start.value >= limit.value)
    return {
      status: 'ok',
      messages: [],
      notes: [...notes, '初始条件不成立，零次迭代，原样返回状态。'],
      outputs: Object.fromEntries(states.map((s) => [s, clone(inputs[s])])),
    };
  const child = evaluate({ ...clone(inputs), i: { kind: 'int', value: 'i' } });
  const errors = [
    ...child.issues,
    ...Object.entries(child.nodes)
      .filter(([, r]) => ['error', 'blocked'].includes(r.status))
      .map(([id, r]) => `${id}: ${r.messages[0]}`),
  ];
  if (!errors.length) {
    if (!child.outputs.i || !isIntegerSpec(child.outputs.i) || child.outputs.i.value !== '(i + 1)')
      errors.push('循环体必须输出 i + 1 到下一轮 i，不能停滞或跳步。');
    for (const s of states)
      if (!sameState(inputs[s], child.outputs[s]))
        errors.push(`循环状态 ${s} 的 shape/dtype 必须在一轮前后保持不变。`);
  }
  const warnings = Object.values(child.nodes).flatMap((r) =>
    r.status === 'warning' ? r.messages : [],
  );
  return {
    status: errors.length ? 'error' : warnings.length ? 'warning' : 'ok',
    child,
    outputs: errors.length
      ? {}
      : Object.fromEntries(states.map((s) => [s, clone(child.outputs[s])])),
    messages: errors.length ? errors : [...new Set(warnings)],
    notes: [
      ...notes,
      `迭代范围：${start.value} <= i < ${limit.value}；这里只展示模板，不展示第 0 轮或第 1 轮的真实状态。`,
    ],
  };
}
