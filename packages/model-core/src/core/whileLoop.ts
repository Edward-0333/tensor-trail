import type { Graph, GraphResult, ModelNode, NodeResult, ValueSpec } from './types';
import { clone, isIntegerSpec, isTensorSpec } from './types';
import { mergeLineages, valueLineage } from './informationSources';

export const MAX_WHILE_LINEAGE_PASSES = 32;

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
/** Infer a symbolic body and optionally close its possible value dependencies. */
export function inferWhile(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  evaluate: (v: Record<string, ValueSpec>) => GraphResult,
  lineageEnabled = false,
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
    '展示一轮符号化循环体并检查状态形状不变量，不执行 Python、不分配 Tensor、不生成分配结果或分支轨迹。',
    '状态 Outlet 代表下一轮同名变量；由循环语义反馈，不通过普通画布连线制造环路，也不表示新建 Tensor 副本。',
  ];
  if (typeof limit.value === 'number' && start.value >= limit.value)
    return {
      status: 'ok',
      messages: [],
      notes: [...notes, '初始条件不成立，零次迭代，原样返回状态。'],
      outputs: Object.fromEntries(states.map((s) => [s, clone(inputs[s])])),
    };
  const symbolicInputs: Record<string, ValueSpec> = { ...clone(inputs), i: { ...clone(start), value: 'i' } };
  let child = evaluate(symbolicInputs);
  const check = (body: GraphResult) => {
    const errors = [
      ...body.issues,
      ...Object.entries(body.nodes)
        .filter(([, r]) => ['error', 'blocked'].includes(r.status))
        .map(([id, r]) => `${id}: ${r.messages[0]}`),
    ];
    if (!errors.length) {
      if (!body.outputs.i || !isIntegerSpec(body.outputs.i) || body.outputs.i.value !== '(i + 1)')
        errors.push('循环体必须输出 i + 1 到下一轮 i，不能停滞或跳步。');
      for (const s of states)
        if (!sameState(inputs[s], body.outputs[s]))
          errors.push(`循环状态 ${s} 的 shape/dtype 必须在一轮前后保持不变。`);
    }
    return errors;
  };
  let errors = check(child);
  const warnings = new Set<string>();
  const collectWarnings = () => Object.values(child.nodes).forEach(r => {
    if (r.status === 'warning') r.messages.forEach(m => warnings.add(m));
  });
  collectWarnings();
  // Replay only lineage changes. Shapes, autograd and runtime observations are not
  // iterated, and the canvas still displays one symbolic body rather than N copies.
  const outputs = errors.length ? {} : Object.fromEntries(states.map(s => [s, clone(child.outputs[s])]));
  if (lineageEnabled && !errors.length) {
    const control = mergeLineages([valueLineage(start), valueLineage(limit)]);
    const count = typeof limit.value === 'number' ? limit.value - start.value : undefined;
    let head = Object.fromEntries(states.map(s => [s, valueLineage(inputs[s])]));
    for (let pass = 1; pass <= MAX_WHILE_LINEAGE_PASSES; pass++) {
      const next = Object.fromEntries(states.map(s => [s,
        mergeLineages([valueLineage(child.outputs[s]), control]),
      ]));
      const joined = Object.fromEntries(states.map(s => [s, mergeLineages([head[s], next[s]])]));
      const stable = states.every(s => JSON.stringify(joined[s]) === JSON.stringify(mergeLineages([head[s]])));
      const finished = stable || pass === count;
      for (const s of states) {
        // A symbolic bound may mean zero iterations; a known positive bound may not.
        outputs[s].lineage = count === undefined
          ? mergeLineages([valueLineage(inputs[s]), next[s]]) : next[s];
        if (!finished && pass === MAX_WHILE_LINEAGE_PASSES)
          outputs[s].lineage = { ...outputs[s].lineage!, unknown: true };
      }
      if (finished || pass === MAX_WHILE_LINEAGE_PASSES) {
        notes.push(`来源按状态依赖传播，进行了 ${pass} 次静态分析；合并可能路径，不表示实际执行或特征融合。`);
        if (!finished) warnings.add('While 来源传播达到分析上限，已知来源可能不完整，保留待确认标记。');
        break;
      }
      head = joined;
      const external = clone(symbolicInputs);
      for (const s of states) external[s].lineage = head[s];
      child = evaluate(external);
      errors = check(child);
      collectWarnings();
      if (errors.length) break;
    }
  }
  return {
    status: errors.length ? 'error' : warnings.size ? 'warning' : 'ok',
    child,
    outputs: errors.length ? {} : outputs,
    messages: errors.length ? errors : [...new Set(warnings)],
    notes: [
      ...notes,
      `迭代范围：${start.value} <= i < ${limit.value}；这里只展示模板，不展示第 0 轮或第 1 轮的真实状态。`,
    ],
  };
}
