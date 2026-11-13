import type { Graph, GraphResult, ModelNode, NodeResult, ValueSpec } from './types';
import { clone, isIntegerSpec, isTensorSequenceSpec, isTensorSpec } from './types';
import { MAX_LOOP_ITEMS } from './forEach';

const boundary = (g: Graph, type: string) =>
  g.nodes.filter(n => n.type === type && n.valueSource === undefined);

export function forLoopStates(node: ModelNode): string[] {
  const states = String(node.params.states).split(',').map(s => s.trim());
  if (!states.length || states.length > 8 || new Set(states).size !== states.length ||
      states.some(s => !/^[A-Za-z][A-Za-z0-9_]*$/.test(s) ||
        ['index', 'constructor', 'prototype'].includes(s)))
    throw Error('For 状态需要 1-8 个不重复标识符，不能使用保留名 index。');
  return states;
}

export function checkForLoopBody(node: ModelNode): void {
  if (!node.subgraph) throw Error('ForLoop 缺少循环体。');
  const states = forLoopStates(node);
  const ins = boundary(node.subgraph, 'Inlet').map(n => String(n.params.port));
  const outs = boundary(node.subgraph, 'Outlet').map(n => String(n.params.port));
  if (!['index', ...states].every(s => ins.includes(s)) || new Set(ins).size !== ins.length)
    throw Error('For 循环体需要唯一 index 和每个状态的 Inlet；其余输入为只读捕获值。');
  if (node.params.iterations_input === true && ins.includes('iterations'))
    throw Error('输入控制次数时，iterations 是循环控制端口，不能再作为循环体 Inlet 或状态。');
  if (outs.length !== states.length || !states.every(s => outs.includes(s)))
    throw Error('For 循环体 Outlet 必须恰好对应全部状态，表示下一轮的值。');
}

export function forLoopPorts(node: ModelNode) {
  return {
    inputs: [...(node.params.iterations_input === true ? [{ id: 'iterations', label: '循环次数' }] : []), ...boundary(node.subgraph ?? { nodes: [], edges: [] }, 'Inlet')
      .filter(n => n.params.port !== 'index')
      .map(n => ({ id: String(n.params.port), label: n.label }))],
    outputs: forLoopStates(node).map(id => ({ id, label: `${id} (final)` })),
  };
}

export function forLoopCount(node: ModelNode, inputs: Record<string, ValueSpec> = {}) {
  if (node.params.iterations_input !== true) return node.params.iterations;
  const count = inputs.iterations;
  return count && isIntegerSpec(count) ? count.value : undefined;
}

function kind(v: ValueSpec): string | undefined {
  if (isTensorSpec(v)) return 'tensor';
  if (isTensorSequenceSpec(v)) return v.kind;
  if (isIntegerSpec(v)) return 'int';
}

/** Carry concrete specifications, not real tensors, between bounded iterations. */
export function inferForLoop(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  evaluate: (external: Record<string, ValueSpec>) => GraphResult,
): Omit<NodeResult, 'inputs'> {
  checkForLoopBody(node);
  if (node.params.iterations_input !== undefined && typeof node.params.iterations_input !== 'boolean')
    throw Error('iterations_input 必须为布尔值。');
  const count = forLoopCount(node, inputs);
  if (typeof count !== 'number' || !Number.isSafeInteger(count) || count < 0 || count > MAX_LOOP_ITEMS)
    throw Error(`For 静态推导需要 0-${MAX_LOOP_ITEMS} 次已知迭代。`);
  const states = forLoopStates(node);
  for (const name of states)
    if (!inputs[name] || !kind(inputs[name]))
      throw Error(`For 状态 ${name} 需要 Tensor、Tensor list/tuple 或局部整数。`);
  let current = Object.fromEntries(states.map(s => [s, clone(inputs[s])]));
  const captures = Object.fromEntries(Object.entries(inputs).filter(([s]) => !states.includes(s) &&
    !(node.params.iterations_input === true && s === 'iterations')));
  const iterations: GraphResult[] = [], errors: string[] = [], warnings: string[] = [];
  for (let index = 0; index < count; index++) {
    const child = evaluate({ ...clone(captures), ...clone(current), index: { kind: 'int', value: index } });
    iterations.push(child);
    const failures = [...child.issues, ...Object.entries(child.nodes)
      .filter(([, r]) => ['error', 'blocked'].includes(r.status))
      .map(([id, r]) => `${id}: ${r.messages[0]}`)];
    if (!failures.length) for (const name of states) {
      const next = child.outputs[name];
      if (!next || kind(next) !== kind(current[name]))
        failures.push(`状态 ${name} 丢失或类型改变；可改变 shape/列表长度，不能改变值种类。`);
      if (next && isTensorSequenceSpec(next) && next.items.length > 256)
        failures.push(`状态 ${name} 的列表超过 256 个 Tensor。`);
    }
    if (failures.length) {
      errors.push(`迭代 ${index}: ${failures.slice(0, 3).join('；')}`);
      break; // A failed recurrence cannot supply valid inputs to later iterations.
    }
    current = Object.fromEntries(states.map(s => [s, clone(child.outputs[s])]));
    warnings.push(...Object.values(child.nodes).flatMap(r =>
      r.status === 'warning' ? r.messages.map(m => `迭代 ${index}: ${m}`) : []));
  }
  return {
    status: errors.length ? 'error' : warnings.length ? 'warning' : 'ok',
    outputs: errors.length ? {} : current,
    messages: errors.length ? errors : [...new Set(warnings)],
    notes: [`for index in range(${count})；同名状态 Outlet 反馈到下一轮 Inlet，其他输入保持不变。`,
      '允许各轮 Tensor 形状变化及列表 append；零次迭代原样返回初始状态。',
      '仅静态规格推导，不执行 Python、分配 Tensor 或假定网络层共享参数；不支持 break/continue。'],
    iterations,
    ...(iterations.length ? { child: iterations[0] } : {}),
  };
}
