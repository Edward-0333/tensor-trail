import type { ModelNode, ModuleDefinition } from '../core/types';
import { bool, num, text } from './primitives';

export function forLoopCode(node: ModelNode): string | undefined {
  if (node.type !== 'ForLoop') return;
  return `# 状态：${node.params.states}；其余 Inlet 为只读捕获值。
for index in range(${node.params.iterations_input === true ? 'iterations' : node.params.iterations}):
    states = body(index, states, captures)
# 每轮 Outlet 传给下一轮同名 Inlet；外层返回最终状态。
# 同一个循环体不表示网络权重共享；只推导规格，不执行 Python。
# 不支持 break/continue；可逐轮查看 shape 和列表长度。`;
}

export const forLoop: ModuleDefinition = {
  type: 'ForLoop', label: 'For · 带状态循环', category: '网络结构',
  description: '有界 for index in range(N)，上一轮输出成为下一轮状态；允许形状变化与列表收集，支持逐轮查看。',
  fields: [bool('iterations_input', '通过整数端口传入循环次数', false), num('iterations', '循环次数（最多 16）', 6, 0), text('states', '循环状态（逗号分隔）', 'value')],
  getFields: p => forLoop.fields.filter(f => f.key !== 'iterations' || p.iterations_input !== true),
  inputs: [{ id: 'value', label: '初始 value' }], outputs: [{ id: 'value', label: '最终 value' }],
  composite: true,
  build: p => {
    const states = String(p.states).split(',').map(s => s.trim());
    return {
      nodes: [
        { id: 'index', type: 'Inlet', definitionVersion: 1, label: 'index', params: { port: 'index' }, position: { x: 80, y: 0 } },
        ...states.flatMap<ModelNode>((s, i) => [
          { id: `in_${s}`, type: 'Inlet', definitionVersion: 1, label: `${s} (current)`, params: { port: s }, position: { x: 80, y: 160 + i * 180 } },
          { id: `out_${s}`, type: 'Outlet', definitionVersion: 1, label: `${s} (next)`, params: { port: s }, position: { x: 600, y: 160 + i * 180 } },
        ]),
      ],
      edges: states.map((s, i) => ({ id: `state_${i}`, source: `in_${s}`, sourcePort: 'out', target: `out_${s}`, targetPort: 'in' })),
    };
  },
};
