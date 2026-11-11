import type { Graph, ModelNode } from '../core/types';
import { whileStates } from '../core/whileLoop';
import type { ModuleRegistry } from './registry';
import { text } from './primitives';

export function whileLoopCode(n: ModelNode): string | undefined {
  if (n.type !== 'WhileLoop') return;
  return `# states: ${n.params.states}\nwhile i < num_points:\n    # 展开子图：本轮状态输入 -> 下一轮状态输出\n    ...\n    i += 1\n# 仅展示一轮符号模板并检查 shape/dtype 不变量，不展开 num_points 份。\n# 不执行 Tensor 运算，不生成实际索引分配、分支结果或中间值。`;
}
export function installWhileLoop(registry: ModuleRegistry) {
  registry.register({
    type: 'WhileLoop',
    label: 'WhileLoop · 状态循环',
    category: '网络结构',
    description:
      '有界状态循环：i < num_points，每轮 i += 1。Tensor 状态经 Outlet 反馈给下一轮；双击查看符号模板，不执行实际循环。',
    fields: [text('states', '循环携带的 Tensor 状态', 'maps,inds')],
    inputs: [],
    outputs: [],
    composite: true,
    build: (params) => {
      const states = whileStates({ params } as ModelNode),
        g: Graph = { nodes: [], edges: [] };
      const add = (id: string, type: string, x: number, y: number, p = {}) => {
        const n = registry.create(type, p, id);
        n.position = { x, y };
        n.label = id;
        g.nodes.push(n);
        return n;
      };
      const edge = (source: string, target: string) =>
        g.edges.push({
          id: `${source}_${target}`,
          source,
          target,
          sourcePort: 'out',
          targetPort: 'in',
        });
      add('i', 'Inlet', 0, 0, { port: 'i' });
      add('num_points', 'Inlet', 0, 100, { port: 'num_points' });
      add('increment', 'IntegerAddOne', 250, 0);
      add('next_i', 'Outlet', 500, 0, { port: 'i' });
      edge('i', 'increment');
      edge('increment', 'next_i');
      states.forEach((s, i) => {
        add(s, 'Inlet', 0, 240 + i * 160, { port: s });
        add(`next_${s}`, 'Outlet', 500, 240 + i * 160, { port: s });
        edge(s, `next_${s}`);
      });
      return g;
    },
  });
}
