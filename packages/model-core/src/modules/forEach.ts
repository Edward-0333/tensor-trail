import type { ModelNode, ModuleDefinition } from '../core/types';
import { forEachPorts } from '../core/forEach';

export function forEachCode(node: ModelNode): string | undefined {
  if (node.type !== 'ForEachTensor') return;
  return `# 循环体 item 接收当前 Tensor；可选 index 是从 0 开始的整数。
# 其他 Inlet 为每轮共享的捕获值，例如 B。
for index, item in enumerate(items):
    value = body(item, index, captures)  # 展开子图定义的处理；不是实际执行 Python
    initial.append(value)        # out 边界是待收集的单个 Tensor
out = initial
# 可逐轮查看不同 shape；不是 RNN，不进行 stack/concat。
# 空序列跳过循环体，保留初始列表。暂不支持 break/continue 或循环携带状态。`;
}

export const forEachTensor: ModuleDefinition = {
  type: 'ForEachTensor',
  label: 'ForEach · 逐元素循环',
  category: '网络结构',
  description:
    '遍历 Tensor tuple/list，以同一个循环体处理每个元素，按序 append 到初始列表。双击进入并切换迭代；仅静态推导，不执行 Python。',
  fields: [],
  ...forEachPorts({} as ModelNode),
  composite: true,
  build: () => ({
    nodes: [
      {
        id: 'item',
        type: 'Inlet',
        definitionVersion: 1,
        label: '当前元素 item',
        params: { port: 'item' },
        position: { x: 80, y: 100 },
      },
      {
        id: 'collect',
        type: 'Outlet',
        definitionVersion: 1,
        label: 'append(item)',
        params: { port: 'out' },
        position: { x: 400, y: 100 },
      },
    ],
    edges: [
      {
        id: 'collect_item',
        source: 'item',
        sourcePort: 'out',
        target: 'collect',
        targetPort: 'in',
      },
    ],
  }),
};
