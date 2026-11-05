import type { Graph } from './types';

export const ROOT_FORWARD_TITLE = '主模型 · forward()';
export const forwardTitle = (moduleName: string) => `${moduleName} · forward()`;
export const functionTitle = (name: string) =>
  name.trim().endsWith('()') ? name.trim() : `${name.trim()}()`;

export interface ForwardLevel {
  path: string[];
  title: string;
}

/**
 * Shared view contract for manually built and imported models:
 * the root graph is the main forward; subgraphs are called modules' forwards,
 * reusable functions, or explicit calculation groups that retain the original operations.
 * Resolve exactly one level. Never flatten children or add a model wrapper.
 * Navigation is presentation state and must not mutate the model or its ports.
 */
export function resolveForwardView(root: Graph, path: readonly string[] = []) {
  let graph = root;
  const levels: ForwardLevel[] = [{ path: [], title: ROOT_FORWARD_TITLE }];
  for (const [index, id] of path.entries()) {
    const owner = graph.nodes.find((node) => node.id === id);
    if (!owner?.subgraph) throw new Error('该模块没有可进入的 forward 子图，请返回上层。');
    graph = owner.subgraph;
    levels.push({
      path: path.slice(0, index + 1),
      title:
        owner.type === 'CalculationGroup'
          ? `${owner.label} · 内部计算`
          : owner.type === 'ToVector'
            ? functionTitle(owner.label)
            : forwardTitle(owner.label),
    });
  }
  return { graph, levels, current: levels[levels.length - 1] };
}
