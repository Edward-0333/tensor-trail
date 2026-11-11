import type { Graph, ModelNode, Params } from '../core/types';
import { tensorValue } from '../core/types';
import { requireThat } from './math';
import type { ModuleRegistry } from './registry';
import { requireRealTensor } from './sorting';

export function reweightCode(node: ModelNode): string | undefined {
  if (node.type !== 'Reweight') return;
  return 'def reweight(maps):\n    max_map, max_index = maps[:, :-1].max(-1)\n    sort_index = torch.sort(max_map, descending=True)[1]\n    maps = maps[sort_index]\n    return maps\n\n# 最后一列是原始 query 编号，不参与最大值计算，但随整行移动。\n# max_index 未使用；torch.sort 的第 [0] 项也未使用。\n# 静态图展示操作和规格，不声称已执行实际数值排序。';
}

export function installReweight(registry: ModuleRegistry) {
  registry.register({
    type: 'Reweight',
    label: 'reweight',
    category: '组合计算',
    description:
      '可展开函数：排除末列编号，求每行最大分数，降序排序后重排完整行。双击查看各步和返回值。',
    fields: [],
    inputs: [{ id: 'in', label: 'maps（分数列 + 原始编号列）' }],
    outputs: [{ id: 'out', label: '重排后的 maps' }],
    composite: true,
    validate: (inputs) => {
      const x = tensorValue(inputs.in);
      requireRealTensor(x);
      requireThat(x.shape.length === 2, 'reweight 的 maps 必须是二维 Tensor。');
      requireThat(
        typeof x.shape[1] !== 'number' || x.shape[1] >= 2,
        'maps 至少需要一列分数和一列编号。',
      );
      return [];
    },
    build: () => {
      const graph: Graph = { nodes: [], edges: [] };
      const add = (
        id: string,
        type: string,
        label: string,
        x: number,
        y: number,
        params: Params = {},
      ) => {
        const node = registry.create(type, params, id);
        node.label = label;
        node.position = { x, y };
        graph.nodes.push(node);
        return node;
      };
      const link = (source: string, target: string, sourcePort = 'out', targetPort = 'in') =>
        graph.edges.push({
          id: `${source}_${sourcePort}_${target}`,
          source,
          sourcePort,
          target,
          targetPort,
        });
      add('maps', 'Inlet', 'maps', 0, 70, { port: 'in' });
      add('scores_slice', 'TensorIndex', 'maps[:, :-1]（排除编号列）', 240, 70, {
        mode: 'slice',
        dim: '1',
        start: '',
        stop: '-1',
        step: 1,
      });
      add('scores', 'TensorVariable', '分数列 maps[:, :-1]', 450, 70);
      add('row_max', 'MaxWithIndices', '逐行最大值及位置', 700, 70, { dim: -1, keepdim: false });
      add('max_map', 'TensorVariable', 'max_map', 920, 30);
      add('max_index', 'TensorVariable', 'max_index（未使用）', 920, 240);
      add('sort', 'Sort', 'torch.sort(max_map, descending=True)', 1180, 30, {
        dim: -1,
        descending: true,
        stable: false,
      });
      add('sort_index', 'TensorVariable', 'sort_index = sort(...)[1]', 1380, 30);
      add('sorted_values', 'TensorVariable', 'sort(...)[0]（未使用）', 1380, 210);
      const mapsCopy = add('maps_near_index', 'Inlet', 'maps（完整行，关联副本）', 1080, 420, {
        port: 'in',
      });
      mapsCopy.valueSource = 'maps';
      add('row_index', 'RowIndex', 'maps[sort_index]', 1380, 420);
      add('return_maps', 'Outlet', 'return maps', 1610, 420, { port: 'out' });
      link('maps', 'scores_slice');
      link('scores_slice', 'scores');
      link('scores', 'row_max');
      link('row_max', 'max_map', 'values');
      link('row_max', 'max_index', 'indices');
      link('max_map', 'sort');
      link('sort', 'sort_index', 'indices');
      link('sort', 'sorted_values', 'values');
      link('maps_near_index', 'row_index');
      link('sort_index', 'row_index', 'out', 'indices');
      link('row_index', 'return_maps');
      // Keep the index route outside the unused sort-values tensor.
      graph.edges.find((e) => e.source === 'sort_index')!.waypoints = [
        { x: 1590, y: 100 },
        { x: 1590, y: 365 },
        { x: 1350, y: 365 },
        { x: 1350, y: 448 },
      ];
      return graph;
    },
  });
}
