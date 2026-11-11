import type { RecipeGraph, RecipeNode } from '../../src/agent/types';
import type { Params } from '../../src/core/types';

/** Example recipe also used by the local SparseWorld graph migration. */
export function matchingLoopBody(): RecipeGraph {
  const builder = () => {
    const g: RecipeGraph = { nodes: [], connections: [] };
    return {
      g,
      n(
        id: string,
        type: string,
        label: string,
        x: number,
        y: number,
        params: Params = {},
        subgraph?: RecipeGraph,
      ) {
        const node: RecipeNode = {
          id,
          type,
          label,
          position: { x, y },
          params,
          ...(subgraph ? { subgraph } : {}),
        };
        g.nodes.push(node);
        return node;
      },
      e(from: string, to: string) {
        g.connections.push([from, to]);
      },
      inlet(id: string, label: string, x: number, y: number) {
        this.n(id, 'Inlet', label, x, y, { port: id });
      },
      outlet(id: string, label: string, x: number, y: number, port: string) {
        this.n(id, 'Outlet', label, x, y, { port });
      },
    };
  };
  const select = builder();
  select.inlet('maps', '本轮 maps', 0, 100);
  select.inlet('i', '本轮 i', 0, 280);
  select.n('read_row', 'TensorRowAt', 'maps[i]', 250, 100);
  select.n('row', 'TensorVariable', 'maps[i]', 470, 70);
  select.n('scores', 'TensorIndex', 'maps[i, :-1]', 720, 100, {
    mode: 'slice',
    dim: '0',
    start: '',
    stop: '-1',
    step: 1,
  });
  select.n('argmax', 'Argmax', 'torch.argmax(..., -1)', 950, 100);
  select.n('item', 'TensorItemInt', 'ind = argmax.item()', 1180, 100, {
    symbol: 'ind',
    cast: false,
  });
  select.outlet('ind_out', 'ind：选中的时间戳', 1440, 80, 'ind');
  select.n('id_column', 'TensorIndex', 'maps[i, -1]', 720, 400, {
    mode: 'index',
    dim: '0',
    index: '-1',
  });
  select.n('int_id', 'TensorItemInt', 'int(maps[i, -1])', 1050, 400, {
    symbol: 'original_id',
    cast: true,
  });
  select.outlet('id_out', 'original_id：原始 query 编号', 1440, 380, 'original_id');
  for (const [a, b] of [
    ['maps.out', 'read_row.in'],
    ['i.out', 'read_row.index'],
    ['read_row.out', 'row.in'],
    ['row.out', 'scores.in'],
    ['scores.out', 'argmax.in'],
    ['argmax.out', 'item.in'],
    ['item.out', 'ind_out.in'],
    ['row.out', 'id_column.in'],
    ['id_column.out', 'int_id.in'],
    ['int_id.out', 'id_out.in'],
  ])
    select.e(a, b);

  const count = builder();
  count.inlet('inds', '本轮写入后的 inds', 0, 80);
  count.inlet('ind', 'ind', 0, 260);
  count.n('equal', 'TensorEqualInt', 'inds == ind', 260, 100);
  count.n('sum', 'ReduceSum', '(inds == ind).sum()', 510, 100, { dim: -1, keepdim: false });
  count.n('item', 'TensorItemInt', 'sum.item()', 760, 100, {
    symbol: 'assigned_count',
    cast: false,
  });
  count.outlet('out', 'assigned_count', 1010, 100, 'out');
  for (const [a, b] of [
    ['inds.out', 'equal.in'],
    ['ind.out', 'equal.value'],
    ['equal.out', 'sum.in'],
    ['sum.out', 'item.in'],
    ['item.out', 'out.in'],
  ])
    count.e(a, b);

  const update = builder();
  update.inlet('maps', 'True 分支的 maps', 0, 80);
  update.inlet('start', 'start = i + 1', 0, 280);
  update.inlet('ind', 'ind', 0, 460);
  update.n('zero_column', 'ZeroTailColumn', 'maps[i+1:, ind] = 0', 300, 100);
  update.n('zeroed', 'TensorVariable', 'maps（剩余行的 ind 列已清零）', 650, 70);
  update.n('tail', 'SliceRowsFrom', 'maps[i+1:]', 950, 100);
  update.n('tail_value', 'TensorVariable', '剩余行（末轮可为空）', 1180, 70);
  update.n('reweight', 'Reweight', 'reweight', 1500, 100);
  update.n('sorted_tail', 'TensorVariable', '重排后的剩余行', 1750, 70);
  update.n('write_tail', 'WriteTailRows', 'maps[i+1:] = reweight(...)', 2000, 350);
  update.outlet('out', 'maps（仅剩余行更新）', 2350, 320, 'out');
  for (const [a, b] of [
    ['maps.out', 'zero_column.in'],
    ['start.out', 'zero_column.start'],
    ['ind.out', 'zero_column.column'],
    ['zero_column.out', 'zeroed.in'],
    ['zeroed.out', 'tail.in'],
    ['start.out', 'tail.start'],
    ['tail.out', 'tail_value.in'],
    ['tail_value.out', 'reweight.in'],
    ['reweight.out', 'sorted_tail.in'],
    ['zeroed.out', 'write_tail.in'],
    ['start.out', 'write_tail.start'],
    ['sorted_tail.out', 'write_tail.rows'],
    ['write_tail.out', 'out.in'],
  ])
    update.e(a, b);

  const g = builder();
  g.inlet('maps', 'maps（本轮状态）', 0, 70);
  g.inlet('i', 'i（本轮符号）', 0, 300);
  g.inlet('inds', 'inds（本轮状态）', 0, 610);
  g.inlet('num_points', 'num_points（循环上界）', 0, 850);
  g.inlet('num_inds', 'num_inds（时间戳配额）', 0, 1070);
  g.n('select_query', 'CalculationGroup', '读取本轮 query', 350, 80, {}, select.g);
  g.n('ind', 'LocalParameter', 'ind', 850, 100);
  g.n('original_id', 'LocalParameter', 'original_id', 850, 310);
  g.n('assign', 'ScalarIndexWrite', 'inds[original_id] = ind', 1120, 430);
  g.n('assigned_inds', 'TensorVariable', 'inds（本轮写入后）', 1420, 400);
  g.n('count_assigned', 'CalculationGroup', '(inds == ind).sum().item()', 1730, 400, {}, count.g);
  g.n('assigned_count', 'LocalParameter', 'assigned_count', 2180, 430);
  g.n('read_quota', 'IntegerListGet', 'num_inds[ind]', 1730, 740);
  g.n('quota', 'LocalParameter', 'quota = num_inds[ind]', 2180, 740);
  g.n('quota_full', 'IfCondition', '配额已满？', 2500, 500);
  g.n('true_maps', 'BranchValue', 'True：更新剩余行', 2820, 450, { branch: true });
  g.n('false_maps', 'BranchValue', 'False：maps 不修改', 2820, 930, { branch: false });
  g.n('update_tail', 'Submodule', 'True 分支：清零并重排剩余行', 3150, 450, {}, update.g);
  g.n('merge_maps', 'MergeBranches', '下一轮 maps：合并分支状态', 3550, 780);
  g.outlet('next_maps', '下一轮 maps', 3850, 750, 'maps');
  g.n('increment', 'IntegerAddOne', 'i + 1', 2800, 1280);
  g.n('start', 'LocalParameter', 'i_next = i + 1', 3100, 1280);
  g.outlet('next_i', '下一轮 i', 3550, 1280, 'i');
  g.outlet('next_inds', '下一轮 inds（保留本轮分配）', 1730, 1110, 'inds');
  for (const [a, b] of [
    ['maps.out', 'select_query.maps'],
    ['i.out', 'select_query.i'],
    ['select_query.ind', 'ind.in'],
    ['select_query.original_id', 'original_id.in'],
    ['inds.out', 'assign.in'],
    ['original_id.out', 'assign.index'],
    ['ind.out', 'assign.value'],
    ['assign.out', 'assigned_inds.in'],
    ['assigned_inds.out', 'count_assigned.inds'],
    ['ind.out', 'count_assigned.ind'],
    ['count_assigned.out', 'assigned_count.in'],
    ['num_inds.out', 'read_quota.in'],
    ['ind.out', 'read_quota.index'],
    ['read_quota.out', 'quota.in'],
    ['assigned_count.out', 'quota_full.count'],
    ['quota.out', 'quota_full.quota'],
    ['maps.out', 'true_maps.in'],
    ['quota_full.true', 'true_maps.control'],
    ['maps.out', 'false_maps.in'],
    ['quota_full.false', 'false_maps.control'],
    ['true_maps.out', 'update_tail.maps'],
    ['ind.out', 'update_tail.ind'],
    ['start.out', 'update_tail.start'],
    ['quota_full.out', 'merge_maps.condition'],
    ['update_tail.out', 'merge_maps.true'],
    ['false_maps.out', 'merge_maps.false'],
    ['merge_maps.out', 'next_maps.in'],
    ['i.out', 'increment.in'],
    ['increment.out', 'start.in'],
    ['start.out', 'next_i.in'],
    ['assigned_inds.out', 'next_inds.in'],
  ])
    g.e(a, b);
  return g.g;
}
