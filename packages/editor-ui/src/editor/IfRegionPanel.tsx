import type { Graph, ModelNode } from '@tensor-trail/model-core/core/types';
import { ifRegionMembers, isCodeRegion } from '@tensor-trail/model-core/core/ifRegions';
import { useEditor } from './store';

export function IfRegionPanel({ graph, node }: { graph: Graph; node: ModelNode }) {
  const selected = new Set(ifRegionMembers(node));
  const occupied = new Set(
    graph.nodes.filter((n) => isCodeRegion(n) && n.id !== node.id).flatMap(ifRegionMembers),
  );
  const parents = new Map(
    graph.nodes.filter(isCodeRegion)
      .flatMap((region) => ifRegionMembers(region).map((id) => [id, region.id] as const)),
  );
  const ancestors = new Set<string>();
  let parent = parents.get(node.id);
  while (parent !== undefined && !ancestors.has(parent)) {
    ancestors.add(parent);
    parent = parents.get(parent);
  }
  return (
    <section className="if-region-members">
      <div className="section-label">{node.type === 'LoopRegion' ? '循环体节点' : '条件体节点'} · {selected.size}</div>
      <p className="microcopy">
        勾选同一画布的直接成员，可包含条件框或循环框；每个节点只能有一个直接父框，不能包含祖先框。虚线框自动调整，不移动既有布局。删除框保留代码并提升到父框；复制框携带全部后代。
      </p>
      <div className="if-region-member-list">
        {graph.nodes
          .filter((n) => n.id !== node.id && n.type !== 'LayoutRegion')
          .map((member) => (
            <label key={member.id}>
              <input
                type="checkbox"
                checked={selected.has(member.id)}
                disabled={occupied.has(member.id) || ancestors.has(member.id)}
                onChange={(event) => {
                  const ids = new Set(selected);
                  if (event.target.checked) ids.add(member.id);
                  else ids.delete(member.id);
                  useEditor.getState().setParam(node.id, 'members', [...ids].join(','));
                }}
              />
              <span>{member.label}</span>
            </label>
          ))}
      </div>
      <p className="microcopy">
        {node.type === 'LoopRegion'
          ? '循环语句只作范围说明，框内参数准备每轮发生。静态图仅推导当前一轮，不模拟迭代次数、状态更新或最终输出。'
          : '条件是代码说明，不求值。框内所有节点只做静态规格推导，不代表分支已执行。跨框属性读取需明确选择赋值版本。'}
      </p>
    </section>
  );
}
