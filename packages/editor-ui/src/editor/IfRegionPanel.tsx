import type { Graph, ModelNode } from '@tensor-trail/model-core/core/types';
import { ifRegionMembers } from '@tensor-trail/model-core/core/ifRegions';
import { useEditor } from './store';

export function IfRegionPanel({ graph, node }: { graph: Graph; node: ModelNode }) {
  const selected = new Set(ifRegionMembers(node));
  const occupied = new Set(
    graph.nodes.filter((n) => n.type === 'IfRegion' && n.id !== node.id).flatMap(ifRegionMembers),
  );
  return (
    <section className="if-region-members">
      <div className="section-label">条件体节点 · {selected.size}</div>
      <p className="microcopy">
        勾选同层节点。虚线框随内部节点移动和尺寸自动调整，不移动既有布局。删除框只取消分组，不删除内部代码。复制框会携带内部节点。
      </p>
      <div className="if-region-member-list">
        {graph.nodes
          .filter((n) => n.type !== 'IfRegion')
          .map((member) => (
            <label key={member.id}>
              <input
                type="checkbox"
                checked={selected.has(member.id)}
                disabled={occupied.has(member.id)}
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
        条件是代码说明，不求值。框内所有节点只做静态规格推导，不代表分支已执行。跨框属性读取需明确选择赋值版本。
      </p>
    </section>
  );
}
