import type { ModelNode } from '@tensor-trail/model-core/core/types';
import {
  attributeBinding,
  attributeOwnerLabel,
  planAttributes,
} from '@tensor-trail/model-core/core/attributes';
import { graphAt } from '@tensor-trail/model-core/core/graph';
import { useEditor } from './store';

export function AttributeBindingPanel({ node }: { node: ModelNode }) {
  const project = useEditor((s) => s.project),
    path = useEditor((s) => s.path);
  let binding;
  try {
    binding = attributeBinding(node.params);
  } catch {
    return <p className="diagnostic error">属性绑定参数待修正。</p>;
  }
  const plan = planAttributes(project.graph),
    current = JSON.stringify([...path, node.id]);
  const partners = [...plan.bindings.entries()].filter(
    ([key, value]) => key !== current && value.key === binding.key,
  );
  return (
    <section className="attribute-binding-panel">
      <div className="section-label">跨层属性绑定</div>
      <code>
        {attributeOwnerLabel(binding.owner)}.{binding.name}
      </code>
      <p className="microcopy">
        按模块节点路径、属性名和赋值版本关联，不占用 forward 形参端口。读取沿用同一 Tensor 的规格与梯度标注。
      </p>
      {binding.version && <p className="microcopy">赋值版本：<code>{binding.version}</code>。仅区分同一属性的写入点，不创建新属性，不自动判定条件或选择最终值。</p>}
      {plan.errors.has(current) && <p className="diagnostic error">{plan.errors.get(current)}</p>}
      {partners.map(([key]) => {
        const target: string[] = JSON.parse(key),
          scope = target.slice(0, -1),
          id = target.at(-1)!;
        const linked = graphAt(project.graph, scope).nodes.find((n) => n.id === id)!;
        return (
          <button
            key={key}
            className="soft-button"
            onClick={() => {
              useEditor.getState().navigate(scope);
              useEditor.getState().select(id);
            }}
          >
            定位 {linked.type === 'SetAttribute' ? 'SET' : 'GET'}：{linked.label}
          </button>
        );
      })}
      {!partners.length && <p className="microcopy">尚无对应读写节点。</p>}
    </section>
  );
}
