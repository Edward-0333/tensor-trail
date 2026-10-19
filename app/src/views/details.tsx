import type { ComponentType } from 'react';
import type { ModelNode, NodeResult } from '../core/types';
import { shapeText } from '../core/types';

export interface DetailProps {
  node: ModelNode;
  result?: NodeResult;
}
// Views consume inference results without adding shape rules to the editor.
const details = new Map<string, ComponentType<DetailProps>>();
export const registerDetailView = (type: string, view: ComponentType<DetailProps>) =>
  details.set(type, view);

function AttentionShapes({ node, result }: DetailProps) {
  const inner = node.type === 'Transformer' ? result?.child?.nodes.attention?.child : result?.child;
  if (!inner?.nodes.scores?.outputs.out) return null;
  return (
    <section className="attention-detail">
      <div className="section-label">注意力内部快照</div>
      {[
        ['q_heads', 'Q · 拆头后'],
        ['k_heads', 'K · 拆头后'],
        ['v_heads', 'V · 拆头后'],
        ['scores', 'Q × Kᵀ'],
      ].map(([id, label]) => (
        <div key={id}>
          <span>{label}</span>
          <code>{shapeText(inner.nodes[id]?.outputs.out)}</code>
        </div>
      ))}
      <p>分数矩阵的两个序列轴分别对应 query 和 key。展开 Attention 可继续观察每一步。</p>
    </section>
  );
}
registerDetailView('Transformer', AttentionShapes);
registerDetailView('SelfAttention', AttentionShapes);
export function ModuleDetail(props: DetailProps) {
  const View = details.get(props.node.type);
  return View ? <View {...props} /> : null;
}
