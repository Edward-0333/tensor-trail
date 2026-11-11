import type { ModelNode } from '@tensor-trail/model-core/core/types';
import { ifRegionMembers } from '@tensor-trail/model-core/core/ifRegions';

export function IfRegionNode({ model, selected }: { model: ModelNode; selected: boolean }) {
  const count = ifRegionMembers(model).length;
  return (
    <div
      className={`if-region-node ${selected ? 'selected' : ''}`}
      data-testid={`node-${model.id}`}
    >
      <header className="if-region-heading" title="选中以编辑条件和成员；边框随内部节点自动调整">
        <code title={`if ${model.params.condition}:`}>if {String(model.params.condition)}:</code>
        <small>{String(model.params.note)}</small>
      </header>
      {!count && (
        <p className="if-region-empty">在右侧勾选框内节点，或选中多个节点后点击「if 框」。</p>
      )}
    </div>
  );
}
