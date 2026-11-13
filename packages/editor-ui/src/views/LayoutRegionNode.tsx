import type { CSSProperties } from 'react';
import { NodeResizer, useReactFlow } from '@xyflow/react';
import type { ModelNode } from '@tensor-trail/model-core/core/types';
import { layoutRegionMembers, REGION_HEADER, REGION_PADDING } from '@tensor-trail/model-core/core/layoutRegions';
import { useEditor } from '../editor/store';
import type { TrailNode } from './ModuleNode';

export function LayoutRegionNode({ model, selected }: { model: ModelNode; selected: boolean }) {
  const { getNodes } = useReactFlow<TrailNode>();
  const ids = new Set(layoutRegionMembers(model));
  return <div className={`layout-region-node ${selected ? 'selected' : ''}`} data-testid={`node-${model.id}`}
    style={{ '--region-color': String(model.params.color) } as CSSProperties}>
    <NodeResizer nodeId={model.id} isVisible={selected} minWidth={360} minHeight={200}
      color={String(model.params.color)}
      shouldResize={(_, rect) => getNodes().filter(n => ids.has(n.id)).every(n =>
        n.position.x >= rect.x + REGION_PADDING && n.position.y >= rect.y + REGION_HEADER &&
        n.position.x + (n.measured?.width ?? 250) <= rect.x + rect.width - REGION_PADDING &&
        n.position.y + (n.measured?.height ?? 160) <= rect.y + rect.height - REGION_PADDING)}
      onResizeEnd={(_, rect) => useEditor.getState().resizeLayoutRegion(model.id, rect)} />
    <header className="layout-region-heading" title="拖动标题整体移动；选中后拖动边框调节大小。成员只能在分区内部拖动。">
      <div className="layout-region-title"><strong>{model.label}</strong><span>{ids.size} 个模块 · {model.params.sizing === 'auto' ? '自动适配' : '手动尺寸'}</span></div>
      {model.params.note && <p className="layout-region-note" title={String(model.params.note)}>{String(model.params.note)}</p>}
    </header>
    {!ids.size && <p className="layout-region-empty">在右侧添加成员，或选中模块后设置「所属分区」。</p>}
  </div>;
}
