import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Copy, ClipboardPaste, Link, Trash2, Undo2, Redo2, Scan, SquareDashed, FolderPlus, FolderInput, FolderMinus, ChevronDown, Route, Group, Ungroup, ListCollapse } from 'lucide-react';
import type { Graph } from '@tensor-trail/model-core/core/types';
import { isLayoutRegion, layoutRegionOf } from '@tensor-trail/model-core/core/layoutRegions';
import { isValueNode } from '@tensor-trail/model-core/core/valueLinks';
import { canMergeCalculationSet } from '@tensor-trail/model-core/core/calculationGroups';
import { registry } from '@tensor-trail/model-core/modules';
import { contextTargets, menuPosition, type CanvasMenuState } from './canvasContext';
import type { createModuleClipboard } from './moduleClipboard';
import { useEditor } from './store';

export function CanvasContextMenu({ menu, graph, clipboard, close, selectAll, fitView }: {
  menu: CanvasMenuState; graph: Graph; clipboard: ReturnType<typeof createModuleClipboard>;
  close: () => void; selectAll: () => void; fitView: () => void;
}) {
  const state = useEditor(), ref = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false), [position, setPosition] = useState({ x: menu.x, y: menu.y });
  const targets = contextTargets(graph, menu.nodeIds);
  const regions = graph.nodes.filter(isLayoutRegion);
  const edges = graph.edges.filter(e => menu.edgeIds.includes(e.id));
  const single = targets.nodes.length === 1 ? targets.nodes[0] : undefined;
  useLayoutEffect(() => {
    const bounds = ref.current?.getBoundingClientRect();
    if (bounds) setPosition(menuPosition(menu, bounds, { width: window.innerWidth, height: window.innerHeight }));
  }, [menu, expanded]);
  useEffect(() => {
    const previous = document.activeElement;
    ref.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus();
    const outside = (event: PointerEvent) => { if (!ref.current?.contains(event.target as Node)) close(); };
    const dismiss = () => close();
    document.addEventListener('pointerdown', outside, true);
    window.addEventListener('blur', dismiss); window.addEventListener('resize', dismiss);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      window.removeEventListener('blur', dismiss); window.removeEventListener('resize', dismiss);
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus({ preventScroll: true });
    };
  }, [close]);
  const run = (action: () => void) => { close(); action(); };
  const item = (label: string, icon: ReactNode, action: () => void, disabled = false, shortcut?: string, danger = false) =>
    <button type="button" role="menuitem" disabled={disabled} className={danger ? 'danger' : undefined}
      onClick={() => run(action)}>{icon}<span>{label}</span>{shortcut && <kbd>{shortcut}</kbd>}</button>;
  const copy = () => { state.selectMany(menu.nodeIds); clipboard.copy(); };
  return createPortal(<div ref={ref} role="menu" aria-label="画布右键菜单" className="canvas-context-menu nodrag nopan nowheel"
    style={{ left: position.x, top: position.y }}
    onContextMenu={e => { e.preventDefault(); e.stopPropagation(); }}
    onPointerDown={e => e.stopPropagation()} onWheel={e => e.stopPropagation()}
    onKeyDown={e => {
      e.stopPropagation();
      if (e.key === 'Escape' || e.key === 'Tab') { e.preventDefault(); close(); return; }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
        e.preventDefault();
        const buttons = [...ref.current!.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
        const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1
          : (index + (e.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next]?.focus();
      }
    }}>
    <div className="canvas-context-title">{targets.nodes.length ? `已选择 ${targets.nodes.length} 个模块` : edges.length ? `已选择 ${edges.length} 条连线` : '画布'}</div>
    {item('复制', <Copy />, copy, !targets.nodes.length, 'Ctrl+C')}
    {item('粘贴到此处', <ClipboardPaste />, () => { clipboard.paste(menu.position); }, !state.clipboard?.nodes.length, 'Ctrl+V')}
    {single && isValueNode(single.type) && item('创建关联副本', <Link />, () => { state.duplicateLinked(single.id); })}
    {item('删除', <Trash2 />, () => {
      if (targets.deletable.length) state.removeNodes(targets.deletable);
      else state.removeEdges(edges.map(e => e.id));
    }, !targets.deletable.length && !edges.length, 'Delete', true)}
    {!!edges.length && !targets.nodes.length && item('恢复自动走线', <Route />, () => {
      const ids = new Set(edges.map(e => e.id));
      state.edit(g => { for (const edge of g.edges) if (ids.has(edge.id)) { delete edge.waypoints; delete edge.labelPosition; } }, false);
    }, !edges.some(e => e.waypoints?.length || e.labelPosition !== undefined))}
    <div role="separator" />
    <button type="button" role="menuitem" disabled={!targets.regionMembers.length || !regions.length}
      aria-expanded={expanded} aria-controls="canvas-region-choices" onClick={() => setExpanded(!expanded)}>
      <FolderInput /><span>添加至分区</span><ChevronDown className={expanded ? 'expanded' : ''} />
    </button>
    {expanded && <div id="canvas-region-choices" role="group" aria-label="可用分区" className="canvas-region-choices">
      {regions.map(region => <button type="button" role="menuitem" key={region.id} title={region.label}
        disabled={targets.regionMembers.every(id => layoutRegionOf(graph, id)?.id === region.id)}
        onClick={() => run(() => { state.assignLayoutRegion(targets.regionMembers, region.id); })}>
        <i style={{ background: String(region.params.color) }} /><span>{region.label}</span>
      </button>)}
    </div>}
    {item('新建分区并添加', <FolderPlus />, () => state.createLayoutRegion(targets.regionMembers), !targets.regionMembers.length)}
    {!!targets.assigned.length && item('解除分区归属', <FolderMinus />, () => { state.assignLayoutRegion(targets.assigned, null); })}
    <div role="separator" />
    {canMergeCalculationSet(graph, menu.nodeIds, registry) &&
      item('合并为语义计算块', <Group />, () => state.mergeCalculationNodes(menu.nodeIds))}
    {item('精简胶水链', <ListCollapse />, state.autoSimplifyCurrentGraph)}
    {item('展开语义块', <Ungroup />, state.expandAllCalculationsInCurrentGraph, !graph.nodes.some(n => n.type === 'CalculationGroup'))}
    <div role="separator" />
    {item('撤销', <Undo2 />, state.undo, !state.past.length, 'Ctrl+Z')}
    {item('重做', <Redo2 />, state.redo, !state.future.length, 'Ctrl+Shift+Z')}
    {item('全选模块', <SquareDashed />, selectAll, !graph.nodes.length)}
    {item('适配全图', <Scan />, fitView, !graph.nodes.length)}
  </div>, document.body);
}
