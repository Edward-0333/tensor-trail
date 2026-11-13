import { useState } from 'react';
import type { Graph, ModelNode } from '@tensor-trail/model-core/core/types';
import {
  canJoinLayoutRegion, isLayoutRegion, layoutRegionMembers, layoutRegionOf, layoutRegionOutputs, layoutRegionRect,
} from '@tensor-trail/model-core/core/layoutRegions';
import { useEditor } from './store';

export function LayoutRegionPanel({ graph, node }: { graph: Graph; node: ModelNode }) {
  const state = useEditor(), [search, setSearch] = useState('');
  const selected = new Set(layoutRegionMembers(node));
  const outputs = new Set(layoutRegionOutputs(node));
  const rect = layoutRegionRect(graph, node, state.nodeSizes);
  return <section className="layout-region-panel">
    <div className="section-label">布局分区</div>
    <p className="microcopy">仅整理画布，不改变计算。加入时移入，拖动标题整体移动；解除所属分区后才能移出。删除分区保留所有模块。</p>
    <label>分区注释<textarea key={`${node.id}-${node.params.note ?? ''}`} aria-label="分区注释" rows={2} maxLength={240}
      defaultValue={String(node.params.note ?? '')} placeholder="用一句话描述本分区的工作"
      onBlur={e => { const note = e.target.value.trim(); if (note !== (node.params.note ?? '')) state.setParam(node.id, 'note', note); }} /></label>
    <label>尺寸模式<select aria-label="分区尺寸模式" value={String(node.params.sizing)} onChange={e => {
      if (e.target.value === 'auto') state.fitLayoutRegion(node.id);
      else state.resizeLayoutRegion(node.id, rect);
    }}><option value="auto">自动适配成员</option><option value="manual">手动尺寸</option></select></label>
    <div className="layout-region-dimensions">
      {(['width', 'height'] as const).map(key => <label key={key}>{key === 'width' ? '宽度' : '高度'}
        <input key={`${node.id}-${rect[key]}`} type="number" min={key === 'width' ? 360 : 200} max={1000000}
          aria-label={`分区${key === 'width' ? '宽度' : '高度'}`} defaultValue={Math.ceil(rect[key])}
          onBlur={e => {
            if (e.target.value && Number(e.target.value) !== Math.ceil(rect[key]))
              state.resizeLayoutRegion(node.id, { ...rect, [key]: Number(e.target.value) });
            e.target.value = String(Math.ceil(rect[key]));
          }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }} />
      </label>)}
    </div>
    <label className="layout-region-color">背景与边框颜色<input type="color" aria-label="分区颜色"
      value={String(node.params.color)} onChange={e => state.setParam(node.id, 'color', e.target.value)} /></label>
    <button className="soft-button" onClick={() => state.fitLayoutRegion(node.id)}>自动适配大小</button>
    <div className="section-label">分区成员 · {selected.size}</div>
    <input type="search" aria-label="搜索分区成员" placeholder="搜索同层模块" value={search} onChange={e => setSearch(e.target.value)} />
    <div className="if-region-member-list">
      {graph.nodes.filter(canJoinLayoutRegion).filter(n => `${n.label} ${n.id}`.toLowerCase().includes(search.toLowerCase())).map(member => {
        const owner = layoutRegionOf(graph, member.id);
        return <label key={member.id}><input type="checkbox" checked={selected.has(member.id)}
          disabled={Boolean(owner && owner.id !== node.id)}
          onChange={e => state.assignLayoutRegion([member.id], e.target.checked ? node.id : null)} />
          <span>{member.label}{owner && owner.id !== node.id ? `（属于 ${owner.label}）` : ''}</span></label>;
      })}
    </div>
    <div className="section-label">分区输出 · {outputs.size}</div>
    <div className="layout-region-output-list">
      {graph.nodes.filter(n => selected.has(n.id)).map(member => <label key={member.id}>
        <input type="checkbox" aria-label={`分区输出 ${member.label}`} checked={outputs.has(member.id)}
          onChange={e => state.setParam(node.id, 'outputs', (e.target.checked
            ? [...outputs, member.id] : [...outputs].filter(id => id !== member.id)).join(','))} />
        <span>{member.label}</span>
      </label>)}
    </div>
    <p className="microcopy">输出标记只说明分区对外提供的结果，不新增 forward 接口、不代表最终返回值或停止梯度。</p>
    <p className="microcopy">分区不嵌套；if/for 虚线框仍独立表示代码范围，不作为布局成员。</p>
  </section>;
}

export function LayoutRegionMembership({ graph, node }: { graph: Graph; node: ModelNode }) {
  const state = useEditor();
  const ids = (state.selectedIds.length > 1 ? state.selectedIds : [node.id])
    .filter(id => graph.nodes.some(n => n.id === id && canJoinLayoutRegion(n)));
  const owner = layoutRegionOf(graph, node.id);
  const mixed = ids.some(id => layoutRegionOf(graph, id)?.id !== owner?.id);
  return <section className="layout-region-membership">
    <label>{ids.length > 1 ? `所选 ${ids.length} 个模块的分区` : '所属分区'}
      <select aria-label="所属分区" value={mixed ? '__mixed' : owner?.id ?? ''}
        onChange={e => { if (e.target.value !== '__mixed') state.assignLayoutRegion(ids, e.target.value || null); }}>
        {mixed && <option value="__mixed" disabled>多个分区</option>}
        <option value="">无分区（解除归属）</option>
        {graph.nodes.filter(isLayoutRegion).map(region => <option key={region.id} value={region.id}>{region.label}</option>)}
      </select>
    </label>
    {(owner || mixed) && <button className="text-button" onClick={() => state.assignLayoutRegion(ids, null)}>解除分区归属</button>}
    {owner && ids.length === 1 && <label className="layout-region-output-toggle">
      <input type="checkbox" aria-label="标记为分区输出" checked={layoutRegionOutputs(owner).includes(node.id)}
        onChange={e => state.setParam(owner.id, 'outputs', (e.target.checked
          ? [...layoutRegionOutputs(owner), node.id] : layoutRegionOutputs(owner).filter(id => id !== node.id)).join(','))} />
      标记为分区输出
    </label>}
  </section>;
}
