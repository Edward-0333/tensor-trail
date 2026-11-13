import { useReactFlow } from '@xyflow/react';
import type { Graph } from '@tensor-trail/model-core/core/types';
import { levelInputs } from '../views/levelInputs';
import { useEditor } from './store';

export function LevelInputsBar({ graph, root }: { graph: Graph; root: boolean }) {
  const inputs = levelInputs(graph, root);
  const selected = useEditor(state => state.selected);
  const lineageEnabled = useEditor(state => state.project.lineageEnabled === true);
  const { fitView } = useReactFlow();
  return (
    <nav className="level-inputs-bar" aria-label="本层已绘制输入" data-testid="level-inputs-bar">
      <strong title="仅统计本层已绘制的输入接口，不代表代码参数已完整录入；关联副本和内部参数不计入。">
        已绘制输入 <span className="level-inputs-count">{inputs.length}</span>
      </strong>
      <div className="level-inputs-list">
        {inputs.map(input => (
          <button key={input.id} type="button" data-input-id={input.id}
            aria-label={`定位输入 ${input.index}/${input.total}：${input.label}`}
            aria-pressed={selected === input.id}
            title={`${input.label}${input.port ? ` · 端口 ${input.port}` : ''} · 点击定位，不移动节点`}
            onClick={() => {
              useEditor.getState().select(input.id);
              void fitView({ nodes: [{ id: input.id }], padding: 0.6, maxZoom: 1.1, duration: 240 });
            }}>
            <span className="level-input-number" aria-hidden="true">{input.index}</span>
            <span className="level-input-name">{input.label}</span>
          </button>
        ))}
        {!inputs.length && <span className="level-inputs-empty">尚未绘制输入</span>}
      </div>
      <label className="lineage-toggle" title="显示根层输入的来源编号；子模块只继承编号，不重新分配。设置随项目保存。">
        <input type="checkbox" aria-label="显示信息来源" checked={lineageEnabled}
          onChange={event => useEditor.getState().setLineageEnabled(event.target.checked)} />
        输入来源
      </label>
    </nav>
  );
}
