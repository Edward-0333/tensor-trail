import type { Graph } from '@tensor-trail/model-core/core/types';
import { rootInputSources } from '@tensor-trail/model-core/core/rootInputs';
import { RootSourceBadge } from '../views/InformationSources';
import { useEditor } from './store';

export function InformationSourcePanel({ graph }: { graph: Graph }) {
  const enabled = useEditor((state) => state.project.lineageEnabled === true);
  const inputs = rootInputSources(graph);
  return (
    <section className="information-source-settings" aria-label="根层输入编号图例">
      <div className="section-label">根层输入编号</div>
      <div className="lineage-legend">
        {inputs.map((source) => (
          <div key={source.id}>
            <RootSourceBadge source={source} />
            <span>{source.label}</span>
          </div>
        ))}
      </div>
      {!inputs.length && <p className="microcopy">根层尚未绘制输入。</p>}
      <p className="microcopy">
        只有根层原始输入创建编号，按接口顺序自动分配；子模块、关联副本和派生张量只继承编号。IN
        是本层接口序号，不是新的来源编号。
      </p>
      {!enabled && (
        <button
          className="soft-button"
          onClick={() => useEditor.getState().setLineageEnabled(true)}
        >
          开启输入来源追踪
        </button>
      )}
    </section>
  );
}
