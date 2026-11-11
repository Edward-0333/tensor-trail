import { useRef } from 'react';
import {
  activeRuntimeSnapshot,
  parseRuntimeSnapshot,
} from '@tensor-trail/model-core/core/runtimeSnapshot';
import { useEditor } from './store';
import { loadActiveRuntime, projectFiles } from './platform';

export function RuntimePanel() {
  const state = useEditor();
  const picker = useRef<HTMLInputElement>(null);
  const reference = state.project.runtime?.snapshots.find(
    (ref) => ref.id === state.project.runtime?.active,
  );
  const snapshot = activeRuntimeSnapshot(state.project, state.runtimeSnapshot);
  return (
    <section className="runtime-panel" aria-label="运行快照">
      <div className="section-label">静态图与运行数据分离</div>
      <p className="microcopy">
        图文件保存结构、布局和输入契约；快照保存某一次运行的实际输入元数据。不执行
        Python，也不读取张量数值。
      </p>
      <label className="param-field">
        <span>当前显示</span>
        <select
          aria-label="当前运行快照"
          value={reference?.id ?? ''}
          disabled={!state.hasProject}
          onChange={(event) => state.selectRuntime(event.target.value || undefined)}
        >
          <option value="">仅静态图</option>
          {state.project.runtime?.snapshots.map((ref) => (
            <option key={ref.id} value={ref.id}>
              {ref.name}
            </option>
          ))}
        </select>
      </label>
      {reference && (
        <>
          <p className="microcopy runtime-path">{reference.file}</p>
          {state.runtimeError && (
            <div className="diagnostic error" role="alert">
              {state.runtimeError}
            </div>
          )}
          <p className="microcopy">
            {snapshot
              ? '快照已加载，画布显示运行时输入；静态图未被覆盖。'
              : '快照未加载。不能把静态形状和未知梯度状态当作本次运行结果。'}
          </p>
          <div className="runtime-actions">
            <button
              className="soft-button"
              onClick={() => {
                state.setRuntimeSnapshot();
                void loadActiveRuntime();
              }}
            >
              重新加载引用
            </button>
            <button className="soft-button" onClick={() => picker.current?.click()}>
              手动选择快照
            </button>
          </div>
          <input
            type="file"
            accept=".json,application/json"
            hidden
            ref={picker}
            onChange={async (event) => {
              const file = event.target.files?.[0];
              event.target.value = '';
              if (!file) return;
              const documentId = state.documentId,
                active = reference.id;
              try {
                if (file.size > 3_000_000) throw Error('快照超过 3 MB。');
                const loaded = parseRuntimeSnapshot(await file.text());
                const current = useEditor.getState();
                if (current.documentId !== documentId || current.project.runtime?.active !== active)
                  return;
                current.setRuntimeSnapshot(loaded);
              } catch (error) {
                const current = useEditor.getState();
                if (current.documentId === documentId && current.project.runtime?.active === active)
                  useEditor.setState({
                    runtimeSnapshot: undefined,
                    runtimeError: (error as Error).message,
                  });
              }
            }}
          />
        </>
      )}
      {snapshot && (
        <>
          <div className="section-label">采集上下文</div>
          <p className="microcopy">
            {snapshot.capturedAt} · {snapshot.bindings.length} 个输入绑定
          </p>
          <pre className="runtime-context">{JSON.stringify(snapshot.context, null, 2)}</pre>
        </>
      )}
      <button
        className="soft-button"
        disabled={!state.hasProject || (!!reference && !snapshot)}
        onClick={() => void projectFiles.exportPortable(state.runtimeSnapshot)}
      >
        导出当前运行单文件
      </button>
      <p className="microcopy">
        普通保存只写图与引用。便携副本附带当前快照，原项目不变。切换运行不会修改 Python 的训练模式。
      </p>
    </section>
  );
}
