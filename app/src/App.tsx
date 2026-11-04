import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  ChevronRight,
  CircleCheck,
  CircleAlert,
  Network,
  Moon,
  Sun,
  PanelLeft,
  PanelRight,
  Route,
  Pause,
  Play,
  Redo2,
  SkipBack,
  StepForward,
  Undo2,
  FilePlus2,
  Save,
} from 'lucide-react';
import { ReactFlowProvider } from '@xyflow/react';
import { resolveForwardView } from './core/hierarchy';
import { inferProject, resultAt } from './core/engine';
import { shapeText } from './core/types';
import { registry } from './modules';
import { example, type ExampleKind } from './modules/examples';
import { useEditor } from './editor/store';
import { Canvas } from './editor/Canvas';
import { Inspector } from './editor/Inspector';
import { ForwardHierarchyHelp } from './editor/ForwardHierarchyHelp';
import { AgentPanel } from './editor/AgentPanel';
import { NodeLibrary } from './editor/NodeLibrary';
import { isTensorNode } from './views/ModuleNode';
import './editor/workbench.css';
import { clipboardText, readClipboard } from './editor/clipboard';
import { canOpenSourceFile, projectFiles } from './editor/fileActions';
import { isContainerInput } from './modules/containers';

function saveProject(saveAs = false) {
  // Inspector fields commit on blur; include the field being edited in this save.
  if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
  void projectFiles.save(saveAs);
}

export default function App() {
  const state = useEditor();
  const fileState = useSyncExternalStore(projectFiles.subscribe, projectFiles.getSnapshot);
  const [agentOpen, setAgentOpen] = useState(false);
  const [flowExpanded, setFlowExpanded] = useState(false);
  const [step, setStep] = useState(-1),
    [playing, setPlaying] = useState(false),
    [mobilePane, setMobilePane] = useState('canvas'),
    [libraryVisible, setLibraryVisible] = useState(true),
    [inspectorVisible, setInspectorVisible] = useState(true),
    [theme, setTheme] = useState<'dark' | 'light'>(() => {
      try {
        return localStorage.getItem('tensor-trail-theme') === 'light' ? 'light' : 'dark';
      } catch {
        return 'dark';
      }
    });
  const searchRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('tensor-trail-theme', theme);
    } catch {
      /* Theme still works without storage. */
    }
  }, [theme]);
  const fileRef = useRef<HTMLInputElement>(null),
    pathKey = state.path.join('/');
  const openProject = () =>
    canOpenSourceFile ? void projectFiles.open() : fileRef.current?.click();
  const rootResult = useMemo(() => inferProject(state.project, registry), [state.project]);
  const forwardView = resolveForwardView(state.project.graph, state.path);
  const { graph } = forwardView;
  const result = resultAt(rootResult, state.path) ?? {
    order: [],
    nodes: {},
    outputs: {},
    issues: ['当前子图无法推导，请返回父层检查输入连接和模块参数。'],
  };
  const order = result?.order ?? [];
  const failures =
    result?.issues.length ||
    Object.values(result?.nodes ?? {}).filter((r) => r.status === 'error' || r.status === 'blocked')
      .length ||
    0;
  const warnings = Object.values(result?.nodes ?? {}).filter((r) => r.status === 'warning').length;
  useEffect(() => {
    setStep(-1);
    setPlaying(false);
  }, [pathKey, state.project.graph]);
  useEffect(() => setMobilePane('canvas'), [pathKey]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setStep((current) => {
          if (current >= order.length - 1) {
            setPlaying(false);
            return current;
          }
          return current + 1;
        }),
      1300,
    );
    return () => clearInterval(timer);
  }, [playing, order.length]);
  useEffect(() => {
    if (step >= 0 && order[step]) useEditor.getState().select(order[step]);
  }, [step, order]);
  useEffect(() => {
    if (playing || step >= 0) setFlowExpanded(true);
  }, [playing, step]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        if (document.querySelector('dialog[open]'))
          useEditor.getState().toast('请先完成或关闭当前对话框，再保存项目。');
        else saveProject(e.shiftKey);
        return;
      }
      if (document.querySelector('dialog[open]')) return;
      if (
        e.key === '/' &&
        !(e.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')
      ) {
        e.preventDefault();
        setLibraryVisible(true);
        setMobilePane('library');
        requestAnimationFrame(() => searchRef.current?.focus());
        return;
      }
      if (!(e.ctrlKey || e.metaKey)) return;
      if ((e.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]'))
        return;
      if (e.key.toLowerCase() === 'z') {
        e.preventDefault();
        e.shiftKey ? useEditor.getState().redo() : useEditor.getState().undo();
      }
    };
    const editing = (target: EventTarget | null) =>
      document.querySelector('dialog[open]') ||
      (target instanceof HTMLElement &&
        target.closest('input,textarea,select,[contenteditable="true"]'));
    const copy = (event: ClipboardEvent) => {
      if (editing(event.target)) return;
      const state = useEditor.getState();
      if (!state.selectedIds.length && !state.selected) return;
      state.copy();
      const fragment = useEditor.getState().clipboard;
      if (fragment?.nodes.length && event.clipboardData) {
        event.clipboardData.setData('text/plain', clipboardText(fragment));
        event.preventDefault();
      }
    };
    const paste = (event: ClipboardEvent) => {
      if (editing(event.target)) return;
      try {
        const fragment = readClipboard(event.clipboardData?.getData('text/plain') ?? '');
        if (!fragment) return;
        event.preventDefault();
        const state = useEditor.getState();
        if (!state.clipboard || clipboardText(state.clipboard) !== clipboardText(fragment))
          useEditor.setState({ clipboard: fragment, pasteCount: 0 });
        useEditor.getState().paste();
      } catch (e) {
        useEditor.getState().toast(`粘贴失败：${(e as Error).message}`);
      }
    };
    window.addEventListener('keydown', handler);
    window.addEventListener('copy', copy);
    window.addEventListener('paste', paste);
    return () => {
      window.removeEventListener('keydown', handler);
      window.removeEventListener('copy', copy);
      window.removeEventListener('paste', paste);
    };
  }, []);
  const currentStep = step >= 0 ? graph.nodes.find((n) => n.id === order[step]) : undefined;
  return (
    <div
      className={`app-shell mobile-pane-${mobilePane} ${libraryVisible ? '' : 'library-hidden'} ${inspectorVisible ? '' : 'inspector-hidden'}`}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Network size={20} />
          </div>
          <div>
            <strong>Tensor Trail</strong>
          </div>
        </div>
        <div className="project-heading">
          <span className="project-dot" />
          <input
            aria-label="项目名称"
            disabled={!state.hasProject}
            key={state.project.name}
            defaultValue={state.project.name}
            onBlur={(e) => state.renameProject(e.target.value)}
          />
          <span className="version">MIT</span>
        </div>
        <div className="header-actions">
          <button
            className="text-button new-project-button"
            onClick={state.newProject}
            title="新建项目，原项目可撤销恢复"
          >
            <FilePlus2 size={15} />
            新建项目
          </button>
          <div className="view-actions">
            <button
              className="icon-button"
              aria-label="切换模块库与计算库"
              title="显示 / 隐藏模块库与计算库"
              aria-pressed={libraryVisible}
              onClick={() => setLibraryVisible((v) => !v)}
            >
              <PanelLeft size={17} />
            </button>
            <button
              className="icon-button"
              aria-label="切换属性面板"
              title="显示 / 隐藏属性面板"
              aria-pressed={inspectorVisible}
              onClick={() => setInspectorVisible((v) => !v)}
            >
              <PanelRight size={17} />
            </button>
          </div>
          <button
            className="icon-button"
            aria-label={theme === 'dark' ? '切换浅色主题' : '切换深色主题'}
            title={theme === 'dark' ? '切换浅色主题' : '切换深色主题'}
            onClick={() => setTheme((v) => (v === 'dark' ? 'light' : 'dark'))}
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <span
            className="save-state"
            title={
              fileState.fileName ||
              (state.hasProject ? '当前项目尚未保存到文件' : '打开已有项目，或新建项目')
            }
          >
            <span />{' '}
            {fileState.busy
              ? '文件处理中…'
              : fileState.fileName
                ? `${fileState.dirty ? '待保存' : '已保存'} · ${fileState.fileName}`
                : state.hasProject
                  ? '未保存文件'
                  : '未打开项目'}
          </span>
          <div className="history-buttons">
            <button
              className="icon-button"
              aria-label="撤销"
              title="撤销 Ctrl+Z"
              disabled={!state.past.length}
              onClick={state.undo}
            >
              <Undo2 size={17} />
            </button>
            <button
              className="icon-button"
              aria-label="重做"
              title="重做 Ctrl+Shift+Z"
              disabled={!state.future.length}
              onClick={state.redo}
            >
              <Redo2 size={17} />
            </button>
          </div>
          <button className="soft-button" onClick={() => setAgentOpen(true)}>
            AI 构建
          </button>
          <button
            className="text-button"
            disabled={fileState.busy}
            onClick={openProject}
            title="打开项目文件，后续 Ctrl+S 保存回此文件"
          >
            <ArrowUpFromLine size={15} />
            打开
          </button>
          <button
            className="text-button"
            disabled={fileState.busy || !state.hasProject}
            onClick={() => saveProject(true)}
            title="另存为 Ctrl+Shift+S"
          >
            <ArrowDownToLine size={15} />
            另存为
          </button>
          <button
            className="primary-button"
            disabled={fileState.busy || !state.hasProject}
            onClick={() => saveProject()}
            title={
              fileState.fileName
                ? `保存到 ${fileState.fileName} · Ctrl+S`
                : '保存项目 Ctrl+S，首次选择保存位置'
            }
          >
            <Save size={15} />
            保存
          </button>
        </div>
        <input
          ref={fileRef}
          className="hidden"
          type="file"
          accept=".json,application/json"
          aria-label="打开项目文件"
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) {
              const documentId = useEditor.getState().documentId;
              try {
                if (file.size > 3_000_000) throw new Error('文件超过 3 MB。');
                const text = await file.text();
                if (useEditor.getState().documentId === documentId)
                  useEditor.getState().importText(text);
              } catch (error) {
                useEditor.getState().toast(`打开失败：${(error as Error).message}`);
              }
            }
          }}
        />
      </header>
      <nav className="mobile-tabs" aria-label="工作区面板">
        <button
          className={mobilePane === 'library' ? 'active' : ''}
          onClick={() => setMobilePane('library')}
        >
          模块与计算
        </button>
        <button
          className={mobilePane === 'canvas' ? 'active' : ''}
          onClick={() => setMobilePane('canvas')}
        >
          画布
        </button>
        <button
          className={mobilePane === 'inspector' ? 'active' : ''}
          onClick={() => setMobilePane('inspector')}
        >
          参数与形状
        </button>
      </nav>
      <main className="workspace">
        <NodeLibrary searchRef={searchRef} />
        <section className="center-stage">
          <div className="canvas-toolbar">
            <nav className="breadcrumbs" aria-label="模型层级">
              <button
                aria-current={state.path.length === 0 ? 'page' : undefined}
                onClick={() => state.navigate([])}
              >
                <Network size={14} />
                {forwardView.levels[0].title}
              </button>
              {forwardView.levels.slice(1).map((level, i) => (
                <span key={level.path.join('/')}>
                  <ChevronRight size={12} />
                  <button
                    aria-current={i === state.path.length - 1 ? 'page' : undefined}
                    onClick={() => state.navigate(level.path)}
                  >
                    {level.title}
                  </button>
                </span>
              ))}
            </nav>
            <ForwardHierarchyHelp />
            <select
              aria-label="载入示例"
              value=""
              onChange={(e) => {
                if (e.target.value) {
                  const project = example(e.target.value as ExampleKind);
                  project.variants = state.project.variants;
                  state.load(project);
                }
              }}
            >
              <option value="" disabled>
                探索示例
              </option>
              <option value="transformer">走进 Transformer</option>
              <option value="mixed">Transformer → CNN → LSTM</option>
              <option value="blank">空白画布</option>
              <option value="pluto-input">PLUTO：data 网络输入</option>
            </select>
          </div>
          <div className="canvas-region">
            {state.hasProject && (
              <div className="canvas-caption">
                <strong>{forwardView.current.title}</strong>
                <small>
                  本层 {graph.nodes.length} 个节点 · {graph.edges.length} 条连接
                </small>
              </div>
            )}
            <ReactFlowProvider key={`${pathKey}:${state.canvasVersion}`}>
              <Canvas graph={graph} result={result} step={step} theme={theme} />
            </ReactFlowProvider>
            {!state.hasProject && (
              <div className="canvas-empty-state">
                <div>
                  <h2>开始绘制架构</h2>
                  <p>打开已有项目，或新建一个空白项目。</p>
                  <div className="canvas-empty-actions">
                    <button
                      className="primary-button"
                      disabled={fileState.busy}
                      onClick={openProject}
                    >
                      <ArrowUpFromLine size={16} />
                      打开项目
                    </button>
                    <button
                      className="soft-button"
                      disabled={fileState.busy}
                      onClick={state.newProject}
                    >
                      <FilePlus2 size={16} />
                      新建项目
                    </button>
                  </div>
                </div>
              </div>
            )}
            <div className="canvas-instruction">
              左键框选 <span>·</span> 中键平移 <span>·</span> 双击模块进入 forward()
            </div>
          </div>
          {state.hasProject && (
            <section className={`flow-strip ${flowExpanded ? '' : 'flow-collapsed'}`}>
              <div className="flow-strip-heading">
                <div>
                  <button
                    className="flow-tab"
                    aria-label="展开或收起数据流观察"
                    aria-expanded={flowExpanded}
                    onClick={() => setFlowExpanded((v) => !v)}
                  >
                    <Route size={14} />
                    数据流观察
                  </button>
                  <span className={`validation-pill ${failures ? 'bad' : warnings ? 'warn' : ''}`}>
                    {failures ? <CircleAlert size={12} /> : <CircleCheck size={12} />}{' '}
                    {failures
                      ? `${failures} 处待检查`
                      : warnings
                        ? '包含待验证约束'
                        : '形状检查通过'}
                  </span>
                </div>
                <div className="play-controls">
                  <button
                    className="icon-button"
                    aria-label="显示全部步骤"
                    title="显示全部"
                    onClick={() => {
                      setPlaying(false);
                      setStep(-1);
                    }}
                  >
                    <SkipBack size={14} />
                  </button>
                  <button
                    className="play-button"
                    disabled={!order.length}
                    onClick={() => {
                      if (playing) setPlaying(false);
                      else {
                        if (step < 0 || step >= order.length - 1) setStep(0);
                        setPlaying(true);
                      }
                    }}
                  >
                    {playing ? <Pause size={13} /> : <Play size={13} />}{' '}
                    {playing ? '暂停' : '逐步观察'}
                  </button>
                  <button
                    className="icon-button"
                    aria-label="下一步"
                    disabled={!order.length || step >= order.length - 1}
                    onClick={() => {
                      setPlaying(false);
                      setStep((v) => v + 1);
                    }}
                  >
                    <StepForward size={15} />
                  </button>
                </div>
              </div>
              {result?.issues.length ? (
                <div className="diagnostic error">{result.issues.join(' ')}</div>
              ) : (
                <div className="flow-steps">
                  {order.map((id, i) => {
                    const node = graph.nodes.find((n) => n.id === id)!,
                      r = result?.nodes[id],
                      spec = r?.outputs.out ?? Object.values(r?.outputs ?? {})[0];
                    return (
                      <button
                        key={id}
                        className={`flow-step ${isTensorNode(node.type) || isContainerInput(node.type) ? 'tensor-step' : ''} ${i === step ? 'active' : ''} ${r?.status === 'error' ? 'error' : ''}`}
                        onClick={() => {
                          setStep(i);
                          setPlaying(false);
                        }}
                      >
                        <span>{String(i + 1).padStart(2, '0')}</span>
                        <div>
                          <strong>{node.label}</strong>
                          <code>{spec ? shapeText(spec) : '待推导'}</code>
                        </div>
                        {i < order.length - 1 && <ChevronRight size={12} />}
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="step-description">
                {currentStep
                  ? `第 ${step + 1} 步 / ${order.length} · ${result?.nodes[currentStep.id]?.notes[0] ?? result?.nodes[currentStep.id]?.messages[0] ?? currentStep.label}`
                  : '尺寸标签由当前参数实时推导。点击任意步骤，查看该模块的输入与输出。'}
              </div>
            </section>
          )}
        </section>
        <Inspector graph={graph} result={result} />
      </main>
      {agentOpen && <AgentPanel onClose={() => setAgentOpen(false)} />}
      <footer className="statusbar">
        <div className="status-message" role="status">
          <span className="status-indicator" />
          {state.notice}
        </div>
        <span>
          本地工作区 <span className="status-divider">/</span> Tensor Trail
        </span>
      </footer>
    </div>
  );
}
