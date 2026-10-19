import { useEffect, useRef, useState } from 'react';
import { Braces, Check, Download, X } from 'lucide-react';
import { canvasApi } from './agentBridge';
import type { ModelResponse } from '../agent';
import { shapeText } from '../core/types';
import example from '../../examples/cnn-transformer.recipe.json';
import './agent.css';

const stringify = (v: unknown) => JSON.stringify(v, null, 2);
function download(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([stringify(value)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function AgentPanel({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [mode, setMode] = useState<'build' | 'patch'>('build');
  const [text, setText] = useState(stringify(example));
  const [result, setResult] = useState<ModelResponse>();
  const [revision, setRevision] = useState('');
  const [message, setMessage] = useState('');
  const [allowWarnings, setAllowWarnings] = useState(false);
  useEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => element.close();
  }, []);
  function reset(value: string) {
    setText(value);
    setResult(undefined);
    setMessage('');
    setAllowWarnings(false);
  }
  function switchMode(next: 'build' | 'patch') {
    setMode(next);
    reset(
      stringify(
        next === 'build'
          ? example
          : {
              patchVersion: 1,
              operations: [
                { op: 'rename_project', name: `${canvasApi.getModel().project.name} · AI 修改` },
              ],
            },
      ),
    );
  }
  function check() {
    setMessage('');
    setAllowWarnings(false);
    try {
      if (new TextEncoder().encode(text).length > 3_000_000) throw new Error('JSON 超过 3 MB。');
      const value = JSON.parse(text),
        snapshot = canvasApi.getModel();
      let next =
        mode === 'build'
          ? canvasApi.buildModel(value)
          : canvasApi.patchModel(snapshot.project, value);
      if (mode === 'build' && next.project) {
        next.project.variants = snapshot.project.variants;
        next = { ...canvasApi.validateModel(next.project), defaultsApplied: next.defaultsApplied };
      }
      setRevision(snapshot.revision);
      setResult(next);
    } catch (e) {
      setResult(undefined);
      setMessage(`JSON 无法读取：${(e as Error).message}`);
    }
  }
  const warnings = result?.diagnostics.filter((d) => d.severity === 'warning') ?? [];
  return (
    <dialog ref={dialog} className="agent-dialog" aria-labelledby="agent-title" onCancel={onClose}>
      <div className="agent-heading">
        <div className="agent-title">
          <Braces size={21} />
          <div>
            <h2 id="agent-title">AI 构建工作区</h2>
            <p>粘贴架构 JSON，检查数据流，再应用到画布</p>
          </div>
        </div>
        <button className="icon-button" aria-label="关闭 AI 构建" onClick={onClose}>
          <X size={19} />
        </button>
      </div>
      <div className="agent-toolbar">
        <div className="agent-modes" aria-label="构建方式">
          <button className={mode === 'build' ? 'active' : ''} onClick={() => switchMode('build')}>
            新建模型
          </button>
          <button className={mode === 'patch' ? 'active' : ''} onClick={() => switchMode('patch')}>
            修改当前模型
          </button>
        </div>
        <button
          className="text-button"
          onClick={() => download(canvasApi.describeModules(true), 'tensor-trail.modules.json')}
        >
          模块说明
        </button>
        <button
          className="text-button"
          onClick={() =>
            download(
              canvasApi.getSchemas()[mode === 'build' ? 'recipe' : 'patch'],
              `tensor-trail.${mode}.schema.json`,
            )
          }
        >
          JSON Schema
        </button>
        <button
          className="text-button"
          onClick={() => download(canvasApi.getModel(), 'tensor-trail.snapshot.json')}
        >
          当前快照
        </button>
      </div>
      <div className="agent-body">
        <div className="agent-source">
          <label htmlFor="agent-json">
            {mode === 'build' ? '架构配方 · Recipe v1' : '批量修改 · Patch v1'}
          </label>
          <textarea
            id="agent-json"
            spellCheck={false}
            value={text}
            onChange={(e) => reset(e.target.value)}
          />
          <p>
            {mode === 'build'
              ? '自动补全默认参数、展开组合模块和排列节点。应用后保留“我的变体”。'
              : 'path 指向子图，例如 ["encoder", "attention"]。整批修改作为一次操作应用。'}
          </p>
        </div>
        <div className="agent-preview" aria-live="polite">
          <span className="agent-eyebrow">VALIDATION / 形状校验</span>
          {message && (
            <p className="agent-error" role="alert">
              {message}
            </p>
          )}
          {!result ? (
            <div className="agent-empty">
              <Braces size={32} />
              <h3>从架构描述到数据流</h3>
              <p>让 AI 阅读模块说明并生成左侧格式。点击“检查形状”，查看每一步的尺寸和问题。</p>
              <code>window.tensorTrail</code>
              <p>浏览器 API 与命令行共用同一套构建规则。</p>
            </div>
          ) : (
            <>
              <h3 className={result.ok ? 'agent-success' : 'agent-error'}>
                {result.ok ? '模型校验通过' : '模型需要修正'}
              </h3>
              <p>
                {result.nodes.length} 个展开节点 · 补全 {result.defaultsApplied.length} 项默认参数
              </p>
              <div className="agent-outputs">
                {Object.entries(result.outputs).map(([id, spec]) => (
                  <div key={id}>
                    <span>{id}</span>
                    <code>{shapeText(spec)}</code>
                  </div>
                ))}
              </div>
              {!!result.diagnostics.length && (
                <div className="agent-diagnostics">
                  {result.diagnostics.map((d, i) => (
                    <div
                      key={i}
                      className={d.severity === 'error' ? 'agent-error' : 'agent-warning'}
                    >
                      <code>
                        {d.path.join(' / ') || '模型'} · {d.code}
                      </code>
                      <p>{d.message}</p>
                    </div>
                  ))}
                </div>
              )}
              <details>
                <summary>逐节点形状</summary>
                {result.nodes.map((n) => (
                  <div className="agent-trace" key={n.path.join('/')}>
                    <code>{n.path.join(' / ')}</code>
                    {Object.entries(n.outputs).map(([port, spec]) => (
                      <span key={port}>
                        {port} {shapeText(spec)}
                      </span>
                    ))}
                  </div>
                ))}
              </details>
              {!!result.defaultsApplied.length && (
                <details>
                  <summary>补全的默认参数</summary>
                  {result.defaultsApplied.map((d, i) => (
                    <div className="agent-trace" key={i}>
                      <code>
                        {d.path.join('/')} · {d.parameter}
                      </code>
                      <span>{String(d.value)}</span>
                    </div>
                  ))}
                </details>
              )}
            </>
          )}
        </div>
      </div>
      <div className="agent-footer">
        <div>
          {warnings.length > 0 ? (
            <label className="agent-ack">
              <input
                type="checkbox"
                checked={allowWarnings}
                onChange={(e) => setAllowWarnings(e.target.checked)}
              />
              我已检查形状警告
            </label>
          ) : (
            <span>应用前不更改画布 · 应用后可撤销</span>
          )}
        </div>
        <button
          className="text-button"
          disabled={!result}
          onClick={() => download(result, 'tensor-trail.validation.json')}
        >
          <Download size={14} />
          报告
        </button>
        <button className="text-button" onClick={check}>
          <Check size={14} />
          检查形状
        </button>
        <button
          className="primary-button"
          disabled={!result?.ok || (warnings.length > 0 && !allowWarnings)}
          onClick={() => {
            const applied = canvasApi.applyModel({
              project: result?.project,
              expectedRevision: revision,
              allowWarnings,
            });
            if (applied.ok) onClose();
            else {
              setResult(applied);
              setMessage('没有应用修改，请重新检查。');
            }
          }}
        >
          应用到画布
        </button>
      </div>
    </dialog>
  );
}
