import { useEffect, useState } from 'react';
import { ArrowUpRight, BookOpen, BookmarkPlus, SlidersHorizontal, Trash2 } from 'lucide-react';
import type { Field, Graph, GraphResult, ModelNode } from '../core/types';
import { graphAt, ports } from '../core/graph';
import { registry } from '../modules';
import { TensorView } from '../views/TensorView';
import { ModuleDetail } from '../views/details';
import { useEditor } from './store';

function ParamField({
  field,
  node,
  disabled,
}: {
  field: Field;
  node: ModelNode;
  disabled: boolean;
}) {
  const value = node.params[field.key];
  const label = `${field.label} (${field.key})`;
  const update = (v: string | number | boolean) =>
    useEditor.getState().setParam(node.id, field.key, v);
  return (
    <label className={`param-field ${field.kind === 'boolean' ? 'boolean-field' : ''}`}>
      <span>
        {field.label}
        <small>{field.key}</small>
      </span>
      {field.kind === 'boolean' ? (
        <input
          aria-label={label}
          type="checkbox"
          disabled={disabled}
          checked={Boolean(value)}
          onChange={(e) => update(e.target.checked)}
        />
      ) : field.kind === 'select' ? (
        <select
          aria-label={label}
          disabled={disabled}
          value={String(value)}
          onChange={(e) => update(e.target.value)}
        >
          {field.options?.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </select>
      ) : (
        <input
          aria-label={label}
          disabled={disabled}
          type={field.kind === 'number' ? 'number' : 'text'}
          min={field.min}
          step={1}
          value={String(value ?? '')}
          onChange={(e) =>
            update(field.kind === 'number' ? Number(e.target.value) : e.target.value)
          }
        />
      )}
    </label>
  );
}

export function Inspector({ graph, result }: { graph: Graph; result?: GraphResult }) {
  const state = useEditor(),
    node = graph.nodes.find((n) => n.id === state.selected);
  const owner = node?.subgraph
    ? node
    : state.path.length
      ? graphAt(state.project.graph, state.path.slice(0, -1)).nodes.find(
          (n) => n.id === state.path.at(-1),
        )
      : undefined;
  const [variantName, setVariantName] = useState('我的 Transformer');
  useEffect(() => {
    if (owner) setVariantName(`${owner.label} · 变体`);
  }, [owner?.id]);
  const def = node ? registry.get(node.type) : undefined;
  const boundary = node && ['Inlet', 'Outlet'].includes(node.type);
  const replaceable =
    node && !node.subgraph && !['Inlet', 'Outlet', 'Input', 'Output'].includes(node.type);
  const compatible = replaceable
    ? registry
        .list()
        .filter(
          (d) =>
            !d.composite &&
            !['Input', 'Output'].includes(d.type) &&
            JSON.stringify(d.inputs.map((p) => p.id)) ===
              JSON.stringify(ports(node, registry).inputs.map((p) => p.id)) &&
            JSON.stringify(d.outputs.map((p) => p.id)) ===
              JSON.stringify(ports(node, registry).outputs.map((p) => p.id)),
        )
    : [];
  return (
    <aside className="inspector">
      <div className="panel-heading">
        <SlidersHorizontal size={16} />
        <h2>模块检查器</h2>
        <span className="tiny-badge">INSPECT</span>
      </div>
      <div className="inspector-scroll">
        {node && def ? (
          <>
            <div className="inspector-title">
              <span className="section-label">
                {def.category} / {node.type}
              </span>
              <input
                aria-label="模块名称"
                key={`${node.id}:${node.label}`}
                defaultValue={node.label}
                onBlur={(e) => state.renameNode(node.id, e.target.value)}
              />
              <p>{def.description}</p>
            </div>
            <div className="inspector-actions">
              {node.subgraph && (
                <button
                  className="soft-button"
                  onClick={() => state.navigate([...state.path, node.id])}
                >
                  <ArrowUpRight size={14} />
                  进入内部结构
                </button>
              )}
              {!boundary && (
                <button
                  aria-label="删除当前模块"
                  title="删除当前模块"
                  className="icon-button danger"
                  onClick={() => state.removeNodes([node.id])}
                >
                  <Trash2 size={15} />
                </button>
              )}
            </div>
            {node.customized && node.subgraph && (
              <div className="info-note">
                此模块已自定义。进入内部修改各算子，父层参数保留为来源记录。
              </div>
            )}
            {!boundary && def.fields.length > 0 && (
              <section className="parameters">
                <div className="section-label">模块参数</div>
                {def.fields.map((field) => (
                  <ParamField
                    key={field.key}
                    field={field}
                    node={node}
                    disabled={Boolean(node.customized && node.subgraph)}
                  />
                ))}
              </section>
            )}
            {replaceable && (
              <label className="replace-field">
                替换为兼容端口的算子
                <select
                  aria-label="替换算子"
                  value={node.type}
                  onChange={(e) => state.replace(node.id, e.target.value)}
                >
                  {compatible.map((d) => (
                    <option key={d.type} value={d.type}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <TensorView result={result?.nodes[node.id]} />
            <ModuleDetail node={node} result={result?.nodes[node.id]} />
          </>
        ) : (
          <div className="inspector-empty">
            <div className="empty-symbol">
              <BookOpen size={25} />
            </div>
            <h3>从一个模块开始观察</h3>
            <p>点击画布上的模块，查看参数、输入输出和每一维的含义。</p>
            <div className="tip-card">
              <strong>进入更细的一层</strong>
              <p>双击 Transformer 或 Attention，沿着同一份模型深入内部。</p>
            </div>
          </div>
        )}
        {owner?.subgraph && (
          <section className="variant-save">
            <div className="section-label">保存当前组合</div>
            <p>将「{owner.label}」作为自己的模块重复使用。</p>
            <input
              aria-label="变体名称"
              value={variantName}
              maxLength={200}
              onChange={(e) => setVariantName(e.target.value)}
            />
            <button className="soft-button" onClick={() => state.saveVariant(variantName)}>
              <BookmarkPlus size={15} />
              保存为我的变体
            </button>
          </section>
        )}
      </div>
    </aside>
  );
}
