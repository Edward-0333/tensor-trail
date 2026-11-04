import { useEffect, useState } from 'react';
import { ArrowUpRight, MousePointer2, BookmarkPlus, SlidersHorizontal, Trash2 } from 'lucide-react';
import type { Field, Graph, GraphResult, ModelNode, PortSide } from '../core/types';
import { graphAt, ports } from '../core/graph';
import { registry } from '../modules';
import { TensorView } from '../views/TensorView';
import { ModuleDetail } from '../views/details';
import { tone, isTensorNode } from '../views/ModuleNode';
import { dimensions } from '../modules/math';
import { CalculationEditor } from './CalculationEditor';
import { isContainerInput } from '../modules/containers';
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
  if (node.type === 'Compute' && field.key === 'operations')
    return <CalculationEditor node={node} />;
  if (isContainerInput(node.type) && field.key === 'text')
    return (
      <label className="container-text-field">
        <span>备注（可选）</span>
        <textarea
          aria-label="Forward 输入备注"
          key={`${node.id}:${value}`}
          defaultValue={String(value ?? '')}
          maxLength={1999}
          rows={6}
          spellCheck={false}
          onBlur={(event) => {
            if (event.target.value !== value) update(event.target.value);
          }}
        />
        <small>备注只保存在属性中；画布显示上方的模块名称。</small>
      </label>
    );
  if (node.type === 'Input' && field.key === 'axes') {
    let shape: (string | number)[] = [];
    try {
      shape = dimensions(node.params.shape);
    } catch {
      /* Correct the shape first. */
    }
    const names = String(value ?? '')
      .split(',')
      .map((s) => s.trim());
    return (
      <fieldset className="axis-name-fields">
        <legend>轴含义</legend>
        {shape.map((size, i) => (
          <label key={i}>
            <span>
              轴 {i} · {size}
            </span>
            <input
              aria-label={`轴 ${i} 的含义`}
              key={`${node.id}:${i}:${names[i]}`}
              defaultValue={names[i] || `axis${i}`}
              onBlur={(e) => useEditor.getState().setAxis(node.id, i, e.target.value)}
            />
          </label>
        ))}
        {!shape.length && <small>先填写有效的输入形状。</small>}
      </fieldset>
    );
  }
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
  const container = node && isContainerInput(node.type);
  const boundary = node && ['Inlet', 'Outlet'].includes(node.type);
  const replaceable =
    node &&
    !node.subgraph &&
    !container &&
    !['Inlet', 'Outlet', 'Input', 'Output'].includes(node.type);
  const compatible = replaceable
    ? registry
        .list()
        .filter(
          (d) =>
            !d.composite &&
            !['Input', 'Output'].includes(d.type) &&
            !isContainerInput(d.type) &&
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
        <h2>属性</h2>
        <span className="tiny-badge">{node ? '模块' : '总览'}</span>
      </div>
      <div className="inspector-scroll">
        {!state.hasProject ? (
          <div className="model-summary">
            <div className="section-label">尚未打开项目</div>
            <p className="muted">选择 JSON 项目文件后，在这里查看模块属性。</p>
          </div>
        ) : node && def ? (
          <>
            <div className="inspector-title">
              <span className="section-label">
                {def.category} / {node.type}
              </span>
              {container ? (
                <textarea
                  className="forward-input-name"
                  aria-label="模块名称"
                  key={`${node.id}:${node.label}`}
                  defaultValue={node.label}
                  maxLength={200}
                  rows={2}
                  onBlur={(event) => state.renameNode(node.id, event.target.value)}
                />
              ) : (
                <input
                  aria-label="模块名称"
                  key={`${node.id}:${node.label}`}
                  defaultValue={node.label}
                  onBlur={(e) => state.renameNode(node.id, e.target.value)}
                />
              )}
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
            <section className="port-layout-settings">
              <div className="section-label">端口布局</div>
              {(['input', 'output'] as const).map((kind) => (
                <label className="param-field" key={kind}>
                  <span>{kind === 'input' ? '输入位置' : '输出位置'}</span>
                  <select
                    aria-label={kind === 'input' ? '输入端口位置' : '输出端口位置'}
                    value={node.portLayout?.[kind] ?? 'auto'}
                    disabled={
                      !ports(node, registry)[kind === 'input' ? 'inputs' : 'outputs'].length
                    }
                    onChange={(event) =>
                      state.setPortSide(node.id, kind, event.target.value as PortSide | 'auto')
                    }
                  >
                    <option value="auto">自动</option>
                    <option value="left">左侧</option>
                    <option value="right">右侧</option>
                    <option value="top">上方</option>
                    <option value="bottom">下方</option>
                  </select>
                </label>
              ))}
            </section>
            {(isTensorNode(node.type) || container) && (
              <section className="tensor-color-settings">
                <div className="section-label">{container ? '输入背景' : 'Tensor 背景'}</div>
                <div className="tensor-color-options">
                  <button
                    className={!node.tensorColor ? 'active' : ''}
                    onClick={() => state.setTensorColor(node.id)}
                    aria-label={container ? '输入背景透明' : 'Tensor 背景透明'}
                  >
                    透明
                  </button>
                  {[
                    ['紫色', '#ede9fe'],
                    ['蓝色', '#dbeafe'],
                    ['绿色', '#dcfce7'],
                    ['橙色', '#ffedd5'],
                    ['粉色', '#fce7f3'],
                    ['灰色', '#e5e7eb'],
                  ].map(([label, color]) => (
                    <button
                      key={color}
                      title={label}
                      aria-label={`${container ? '输入背景' : 'Tensor 背景'}${label}`}
                      aria-pressed={node.tensorColor === color}
                      style={{ background: color }}
                      onClick={() => state.setTensorColor(node.id, color)}
                    />
                  ))}
                  <input
                    aria-label={container ? '自定义输入背景色' : '自定义 Tensor 背景色'}
                    type="color"
                    value={node.tensorColor ?? '#ede9fe'}
                    onChange={(e) => state.setTensorColor(node.id, e.target.value)}
                  />
                </div>
              </section>
            )}
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
            {!container && <TensorView result={result?.nodes[node.id]} />}
            <ModuleDetail node={node} result={result?.nodes[node.id]} />
          </>
        ) : (
          <div className="model-summary">
            <div className="section-label">当前层级概览</div>
            <dl className="summary-metrics">
              <div>
                <dt>模块</dt>
                <dd>{graph.nodes.length}</dd>
              </div>
              <div>
                <dt>连接</dt>
                <dd>{graph.edges.length}</dd>
              </div>
              <div>
                <dt>可展开的组合</dt>
                <dd>{graph.nodes.filter((n) => n.subgraph).length}</dd>
              </div>
              <div>
                <dt>已推导输出的模块</dt>
                <dd>
                  {
                    Object.values(result?.nodes ?? {}).filter(
                      (r) => Object.keys(r.outputs).length > 0,
                    ).length
                  }
                </dd>
              </div>
            </dl>
            <div className="section-label">模块分布</div>
            <div className="category-breakdown">
              {[...new Set(graph.nodes.map((n) => registry.get(n.type)?.category ?? '其他'))].map(
                (category) => (
                  <div className={`tone-${tone(category)}`} key={category}>
                    <span className="category-dot" />
                    <span>{category}</span>
                    <code>
                      {
                        graph.nodes.filter(
                          (n) => (registry.get(n.type)?.category ?? '其他') === category,
                        ).length
                      }
                    </code>
                  </div>
                ),
              )}
            </div>
            <div className="tip-card">
              <MousePointer2 size={18} />
              <strong>选择模块，观察 Tensor</strong>
              <p>单击查看参数和形状。双击 Transformer 或 Attention，继续探索内部结构。</p>
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
