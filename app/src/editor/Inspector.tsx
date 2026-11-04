import { useEffect, useState } from 'react';
import { ArrowUpRight, MousePointer2, BookmarkPlus, SlidersHorizontal, Trash2 } from 'lucide-react';
import type { Field, Graph, GraphResult, ModelNode, PortSide, ValueSpec } from '../core/types';
import { graphAt, ports } from '../core/graph';
import { resolveForwardView } from '../core/hierarchy';
import { registry } from '../modules';
import { TensorView } from '../views/TensorView';
import { ModuleDetail } from '../views/details';
import { tone, isTensorNode } from '../views/ModuleNode';
import { dimensions } from '../modules/math';
import { emptyDictionary } from '../core/dictionary';
import { CalculationEditor } from './CalculationEditor';
import { DictionaryInputEditor } from './DictionaryInputEditor';
import { isContainerInput } from '../modules/containers';
import { useEditor } from './store';
import { HyperparametersPanel } from './HyperparametersPanel';
import { DictionaryPathField, IndexExpressionField } from './IndexingFields';

function ParamField({
  field,
  node,
  disabled,
  input,
}: {
  field: Field;
  node: ModelNode;
  disabled: boolean;
  input?: ValueSpec;
}) {
  const value = node.params[field.key];
  const label = `${field.label} (${field.key})`;
  const update = (v: string | number | boolean) =>
    useEditor.getState().setParam(node.id, field.key, v);
  if (node.type === 'TensorIndex') return <IndexExpressionField node={node} field={field} />;
  if (node.type === 'DictionaryGet') return <DictionaryPathField node={node} input={input} />;
  if (isContainerInput(node.type) && field.key === 'schema') {
    if (node.type !== 'DictionaryInput' && (value === undefined || value === ''))
      return (
        <div>
          <p className="muted">输入数据类型尚未指定。</p>
          <button className="soft-button" onClick={() => update(JSON.stringify(emptyDictionary()))}>
            添加字典类型与尺寸
          </button>
        </div>
      );
    return (
      <div>
        <div className="section-label">输入数据 · dict（字典）</div>
        <p className="muted">字典本身没有统一尺寸；展开字段查看各 Tensor 的形状和类型。</p>
        <DictionaryInputEditor key={node.id} node={node} />
      </div>
    );
  }
  if (node.type === 'Compute' && field.key === 'operations')
    return <CalculationEditor node={node} />;
  if (isContainerInput(node.type) && field.key === 'text')
    return (
      <label className="container-text-field">
        <span>备注（可选）</span>
        <textarea
          aria-label="网络输入备注"
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
  const [tab, setTab] = useState<'node' | 'hyperparameters'>('node');
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
    ![
      'Inlet',
      'Outlet',
      'Input',
      'Output',
      'TensorSize',
      'LocalParameter',
      'TensorVariable',
    ].includes(node.type);
  const compatible = replaceable
    ? registry
        .list()
        .filter(
          (d) =>
            !d.composite &&
            !['Input', 'Output', 'LocalParameter', 'TensorVariable'].includes(d.type) &&
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
      <div className="inspector-tabs" role="tablist" aria-label="属性面板内容">
        {(
          [
            ['node', '节点属性'],
            ['hyperparameters', '模型超参数'],
          ] as const
        ).map(([key, label], index) => (
          <button
            key={key}
            id={`inspector-tab-${key}`}
            role="tab"
            aria-selected={tab === key}
            aria-controls={`inspector-panel-${key}`}
            tabIndex={tab === key ? 0 : -1}
            onClick={() => setTab(key)}
            onKeyDown={(event) => {
              if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
                event.preventDefault();
                const next =
                  event.key === 'Home'
                    ? 'node'
                    : event.key === 'End'
                      ? 'hyperparameters'
                      : index
                        ? 'node'
                        : 'hyperparameters';
                setTab(next);
                document.getElementById(`inspector-tab-${next}`)?.focus();
              }
            }}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === 'hyperparameters' ? (
        <div
          className="inspector-scroll"
          id="inspector-panel-hyperparameters"
          role="tabpanel"
          aria-labelledby="inspector-tab-hyperparameters"
        >
          <HyperparametersPanel />
        </div>
      ) : (
        <div
          className="inspector-scroll"
          id="inspector-panel-node"
          role="tabpanel"
          aria-labelledby="inspector-tab-node"
        >
          {!state.hasProject ? (
            <div className="model-summary">
              <div className="section-label">尚未打开项目</div>
              <p className="muted">打开或新建项目后，在这里查看模块属性。</p>
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
                    {node.type === 'CalculationGroup' ? '编辑内部操作' : '进入 forward()'}
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
              {node.type === 'CalculationGroup' && (
                <section className="calculation-group-actions">
                  <p className="microcopy">
                    操作与参数完整保存在组合内部；外部连接保留。可双击进入编辑，或拆开恢复独立节点。
                  </p>
                  <button className="soft-button" onClick={() => state.splitCalculation(node.id)}>
                    拆开组合
                  </button>
                </section>
              )}
              {node.customized && node.subgraph && node.type !== 'CalculationGroup' && (
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
                      input={result?.nodes[node.id]?.inputs.in}
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
              {!container && (
                <TensorView
                  result={result?.nodes[node.id]}
                  outputPorts={
                    node.type === 'TensorSize' ? ports(node, registry).outputs : undefined
                  }
                />
              )}
              <ModuleDetail node={node} result={result?.nodes[node.id]} />
            </>
          ) : (
            <div className="model-summary">
              <div className="section-label">当前层级概览</div>
              <p className="forward-scope-title">
                {resolveForwardView(state.project.graph, state.path).current.title}
              </p>
              <p className="muted">
                本层展示直接计算和模块调用；子模块保持封装，双击进入内部 forward()。
              </p>
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
                <p>单击查看参数和形状；双击有子图的模块进入 forward()，通过面包屑返回上层。</p>
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
      )}
    </aside>
  );
}
