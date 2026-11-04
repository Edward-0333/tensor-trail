import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { useUpdateNodeInternals, type Node, type NodeProps } from '@xyflow/react';
import { ChevronRight, AlertCircle, ArrowUpRight } from 'lucide-react';
import type { ModelNode, NodeResult, Port } from '../core/types';
import { isDictionarySpec, isTensorSpec, isIntegerSpec, shapeText } from '../core/types';
import { ModuleGlyph } from './ModuleGlyph';
import { PortHandles } from './PortHandles';
import { useEditor } from '../editor/store';
import { parseCalculations, calculationLabel, sliceLabel } from '../modules/calculation';
import { isContainerInput } from '../modules/containers';
import { parseFieldPath } from '../modules/indexing';
import { orderGraph } from '../core/graph';

export type TrailNode = Node<
  {
    model: ModelNode;
    result?: NodeResult;
    inputs: Port[];
    outputs: Port[];
    category: string;
    reverse: boolean;
    active: boolean;
    dimmed: boolean;
    open: () => void;
  },
  'module'
>;
export const tone = (category: string) =>
  category === 'Transformer' || category === '注意力算子'
    ? 'orange'
    : category === '卷积'
      ? 'purple'
      : category === '循环网络'
        ? 'cyan'
        : category === '输入输出' || category === '内部'
          ? 'teal'
          : category === '归一化'
            ? 'green'
            : ['维度操作', '索引与切片', '形状查询'].includes(category)
              ? 'sky'
              : category === '分支运算'
                ? 'pink'
                : 'blue';

export const categoryColor = (category: string) => `var(--category-${tone(category)})`;

export const isTensorNode = (type: string) =>
  ['Input', 'Output', 'Inlet', 'Outlet', 'TensorVariable'].includes(type);

function operationText(node: ModelNode): string {
  try {
    if (node.type === 'DictionaryGet')
      return parseFieldPath(node.params.path)
        .map((key) => `[${JSON.stringify(key)}]`)
        .join('');
    if (node.type === 'Slice') return sliceLabel(node.params);
    if (node.type === 'TensorIndex') return `轴 ${node.params.dim} · 索引 ${node.params.index}`;
    if (node.type === 'TensorSize') return `shape · 读取轴 ${node.params.axes}`;
    if (node.type === 'Compute')
      return parseCalculations(node.params.operations).map(calculationLabel).join(' › ');
  } catch {
    /* Incomplete parameters remain editable inside the group. */
  }
  return node.label;
}

export function ModuleNode({ data, selected }: NodeProps<TrailNode>) {
  const { model, result, inputs, outputs } = data;
  const [editing, setEditing] = useState(false);
  const cancelRename = useRef(false);
  const background = model.tensorColor;
  const rgb = background
    ? [1, 3, 5].map((offset) => parseInt(background.slice(offset, offset + 2), 16))
    : [];
  const ink =
    rgb.length && rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114 < 145 ? '#f8fafc' : '#172033';
  const spec = result?.outputs.out ?? Object.values(result?.outputs ?? {})[0] ?? result?.inputs.in;
  const axisLabels = spec && isTensorSpec(spec) && (
    <div className="node-axis-labels" aria-label="输出轴含义">
      {spec.axes.length ? (
        spec.axes.map((axis, i) => (
          <span key={i} title={`轴 ${i} · ${spec.shape[i]}`}>
            {axis}
          </span>
        ))
      ) : (
        <span>标量 · 无轴</span>
      )}
    </div>
  );
  const inputSide = model.portLayout?.input ?? (data.reverse ? 'right' : 'left');
  const outputSide = model.portLayout?.output ?? (data.reverse ? 'left' : 'right');
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(model.id);
  }, [model.id, inputSide, outputSide, inputs.length, outputs.length, updateNodeInternals]);
  const bad = result?.status === 'error' || result?.status === 'blocked';
  const container = isContainerInput(model.type);
  if (model.type === 'LocalParameter')
    return (
      <div
        className={`local-parameter-node tone-orange ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`局部参数 ${model.label}`}
        title={
          bad
            ? result?.messages.join('\n')
            : `${model.label} · ${shapeText(result?.outputs.out)} · 双击修改名称`
        }
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        {editing ? (
          <input
            className="local-parameter-editor nodrag nopan nowheel"
            aria-label="编辑局部参数名称"
            autoFocus
            defaultValue={model.label}
            maxLength={200}
            style={{ width: `${Math.max(2, Math.min(model.label.length, 12))}ch` }}
            onBlur={(event) => {
              if (!cancelRename.current)
                useEditor.getState().renameNode(model.id, event.target.value);
              setEditing(false);
            }}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) return;
              if (event.key === 'Enter') {
                event.preventDefault();
                event.currentTarget.blur();
              }
              if (event.key === 'Escape') {
                cancelRename.current = true;
                setEditing(false);
              }
            }}
            onDoubleClick={(event) => event.stopPropagation()}
          />
        ) : (
          <span
            className="local-parameter-name"
            onDoubleClick={(event) => {
              event.stopPropagation();
              cancelRename.current = false;
              setEditing(true);
            }}
          >
            {model.label}
          </span>
        )}
      </div>
    );
  if (container)
    return (
      <div
        className={`${model.type === 'DictionaryInput' ? 'dictionary-input-node' : 'tensor-node forward-input-node'} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        style={
          background
            ? ({ '--tensor-bg': background, '--tensor-ink': ink } as CSSProperties)
            : undefined
        }
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        {editing ? (
          <textarea
            className="forward-input-editor nodrag nopan nowheel"
            aria-label="编辑网络输入名称"
            autoFocus
            defaultValue={model.label}
            maxLength={200}
            rows={2}
            onBlur={(event) => {
              if (!cancelRename.current)
                useEditor.getState().renameNode(model.id, event.target.value);
              setEditing(false);
            }}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) return;
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.blur();
              }
              if (event.key === 'Escape') {
                cancelRename.current = true;
                setEditing(false);
              }
            }}
            onDoubleClick={(event) => event.stopPropagation()}
          />
        ) : (
          <div
            className="forward-input-text"
            title={`${model.label} · 双击修改文字`}
            onDoubleClick={(event) => {
              event.stopPropagation();
              cancelRename.current = false;
              setEditing(true);
            }}
          >
            {model.label}
          </div>
        )}
        {model.type === 'DictionaryInput' && (
          <div className="dictionary-node-summary">
            <strong>{spec && isDictionarySpec(spec) ? shapeText(spec) : 'dict · 待修正'}</strong>
            {spec && isDictionarySpec(spec) && (
              <span title={Object.keys(spec.fields).join(', ')}>
                {Object.keys(spec.fields).join(' · ') || '在属性面板中添加字段'}
              </span>
            )}
          </div>
        )}
      </div>
    );
  if (isTensorNode(model.type)) {
    const spec = result?.outputs.out ?? result?.inputs.in;
    const shape = spec ? shapeText(spec) : bad ? '待修正' : '等待输入';
    const tensorRole =
      model.type === 'TensorVariable'
        ? 'tensor-variable'
        : ['Input', 'Inlet'].includes(model.type)
          ? 'tensor-input'
          : 'tensor-output';
    return (
      <div
        className={`tensor-node ${spec && isIntegerSpec(spec) ? 'integer-value-node' : ''} ${tensorRole} ${data.reverse ? 'flow-left' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={model.type === 'TensorVariable' ? `张量变量 ${model.label}` : undefined}
        style={
          background
            ? ({ '--tensor-bg': background, '--tensor-ink': ink } as CSSProperties)
            : undefined
        }
      >
        {editing ? (
          <input
            className="tensor-label-input nodrag nowheel"
            aria-label={spec && isIntegerSpec(spec) ? '编辑变量名称' : '编辑 Tensor 名称'}
            autoFocus
            defaultValue={model.label}
            maxLength={200}
            onBlur={(e) => {
              if (!cancelRename.current) useEditor.getState().renameNode(model.id, e.target.value);
              setEditing(false);
            }}
            onKeyDown={(e) => {
              if (e.nativeEvent.isComposing) return;
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') {
                cancelRename.current = true;
                setEditing(false);
              }
            }}
            onDoubleClick={(e) => e.stopPropagation()}
          />
        ) : (
          <div
            className="tensor-node-label"
            title={`${model.label} · 双击修改名称`}
            onDoubleClick={(e) => {
              e.stopPropagation();
              cancelRename.current = false;
              setEditing(true);
            }}
          >
            {model.label}
          </div>
        )}
        <div className="tensor-node-shape">
          <PortHandles
            inputs={inputs}
            outputs={outputs}
            inputSide={inputSide}
            outputSide={outputSide}
            labels={false}
          />
          <code title={spec && isTensorSpec(spec) ? `${shape} · ${spec.axes.join(', ')}` : shape}>
            {shape}
          </code>
        </div>
        {axisLabels}
      </div>
    );
  }

  if (model.type === 'TensorSize') {
    return (
      <div
        className={`compact-operation-node size-query-node tone-sky ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${model.label}，读取轴 ${model.params.axes}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · 每个输出为整数尺寸`}
        style={{
          minHeight: Math.max(
            34,
            ((inputSide === outputSide
              ? inputs.length + outputs.length
              : Math.max(inputs.length, outputs.length)) +
              1) *
              12,
          ),
        }}
      >
        <code title={`读取轴 ${model.params.axes}`}>Size · 轴 {model.params.axes}</code>
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
      </div>
    );
  }

  if (model.type === 'DictionaryGet' || model.type === 'Slice') {
    const isSlice = model.type === 'Slice';
    const kind = isSlice ? '切片 Slice' : '字典字段读取';
    let expression = isSlice ? '切片参数待修正' : '选择字典字段';
    try {
      expression = isSlice
        ? sliceLabel(model.params)
        : parseFieldPath(model.params.path)
            .map((key) => `[${JSON.stringify(key)}]`)
            .join('');
    } catch {
      /* The inspector shows validation errors for incomplete parameters. */
    }
    return (
      <div
        className={`compact-operation-node ${isSlice ? 'slice-node' : 'field-read-node'} tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${kind}：${expression}`}
        title={
          bad ? `${expression}\n${result?.messages.join('\n')}` : `${model.label} · ${expression}`
        }
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <code>{expression}</code>
      </div>
    );
  }

  if (model.type === 'CalculationGroup' && model.subgraph) {
    const child = model.subgraph;
    const steps = orderGraph(child)
      .map((id) => child.nodes.find((n) => n.id === id)!)
      .filter((n) => !['Inlet', 'Outlet'].includes(n.type));
    return (
      <div
        className={`calculation-group-node tone-sky ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${model.label}，${steps.length} 个操作，双击进入编辑`}
        title={bad ? result?.messages.join('\n') : `${model.label} · 双击编辑内部操作`}
        style={{ minHeight: Math.max(inputs.length, outputs.length) * 24 + 16 }}
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={inputs.length > 1 || outputs.length > 1}
        />
        <ol className="calculation-group-steps">
          {steps.map((node) => (
            <li key={node.id}>
              <code title={`${node.label} · ${operationText(node)}`}>{operationText(node)}</code>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  if (model.type === 'Concat')
    return (
      <div
        className={`cat-node tone-pink ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · 沿指定轴拼接`}
      >
        <strong>Cat</strong>
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          horizontalArea={{ top: 12, height: 64 }}
        />
        {axisLabels}
      </div>
    );
  let operations: string[] = [];
  if (model.type === 'Compute') {
    try {
      operations = parseCalculations(model.params.operations).map(calculationLabel);
    } catch {
      operations = ['步骤无效'];
    }
  }
  if (model.type === 'TensorIndex')
    operations = [`轴 ${model.params.dim}`, `索引 ${model.params.index}`];
  const rows = Math.max(inputs.length, outputs.length, 1);
  const indexing = model.type === 'TensorIndex';
  const bodyHeight = Math.max(
    rows * 27 + 12,
    indexing
      ? operations.length * 34 + 24
      : operations.length
        ? Math.ceil(operations.length / 3) * 34 + 18
        : 72,
  );
  const hasTop =
    (inputs.length > 0 && inputSide === 'top') || (outputs.length > 0 && outputSide === 'top');
  const hasBottom =
    (inputs.length > 0 && inputSide === 'bottom') ||
    (outputs.length > 0 && outputSide === 'bottom');
  return (
    <div
      className={`module-node ${indexing ? 'indexing-node' : ''} tone-${tone(data.category)} ${hasTop ? 'has-top-ports' : ''} ${hasBottom ? 'has-bottom-ports' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
      data-testid={`node-${model.id}`}
      title={model.subgraph ? `双击进入 ${model.label} 的 forward()` : undefined}
    >
      <div className="node-heading">
        <div className="node-title">
          <strong className="node-kind" title={`${model.label} · ${model.type}`}>
            {model.label}
          </strong>
        </div>
        {model.subgraph && <ArrowUpRight className="node-expand-icon" size={15} />}
      </div>
      <div className="node-ports" style={{ minHeight: bodyHeight }}>
        <div className="node-operation">
          {['Compute', 'TensorIndex'].includes(model.type) ? (
            <div className="calculation-chips">
              {operations.map((label, i) => (
                <span key={i} title={label}>
                  {label}
                </span>
              ))}
            </div>
          ) : (
            <ModuleGlyph type={model.type} />
          )}
        </div>
      </div>
      <PortHandles
        inputs={inputs}
        outputs={outputs}
        inputSide={inputSide}
        outputSide={outputSide}
        horizontalArea={{ top: 3 + (hasTop ? 60 : 47), height: bodyHeight }}
      />
      <div className="node-footer">
        {bad ? (
          <span className="node-error">
            <AlertCircle size={12} />
            {result.status === 'blocked' ? '等待输入或上游修正' : '形状不兼容'}
          </span>
        ) : model.subgraph ? (
          <button className="nodrag" onClick={data.open}>
            进入 forward() <ChevronRight size={12} />
          </button>
        ) : (
          <span>
            {result?.status === 'warning' ? '符号约束待验证' : result ? '形状已推导' : '等待连接'}
          </span>
        )}
      </div>
      {axisLabels}
    </div>
  );
}
