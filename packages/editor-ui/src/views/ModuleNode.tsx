import { plutoCodeTypes, plutoOperationLabel } from '@tensor-trail/model-core/modules/plutoForwardOps';
import { imageMetadataCodeTypes } from '@tensor-trail/model-core/modules/imageMetadata';
import { parameterName } from '@tensor-trail/model-core/modules/moduleReference';
import { matchingPreparationLabel } from '@tensor-trail/model-core/modules/matchingPreparation';
import { sortingLabel } from '@tensor-trail/model-core/modules/sorting';
import { loopOperationLabels } from '@tensor-trail/model-core/modules/loopOps';
import { attributeBinding, attributeOwnerLabel, attributeTypes } from '@tensor-trail/model-core/core/attributes';
import { plutoCompositeTypes } from '@tensor-trail/model-core/modules/plutoForward';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { useUpdateNodeInternals, type Node, type NodeProps } from '@xyflow/react';
import { ChevronRight, AlertCircle, ArrowUpRight } from 'lucide-react';
import type { ModelNode, NodeResult, Port } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSpec, isIntegerSpec, isIntegerListSpec, isControlSpec, isModuleSpec, isDeviceSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { ModuleGlyph } from './ModuleGlyph';
import { PortHandles } from './PortHandles';
import { canvasValueLabel } from './canvasValueLabel';
import { valueDisplayText } from './valueDisplayText';
import { IfRegionNode } from './IfRegionNode';
import { useEditor } from '../editor/store';
import { parseCalculations, calculationLabel, sliceLabel } from '@tensor-trail/model-core/modules/calculation';
import { isContainerInput } from '@tensor-trail/model-core/modules/containers';
import { parseFieldPath, tensorIndexLabel } from '@tensor-trail/model-core/modules/indexing';
import { orderGraph } from '@tensor-trail/model-core/core/graph';
import { unsqueezeLabel } from '@tensor-trail/model-core/modules/unsqueeze';
import { anyLabel } from '@tensor-trail/model-core/modules/logical';
import { castLabel } from '@tensor-trail/model-core/modules/cast';
import { functionTitle } from '@tensor-trail/model-core/core/hierarchy';
import type { LocalParameterColor } from './localParameterColors';
import { agentCompositeTypes, agentCodeTypes } from '@tensor-trail/model-core/modules/agentCompletion';
import { mapCompositeTypes } from '@tensor-trail/model-core/modules/mapEncoder';
import { natCompositeTypes } from '@tensor-trail/model-core/modules/natSequence';
import { DEFAULT_TENSOR_PALETTE, gradLabel, gradSourceLabel, tensorGradState, tensorInk } from '@tensor-trail/model-core/core/tensorAppearance';

export type TrailNode = Node<
  {
    model: ModelNode;
    localParameterColor?: LocalParameterColor;
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
            : ['维度操作', '索引与切片', '形状查询', '数学运算', '逻辑与归约'].includes(category)
              ? 'sky'
              : category === '分支运算'
                ? 'pink'
                : 'blue';

export const categoryColor = (category: string) => `var(--category-${tone(category)})`;

export const isTensorNode = (type: string) =>
  ['Input', 'Output', 'Inlet', 'Outlet', 'TensorVariable', 'ValueVariable', 'BufferSource'].includes(type);

const elementwiseLabels: Record<string, string> = {
  BitwiseAnd: '&',
  Subtract: '−',
  Where: 'where',
  ZerosLike: 'zeros_like',
  AlignTrailingDims: '尾部补维',
  Divide: 'a / b',
};
const featureOperations = [
  ...Object.keys(loopOperationLabels),
  'IntegerGreater',
  'Squeeze',
  'Cos',
  'Sin',
  'IntegerMultiply',
  'View',
  'Permute',
  'Flatten',
  'BooleanIndex',
  'Contiguous',
  'Zeros',
  'NewZeros',
  'OnesScaled',
  'Arange',
  'SortRowsByMax',
  'MaxWithIndices',
  'Sort',
  'RowIndex',
  'GetDevice',
  'BooleanAssign',
  'IndexAssign',
  'RepeatBatch',
  'Repeat',
  'Reshape',
  'ReduceMax',
  'ReduceSum',
];

function operationText(node: ModelNode, result?: NodeResult): string {
  try {
    if (node.type === 'TensorItemInt' && node.params.cast) return 'int(x)';
    if (loopOperationLabels[node.type]) return loopOperationLabels[node.type];
    if (node.type === 'GetDevice') return '.device';
    const matchingLabel = matchingPreparationLabel(node);
    if (matchingLabel) return matchingLabel;
    const sortLabel = sortingLabel(node);
    if (sortLabel) return sortLabel;
    const plutoLabel = plutoOperationLabel(node);
    if (plutoLabel) return plutoLabel;
    if (elementwiseLabels[node.type]) return elementwiseLabels[node.type];
    if (node.type === 'Cos') return 'cos()';
    if (node.type === 'Sin') return 'sin()';
    if (node.type === 'IntegerMultiply') return '× int';
    if (node.type === 'View') return 'view()';
    if (node.type === 'Permute') return 'permute()';
    if (node.type === 'BooleanIndex') return 'x[mask]';
    if (node.type === 'Contiguous') return 'contiguous()';
    if (node.type === 'Repeat') return 'repeat()';
    if (node.type === 'Reshape') return 'reshape()';
    if (node.type === 'ReduceMax') return 'max()';
    if (node.type === 'ReduceSum') return 'sum()';
    if (node.type === 'RepeatBatch') return 'repeat(B)';
    if (node.type === 'IndexAssign')
      return node.params.dim === 1 ? `x[:, ${node.params.index}] = v` : 'index_put()';
    if (node.type === 'Zeros') return 'zeros()';
    if (node.type === 'NewZeros') return 'new_zeros()';
    if (node.type === 'BooleanAssign') return 'x[mask] = v';
    if (node.type === 'Flatten')
      return `flatten(${node.params.start_dim ?? 1}, ${node.params.end_dim ?? -1})`;
    if (node.type === 'ToVector') return functionTitle(node.label);
    if (node.type === 'GetParameter') return `读取 .${parameterName(node.params.name)}`;
    if (node.type === 'DictionaryGet')
      return parseFieldPath(node.params.path)
        .map((key) => `[${JSON.stringify(key)}]`)
        .join('');
    if (node.type === 'Slice') return sliceLabel(node.params);
    if (node.type === 'TensorIndex') {
      const input = result?.inputs.in;
      return tensorIndexLabel(node.params, input && isTensorSpec(input) ? input.shape.length : undefined);
    }
    if (node.type === 'Unsqueeze') return unsqueezeLabel(node.params);
    if (node.type === 'Any') return anyLabel(node.params);
    if (node.type === 'Invert') return '~';
    if (node.type === 'Cast') return castLabel(node.params);
    if (node.type === 'TensorSize') return 'size()';
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
  const palette = useEditor((state) => state.project.tensorPalette) ?? DEFAULT_TENSOR_PALETTE;
  const spec = result?.outputs.out ?? Object.values(result?.outputs ?? {})[0] ?? result?.inputs.in;
  const tensor = spec && isTensorSpec(spec) ? spec : undefined;
  const showGrad = !!tensor || (!spec && isTensorNode(model.type) && model.type !== 'Inlet');
  const gradState = tensorGradState(tensor);
  const background = showGrad ? palette[gradState] : undefined;
  const border = model.tensorBorderColor ?? model.tensorColor ?? (tensor?.parameterRef ? '#60a5fa' : tensor?.bufferRef ? '#059669' : undefined);
  const tensorStyle = {
    ...(background ? { '--tensor-bg': background, '--tensor-ink': tensorInk(background) } : {}),
    ...(border ? { '--tensor-border': border } : {}),
  } as CSSProperties;
  const gradBadge = showGrad && <span className="tensor-grad-badge" data-grad-state={gradState}
    title={`requires_grad=${gradState === 'unknown' ? '未知' : gradState === 'on' ? 'True' : 'False'} · ${tensor ? gradSourceLabel(tensor) : '尚未确定'}；不代表已执行 backward`}>{gradLabel(gradState)}</span>;
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
  const sideCount = (side: string) =>
    (inputSide === side ? inputs.length : 0) + (outputSide === side ? outputs.length : 0);
  const verticalPorts = Math.max(sideCount('left'), sideCount('right'));
  const horizontalPorts = Math.max(sideCount('top'), sideCount('bottom'));
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(model.id);
  }, [model.id, inputSide, outputSide, inputs.length, outputs.length, updateNodeInternals]);
  const bad = result?.status === 'error' || result?.status === 'blocked';
  const container = isContainerInput(model.type);
  if (model.type === 'IfRegion') return <IfRegionNode model={model} selected={selected} />;
  if (model.type === 'WhileLoop') return (
    <div className={`while-loop-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={`node-${model.id}`} title="双击进入循环体；只展示一轮符号模板，未执行循环" style={{ minHeight: Math.max(140,(verticalPorts+1)*24) }}>
      <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels />
      <span className="control-role-badge">WHILE <ArrowUpRight size={14} /></span>
      <strong>{model.label}</strong><code>i &lt; num_points</code>
      <small>状态：{String(model.params.states)} · i += 1</small>
      <small>一轮符号模板 · 未执行</small>
    </div>
  );
  if (model.type === 'IfCondition') return (
    <div className={`condition-test-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={`node-${model.id}`} title="count == quota；虚线表示控制依赖，实际分支未采集">
      <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels />
      <strong>{model.label}</strong><code>count == quota</code><small>条件比较 · 未执行</small>
    </div>
  );
  if (model.type === 'BranchValue') return (
    <div className={`branch-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={`node-${model.id}`} title="仅静态展示此分支，未执行数值计算">
      <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
      <strong>{model.params.branch ? 'True 分支' : 'False 分支'}</strong>
      <small>{model.params.branch ? '更新剩余行' : 'maps 不修改'}</small>
    </div>
  );
  if (model.type === 'IntegerList' || (spec && (isIntegerListSpec(spec) || isControlSpec(spec)) && isTensorNode(model.type))) return (
    <div className={`control-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`} data-testid={`node-${model.id}`} title={spec ? shapeText(spec) : '待连接'}>
      <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
      <strong>{model.label}</strong><code>{spec ? shapeText(spec) : 'list[int]'}</code>
    </div>
  );
  if (model.type === 'DeviceVariable' || (spec && isDeviceSpec(spec) && isTensorNode(model.type))) {
    return (
      <div
        className={`device-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`设备值 ${model.label}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · ${spec ? shapeText(spec) : '待连接 Device'} · 设备元数据，不是 Tensor；实际设备未采集`}
      >
        <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
        <span className="device-role-badge">DEVICE</span>
        <strong>{model.label}</strong>
      </div>
    );
  }
  if (attributeTypes.has(model.type)) {
    const write = model.type === 'SetAttribute';
    let owner = '属性路径待修正', name = String(model.params.name ?? '属性');
    try {
      const binding = attributeBinding(model.params);
      owner = attributeOwnerLabel(binding.owner);
      name = binding.name;
    } catch { /* Invalid bindings remain editable in the inspector. */ }
    return (
      <div
        className={`attribute-node ${write ? 'attribute-set-node' : 'attribute-get-node'} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${write ? '写属性' : '读属性'} ${owner}.${name}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · ${owner}.${name} · 属性绑定，不是 forward 形参`}
        style={tensorStyle}
      >
        <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
        <div className="attribute-node-heading"><span className="attribute-role-badge">{write ? 'SET' : 'GET'}</span><strong>{name}</strong></div>
        <code className="attribute-owner">{owner}</code>
        {model.params.version && <small className="attribute-version">赋值版本：{String(model.params.version)}</small>}
        <div className="attribute-value-summary">{spec && <code>{valueDisplayText(spec)}</code>}{gradBadge}</div>
      </div>
    );
  }
  if (model.type === 'LocalParameter')
    return (
      <div
        className={`local-parameter-node tone-orange ${data.localParameterColor && data.localParameterColor.count > 1 ? 'distinct-local-parameter' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`局部参数 ${model.label}`}
        style={
          data.localParameterColor
            ? ({ '--local-parameter-tone': data.localParameterColor.color } as CSSProperties)
            : undefined
        }
        title={
          bad
            ? result?.messages.join('\n')
            : `${model.label} · ${shapeText(result?.outputs.out)}${data.localParameterColor && data.localParameterColor.count > 1 ? ` · 同名独立定义 ${data.localParameterColor.index + 1}/${data.localParameterColor.count} · 来源 ${data.localParameterColor.ownerId}` : ''} · 双击修改名称`
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
  if (container || model.type === 'Inlet')
    return (
      <div
        className={`${model.type === 'DictionaryInput' ? 'dictionary-input-node' : 'tensor-node forward-input-node'} ${model.type === 'Inlet' ? 'module-input-node' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        style={tensorStyle}
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
            aria-label={model.type === 'Inlet' ? '编辑模块输入名称' : '编辑网络输入名称'}
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
        {gradBadge}
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
    const moduleReference = spec && isModuleSpec(spec);
    const shape = spec ? canvasValueLabel(spec, 'node') : bad ? '待修正' : '等待输入';
    const tensorRole = ['TensorVariable', 'ValueVariable'].includes(model.type)
      ? 'tensor-variable'
      : ['Input', 'Inlet'].includes(model.type)
        ? 'tensor-input'
        : 'tensor-output';
    return (
      <div
        className={`tensor-node ${spec && isModuleSpec(spec) ? 'module-reference-node' : ''} ${spec && isIntegerSpec(spec) ? 'integer-value-node' : ''} ${tensorRole} ${data.reverse ? 'flow-left' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={
          ['TensorVariable', 'ValueVariable'].includes(model.type)
            ? `${spec && !isTensorSpec(spec) ? '值变量' : '张量变量'} ${model.label}`
            : undefined
        }
        style={tensorStyle}
      >
        {editing ? (
          <input
            className="tensor-label-input nodrag nowheel"
            aria-label={spec && !isTensorSpec(spec) ? '编辑变量名称' : '编辑 Tensor 名称'}
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
        {tensor?.parameterRef && <span className="parameter-role-badge" title={`${tensor.parameterRef.module}.${tensor.parameterRef.name} · 注册参数，不等于已执行反向传播`}>Parameter</span>}
        {tensor?.bufferRef && <span className="buffer-role-badge" title={`${tensor.bufferRef.reference} · 注册 buffer，不是 nn.Parameter；persistent=${tensor.bufferRef.persistent}`}>BUFFER</span>}
        {gradBadge}
        {moduleReference ? (
          <PortHandles
            inputs={inputs}
            outputs={outputs}
            inputSide={inputSide}
            outputSide={outputSide}
            labels={false}
          />
        ) : <div className="tensor-node-shape">
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
        </div>}
        {axisLabels}
      </div>
    );
  }

  if (
    model.type === 'Submodule' ||
    natCompositeTypes.has(model.type) ||
    agentCompositeTypes.has(model.type) ||
    mapCompositeTypes.has(model.type) ||
    plutoCompositeTypes.has(model.type)
  ) {
    const pending = outputs.length === 0;
    const bodyHeight = Math.max(48, Math.max(inputs.length, outputs.length) * 24 + 24);
    const hasTop = inputSide === 'top' || (outputs.length > 0 && outputSide === 'top');
    const hasBottom = inputSide === 'bottom' || (outputs.length > 0 && outputSide === 'bottom');
    return (
      <div
        className={`module-node submodule-node tone-${tone(data.category)} ${hasTop ? 'has-top-ports' : ''} ${hasBottom ? 'has-bottom-ports' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`子模块 ${model.label}`}
        title={`双击进入 ${model.label} 的 forward()`}
      >
        <div className="node-heading">
          <div className="node-title">
            <strong className="node-kind">{model.label}</strong>
          </div>
          <ArrowUpRight className="node-expand-icon" size={15} />
        </div>
        <div
          className={`submodule-body ${pending ? 'pending' : ''}`}
          style={{ minHeight: bodyHeight }}
        >
          {bad
            ? '查看输入或内部提示'
            : pending
              ? '内部流程待完善'
              : model.type === 'EgoHistoryBranch'
                ? model.params.use_ego_history
                  ? '条件为假 · 保留历史编码'
                  : '条件为真 · 替换自车编码'
                : model.type === 'ReferencePlanningBranch'
                  ? model.params.available
                    ? 'R > 0 · 规划解码'
                    : 'R = 0 · 返回 None'
                  : '子模块 · forward()'}
        </div>
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          horizontalArea={{ top: 3 + (hasTop ? 60 : 47), height: bodyHeight }}
        />
        <div className="node-footer">
          <button className="nodrag" onClick={data.open}>
            进入 forward() <ChevronRight size={12} />
          </button>
        </div>
      </div>
    );
  }

  if (model.type === 'IntegerParameter') {
    return (
      <div
        className={`compact-operation-node tone-orange ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        title={
          bad ? result?.messages.join('\n') : `${model.params.expression} → ${shapeText(spec)}`
        }
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <code>{model.label}</code>
      </div>
    );
  }

  if (['ToVector', 'Reweight'].includes(model.type)) {
    return (
      <div
        className={`compact-operation-node function-node to-vector-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`函数：${functionTitle(model.label)}，双击进入函数`}
        title={
          bad
            ? `${functionTitle(model.label)}\n${result?.messages.join('\n')}`
            : `${functionTitle(model.label)} · 双击进入函数`
        }
        style={{
          minHeight: Math.max(
            44,
            ((inputSide === outputSide
              ? inputs.length + outputs.length
              : Math.max(inputs.length, outputs.length)) +
              1) *
              14,
          ),
        }}
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <span className="function-symbol" aria-hidden="true">
          ƒ
        </span>
        <strong>{model.type === 'Reweight' ? functionTitle(model.label) : model.label}</strong>
        {model.type === 'Reweight' && <ArrowUpRight size={14} aria-hidden="true" />}
      </div>
    );
  }

  if (
    model.type === 'Function' ||
    model.type === 'CodeFunction' ||
    model.type === 'TensorSequencePack' ||
    agentCodeTypes.has(model.type) ||
    imageMetadataCodeTypes.has(model.type) ||
    plutoCodeTypes.has(model.type)
  ) {
    return (
      <div
        className={`compact-operation-node function-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`函数：${model.label}`}
        title={`${model.label} · ${model.type === 'CodeFunction' || model.type === 'TensorSequencePack' || agentCodeTypes.has(model.type) || plutoCodeTypes.has(model.type) || imageMetadataCodeTypes.has(model.type) ? '在右侧查看函数代码与参数' : '在右侧查看和编辑数学公式'}`}
        style={verticalPorts > 2 ? { minHeight: (verticalPorts + 1) * 16 } : undefined}
      >
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <span className="function-symbol" aria-hidden="true">
          ƒ
        </span>
        <strong>{model.label}</strong>
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
        <code>{operationText(model)}</code>
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

  if (featureOperations.includes(model.type)) {
    const expression = operationText(model);
    return (
      <div
        className={`compact-operation-node feature-operation-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${model.label}：${expression}`}
        title={
          bad ? `${expression}\n${result?.messages.join('\n')}` : `${model.label} · ${expression}`
        }
        style={{
          minHeight: Math.max(34, (verticalPorts + 1) * 14),
          minWidth: Math.max(64, (horizontalPorts + 1) * 16),
        }}
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

  if (['Any', 'Invert', 'Cast'].includes(model.type) || elementwiseLabels[model.type]) {
    const expression = operationText(model);
    return (
      <div
        className={`compact-operation-node ${model.type === 'Invert' ? 'invert-node' : 'any-node'} tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${model.label}：${expression}`}
        title={
          bad ? `${expression}\n${result?.messages.join('\n')}` : `${model.label} · ${expression}`
        }
        style={{
          minHeight: Math.max(
            34,
            ((inputSide === outputSide
              ? inputs.length + outputs.length
              : Math.max(inputs.length, outputs.length)) +
              1) *
              14,
          ),
        }}
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

  if (['DictionaryGet', 'GetParameter', 'Slice', 'TensorIndex', 'Unsqueeze'].includes(model.type)) {
    const isSlice = model.type === 'Slice';
    const isIndex = model.type === 'TensorIndex';
    const isUnsqueeze = model.type === 'Unsqueeze';
    const isParameter = model.type === 'GetParameter';
    const kind = isParameter ? '读取参数' : isUnsqueeze ? '增加维度' : isSlice || isIndex ? '索引与切片' : '字典字段读取';
    let expression = isParameter ? '参数名待修正' : isSlice ? '切片参数待修正' : '选择字典字段';
    try {
      expression = isParameter
        ? `读取 .${parameterName(model.params.name)}`
        : isSlice
        ? sliceLabel(model.params)
        : isIndex
          ? operationText(model, result)
          : isUnsqueeze
            ? unsqueezeLabel(model.params)
            : parseFieldPath(model.params.path)
                .map((key) => `[${JSON.stringify(key)}]`)
                .join('');
    } catch {
      /* The inspector shows validation errors for incomplete parameters. */
    }
    return (
      <div
        className={`compact-operation-node ${isParameter ? 'parameter-read-node' : isSlice ? 'slice-node' : isIndex ? 'tensor-index-node' : isUnsqueeze ? 'unsqueeze-node' : 'field-read-node'} tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
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
              <code title={`${node.label} · ${operationText(node, result?.child?.nodes[node.id])}`}>{operationText(node, result?.child?.nodes[node.id])}</code>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  if (model.type === 'Concat' || model.type === 'Stack') {
    const stack = model.type === 'Stack';
    const height = verticalPorts >= 5 ? (verticalPorts + 1) * 16 + 24 : 88;
    const width = horizontalPorts >= 5 ? (horizontalPorts + 1) * 16 : 88;
    return (
      <div
        className={`cat-node ${stack ? 'stack-node' : ''} ${width > 88 || height > 88 ? 'many-port-node' : ''} tone-pink ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={`node-${model.id}`}
        aria-label={`${model.label}，${inputs.length} 路输入`}
        title={
          bad
            ? result?.messages.join('\n')
            : `${model.label} · ${inputs.length} 路输入 · ${stack ? '沿新轴堆叠' : '沿已有轴拼接'} · dim=${model.params.dim}`
        }
        style={{ width, height }}
      >
        <strong>{stack ? 'Stack' : 'Cat'}</strong>
        <PortHandles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          horizontalArea={{ top: 12, height: height - 24 }}
        />
        {axisLabels}
      </div>
    );
  }
  let operations: string[] = [];
  if (model.type === 'Compute') {
    try {
      operations = parseCalculations(model.params.operations).map(calculationLabel);
    } catch {
      operations = ['步骤无效'];
    }
  }
  const rows = Math.max(inputs.length, outputs.length, 1);
  const bodyHeight = Math.max(
    rows * 27 + 12,
    operations.length ? Math.ceil(operations.length / 3) * 34 + 18 : 72,
  );
  const hasTop =
    (inputs.length > 0 && inputSide === 'top') || (outputs.length > 0 && outputSide === 'top');
  const hasBottom =
    (inputs.length > 0 && inputSide === 'bottom') ||
    (outputs.length > 0 && outputSide === 'bottom');
  return (
    <div
      className={`module-node tone-${tone(data.category)} ${hasTop ? 'has-top-ports' : ''} ${hasBottom ? 'has-bottom-ports' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
      data-testid={`node-${model.id}`}
      title={model.subgraph ? `双击进入 ${model.label} 的${['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? '参数初始化（非 forward）' : model.type === 'ForEachTensor' ? '循环体' : ' forward()'}` : undefined}
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
          {model.type === 'Compute' ? (
            <div className="calculation-chips">
              {operations.map((label, i) => (
                <span key={i} title={label}>
                  {label}
                </span>
              ))}
            </div>
          ) : model.type === 'ForEachTensor' ? (
            <strong>ForEach · {result?.iterations?.length ?? '?'} 次</strong>
          ) : ['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? (
            <strong>仅构造阶段 · {model.type === 'EmbeddingXYZModuleInit' ? '输出模块引用' : '输出 weight'}</strong>
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
            {['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? '进入初始化' : model.type === 'ForEachTensor' ? '进入循环体' : '进入 forward()'} <ChevronRight size={12} />
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
