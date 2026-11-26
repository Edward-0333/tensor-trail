import { plutoCodeTypes, plutoOperationLabel } from '@tensor-trail/model-core/modules/plutoForwardOps';
import { imageMetadataCodeTypes } from '@tensor-trail/model-core/modules/imageMetadata';
import { parameterName } from '@tensor-trail/model-core/modules/moduleReference';
import { matchingPreparationLabel } from '@tensor-trail/model-core/modules/matchingPreparation';
import { literalDeviceName } from '@tensor-trail/model-core/modules/device';
import { sortingLabel } from '@tensor-trail/model-core/modules/sorting';
import { loopOperationLabels } from '@tensor-trail/model-core/modules/loopOps';
import { attributeBinding, attributeOwnerLabel, attributeTypes } from '@tensor-trail/model-core/core/attributes';
import { plutoCompositeTypes } from '@tensor-trail/model-core/modules/plutoForward';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { useUpdateNodeInternals, type Node, type NodeProps } from '@xyflow/react';
import { ChevronRight, AlertCircle, ArrowUpRight, Replace, SquarePlus, Expand } from 'lucide-react';
import type { ModelNode, NodeResult, Port } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSpec, isIntegerSpec, isIntegerListSpec, isControlSpec, isModuleSpec, isDeviceSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { resolveSubmoduleVisual } from '@tensor-trail/model-core/core/submoduleVisual';
import { ModuleGlyph, SubmoduleGlyph } from './ModuleGlyph';
import { TensorGeometryView } from './TensorGeometryView';
import type { TensorGeometryDiagram } from '@tensor-trail/model-core/core/tensorGeometry';
import { NodeVisualMedia } from './NodeVisualMedia';
import { PortHandles, SubmodulePortSlots } from './PortHandles';
import { canvasValueLabel } from './canvasValueLabel';
import { valueDisplayText } from './valueDisplayText';
import { IfRegionNode } from './IfRegionNode';
import { LayoutRegionNode } from './LayoutRegionNode';
import { RegionOutputBadge, type RegionOutputInfo } from './RegionOutputBadge';
import {
  compactJunctionTypes,
  supportsCompactPresentation,
} from '@tensor-trail/model-core/core/nodePresentation';
import { isCodeRegion } from '@tensor-trail/model-core/core/ifRegions';
import { useEditor } from '../editor/store';
import { parseCalculations, calculationLabel, sliceLabel } from '@tensor-trail/model-core/modules/calculation';
import { isContainerInput } from '@tensor-trail/model-core/modules/containers';
import { parseFieldPath, tensorIndexLabel } from '@tensor-trail/model-core/modules/indexing';
import { scalarComparisonSymbol } from '@tensor-trail/model-core/modules/forecastMath';
import { linspaceCall } from '@tensor-trail/model-core/modules/linspace';
import { normalizeChannelCount } from '@tensor-trail/model-core/modules/normalize';
import { selectionLabels } from '@tensor-trail/model-core/modules/selectionOps';
import { orderGraph } from '@tensor-trail/model-core/core/graph';
import { forLoopCount } from '@tensor-trail/model-core/core/forLoop';
import { unsqueezeLabel } from '@tensor-trail/model-core/modules/unsqueeze';
import { anyLabel } from '@tensor-trail/model-core/modules/logical';
import { castLabel } from '@tensor-trail/model-core/modules/cast';
import { functionTitle } from '@tensor-trail/model-core/core/hierarchy';
import type { LocalParameterColor } from './localParameterColors';
import type { LevelInput } from './levelInputs';
import { FINAL_OUTPUT_TITLE } from './finalOutputs';
import { LineageBadges } from './InformationSources';
import { agentCompositeTypes, agentCodeTypes } from '@tensor-trail/model-core/modules/agentCompletion';
import { mapCompositeTypes } from '@tensor-trail/model-core/modules/mapEncoder';
import { natCompositeTypes } from '@tensor-trail/model-core/modules/natSequence';
import { DEFAULT_TENSOR_PALETTE, gradLabel, gradSourceLabel, tensorGradState, tensorInk } from '@tensor-trail/model-core/core/tensorAppearance';

export type TrailNode = Node<
  {
    model: ModelNode;
    tensorDiagram?: TensorGeometryDiagram;
    inputBoundary?: LevelInput;
    inputAlias?: LevelInput;
    finalOutput?: boolean;
    regionOutputs?: RegionOutputInfo[];
    localParameterColor?: LocalParameterColor;
    result?: Omit<NodeResult, 'status'> & { status: NodeResult['status'] | 'incomplete' };
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
  OnesLike: 'ones_like',
  AlignTrailingDims: '尾部补维',
  Divide: 'a / b',
};
const featureOperations = [
  'Sigmoid', 'Log', 'Sqrt', 'Clamp', 'Chunk', 'TopK',
  'Unflatten', 'TensorCompareScalar', 'LogicalAnd', 'MatrixInverse', 'TensorToDevice',
  'DictionaryUpdate', 'SequenceLength', 'SelectValue', 'DynamicSlice', 'BooleanParameter',
  'BooleanAnd', 'BooleanNot', 'IntegerCompare', 'SingleIterationPreview', 'ReshapeLike',
  'TensorFieldContract', 'BatchMetaStack', 'ArrayToTensorExplicit',
  'Exp', 'Softplus', 'L2Norm', 'NoGradResult',
  'NanToNum',
  ...Object.keys(loopOperationLabels),
  'IntegerGreater',
  'Squeeze',
  'Cos',
  'Sin',
  'IntegerMultiply',
  'IntegerFloorDivide',
  'TensorListAppend',
  'PointRangeTransform',
  'ReduceMean',
  'View',
  'Permute',
  'Flatten',
  'BooleanIndex',
  'IntegerSubtract', 'IntegerMinimum', 'IntegerMaximum',
  'Contiguous',
  'Zeros',
  'NewZeros',
  'NewOnes',
  'OnesScaled',
  'Arange',
  'SortRowsByMax',
  'MaxWithIndices',
  'Sort',
  'RowIndex',
  'GetDevice',
  'GetDtype',
  'BooleanAssign',
  'IndexAssign',
  'RepeatBatch',
  'Repeat',
  'Reshape',
  'ReduceMax',
  'ReduceSum',
];

function operationText(node: ModelNode, result?: Pick<NodeResult, 'inputs'>): string {
  try {
    if (node.type === 'Sigmoid') return 'sigmoid()';
    if (node.type === 'Log') return node.params.log1p ? 'log1p()' : 'log()';
    if (node.type === 'Sqrt') return 'sqrt()';
    if (node.type === 'Clamp') return `clamp(${node.params.min || 'None'}, ${node.params.max || 'None'})`;
    if (node.type === 'Chunk') return `chunk(${node.params.chunks}, dim=${node.params.dim})`;
    if (node.type === 'Unflatten') return `unflatten(${node.params.dim}, (${node.params.sizes}))`;
    if (node.type === 'TensorCompareScalar') return `x ${scalarComparisonSymbol(node.params.operator)} ${node.params.value}`;
    if (node.type === 'LogicalAnd') return 'logical_and';
    if (node.type === 'MatrixInverse') return 'linalg.inv';
    if (node.type === 'TensorToDevice') return `.${node.params.target}()`;
    if (node.type === 'DictionaryUpdate') return 'dict.update';
    if (node.type === 'SequenceLength') return 'len(list)';
    if (node.type === 'SelectValue') return '条件后取值';
    if (node.type === 'DynamicSlice') return `[start:stop] (dim=${node.params.dim})`;
    if (node.type === 'BooleanParameter') return node.params.value ? 'True' : 'False';
    if (node.type === 'BooleanAnd') return 'and';
    if (node.type === 'BooleanNot') return 'not';
    if (node.type === 'IntegerCompare') return `a ${{lt:'<',le:'<=',eq:'==',gt:'>',ge:'>='}[String(node.params.operator)] ?? '?'} b`;
    if (node.type === 'SingleIterationPreview') return '单轮预览 · interval=0';
    if (node.type === 'ReshapeLike') return 'reshape(like.shape)';
    if (node.type === 'TensorFieldContract') return `[${JSON.stringify(node.params.field)}] · 规格声明`;
    if (node.type === 'BatchMetaStack') return `stack(meta[${JSON.stringify(node.params.field)}])`;
    if (node.type === 'ArrayToTensorExplicit') return `tensor(${node.params.dtype}, ${node.params.device})`;
    if (node.type === 'Exp') return 'exp()';
    if (node.type === 'Softplus') return 'softplus()';
    if (node.type === 'L2Norm') return `norm(dim=${node.params.dim})`;
    if (node.type === 'NoGradResult') return 'no_grad 结果';
    if (node.type === 'Detach') return 'detach()';
    if (selectionLabels[node.type]) return selectionLabels[node.type];
    if (node.type === 'BooleanIndexAxis') {
      return `按轴筛选 (dim=${String(node.params.dim ?? '').trim() || '?'})`;
    }
    if (node.type === 'TensorSequenceGet') return `[${String(node.params.index ?? '').trim() || '?'}]`;
    if (node.type === 'NanToNum') return 'nan_to_num()';
    if (node.type === 'TensorItemInt' && node.params.cast) return 'int(x)';
    if (loopOperationLabels[node.type]) return loopOperationLabels[node.type];
    if (node.type === 'GetDevice') return '.device';
    if (node.type === 'GetDtype') return '.dtype';
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
    if (node.type === 'IntegerFloorDivide') return 'a // b';
    if (node.type === 'TensorListAppend') return 'append()';
    if (node.type === 'PointRangeTransform') return node.params.mode === 'decode' ? 'decode_points()' : 'encode_points()';
    if (node.type === 'ReduceMean') return `mean(${node.params.dim})`;
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
    if (node.type === 'NewOnes') return 'new_ones()';
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

const noHandles = () => null;

function portSides({ model, reverse }: TrailNode['data']) {
  return {
    inputSide: model.portLayout?.input ?? (reverse ? 'right' : 'left'),
    outputSide: model.portLayout?.output ?? (reverse ? 'left' : 'right'),
  } as const;
}

const junctionSymbols: Record<string, string> = {
  Add: '+', Mul: '×', Subtract: '−', Divide: '÷', BitwiseAnd: '&', Maximum: 'max', MatMul: '@', Concat: '‖', Stack: '⊞',
};

function compactLayerSpec(model: ModelNode): string {
  const p = model.params;
  switch (model.type) {
    case 'Linear':
      return `Linear ${p.in_features}→${p.out_features}`;
    case 'LinearByDimensions':
      return `Linear →${p.out_features}`;
    case 'LayerNorm':
      return `LN(${p.features})`;
    case 'RMSNorm':
      return `RMSNorm(${p.features})`;
    case 'FFN':
    case 'MMCVFFNContract': {
      const inp = p.embed_dims ?? p.dim ?? p.in_features;
      const hid = p.feedforward_channels ?? p.hidden_dim ?? p.hidden_features;
      return inp !== undefined && hid !== undefined ? `FFN ${inp}→${hid}` : 'FFN';
    }
    case 'Conv1d':
    case 'Conv2d':
      return `${model.type} ${p.in_channels}→${p.out_channels}`;
    case 'Softmax':
      return `softmax(${p.dim})`;
    case 'Linspace':
      try {
        return linspaceCall(p, false);
      } catch {
        return 'linspace(?)';
      }
    case 'OnesLike': return 'ones_like';
    case 'ZerosLike': return 'zeros_like';
    case 'GridSample':
      return `GridSample · ${p.mode}`;
    case 'Normalize': {
      try {
        const count = normalizeChannelCount(p);
        return count === 1 ? 'Normalize · 广播' : `Normalize · C=${count}`;
      } catch {
        // The inference diagnostic explains invalid statistics while the fields are being edited.
        return 'Normalize';
      }
    }
    default:
      return model.type;
  }
}

function compactLayerCaption(model: ModelNode): string | undefined {
  const cleaned = model.label.trim().replace(/^self\./, '');
  if (!cleaned) return undefined;
  const parts = cleaned.split(/\s*[·]\s*/).filter(Boolean);
  const candidate = parts[0] ?? cleaned;
  const normalized = candidate.toLowerCase();
  if (
    normalized === model.type.toLowerCase() ||
    normalized === 'linear' ||
    normalized === 'layernorm' ||
    normalized === 'rmsnorm'
  ) {
    return parts[1] && parts[1].toLowerCase() !== model.type.toLowerCase() ? parts[1] : undefined;
  }
  return candidate;
}

function IntegratedTensorHeader({ data }: NodeProps<TrailNode>) {
  const { model, result } = data;
  const [editing, setEditing] = useState(false);
  const cancelRename = useRef(false);
  const spec = result?.outputs.out ?? Object.values(result?.outputs ?? {})[0] ?? result?.inputs.in;
  const tensor = spec && isTensorSpec(spec) ? spec : undefined;
  const bad = result?.status === 'error' || result?.status === 'blocked';
  const shape = spec ? canvasValueLabel(spec, 'node') : bad ? '待修正' : '等待输入';
  const showGrad = !!tensor || (!spec && isTensorNode(model.type) && model.type !== 'Inlet');
  const gradState = tensorGradState(tensor);
  const role = ['Input', 'Inlet'].includes(model.type)
    ? 'input'
    : ['Output', 'Outlet'].includes(model.type)
      ? 'output'
      : 'variable';
  const dtypeText = data.tensorDiagram
    ? `${data.tensorDiagram.badge} · ${data.tensorDiagram.output.dtype}`
    : tensor?.dtype;
  const axisTitle = tensor?.axes.length ? `${shape} · ${tensor.axes.join(', ')}` : shape;
  const inputTag = data.inputBoundary ?? data.inputAlias;
  const isAliasTag = !data.inputBoundary && !!data.inputAlias;
  return (
    <div className="tensor-card-header">
      <div className="tensor-card-title-row">
        {inputTag ? (
          <span
            className="tensor-card-pill role-in"
            aria-label={
              isAliasTag
                ? `本层输入副本 ${inputTag.index}/${inputTag.total}`
                : `本层输入 ${inputTag.index}/${inputTag.total}`
            }
            title={
              isAliasTag
                ? '同源模块输入拆分副本（不增加外层输入端口，1对1连接单个下游模块）。'
                : '本层已绘制的输入接口；输入身份与梯度状态相互独立。'
            }
          >
            IN · {inputTag.index}/{inputTag.total}
          </span>
        ) : role === 'input' ? (
          <span className="tensor-card-pill role-in">IN</span>
        ) : role === 'output' ? (
          <span className="tensor-card-pill role-out">OUT</span>
        ) : null}
        {tensor?.parameterRef && (
          <span
            className="tensor-card-pill role-param"
            title={`${tensor.parameterRef.module}.${tensor.parameterRef.name} · 注册参数`}
          >
            Param
          </span>
        )}
        {tensor?.bufferRef && (
          <span
            className="tensor-card-pill role-buffer"
            title={`${tensor.bufferRef.reference} · 注册 buffer`}
          >
            Buffer
          </span>
        )}
        {editing ? (
          <input
            className="tensor-card-title-input nodrag nowheel"
            aria-label="编辑 Tensor 名称"
            autoFocus
            defaultValue={model.label}
            onBlur={(e) => {
              if (!cancelRename.current && e.target.value !== model.label) useEditor.getState().renameNode(model.id, e.target.value);
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
          <strong
            className="tensor-card-title"
            title={`${model.label} · 双击修改名称`}
            onDoubleClick={(e) => {
              e.stopPropagation();
              cancelRename.current = false;
              setEditing(true);
            }}
          >
            {model.label}
          </strong>
        )}
      </div>
      <div className="tensor-card-meta-row">
        <code className="tensor-card-shape" title={axisTitle}>
          {shape}
        </code>
        <span className="tensor-card-tags">
          {dtypeText && <span className="tensor-card-dtype">{dtypeText}</span>}
          {showGrad && gradState !== 'unknown' && (
            <span
              className="tensor-card-grad"
              data-grad-state={gradState}
              title={`requires_grad=${gradState === 'on' ? 'True' : 'False'} · ${tensor ? gradSourceLabel(tensor) : '尚未确定'}；不代表已执行 backward`}
            >
              {gradLabel(gradState)}
            </span>
          )}
        </span>
      </div>
      <LineageBadges value={spec} />
    </div>
  );
}

const submoduleDiagnosticLabels = {
  error: '推导错误', blocked: '已阻断', incomplete: '待补齐', warning: '警告',
};

function SubmoduleCard({ data, selected }: NodeProps<TrailNode>) {
  const { model, result, inputs, outputs, tensorDiagram } = data;
  const { icon, tone, summary, status, statusLabel } = resolveSubmoduleVisual(model);
  const { inputSide, outputSide } = portSides(data);
  const diagnostic = result && result.status !== 'ok' ? result.status : undefined;
  const diagnosticTitle = diagnostic && result?.messages.join('\n');
  const invalid = diagnostic === 'error' || diagnostic === 'blocked';
  const singleOutput = outputs.length === 1 && (!diagnostic || diagnostic === 'warning')
    ? result?.outputs[outputs[0].id] : undefined;
  const shape = singleOutput && isTensorSpec(singleOutput) ? shapeText(singleOutput) : '';
  const capCount = (side: 'top' | 'bottom') =>
    (inputSide === side ? inputs.length : 0) + (outputSide === side ? outputs.length : 0);
  const illustrated = model.visualMedia || tensorDiagram;
  return <div className={`module-node submodule-card submodule-tone-${tone} ${illustrated ? 'illustrated-node' : ''} ${selected ? 'selected' : ''} ${invalid ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
    style={{ minWidth: Math.max(200, Math.max(capCount('top'), capCount('bottom')) * 68) }}
    data-testid={`node-${model.id}`} aria-label={`子模块 ${model.label}`}
    title={diagnosticTitle}>
    <div className="submodule-content">
      <div className="submodule-heading">
        <SubmoduleGlyph icon={icon} />
        <strong className="node-kind submodule-title" title={model.label}>{model.label}</strong>
        <button type="button" className="submodule-enter nodrag nopan" aria-label="进入 forward()"
          title="进入 forward()" onDoubleClick={event => event.stopPropagation()}
          onClick={event => { event.stopPropagation(); data.open(); }}>↗</button>
      </div>
      <div className="submodule-details">
        {summary && <div className="submodule-summary" title={summary}>{summary}</div>}
        <div className="submodule-spec"><span className="submodule-status" data-status={diagnostic ?? status} title={diagnosticTitle}>
          {diagnostic ? submoduleDiagnosticLabels[diagnostic] : statusLabel}
        </span>
          {shape && <code title={shape}>{shape}</code>}
        </div>
      </div>
      {model.visualMedia && <NodeVisualMedia media={model.visualMedia} />}
      {tensorDiagram && <TensorGeometryView diagram={tensorDiagram} variant="node" />}
    </div>
    <SubmodulePortSlots inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} />
  </div>;
}

export function ModuleNode(props: NodeProps<TrailNode>) {
  const { model, inputs, outputs, result } = props.data;
  const media = model.visualMedia;
  const diagram = props.data.tensorDiagram;
  const spec = result?.outputs.out ?? Object.values(result?.outputs ?? {})[0] ?? result?.inputs.in;
  const integratedTensor =
    Boolean(media || diagram) &&
    isTensorNode(model.type) &&
    !isContainerInput(model.type) &&
    !(spec && (isIntegerListSpec(spec) || isControlSpec(spec) || isDeviceSpec(spec) || isModuleSpec(spec)));
  const tensorRole = ['Input', 'Inlet'].includes(model.type)
    ? 'input'
    : ['Output', 'Outlet'].includes(model.type)
      ? 'output'
      : 'variable';
  const cardAccent =
    model.tensorBorderColor ??
    model.tensorVisual?.color ??
    model.tensorColor ??
    (tensorRole === 'output'
      ? 'var(--category-green)'
      : diagram
        ? `var(--tv-${diagram.output.tones[0]}-front)`
        : tensorRole === 'input'
          ? 'var(--category-sky)'
          : 'var(--category-blue)');
  const multiple = inputs.length > 1 || outputs.length > 1;
  const sides = portSides(props.data);
  const updateNodeInternals = useUpdateNodeInternals();
  const portSignature = JSON.stringify([sides.inputSide, sides.outputSide, inputs.map(p => p.id), outputs.map(p => p.id), model.type, model.presentation, diagram?.width, integratedTensor]);
  const previousPorts = useRef(portSignature);
  const previousMedia = useRef(media);
  useEffect(() => {
    // ResizeObserver measures tensor height changes. Remeasure handles when the
    // card switches layout or ports move without changing the measured bounds.
    const old = previousMedia.current;
    previousMedia.current = media;
    const mediaChanged = old !== media && (!old || !media || old.layout !== media.layout ||
      old.tileWidth !== media.tileWidth || old.items.length !== media.items.length || old.items.some((item, i) => {
        const next = media.items[i];
        return item.id !== next.id || item.src !== next.src || item.caption !== next.caption ||
          item.badge !== next.badge || item.emphasis !== next.emphasis;
      }));
    if (previousPorts.current !== portSignature || mediaChanged) {
      previousPorts.current = portSignature;
      updateNodeInternals(model.id);
    }
  }, [model.id, media, portSignature, updateNodeInternals]);
  const gutter = (side: string) => {
    const ports = [...(sides.inputSide === side ? inputs : []), ...(sides.outputSide === side ? outputs : [])];
    if (integratedTensor && diagram && !media && !multiple) return 6;
    return multiple && ports.length ? 12 + Math.min(100, Math.max(...ports.map(p => p.id.length)) * 5) : 10;
  };
  return <>
    {model.type === 'Submodule' ? <SubmoduleCard {...props} /> : media || diagram ? <div
      className={`illustrated-node ${integratedTensor ? `illustrated-tensor-node tensor-role-${tensorRole}` : ''} ${props.data.finalOutput ? 'final-output-value' : ''} ${props.data.active ? 'step-active' : ''} ${props.data.dimmed ? 'dimmed' : ''}`}
      data-testid={`node-${model.id}`}
      data-selected={props.selected}
      data-invalid={result?.status === 'error' || result?.status === 'blocked'}
      data-final-output={props.data.finalOutput || undefined}
      title={integratedTensor && props.data.finalOutput ? FINAL_OUTPUT_TITLE : undefined}
      style={{
        ...(integratedTensor && diagram && !media ? { width: diagram.width, minWidth: 0, gridTemplateColumns: 'minmax(0, 1fr)' } : {}),
        paddingLeft: gutter('left'),
        paddingRight: gutter('right'),
        paddingBlock: multiple ? 20 : 10,
        ...(integratedTensor ? ({ '--tensor-card-accent': cardAccent } as CSSProperties) : {}),
      }}
    >
      {integratedTensor ? <IntegratedTensorHeader {...props} /> : <ModuleNodeContent {...props} />}
      {media && <NodeVisualMedia media={media} />}
      {diagram && <TensorGeometryView diagram={diagram} variant={integratedTensor ? 'integrated' : 'node'} />}
      <PortHandles inputs={inputs} outputs={outputs} {...sides} labels={multiple} />
    </div> : <ModuleNodeContent {...props} />}
    {!!props.data.regionOutputs?.length && <div className="region-output-badges nodrag nopan">
      {props.data.regionOutputs.map(region => <RegionOutputBadge key={region.id} region={region} />)}
    </div>}
  </>;
}

function ModuleNodeContent({ data, selected }: NodeProps<TrailNode>) {
  const { model, result, inputs, outputs } = data;
  const illustrated = model.visualMedia || data.tensorDiagram;
  const Handles = illustrated ? noHandles : PortHandles;
  const testId = illustrated ? undefined : `node-${model.id}`;
  const [editing, setEditing] = useState(false);
  const cancelRename = useRef(false);
  const palette = useEditor((state) => state.project.tensorPalette) ?? DEFAULT_TENSOR_PALETTE;
  const output = result?.outputs.out ?? Object.values(result?.outputs ?? {})[0];
  const spec = output ?? result?.inputs.in;
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
  const inputBadge = data.inputBoundary ? <span className="level-input-badge"
    aria-label={`本层输入 ${data.inputBoundary.index}/${data.inputBoundary.total}`}
    title="本层已绘制的输入接口；输入身份与梯度状态相互独立。">
    IN · {data.inputBoundary.index}/{data.inputBoundary.total}
  </span> : data.inputAlias ? <span className="level-input-alias-badge"
    aria-label={`本层输入副本 ${data.inputAlias.index}/${data.inputAlias.total}`}
    title="同源模块输入拆分副本（不增加外层输入端口，1对1连接单个下游模块）。">
    IN · {data.inputAlias.index}/{data.inputAlias.total}
  </span> : undefined;
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
  const { inputSide, outputSide } = portSides(data);
  const sideCount = (side: string) =>
    (inputSide === side ? inputs.length : 0) + (outputSide === side ? outputs.length : 0);
  const verticalPorts = Math.max(sideCount('left'), sideCount('right'));
  const horizontalPorts = Math.max(sideCount('top'), sideCount('bottom'));
  const bad = result?.status === 'error' || result?.status === 'blocked';
  const container = isContainerInput(model.type);
  if (model.presentation === 'compact' && supportsCompactPresentation(model) && !illustrated) {
    const summary = output ? `输出 ${shapeText(output)}` : '暂无有效输出';
    const title = [`${model.label} · ${summary}`, ...(result?.messages ?? [])].join('\n');
    if (compactJunctionTypes.has(model.type)) {
      return <div className={`compact-junction-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId} data-type={model.type} data-status={result?.status} aria-label={`${model.label} · ${model.type}`} title={title}>
        {junctionSymbols[model.type]}
        {result?.status === 'warning' && <span className="compact-junction-warning" aria-label="警告">!</span>}
        {['Concat', 'Stack'].includes(model.type) && <span className="compact-junction-sub">dim={String(model.params.dim)}</span>}
        <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} horizontalArea={{ top: -8, height: 60 }} labels={false} />
      </div>;
    }
    const specLine = compactLayerSpec(model);
    const caption = compactLayerCaption(model);
    const gridSample = model.type === 'GridSample';
    const portSpace = gridSample ? {
      paddingLeft: sideCount('left') ? 40 : 12,
      paddingRight: sideCount('right') ? 40 : 12,
      paddingTop: sideCount('top') ? 24 : 8,
      paddingBottom: sideCount('bottom') ? 24 : 7,
      minHeight: (verticalPorts + 1) * 20,
      minWidth: Math.max(104, (horizontalPorts + 1) * 36),
    } : undefined;
    return <div className={`compact-layer-node ${gridSample ? 'compact-grid-sample' : ''} tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
      style={portSpace}
      data-testid={testId} data-status={result?.status} aria-label={`${model.label} · ${specLine}`} title={model.type === 'Linspace' ? `${specLine}\n${title}` : title}>
      <div className="compact-layer-icon" aria-hidden="true">
        <ModuleGlyph type={model.type === 'LinearByDimensions' ? 'Linear' : model.type} />
      </div>
      <strong className="compact-layer-spec">{specLine}</strong>
      {result?.status === 'warning' && <span className="compact-layer-warning">警告</span>}
      {caption && <span className="compact-layer-name">{caption}</span>}
      <PortHandles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={gridSample} />
    </div>;
  }
  if (isCodeRegion(model)) return <IfRegionNode model={model} selected={selected} />;
  if (model.type === 'LayoutRegion') return <LayoutRegionNode model={model} selected={selected} />;
  if (model.type === 'WhileLoop') return (
    <div className={`while-loop-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={testId} title="双击进入循环体；只展示一轮符号模板，未执行循环" style={{ minHeight: Math.max(140,(verticalPorts+1)*24) }}>
      <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels />
      <span className="control-role-badge">WHILE <ArrowUpRight size={14} /></span>
      <strong>{model.label}</strong><code>i &lt; num_points</code>
      <small>状态：{String(model.params.states)} · i += 1</small>
      <small>一轮符号模板 · 未执行</small>
    </div>
  );
  if (model.type === 'IfCondition') return (
    <div className={`condition-test-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={testId} title="count == quota；虚线表示控制依赖，实际分支未采集">
      <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels />
      <strong>{model.label}</strong><code>count == quota</code><small>条件比较 · 未执行</small>
    </div>
  );
  if (model.type === 'BranchValue') return (
    <div className={`branch-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`}
      data-testid={testId} title="仅静态展示此分支，未执行数值计算">
      <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
      <strong>{model.params.branch ? 'True 分支' : 'False 分支'}</strong>
      <small>{model.params.branch ? '更新剩余行' : 'maps 不修改'}</small>
    </div>
  );
  if (model.type === 'IntegerList' || (spec && (isIntegerListSpec(spec) || isControlSpec(spec)) && isTensorNode(model.type))) return (
    <div className={`control-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''}`} data-testid={testId} title={spec ? shapeText(spec) : '待连接'}>
      <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
      <strong>{model.label}</strong><code>{spec ? shapeText(spec) : 'list[int]'}</code>
    </div>
  );
  if (model.type === 'DeviceConstant' || model.type === 'DtypeConstant') {
    const device = model.type === 'DeviceConstant';
    const value = String(model.params[device ? 'device' : 'dtype']);
    return (
      <div className={`device-value-node metadata-constant-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId} aria-label={`${device ? '设备' : '类型'}常量 ${value}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · 固定元数据 ${value}；不读取张量，配置变化时需同步更新。${model.params.context || ''}`}>
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
        <span className="device-role-badge">{device ? 'DEVICE' : 'DTYPE'}</span><code>{value}</code>
      </div>
    );
  }
  if (model.type === 'DeviceVariable' || (spec && isDeviceSpec(spec) && isTensorNode(model.type))) {
    return (
      <div
        className={`device-value-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`设备值 ${model.label}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · ${spec ? shapeText(spec) : '待连接 Device'} · 设备元数据，不是 Tensor；${spec && isDeviceSpec(spec) && literalDeviceName(spec.reference) ? '固定设备声明，非运行观测' : '实际设备未采集'}`}
      >
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
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
        data-testid={testId}
        aria-label={`${write ? '写属性' : '读属性'} ${owner}.${name}`}
        title={bad ? result?.messages.join('\n') : `${model.label} · ${owner}.${name} · 属性绑定，不是 forward 形参`}
        style={tensorStyle}
      >
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={false} />
        <div className="attribute-node-heading"><span className="attribute-role-badge">{write ? 'SET' : 'GET'}</span><strong>{name}</strong></div>
        <code className="attribute-owner">{owner}</code>
        {model.params.version && <small className="attribute-version">赋值版本：{String(model.params.version)}</small>}
        <div className="attribute-value-summary">{spec && <code>{valueDisplayText(spec)}</code>}{gradBadge}</div>
        <LineageBadges value={spec} />
      </div>
    );
  }
  if (model.type === 'LocalParameter')
    return (
      <div
        className={`local-parameter-node tone-orange ${data.localParameterColor && data.localParameterColor.count > 1 ? 'distinct-local-parameter' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
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
        <Handles
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
            style={{ width: `${Math.max(2, Math.min(model.label.length, 12))}ch` }}
            onBlur={(event) => {
              if (!cancelRename.current && event.target.value !== model.label)
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
        className={`${model.type === 'DictionaryInput' ? 'dictionary-input-node' : 'tensor-node forward-input-node'} ${model.type === 'Inlet' ? 'module-input-node' : ''} ${inputBadge ? 'level-input-node' : ''} ${data.finalOutput ? 'final-output-value' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        data-final-output={data.finalOutput || undefined}
        title={data.finalOutput ? FINAL_OUTPUT_TITLE : undefined}
        style={tensorStyle}
      >
        {inputBadge}
        <Handles
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
            rows={2}
            onBlur={(event) => {
              if (!cancelRename.current && event.target.value !== model.label)
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
        <LineageBadges value={spec} />
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
        className={`tensor-node ${spec && isModuleSpec(spec) ? 'module-reference-node' : ''} ${spec && isIntegerSpec(spec) ? 'integer-value-node' : ''} ${tensorRole} ${inputBadge ? 'level-input-node' : ''} ${data.finalOutput ? 'final-output-value' : ''} ${data.reverse ? 'flow-left' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        data-final-output={data.finalOutput || undefined}
        title={data.finalOutput ? FINAL_OUTPUT_TITLE : undefined}
        aria-label={
          ['TensorVariable', 'ValueVariable'].includes(model.type)
            ? `${spec && !isTensorSpec(spec) ? '值变量' : '张量变量'} ${model.label}`
            : undefined
        }
        style={tensorStyle}
      >
        {inputBadge}
        {editing ? (
          <input
            className="tensor-label-input nodrag nowheel"
            aria-label={spec && !isTensorSpec(spec) ? '编辑变量名称' : '编辑 Tensor 名称'}
            autoFocus
            defaultValue={model.label}
            onBlur={(e) => {
              if (!cancelRename.current && e.target.value !== model.label) useEditor.getState().renameNode(model.id, e.target.value);
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
          <Handles
            inputs={inputs}
            outputs={outputs}
            inputSide={inputSide}
            outputSide={outputSide}
            labels={false}
          />
        ) : <div className="tensor-node-shape">
          <Handles
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
        <LineageBadges value={spec} />
      </div>
    );
  }

  if (
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
        data-testid={testId}
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
        <Handles
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
        data-testid={testId}
        title={
          bad ? result?.messages.join('\n') : `${model.params.expression} → ${shapeText(spec)}`
        }
      >
        <Handles
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
        data-testid={testId}
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
        <Handles
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
        data-testid={testId}
        aria-label={`函数：${model.label}`}
        title={`${model.label} · ${model.type === 'CodeFunction' || model.type === 'TensorSequencePack' || agentCodeTypes.has(model.type) || plutoCodeTypes.has(model.type) || imageMetadataCodeTypes.has(model.type) ? '在右侧查看函数代码与参数' : '在右侧查看和编辑数学公式'}`}
        style={verticalPorts > 2 ? { minHeight: (verticalPorts + 1) * 16 } : undefined}
      >
        <Handles
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
        data-testid={testId}
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
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
      </div>
    );
  }

  if (model.type === 'NumpyAstype') {
    const dtype = String(model.params.dtype ?? 'float32');
    const expression = `astype(np.${dtype})`;
    return (
      <div
        className={`compact-operation-node astype-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}: ${expression}`}
        title={bad ? `${expression}\n${result?.messages.join('\n')}` : `${model.label} · ${expression}`}
        style={{
          minHeight: Math.max(34, (verticalPorts + 1) * 14),
          minWidth: Math.max(64, (horizontalPorts + 1) * 16),
        }}
      >
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <Replace className="astype-icon" size={16} aria-hidden="true" />
        <code>astype</code>
        <code className="astype-dtype">{dtype}</code>
      </div>
    );
  }

  if (model.type === 'DictionaryPack') {
    const names = inputs.map(port => port.id);
    const hint = `${model.label} · dict(${names.join(', ')}) · 按字段名组装字典，不堆叠或融合张量；各字段保留自己的类型、形状与来源，详情见属性面板。`;
    const labelWidth = (name: string) => Math.min(120, name.length * 5.5);
    const sideLabelWidth = (side: string) => Math.max(0,
      ...(inputSide === side ? names.map(labelWidth) : []),
      ...(outputSide === side ? [labelWidth('dict')] : []));
    const horizontalWidth = Math.max(...['top', 'bottom'].map(side =>
      (sideCount(side) + 1) * Math.max(40, sideLabelWidth(side) + 16)));
    return (
      <div
        className={`compact-operation-node dictionary-pack-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}：组装字典，${inputs.length} 个字段`}
        title={bad ? `${hint}\n${result?.messages.join('\n')}` : hint}
        style={{
          minHeight: Math.max(40, (verticalPorts + 1) * 18),
          minWidth: Math.max(136, horizontalWidth),
          paddingLeft: Math.max(12, sideLabelWidth('left') + 16),
          paddingRight: Math.max(12, sideLabelWidth('right') + 16),
          paddingTop: sideCount('top') ? 24 : 8,
          paddingBottom: sideCount('bottom') ? 24 : 8,
        }}
      >
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide}
          outputLabels={{ out: 'dict' }} />
        <ModuleGlyph type="DictionaryPack" />
        <code>dict()</code>
      </div>
    );
  }

  if (model.type === 'BooleanIndexAxis') {
    const dim = String(model.params.dim ?? '').trim() || '?';
    const declared = model.params.declared_count === true;
    const count = result?.inputs.count;
    const countText = count && isIntegerSpec(count) ? String(count.value) : '?';
    const hint = `${model.label} · 沿 dim=${dim} 保留 mask=True 的整项，保持顺序和其他轴；不是逐元素乘法置零或连续区间切片。${declared ? `True 数量契约=${countText}（非实测），只是形状推导声明，不是 Python 索引的第三个数据参数。` : 'True 数量取决于实际 mask，不从掩码长度猜测。'}图标仅为筛选示意。`;
    const portNames: Record<string, string> = { in: 'x', mask: 'mask', count: '契约', out: 'out' };
    const gutter = (side: 'left' | 'right') => Math.max(12, ...[
      ...(inputSide === side ? inputs : []), ...(outputSide === side ? outputs : []),
    ].map(port => 14 + (portNames[port.id] ?? port.id).length * 5));
    return (
      <div
        className={`compact-operation-node axis-filter-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}: 按轴筛选 dim=${dim}`}
        title={bad ? `${hint}\n${result?.messages.join('\n')}` : hint}
        style={{
          minHeight: Math.max(declared ? 72 : 52, (verticalPorts + 1) * 18),
          minWidth: Math.max(152, (horizontalPorts + 1) * 48),
          paddingLeft: gutter('left'), paddingRight: gutter('right'),
          paddingTop: sideCount('top') ? 24 : 8,
          paddingBottom: sideCount('bottom') ? 24 : 8,
        }}
      >
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide}
          inputLabels={{ in: 'x', mask: 'mask', count: '契约' }} />
        <div className="axis-filter-summary">
          <ModuleGlyph type="BooleanIndexAxis" />
          <strong>按轴筛选</strong>
          <code className="axis-filter-dim">dim={dim}</code>
        </div>
        {declared && <span className="axis-filter-contract" title="True 数量契约（非实测）；仅用于形状推导，不参与数值计算">
          数量契约 <code>{countText}</code>
        </span>}
      </div>
    );
  }

  if (model.type === 'TensorSequenceGet' || model.type === 'Detach') {
    const isDetach = model.type === 'Detach';
    const expression = operationText(model);
    const operation = isDetach ? expression : `list/tuple${expression}`;
    const hint = isDetach
      ? `${model.label} · ${expression} · 输出 requires_grad=False；数据、shape、dtype 和来源编号保留，与输入共享存储，不是 clone，也不影响其他分支的梯度`
      : `${model.label} · ${operation} · 读取一个完整 Tensor，不切片 Tensor 轴、不复制或 detach`;
    return (
      <div
        className={`compact-operation-node ${isDetach ? 'detach' : 'tensor-sequence-get'}-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}: ${operation}`}
        title={bad ? `${hint}\n${result?.messages.join('\n')}` : hint}
        style={{
          minHeight: Math.max(34, (verticalPorts + 1) * 14),
          minWidth: Math.max(100, (horizontalPorts + 1) * 24),
        }}
      >
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        <ModuleGlyph type={model.type} />
        <code>{expression}</code>
      </div>
    );
  }

  if (model.type === 'ArrayToTensor' || model.type === 'ExpandAs' || model.type === 'MatMul') {
    const isExpandAs = model.type === 'ExpandAs';
    const isMatMul = model.type === 'MatMul';
    const expression = isMatMul ? 'matmul()' : isExpandAs ? 'expand_as()'
      : model.params.mode === 'new_tensor' ? 'new_tensor()' : 'torch.tensor()';
    const compactKind = isMatMul ? 'matmul' : isExpandAs ? 'expand-as' : 'tensor-create';
    const OperationIcon = isExpandAs ? Expand : SquarePlus;
    const hint = `${model.label} · ${expression}${isExpandAs ? ' · like 只提供目标 shape，不融合其数值' : ''}`;
    const portGutter = (side: 'left' | 'right') => Math.max(12, ...[
      ...(inputSide === side ? inputs : []),
      ...(outputSide === side ? outputs : []),
    ].map(port => 16 + port.id.length * 5));
    return (
      <div
        className={`compact-operation-node ${compactKind}-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}: ${expression}`}
        title={bad ? `${expression}\n${result?.messages.join('\n')}` : hint}
        style={{
          minHeight: Math.max(40, (verticalPorts + 1) * 16),
          minWidth: Math.max(108, (horizontalPorts + 1) * 52),
          paddingLeft: portGutter('left'),
          paddingRight: portGutter('right'),
          paddingTop: sideCount('top') ? 22 : 8,
          paddingBottom: sideCount('bottom') ? 22 : 8,
        }}
      >
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
        />
        {isMatMul ? <ModuleGlyph type="MatMul" />
          : <OperationIcon className={`${compactKind}-icon`} size={16} aria-hidden="true" />}
        <code>{expression}</code>
      </div>
    );
  }

  if (featureOperations.includes(model.type)) {
    const expression = operationText(model);
    return (
      <div
        className={`compact-operation-node feature-operation-node tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${model.label}：${expression}`}
        title={
          bad ? `${expression}\n${result?.messages.join('\n')}` : `${model.label} · ${expression}`
        }
        style={{
          minHeight: Math.max(34, (verticalPorts + 1) * 14),
          minWidth: Math.max(64, (horizontalPorts + 1) * 16),
        }}
      >
        <Handles
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
        data-testid={testId}
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
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        {['OnesLike', 'ZerosLike'].includes(model.type) && <ModuleGlyph type={model.type} />}
        <code>{expression}</code>
      </div>
    );
  }

  if (['DictionaryGet', 'GetParameter', 'Slice', 'TensorIndex', 'Unsqueeze'].includes(model.type)) {
    const isDictionary = model.type === 'DictionaryGet';
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
    const hint = isDictionary
      ? `${model.label} · ${expression} · 按键路径读取字段，不复制或融合数据`
      : `${model.label} · ${expression}`;
    return (
      <div
        className={`compact-operation-node ${isParameter ? 'parameter-read-node' : isSlice ? 'slice-node' : isIndex ? 'tensor-index-node' : isUnsqueeze ? 'unsqueeze-node' : 'field-read-node'} tone-${tone(data.category)} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId}
        aria-label={`${kind}：${expression}`}
        title={bad ? `${hint}\n${result?.messages.join('\n')}` : hint}
        style={isDictionary ? {
          minHeight: Math.max(34, (verticalPorts + 1) * 14),
          minWidth: Math.max(100, (horizontalPorts + 1) * 24),
        } : undefined}
      >
        <Handles
          inputs={inputs}
          outputs={outputs}
          inputSide={inputSide}
          outputSide={outputSide}
          labels={false}
        />
        {isDictionary && <ModuleGlyph type="DictionaryGet" />}
        <code>{expression}</code>
      </div>
    );
  }

  const summary = String(model.params.summary ?? '').trim();
  if (model.type === 'SemanticOp' || (model.type === 'CalculationGroup' && model.subgraph && summary)) {
    const steps = model.subgraph?.nodes.filter(n => !['Inlet', 'Outlet'].includes(n.type)).length;
    const badge = steps === undefined ? '语义步骤' : `折叠 · ${steps} 步`;
    const multi = inputs.length > 1 || outputs.length > 1;
    const labelSpace = (side: string) => multi ? Math.max(0,
      ...(inputSide === side ? inputs : []).map(p => p.id.length),
      ...(outputSide === side ? outputs : []).map(p => p.id.length)) * 5 : 0;
    const left = Math.min(100, labelSpace('left')), right = Math.min(100, labelSpace('right'));
    const formula = String(model.params.formula ?? '').trim();
    return (
      <div className={`semantic-card calculation-group-node ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
        data-testid={testId} aria-label={`${badge}，${summary}`}
        title={bad ? result?.messages.join('\n') : steps !== undefined ? '双击查看内部操作' : model.label}
        style={{
          minHeight: (verticalPorts + 1) * 18,
          width: Math.max(190 + left + right, (horizontalPorts + 1) * (14 + Math.min(100, Math.max(labelSpace('top'), labelSpace('bottom'))))),
          paddingLeft: 14 + left, paddingRight: 14 + right,
          paddingTop: sideCount('top') && multi ? 23 : 12, paddingBottom: sideCount('bottom') && multi ? 23 : 12,
        }}>
        <Handles inputs={inputs} outputs={outputs} inputSide={inputSide} outputSide={outputSide} labels={multi} />
        <span className="semantic-card-badge">{badge}</span>
        <strong>{summary || model.label}</strong>
        {formula && <code className="semantic-card-formula">{formula}</code>}
        {model.type === 'SemanticOp' && (
          <div className="semantic-card-outputs">
            {outputs.map(port => <span key={port.id}>{outputs.length > 1 && `${port.id} `}{shapeText(result?.outputs[port.id])}</span>)}
          </div>
        )}
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
        data-testid={testId}
        aria-label={`${model.label}，${steps.length} 个操作，双击进入编辑`}
        title={bad ? result?.messages.join('\n') : `${model.label} · 双击编辑内部操作`}
        style={{ minHeight: Math.max(inputs.length, outputs.length) * 24 + 16 }}
      >
        <Handles
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
        data-testid={testId}
        aria-label={`${model.label}，${inputs.length} 路输入`}
        title={
          bad
            ? result?.messages.join('\n')
            : `${model.label} · ${inputs.length} 路输入 · ${stack ? '沿新轴堆叠' : '沿已有轴拼接'} · dim=${model.params.dim}`
        }
        style={{ width, height }}
      >
        <strong>{stack ? 'Stack' : 'Cat'}</strong>
        <Handles
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
      data-testid={testId}
      title={model.subgraph ? `双击进入 ${model.label} 的${['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? '参数初始化（非 forward）' : ['ForEachTensor', 'ForLoop'].includes(model.type) ? '循环体' : ' forward()'}` : result?.messages.join('\n') || undefined}
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
          ) : model.type === 'ForLoop' ? (
            <strong>For · {String(forLoopCount(model, result?.inputs) ?? '?')} 次 · 状态递推</strong>
          ) : ['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? (
            <strong>仅构造阶段 · {model.type === 'EmbeddingXYZModuleInit' ? '输出模块引用' : '输出 weight'}</strong>
          ) : (
            <ModuleGlyph type={model.type === 'LinearByDimensions' ? 'Linear' : model.type} />
          )}
        </div>
      </div>
      <Handles
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
            {['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(model.type) ? '进入初始化' : ['ForEachTensor', 'ForLoop'].includes(model.type) ? '进入循环体' : '进入 forward()'} <ChevronRight size={12} />
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
