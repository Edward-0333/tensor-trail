import {
  supportsCompactLayerPresentation,
  supportsCompactPresentation,
  supportsEdgePresentation,
} from '@tensor-trail/model-core/core/nodePresentation';
import { resolveGraphTensorTones, type GraphTensorTones } from '@tensor-trail/model-core/core/tensorGeometry';
import { submoduleIcons, validateSubmoduleSummary } from '@tensor-trail/model-core/core/submoduleVisual';
import { supportsTensorVisual, tensorVisualTones } from '@tensor-trail/model-core/core/tensorVisual';
import { TensorVisualPanel } from './TensorVisualPanel';
import { supportsVisualMedia } from '@tensor-trail/model-core/core/visualMedia';
import { NodeVisualMediaPanel } from './NodeVisualMediaPanel';
import { plutoForwardCode } from '@tensor-trail/model-core/modules/plutoForwardOps';
import { plutoCompositeCode } from '@tensor-trail/model-core/modules/plutoForward';
import { useEffect, useMemo, useState, type FocusEvent } from 'react';
import {
  ArrowUpRight,
  MousePointer2,
  BookmarkPlus,
  SlidersHorizontal,
  Trash2,
  Link2,
  Unlink,
} from 'lucide-react';
import type { Field, Graph, GraphResult, ModelNode, SubmoduleIcon, TensorVisualTone, PortSide, ValueSpec } from '@tensor-trail/model-core/core/types';
import { canMergeCalculationSet } from '@tensor-trail/model-core/core/calculationGroups';
import { graphAt, ports } from '@tensor-trail/model-core/core/graph';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { registry } from '@tensor-trail/model-core/modules/index';
import { TensorView } from '../views/TensorView';
import { InformationSourcePanel } from './InformationSourcePanel';
import { ModuleDetail } from '../views/details';
import { tone, isTensorNode } from '../views/ModuleNode';
import { dimensions } from '@tensor-trail/model-core/modules/math';
import { emptyDictionary } from '@tensor-trail/model-core/core/dictionary';
import { CalculationEditor } from './CalculationEditor';
import { DictionaryInputEditor } from './DictionaryInputEditor';
import { isContainerInput } from '@tensor-trail/model-core/modules/containers';
import { isForwardInput } from '@tensor-trail/model-core/core/inputTypes';
import { useEditor } from './store';
import { HyperparametersPanel } from './HyperparametersPanel';
import { DictionaryPathField, IndexExpressionField } from './IndexingFields';
import { FunctionFormulaField } from './FunctionFormulaField';
import { effectiveValueNode, isValueNode, linkedValueNodes, valueOwner } from '@tensor-trail/model-core/core/valueLinks';
import { localParameterColors } from '../views/localParameterColors';
import { sequenceCode } from '@tensor-trail/model-core/modules/sequenceOps';
import { decoderSupportCode } from '@tensor-trail/model-core/modules/decoderSupport';
import { decoderLayerOpCode } from '@tensor-trail/model-core/modules/decoderLayerOps';
import { attentionMathCode } from '@tensor-trail/model-core/modules/attentionMath';
import { opusCrossAttentionCode } from '@tensor-trail/model-core/modules/opusCrossAttention';
import { forecastMathCode } from '@tensor-trail/model-core/modules/forecastMath';
import { forecastSupportCode } from '@tensor-trail/model-core/modules/forecastSupport';
import { arrayConversionCode } from '@tensor-trail/model-core/modules/arrayConversion';
import { imageMetadataCode } from '@tensor-trail/model-core/modules/imageMetadata';
import { tensorSequenceCode } from '@tensor-trail/model-core/modules/tensorSequence';
import { forEachCode } from '@tensor-trail/model-core/modules/forEach';
import { forLoopCode } from '@tensor-trail/model-core/modules/forLoop';
import { natSource } from '@tensor-trail/model-core/modules/natSequence';
import { agentCompletionCode } from '@tensor-trail/model-core/modules/agentCompletion';
import { embeddingInitializationCode } from '@tensor-trail/model-core/modules/embeddingInitialization';
import { moduleReferenceCode } from '@tensor-trail/model-core/modules/moduleReference';
import { attributeCode } from '@tensor-trail/model-core/modules/attributes';
import { newZerosCode } from '@tensor-trail/model-core/modules/newZeros';
import { selectionCode } from '@tensor-trail/model-core/modules/selectionOps';
import { bufferSourceCode } from '@tensor-trail/model-core/modules/bufferSource';
import { matchingPreparationCode } from '@tensor-trail/model-core/modules/matchingPreparation';
import { deviceCode } from '@tensor-trail/model-core/modules/device';
import { sortingCode } from '@tensor-trail/model-core/modules/sorting';
import { reweightCode } from '@tensor-trail/model-core/modules/reweight';
import { whileLoopCode } from '@tensor-trail/model-core/modules/whileLoop';
import { loopOpsCode } from '@tensor-trail/model-core/modules/loopOps';
import { linspaceCode } from '@tensor-trail/model-core/modules/linspace';
import { elementwiseCode } from '@tensor-trail/model-core/modules/elementwise';
import { AttributeBindingPanel } from './AttributeBindingPanel';
import { IfRegionPanel } from './IfRegionPanel';
import { LayoutRegionPanel, LayoutRegionMembership } from './LayoutRegionPanel';
import { unlinkValueWithConfirmation, valueUnlinkWarning } from './valueLinkActions';
import { canJoinLayoutRegion, isLayoutRegion } from '@tensor-trail/model-core/core/layoutRegions';
import { isCodeRegion } from '@tensor-trail/model-core/core/ifRegions';
import { mapOperationCode } from '@tensor-trail/model-core/modules/mapOps';
import { mapEncoderCode } from '@tensor-trail/model-core/modules/mapEncoder';
import { maskedWriteCode } from '@tensor-trail/model-core/modules/maskedWrite';
import { isTensorSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { canRequireGrad } from '@tensor-trail/model-core/core/tensorAppearance';
import { TensorAppearancePanel } from './TensorAppearancePanel';
import { RuntimePanel } from './RuntimePanel';

const operationHelp: Record<
  string,
  { ports: [string, string][]; explanation: string; expression?: string }
> = {
  Linspace: {
    ports: [['out', '一维序列，形状为 [steps]，axis 指定轴名。']],
    explanation: '用 start 和 end 指定两端点，steps 为非负整数。两个端点须是可绑定的标量，整数字面量范围为 −2⁶³ 至 2⁶⁴−1。零步生成空轴，一步只转换 start，两步及以上转换两端点。转换前按 dtype 检查范围。',
    expression: 'out = torch.linspace(start, end, steps, dtype=dtype)',
  },
  OnesLike: {
    ports: [['in', '提供形状、轴和 dtype 的 Tensor。'], ['out', '同形状、同类型的全一 Tensor。']],
    explanation: '可用作齐次坐标的常量分量。只读取输入规格，默认关闭梯度。',
    expression: 'out = torch.ones_like(x)',
  },
  Maximum: {
    ports: [['a / b', '可从末轴对齐广播的两路 Tensor。'], ['out', '广播后逐元素取较大值的 Tensor。']],
    explanation: '对应轴的尺寸相同或一侧为 1，输出 dtype 按两路输入提升。所有广播后的轴均保留。',
    expression: 'out = torch.maximum(a, b)',
  },
  Normalize: {
    ports: [
      ['in', '浮点 Tensor，dim 指定通道轴。默认 dim=-3 对应 [..., C, H, W] 中的 C。'],
      ['out', '逐通道标准化后的新 Tensor，保留输入的形状、轴名与 dtype。'],
    ],
    explanation: '沿 dim 指定的通道轴广播 mean 和 std，计算 (x - mean) / std。两组统计量等长，转为输入 dtype 后 std 每项须大于 0。单组统计量适用于所有通道。',
    expression: [
      '# 默认 dim=-3，输入形状为 [..., C, H, W]',
      'out = torchvision.transforms.Normalize(mean=mean, std=std)(x)',
      '# 任意通道轴 dim（如 HWC 的 dim=2）',
      'shape = [1] * x.ndim',
      'shape[dim] = -1',
      'out = (x - x.new_tensor(mean).view(shape)) / x.new_tensor(std).view(shape)',
    ].join('\n'),
  },
  GridSample: {
    ports: [
      ['in', '浮点特征张量 [N,C,H,W] 或 [N,C,D,H,W]。'],
      ['grid', '与 in 同秩、同 batch、同 dtype。末维存归一化坐标 (x, y[, z])，依次对应宽、高、深。'],
      ['out', '保留 in 的 batch 和通道，每个 grid 空间位置产生一个采样值（每通道）。'],
    ],
    explanation: 'mode 可选 nearest（最近邻）、bilinear（2D 双线性、3D 三线性）和 bicubic（双三次，仅限 4D）。padding_mode 控制越界采样，可选 zeros（补零）、border（边界值）和 reflection（反射）。align_corners 为 True 时，−1 和 1 对应两端像素中心，为 False 时对应两端像素的外边界。',
    expression: 'out = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)',
  },
  Cos: {
    ports: [
      ['in', '以弧度表示角度的 Tensor。'],
      ['out', '逐元素余弦；形状与输入一致。'],
    ],
    explanation: '对应 Tensor.cos()，不增加或移除维度。',
    expression: 'out = x.cos()',
  },
  Sin: {
    ports: [
      ['in', '以弧度表示角度的 Tensor。'],
      ['out', '逐元素正弦；形状与输入一致。'],
    ],
    explanation: '对应 Tensor.sin()，不增加或移除维度。',
    expression: 'out = x.sin()',
  },
  IntegerMultiply: {
    ports: [
      ['a / b', '整数或符号整数尺寸，例如 batch 和 agent 的数量。'],
      ['out', '两个整数尺寸的乘积，可接到 View 的维度输入。'],
    ],
    explanation: '用于计算 Python 整数尺寸，结果不是 Tensor。',
    expression: 'out = a * b',
  },
  View: {
    ports: [
      ['in', '要重新组织形状的 Tensor。'],
      ['d0, d1, …', '形状表达式引用的整数维度；填写 d0 等名称后出现对应端口。'],
      ['out', '按目标形状组织的 Tensor，元素总数保持不变。'],
    ],
    explanation:
      '形状可填写固定整数、d0 等维度输入，最多一个 -1 表示自动推导。修改完整形状后移开焦点或按 Enter 应用。',
    expression: 'out = x.view(d0, d1, -1)',
  },
  Flatten: {
    ports: [
      ['in', '待展平的 Tensor。'],
      ['out', '将 start_dim 到 end_dim（含两端）合并为一个轴的 Tensor。'],
    ],
    explanation: '区间外的维度保持原状，支持负轴编号。',
    expression: 'out = x.flatten(start_dim, end_dim)',
  },
  Concat: {
    ports: [
      ['a … p', '按端口字母顺序拼接的 Tensor，输入数量可设为 1 至 16。'],
      ['out', '沿已有 dim 轴拼接，其余轴大小必须一致。'],
    ],
    explanation: 'Cat 不增加维数；拼接轴长度等于各输入对应轴长度之和。',
    expression: 'out = torch.cat([a, b, …], dim=dim)',
  },
  Stack: {
    ports: [
      ['a … p', '形状相同的 Tensor，按端口字母顺序堆叠。'],
      ['out', '插入一个长度为输入数量的新轴，输入数量可设为 1 至 16。'],
    ],
    explanation: 'Stack 增加一个维度；dim 指定新轴插入的位置。',
    expression: 'out = torch.stack([a, b, …], dim=dim)',
  },
  ToVector: {
    ports: [
      ['feat', '特征 Tensor，时间维位于轴 2。'],
      ['valid_mask', '布尔有效标记 [B, A, T]，对应特征的前三个维度。'],
      ['out', '相邻有效帧的特征差；无效位置为 0，时间长度变为 max(T − 1, 0)。'],
    ],
    explanation:
      '内部计算相邻有效标记、补齐掩码尾部维度，再对特征求相邻差并按掩码选择。无可训练参数。',
  },
  BitwiseAnd: {
    ports: [
      ['a / b', '两路参与逐元素与运算的 Tensor。'],
      ['out', '布尔输入表示两项都有效；整数输入执行按位与。'],
    ],
    explanation: '从末尾维度对齐并按广播规则计算。',
  },
  Subtract: {
    ports: [
      ['a', '被减数。'],
      ['b', '减数。'],
      ['out', '逐元素 a − b 的结果。'],
    ],
    explanation: '两路输入从末尾维度对齐，支持广播。',
  },
  Where: {
    ports: [
      ['condition', '布尔选择条件。'],
      ['x', '条件为真时取值。'],
      ['y', '条件为假时取值。'],
      ['out', '按条件逐元素选择的结果。'],
    ],
    explanation: '条件和两路取值按广播规则对齐。',
  },
  ZerosLike: {
    ports: [
      ['in', '提供形状、轴含义和数据类型的 Tensor。'],
      ['out', '同形状、同类型的全零 Tensor。'],
    ],
    explanation: '用于为无效位置提供零值。',
  },
  AlignTrailingDims: {
    ports: [
      ['in', '待补维的掩码 mask。'],
      ['like', '用于确定目标维数的特征 feat。'],
      ['out', '在掩码末尾补入大小为 1 的轴，直到与特征维数相同。'],
    ],
    explanation: '对应 to_vector 中的补维循环，已有维度和类型保持不变。',
    expression: 'while mask.ndim < feat.ndim:\n    mask = mask.unsqueeze(-1)',
  },
};

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
  if (['Concat', 'Stack'].includes(node.type) && field.key === 'input_count')
    return (
      <label className="param-field branch-input-count">
        <span>
          输入数量
          <small>input_count</small>
        </span>
        <select
          aria-label="输入数量 (input_count)"
          disabled={disabled}
          value={Number(value ?? 2)}
          onChange={(event) => update(Number(event.target.value))}
        >
          {Array.from({ length: 16 }, (_, index) => index + 1).map((count) => (
            <option key={count} value={count}>
              {count}
            </option>
          ))}
        </select>
      </label>
    );
  if (
    (node.type === 'CalculationGroup' && ['summary', 'formula'].includes(field.key)) ||
    (node.type === 'SemanticOp' && ['summary', 'formula', 'inputs', 'outputs', 'output_specs'].includes(field.key)) ||
    (['View', 'Reshape'].includes(node.type) && field.key === 'shape') ||
    (node.type === 'Repeat' && field.key === 'repeats')
  )
    return (
      <label className="param-field view-shape-field">
        <span>
          {field.label}
          <small>{field.key}</small>
        </span>
        <input
          aria-label={label}
          key={`${node.id}:${String(value)}`}
          disabled={disabled}
          defaultValue={String(value ?? '')}
          spellCheck={false}
          placeholder={field.key === 'repeats' ? '1, d0, 1' : field.key === 'shape' ? 'd0, d1, -1' : undefined}
          onBlur={(event) => {
            if (event.target.value !== value && !update(event.target.value))
              event.target.value = String(value ?? '');
          }}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) return;
            if (event.key === 'Enter') {
              event.preventDefault();
              event.currentTarget.blur();
            }
            if (event.key === 'Escape') {
              event.preventDefault();
              event.currentTarget.value = String(value ?? '');
              event.currentTarget.blur();
            }
          }}
        />
        {['shape', 'repeats'].includes(field.key) && <small>完成后按 Enter 或移开焦点应用；d0、d1 等名称对应整数输入端口。</small>}
      </label>
    );
  if (node.type === 'IntegerParameter' && field.key === 'expression')
    return <IndexExpressionField node={node} field={field} />;
  if (node.type === 'Function' && field.key === 'expression')
    return <FunctionFormulaField node={node} />;
  if (node.type === 'TensorIndex') {
    if (field.key === 'mode')
      return (
        <label className="param-field tensor-selection-mode">
          <span>取值方式</span>
          <select
            aria-label="取值方式"
            value={String(value ?? 'index')}
            onChange={(e) => update(e.target.value)}
          >
            <option value="index">取单个元素（整数索引）</option>
            <option value="slice">取一段范围（切片）</option>
          </select>
          <small>
            {value === 'slice'
              ? '保留操作轴，结束位置不包含在结果中。'
              : '移除操作轴，其余维度保持不变。'}
          </small>
        </label>
      );
    if (field.kind === 'text' && field.key !== 'stopLabel')
      return <IndexExpressionField node={node} field={field} />;
  }
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
          rows={6}
          spellCheck={false}
          onBlur={(event) => {
            if (event.target.value !== value && !update(event.target.value))
              event.target.value = String(value ?? '');
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
          max={field.max}
          step={field.integer === false ? 'any' : 1}
          value={String(value ?? '')}
          onChange={(e) =>
            update(field.kind === 'number' ? Number(e.target.value) : e.target.value)
          }
        />
      )}
    </label>
  );
}

const iconLabels: Record<SubmoduleIcon, string> = {
  auto: '自动', backbone: '骨干网络', attention: '注意力', cross_attention: '交叉注意力',
  mlp: '前馈网络', projection: '投影', encoder: '编码器', decoder: '解码器', head: '预测头', module: '通用模块',
};
const toneLabels: Record<TensorVisualTone, string> = {
  auto: '自动', blue: '蓝', violet: '紫', teal: '青绿', amber: '琥珀', rose: '玫红', slate: '灰蓝',
};

function SubmoduleStylePanel({ node }: { node: ModelNode }) {
  const setVisual = useEditor(s => s.setSubmoduleVisual);
  const visual = node.submoduleVisual;
  const commit = (input: HTMLInputElement) => {
    try {
      validateSubmoduleSummary(input.value);
    } catch (error) {
      input.value = visual?.summary ?? '';
      useEditor.getState().toast((error as Error).message);
      return;
    }
    const summary = input.value.trim();
    if (summary !== (visual?.summary ?? '') && !setVisual(node.id, { ...visual, summary }))
      input.value = visual?.summary ?? '';
    else input.value = summary;
  };
  return <section className="submodule-style-panel" aria-label="封装模块样式">
    <div className="section-label">封装模块样式</div>
    <label>图标原型<select aria-label="图标原型" value={visual?.icon ?? 'auto'}
      onChange={e => setVisual(node.id, { ...visual, icon: e.currentTarget.value as SubmoduleIcon })}>
      {submoduleIcons.map(icon => <option key={icon} value={icon}>{iconLabels[icon]}</option>)}
    </select></label>
    <label>主题色标<select aria-label="主题色标" value={visual?.tone ?? 'auto'}
      onChange={e => setVisual(node.id, { ...visual, tone: e.currentTarget.value as TensorVisualTone })}>
      {tensorVisualTones.map(tone => <option key={tone} value={tone}>{toneLabels[tone]}</option>)}
    </select></label>
    <label>功能摘要<input key={visual?.summary ?? ''} aria-label="功能摘要" defaultValue={visual?.summary ?? ''}
      placeholder="自动提取" onBlur={e => commit(e.currentTarget)}
      onKeyDown={e => { if (e.key === 'Enter' && !e.nativeEvent.isComposing) commit(e.currentTarget); }} /></label>
    <button type="button" className="soft-button" onClick={() => setVisual(node.id, null)}>恢复自动</button>
  </section>;
}

export function Inspector({ graph, result, graphTones, referenceMaxDim, hiddenNodeIds }: { hiddenNodeIds?: ReadonlySet<string>; graph: Graph; result?: GraphResult; graphTones?: GraphTensorTones; referenceMaxDim?: number }) {
  const tones = useMemo(() => graphTones ?? resolveGraphTensorTones(graph, result), [graphTones, graph, result]);
  const state = useEditor();
  const edge = graph.edges.find(e => e.id === state.selectedEdgeId);
  const tabs = [['node', edge ? '连线属性' : '节点属性'], ['hyperparameters', '模型超参数'], ['appearance', '张量配色'], ['runtime', '运行数据']] as const;
  const [tab, setTab] = useState<'node' | 'hyperparameters' | 'appearance' | 'runtime'>('node');
  const storedNode = graph.nodes.find((n) => n.id === state.selected),
    node = storedNode ? effectiveValueNode(graph, storedNode) : undefined,
    linkedMembers = node && isValueNode(node.type) ? linkedValueNodes(graph, node.id) : [],
    linkedOwner = node && linkedMembers.length > 1 ? valueOwner(graph, node) : undefined;
  const owner = node?.subgraph
    ? node
    : state.path.length
      ? graphAt(state.project.graph, state.path.slice(0, -1)).nodes.find(
          (n) => n.id === state.path.at(-1),
        )
      : undefined;
  const presentationModes = node
    ? ([
        'default',
        ...(supportsCompactPresentation(node) ? (['compact'] as const) : []),
        ...(supportsEdgePresentation(node) ? (['edge'] as const) : []),
      ] as const)
    : [];
  const compactLabel = node && supportsCompactLayerPresentation(node) ? '精简图标' : '微型圆徽章';
  const embedded = !!node && !!hiddenNodeIds?.has(node.id);
  useEffect(() => { if (embedded) setTab('node'); }, [embedded, node?.id]);
  useEffect(() => { if (state.selectedEdgeId) setTab('node'); }, [state.selectedEdgeId]);
  const [variantName, setVariantName] = useState('我的 Transformer');
  useEffect(() => {
    if (owner) setVariantName(`${owner.label} · 变体`);
  }, [owner?.id]);
  const def = node ? registry.get(node.type) : undefined;
  const fields = node && def ? [...(def.getFields?.(node.params) ?? def.fields)] : [];
  // Old Cat nodes omit input_count; the editor still exposes its default of two.
  if (
    node &&
    ['Concat', 'Stack'].includes(node.type) &&
    !fields.some((field) => field.key === 'input_count')
  )
    fields.unshift({ key: 'input_count', label: '输入数量', kind: 'number', default: 2, min: 1 });
  if (node?.type === 'BooleanAssign' && node.params.allow_broadcast === undefined)
    fields.push({ key: 'allow_broadcast', label: '允许赋值广播', kind: 'boolean', default: false });
  const help = node ? operationHelp[node.type] : undefined;
  const code = node
    ? ((node.type === 'NanToNum' ? 'out = torch.nan_to_num(x)\n# NaN -> 0; +/-inf -> dtype finite limits; shape and dtype unchanged' : undefined) ??
      plutoForwardCode(node) ??
      plutoCompositeCode(node) ??
      mapEncoderCode(node) ??
      mapOperationCode(node) ??
      agentCompletionCode(node) ??
      embeddingInitializationCode(node) ??
      moduleReferenceCode(node) ??
      attributeCode(node, state.path) ??
      newZerosCode(node) ??
      selectionCode(node) ??
      bufferSourceCode(node) ??
      matchingPreparationCode(node) ??
      sortingCode(node) ??
      reweightCode(node) ??
      whileLoopCode(node) ??
      loopOpsCode(node) ??
      deviceCode(node) ??
      linspaceCode(node) ??
      elementwiseCode(node) ??
      maskedWriteCode(node) ??
      sequenceCode(node) ??
      decoderSupportCode(node) ??
      decoderLayerOpCode(node) ??
      attentionMathCode(node) ??
      opusCrossAttentionCode(node) ??
      forecastMathCode(node) ??
      forecastSupportCode(node) ??
      arrayConversionCode(node) ??
      imageMetadataCode(node) ??
      tensorSequenceCode(node) ??
      forEachCode(node) ??
      forLoopCode(node) ??
      natSource(node))
    : undefined;
  const localColor =
    node?.type === 'LocalParameter' ? localParameterColors(graph).get(node.id) : undefined;
  const container = node && isContainerInput(node.type);
  const namedInput = container || node?.type === 'Inlet';
  const renameOnBlur = (event: FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (node && event.currentTarget.value !== node.label) state.renameNode(node.id, event.currentTarget.value);
  };
  const boundary = node && ['Inlet', 'Outlet'].includes(node.type);
  const unlinkWarning = linkedOwner && node ? valueUnlinkWarning(graph, !state.path.length, node.id) : undefined;
  const tensorOutput = node ? result?.nodes[node.id]?.outputs.out : undefined;
  const replaceable =
    node &&
    linkedMembers.length <= 1 &&
    !isLayoutRegion(node) &&
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
      'DeviceVariable',
      'CodeFunction', // Change its typed template through the operation field instead.
    ].includes(node.type);
  const compatible = replaceable
    ? registry
        .list()
        .filter(
          (d) =>
            !d.composite &&
            !['Input', 'Output', 'LocalParameter', 'TensorVariable', 'DeviceVariable'].includes(d.type) &&
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
        <span className="tiny-badge">{edge ? '连线' : node ? '模块' : '总览'}</span>
      </div>
      <div className="inspector-tabs" role="tablist" aria-label="属性面板内容">
        {tabs.map(([key, label], index) => (
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
                const next = tabs[event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
                  : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length][0];
                setTab(next);
                document.getElementById(`inspector-tab-${next}`)?.focus();
              }
            }}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === 'runtime' ? (
        <div className="inspector-scroll" id="inspector-panel-runtime" role="tabpanel" aria-labelledby="inspector-tab-runtime"><RuntimePanel /></div>
      ) : tab === 'appearance' ? (
        <div className="inspector-scroll" id="inspector-panel-appearance" role="tabpanel" aria-labelledby="inspector-tab-appearance">
          <TensorAppearancePanel />
        </div>
      ) : tab === 'hyperparameters' ? (
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
          ) : edge ? (
            <section className="edge-settings" key={edge.id}>
              <h3>连线与箭头样式</h3>
              <p><code>{edge.source}.{edge.sourcePort} → {edge.target}.{edge.targetPort}</code></p>
              <p>当前推导形状 · {shapeText(result?.nodes[edge.source]?.outputs[edge.sourcePort])}</p>
              <label className="param-field"><span>箭头样式</span>
                <select value={edge.presentation ?? 'default'} onChange={e =>
                  state.setEdgePresentation(edge.id, { presentation: e.target.value === 'default' ? null : e.target.value as 'slide' | 'anchor' })}>
                  <option value="default">默认连线</option><option value="slide">PPT 粗箭头</option>
                  <option value="anchor">悬停锚点 (● 源胶囊)</option>
                </select>
              </label>
              {edge.presentation === 'anchor' && <p className="muted">悬停源胶囊、选中连线或选中源节点时，显示回溯连线。</p>}
              <label className="param-field"><span>箭头说明文字</span>
                <textarea key={edge.annotation ?? ''} defaultValue={edge.annotation ?? ''} maxLength={160} rows={2}
                  onBlur={e => { if (e.target.value !== (edge.annotation ?? '')) state.setEdgePresentation(edge.id, { annotation: e.target.value }); }} />
              </label>
              <button className="soft-button" disabled={edge.labelPosition === undefined}
                onClick={() => state.setEdgeLabelPosition(edge.id)}>恢复默认标签位置</button>
            </section>
          ) : node && def ? (
            <>
              <div className="inspector-title">
                <span className="section-label">
                  {def.category} / {node.type}
                </span>
                {namedInput ? (
                  <textarea
                    className="forward-input-name"
                    aria-label="模块名称"
                    key={`${node.id}:${node.label}`}
                    defaultValue={node.label}
                    rows={2}
                    onBlur={renameOnBlur}
                  />
                ) : (
                  <input
                    aria-label="模块名称"
                    key={`${node.id}:${node.label}`}
                    defaultValue={node.label}
                    onBlur={renameOnBlur}
                  />
                )}
                <p>{def.description}</p>
              </div>
              <div className="inspector-actions">
                {canMergeCalculationSet(graph, state.selectedIds, registry) && (
                  <button className="soft-button" onClick={() => state.mergeCalculationNodes(state.selectedIds)}>
                    合并为语义计算块
                  </button>
                )}
                {node.subgraph && (
                  <button
                    className="soft-button"
                    onClick={() => state.navigate([...state.path, node.id])}
                  >
                    <ArrowUpRight size={14} />
                    {['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(node.type)
                      ? '进入初始化'
                      : node.type === 'WhileLoop'
                      ? '进入循环体（符号模板）'
                      : ['ForLoop', 'ForEachTensor'].includes(node.type)
                      ? '进入循环体'
                      : node.type === 'CalculationGroup'
                      ? '编辑内部操作'
                      : ['ToVector', 'Reweight'].includes(node.type)
                        ? '进入函数'
                        : '进入 forward()'}
                  </button>
                )}
                {(!boundary || storedNode?.valueSource) && (
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
              {presentationModes.length > 1 && <section className="node-presentation-settings">
                <div className="section-label">画布样式</div>
                <div className="presentation-options" role="group" aria-label="画布样式">
                  {presentationModes.map(mode =>
                    <button key={mode} type="button" aria-pressed={(node.presentation ?? 'default') === mode}
                      onClick={() => state.setNodePresentation(node.id, mode)}>
                      {mode === 'default' ? '标准卡片' : mode === 'compact' ? compactLabel : '内嵌连线'}
                    </button>)}
                </div>
                {embedded && <small>当前内嵌于连线上，可随时展开为画布卡片</small>}
              </section>}
              {node.type === 'Submodule' && <SubmoduleStylePanel key={'submodule-' + node.id} node={node} />}
              {supportsVisualMedia(node) && <NodeVisualMediaPanel key={node.id} node={node} />}
              {supportsTensorVisual(node) && <TensorVisualPanel key={'tv-' + node.id} node={node}
                result={result?.nodes[node.id]} graphTones={tones} referenceMaxDim={referenceMaxDim} />}
              {help && (
                <section className="operation-port-help" aria-label="操作输入与输出说明">
                  <div className="section-label">
                    {['ToVector', 'Reweight'].includes(node.type) ? '函数接口' : '操作接口'}
                  </div>
                  <dl>
                    {help.ports.map(([port, description]) => (
                      <div key={port}>
                        <dt>
                          <code>{port}</code>
                        </dt>
                        <dd>{description}</dd>
                      </div>
                    ))}
                  </dl>
                  <p className="microcopy">{help.explanation}</p>
                  {help.expression && <pre>{help.expression}</pre>}
                </section>
              )}
              {code && (
                <section className="operation-port-help source-code-help" aria-label="函数代码">
                  <div className="section-label">函数代码</div>
                  <p className="microcopy">
                    代码用于阅读，对应当前节点参数；形状由接口规则推导，不在网页执行 Python。
                  </p>
                  <pre>
                    <code>{code}</code>
                  </pre>
                </section>
              )}
              {localColor && localColor.count > 1 && (
                <p className="local-parameter-color-note microcopy" aria-label="同名局部参数颜色">
                  <span
                    className="local-parameter-color-dot"
                    style={{ backgroundColor: localColor.color }}
                  />
                  同名参数的第 {localColor.index + 1}/{localColor.count}{' '}
                  个独立定义；其关联副本保持同色。
                </p>
              )}
              {canJoinLayoutRegion(node) && <LayoutRegionMembership graph={graph} node={node} />}
              {isValueNode(node.type) && (() => {
                const isModuleInput =
                  node.type === 'Inlet' ||
                  (!state.path.length && (node.type === 'Input' || isForwardInput(node.type)));
                const outgoingCount = storedNode
                  ? graph.edges.filter((edge) => edge.source === storedNode.id).length
                  : 0;
                return (
                  <section className="value-link-settings" aria-label="复制规则">
                    <p className="microcopy">
                      {node.type === 'LocalParameter'
                        ? '局部参数默认复制为关联副本，可就近连接。'
                        : isModuleInput
                          ? '模块输入可拆分为多份同源 IN（左上角标 IN），每份建议仅 1 对 1 连接下游一个模块，不增加外层接口。'
                          : '默认复制为独立节点；多处使用同一数据时直接从原节点连线。'}
                    </p>
                    <button
                      className="soft-button"
                      aria-label="创建关联副本"
                      onClick={() => state.duplicateLinked(storedNode?.id ?? node.id)}
                    >
                      <Link2 size={13} />
                      {isModuleInput ? '复制一份同源 IN' : '创建关联副本'}
                    </button>
                    {isModuleInput && storedNode && outgoingCount > 1 && (
                      <button
                        className="soft-button"
                        aria-label="按目标拆分为多份 IN (1对1)"
                        onClick={() => state.splitInputFanout(storedNode.id)}
                      >
                        <Link2 size={13} />
                        按目标拆分为 {outgoingCount} 份 IN (1对1)
                      </button>
                    )}
                  </section>
                );
              })()}
              {linkedOwner && (
                <section className="value-link-settings" aria-label="值关联">
                  <div className="value-link-heading">
                    <Link2 size={14} />
                    <strong>值关联</strong>
                    <span>{linkedMembers.length} 个节点</span>
                  </div>
                  <p className="microcopy">点击列表可定位节点；修改名称或数据参数会同步整个关联组。</p>
                  {storedNode?.valueSource && (
                    <p className="microcopy">此副本直接使用主节点的数据。</p>
                  )}
                  <ul className="value-link-members" aria-label="关联节点">
                    {linkedMembers.map((member) => (
                      <li key={member.id}>
                        <button
                          type="button"
                          aria-current={member.id === node.id ? 'true' : undefined}
                          onClick={() => state.focusNode(member.id)}
                          title={`定位到 ${effectiveValueNode(graph, member).label} · ${member.id}`}
                        >
                          <span>{effectiveValueNode(graph, member).label}</span>
                          <code>{member.id}</code>
                          <small>
                            {member.id === linkedOwner.id ? '主节点' : '副本'}
                            {member.id === node.id ? ' · 当前' : ''}
                          </small>
                        </button>
                      </li>
                    ))}
                  </ul>
                  <div className="value-link-actions">
                    {node.id !== linkedOwner.id && (
                      <button className="soft-button" onClick={() => state.focusNode(linkedOwner.id)}>
                        <ArrowUpRight size={13} />
                        跳转主节点
                      </button>
                    )}
                    <button
                      className="soft-button"
                      aria-label="解除当前节点的值关联"
                      onClick={() => unlinkValueWithConfirmation(node.id)}
                    >
                      <Unlink size={13} />
                      解除值关联
                    </button>
                  </div>
                  <p className="microcopy">{unlinkWarning ?? '值关联与分区归属相互独立；此按钮不会将模块移出分区。'}</p>
                </section>
              )}
              {!isLayoutRegion(node) && <section className="port-layout-settings">
                <div className="section-label">端口布局</div>
                {(['input', 'output'] as const).map((kind) => (
                  <label className="param-field" key={kind}>
                    <span>{kind === 'input' ? '输入位置' : '输出位置'}</span>
                    <select
                      aria-label={kind === 'input' ? '输入端口位置' : '输出端口位置'}
                      value={node.portLayout?.[kind] ?? 'auto'}
                      disabled={
                        (kind === 'input' && !!storedNode?.valueSource) ||
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
              </section>}
              {(isTensorNode(node.type) || container) && (
                <section className="tensor-color-settings">
                  <div className="section-label">{namedInput ? '输入边框' : 'Tensor 边框'}</div>
                  <div className="tensor-color-options">
                    <button
                      className={!node.tensorBorderColor ? 'active' : ''}
                      onClick={() => state.setTensorBorderColor(node.id)}
                      aria-label={namedInput ? '输入边框默认' : 'Tensor 边框默认'}
                    >
                      默认
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
                        aria-label={`${namedInput ? '输入边框' : 'Tensor 边框'}${label}`}
                        aria-pressed={node.tensorBorderColor === color}
                        style={{ background: color }}
                        onClick={() => state.setTensorBorderColor(node.id, color)}
                      />
                    ))}
                    <input
                      aria-label={namedInput ? '自定义输入边框色' : '自定义 Tensor 边框色'}
                      type="color"
                      value={node.tensorBorderColor ?? '#64748b'}
                      onChange={(e) => state.setTensorBorderColor(node.id, e.target.value)}
                    />
                  </div>
                  <button className="soft-button" onClick={() => setTab('appearance')}>管理全项目张量背景色</button>
                </section>
              )}
              {tensorOutput && isTensorSpec(tensorOutput) && (
                <section className="tensor-autograd-settings">
                  <label className="param-field">
                    <span>自动求导状态标注<small>out · requires_grad</small></span>
                    <select aria-label="自动求导状态标注" value={node.tensorAutograd ? String(node.tensorAutograd.requiresGrad) : 'auto'}
                      onChange={(event) => state.setTensorAutograd(node.id, event.target.value === 'auto' ? undefined : event.target.value === 'true')}>
                      <option value="auto">跟随输入 / 推导；无法确定则未知</option>
                      <option value="true" disabled={!canRequireGrad(tensorOutput.dtype)}>grad:on · True</option>
                      <option value="false">grad:off · False</option>
                    </select>
                  </label>
                  <p className="microcopy">修改后记为手工标注，不会改变 Python 模型。运行模式或代码改变后需要重新核实。</p>
                </section>
              )}
              {ports(node, registry).outputs.length > 0 && <InformationSourcePanel graph={state.project.graph} />}
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
              {!boundary && fields.length > 0 && (
                <section className="parameters">
                  <div className="section-label">模块参数</div>
                  {fields.map((field) => (
                    <ParamField
                      key={field.key}
                      field={field}
                      node={node}
                      input={result?.nodes[node.id]?.inputs.in}
                      disabled={Boolean(node.customized && node.subgraph && node.type !== 'CalculationGroup')}
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
              {['SetAttribute', 'GetAttribute'].includes(node.type) && <AttributeBindingPanel node={node} />}
              {isCodeRegion(node) && <IfRegionPanel graph={graph} node={node} />}
              {isLayoutRegion(node) && <LayoutRegionPanel graph={graph} node={node} />}
              {!container && !isCodeRegion(node) && !isLayoutRegion(node) && (
                <TensorView
                  node={node} graphTones={tones} referenceMaxDim={referenceMaxDim}
                  observeInput={node.type === 'SetAttribute' || (['Output', 'Outlet'].includes(node.type) && !Object.keys(result?.nodes[node.id]?.outputs ?? {}).length)}
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
                {owner && ['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(owner.type)
                  ? '本层说明模型构造时的参数初始化，不会在每次 forward 时重复执行；出口展示已注册的 weight。'
                  : '本层展示直接计算和模块调用；子模块保持封装，双击进入内部 forward()。'}
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
              <button
                className="soft-button"
                disabled={
                  owner.type === 'Submodule' &&
                  !owner.subgraph.nodes.some((n) => n.type === 'Outlet')
                }
                onClick={() => state.saveVariant(variantName)}
              >
                <BookmarkPlus size={15} />
                保存为我的变体
              </button>
              {owner.type === 'Submodule' &&
                !owner.subgraph.nodes.some((n) => n.type === 'Outlet') && (
                  <p className="microcopy">补充模块输出后可保存为变体。</p>
                )}
            </section>
          )}
        </div>
      )}
    </aside>
  );
}
