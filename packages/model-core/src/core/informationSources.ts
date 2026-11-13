import type {
  InformationSource,
  InitializationKind,
  ModelNode,
  RootInputSource,
  ValueLineage,
  ValueSpec,
} from './types';
import { isDictionarySpec, isModuleSpec, isTensorSequenceSpec, isTensorSpec } from './types';

export const VALUE_ORIGIN_LABELS = {
  parameter: '模型参数',
  constant: '常量',
  random: '随机生成',
  state: '模型状态',
  metadata: '形状/设备元数据',
} as const;
export const INITIALIZATION_LABELS = {
  random: '随机初始化',
  zeros: '全零初始化',
  ones: '全一初始化',
  checkpoint: '加载权重',
} as const;

export function validateInformationSource(value: unknown): asserts value is InformationSource {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw Error('信息来源需要对象。');
  const v = value as Record<string, unknown>;
  if (
    Object.keys(v).some((key) => !['kind', 'initialization'].includes(key)) ||
    typeof v.kind !== 'string' ||
    ![
      'image',
      'ego',
      'input',
      'metadata',
      'target',
      'parameter',
      'constant',
      'random',
      'state',
      'unknown',
    ].includes(v.kind) ||
    (v.initialization !== undefined &&
      (typeof v.initialization !== 'string' ||
        !Object.hasOwn(INITIALIZATION_LABELS, v.initialization)))
  )
    throw Error('信息来源或初始化类型无效。');
}

const info = (value: Partial<ValueLineage> = {}): ValueLineage => ({
  sources: [],
  origins: [],
  initializations: [],
  unknown: false,
  evidence: 'inferred',
  ...value,
});
const unknown = () => info({ unknown: true });
const unique = <T extends string>(items: T[]): T[] => [...new Set(items)].sort();
export function mergeLineages(values: ValueLineage[]): ValueLineage {
  return info({
    sources: [
      ...new Map(values.flatMap((v) => v.sources).map((source) => [source.id, source])).values(),
    ].sort((a, b) => a.index - b.index || a.id.localeCompare(b.id)),
    origins: unique(values.flatMap((v) => v.origins)),
    initializations: unique(values.flatMap((v) => v.initializations)),
    unknown: values.some((v) => v.unknown),
  });
}
export function valueLineage(value?: ValueSpec): ValueLineage {
  if (!value) return unknown();
  if (value.lineage) return value.lineage;
  if (isDictionarySpec(value)) return mergeLineages(Object.values(value.fields).map(valueLineage));
  if (isTensorSequenceSpec(value)) return mergeLineages(value.items.map(valueLineage));
  if (isModuleSpec(value)) return mergeLineages(Object.values(value.parameters).map(valueLineage));
  return unknown();
}

/** Copy descriptions only; never mutate runtime snapshots or upstream field contracts. */
function stamp(value: ValueSpec, lineage: ValueLineage): ValueSpec {
  if (isDictionarySpec(value))
    return {
      ...value,
      lineage,
      fields: Object.fromEntries(
        Object.entries(value.fields).map(([key, child]) => [key, stamp(child, lineage)]),
      ),
    } as ValueSpec;
  if (isTensorSequenceSpec(value))
    return { ...value, lineage, items: value.items.map((item) => ({ ...item, lineage })) };
  if (isModuleSpec(value))
    return {
      ...value,
      lineage,
      parameters: Object.fromEntries(
        Object.entries(value.parameters).map(([key, parameter]) => [
          key,
          { ...parameter, lineage },
        ]),
      ),
    };
  return { ...value, lineage };
}
function preserve(value: ValueSpec): ValueSpec {
  if (isDictionarySpec(value)) {
    const fields = Object.fromEntries(
      Object.entries(value.fields).map(([key, child]) => [key, preserve(child)]),
    );
    return {
      ...value,
      fields,
      lineage: Object.keys(fields).length
        ? mergeLineages(Object.values(fields).map(valueLineage))
        : (value.lineage ?? info()),
    } as ValueSpec;
  }
  if (isTensorSequenceSpec(value)) {
    const items = value.items.map((item) => preserve(item) as typeof item);
    return {
      ...value,
      items,
      lineage: items.length ? mergeLineages(items.map(valueLineage)) : (value.lineage ?? info()),
    };
  }
  if (isModuleSpec(value)) {
    const parameters = Object.fromEntries(
      Object.entries(value.parameters).map(([key, item]) => [key, preserve(item) as typeof item]),
    );
    return {
      ...value,
      parameters,
      lineage: mergeLineages(Object.values(parameters).map(valueLineage)),
    };
  }
  const lineage =
    value.lineage ??
    (isTensorSpec(value) && value.parameterRef ? info({ origins: ['parameter'] }) : unknown());
  return { ...value, lineage: { ...lineage, evidence: 'inferred' } };
}

const valueTransfers = new Set([
  'DictionaryUpdate', 'SelectValue',
  'DictionaryPack',
  'Inlet',
  'Outlet',
  'Output',
  'Identity',
  'TensorVariable',
  'ValueVariable',
  'LocalParameter',
  'DeviceVariable',
  'DictionaryGet',
  'TensorSequencePack',
  'TensorListAppend',
  'TensorSequenceGet',
  'GetParameter',
  'GetAttribute',
  'ModuleParameterState',
]);
const unaryValues = new Set([
  'Unflatten', 'TensorCompareScalar', 'MatrixInverse', 'TensorToDevice', 'DynamicSlice',
  'ReshapeLike', 'ArrayToTensorExplicit',
  'Exp', 'Softplus', 'L2Norm', 'NoGradResult',
  'NanToNum',
  'View',
  'Reshape',
  'Repeat',
  'RepeatBatch',
  'Unsqueeze',
  'Squeeze',
  'Flatten',
  'Permute',
  'Contiguous',
  'ExpandAs',
  'Cast',
  'Slice',
  'TensorIndex',
  'ReLU',
  'GELU',
  'Sigmoid',
  'Tanh',
  'Softmax',
  'ReduceSum',
  'ReduceMean',
  'ReduceMax',
  'MaxWithIndices',
  'Sort',
  'Argmax',
  'Function',
  'Detach',
  'ArrayToTensor',
  'NumpyAstype',
  'Clone',
  'TensorItemInt',
  'AlignTrailingDims',
  'SplitHeads',
  'MergeHeads',
  'Scale',
  'PointRangeTransform',
  'CausalMask',
  'SiLU',
]);
const parameterized = new Set([
  'Linear',
  'LinearByDimensions',
  'MMCVFFNContract',
  'Conv1d',
  'Conv2d',
  'LayerNorm',
  'BatchNorm1d',
  'BatchNorm2d',
]);
const mergedValues = new Set([
  'LogicalAnd', 'BooleanAnd', 'BooleanNot', 'IntegerCompare',
  'IntegerSubtract', 'IntegerMinimum', 'IntegerMaximum', 'IntegerAddOne',
  'TensorRowAt', 'TensorEqualInt', 'IntegerListGet', 'ScalarIndexWrite',
  'ZeroTailColumn', 'SliceRowsFrom', 'WriteTailRows',
  'IfCondition', 'BranchValue', 'MergeBranches',
  'Add',
  'Mul',
  'Subtract',
  'Divide',
  'MatMul',
  'Concat',
  'Stack',
  'RowIndex',
  'BooleanIndex',
]);
const metadataOnly = new Set(['TensorSize', 'TensorShapeSlice', 'GetDevice', 'GetDtype', 'DeviceConstant', 'DtypeConstant', 'SequenceLength', 'SingleIterationPreview']);
const factories = new Set([
  'BooleanParameter',
  'NewZeros',
  'NewOnes',
  'Zeros',
  'ZerosLike',
  'OnesScaled',
  'Arange',
  'IntegerParameter',
  'IntegerList',
]);

/** Whitelisted value semantics; an unknown operator must not inherit specs by accident. */
export function annotateInformationSources(
  node: ModelNode,
  inputs: Record<string, ValueSpec>,
  outputs: Record<string, ValueSpec>,
  rootInput?: RootInputSource,
): Record<string, ValueSpec> {
  const all = (lineage: ValueLineage) =>
    Object.fromEntries(
      Object.entries(outputs).map(([port, value]) => [port, stamp(value, lineage)]),
    );
  // Node labels and old semantic declarations never create or override a source.
  if (rootInput) return all(info({ sources: [rootInput], evidence: 'root' }));
  // inferWhile closes carried-state dependencies; an opaque loop remains unknown.
  if (node.type === 'WhileLoop' && !node.subgraph) return all(unknown());
  if (node.subgraph || valueTransfers.has(node.type))
    return Object.fromEntries(
      Object.entries(outputs).map(([port, value]) => [port, preserve(value)]),
    );
  if (metadataOnly.has(node.type)) return all(info({ origins: ['metadata'] }));
  if (node.type === 'UpdateImageMetaInputShape')
    return all({ ...valueLineage(inputs.img_metas), evidence: 'inferred' });
  if (node.type === 'BatchMetaTensor' || node.type === 'BatchMetaArray' || node.type === 'BatchMetaStack')
    return all({ ...valueLineage(inputs.img_metas), evidence: 'inferred' });
  if (node.type === 'TensorFieldContract')
    return Object.fromEntries(Object.entries(outputs).map(([port, value]) => [port,
      value.lineage ? preserve(value) : stamp(value, { ...valueLineage(inputs.in), evidence: 'inferred' })]));
  if (['OPUSDecoderLayerContract', 'OPUSDecoderLayerAt'].includes(node.type))
    return all(mergeLineages([
      ...Object.entries(inputs).filter(([port]) => port !== 'index').map(([, value]) => valueLineage(value)),
      info({ origins: ['parameter'] }),
    ]));
  if (['OPUSSamplingContract', 'AdaptiveMixingContract', 'OPUSSelfAttentionContract', 'OPUSCrossAttentionContract'].includes(node.type))
    return all(mergeLineages([...Object.values(inputs).map(valueLineage), info({ origins: ['parameter'] })]));
  if (node.type === 'TorchMultiheadAttention')
    return all(mergeLineages([...Object.values(inputs).map(valueLineage),
      info({ origins: Number(node.params.dropout) > 0 ? ['parameter', 'random'] : ['parameter'] })]));
  if (node.type === 'Dropout')
    return all(mergeLineages([valueLineage(inputs.in), info({ origins: Number(node.params.p) > 0 ? ['random'] : [] })]));
  if (factories.has(node.type)) {
    const initialization = ['NewZeros', 'Zeros', 'ZerosLike'].includes(node.type)
      ? 'zeros'
      : ['OnesScaled', 'NewOnes'].includes(node.type)
        ? 'ones'
        : undefined;
    return all(
      info({ origins: ['constant'], initializations: initialization ? [initialization] : [] }),
    );
  }
  if (['EmbeddingModule', 'EmbeddingWeight', 'ModuleReference'].includes(node.type))
    return all(
      info({
        origins: ['parameter'],
        initializations: node.type === 'ModuleReference' ? [] : ['random'],
      }),
    );
  if (node.type === 'UniformInitColumns')
    return all(
      mergeLineages([
        valueLineage(inputs.in),
        info({ origins: ['parameter'], initializations: ['random'] }),
      ]),
    );
  if (node.type === 'BufferSource') {
    const initial = String(node.params.initial);
    return all(
      info({
        origins: ['state'],
        initializations: ['zeros', 'ones'].includes(initial) ? [initial as InitializationKind] : [],
      }),
    );
  }
  if (unaryValues.has(node.type)) return all({ ...valueLineage(inputs.in), evidence: 'inferred' });
  if (node.type === 'BooleanIndexAxis')
    return all(mergeLineages([valueLineage(inputs.in), valueLineage(inputs.mask)]));
  if (parameterized.has(node.type))
    return all(mergeLineages([valueLineage(inputs.in), info({ origins: ['parameter'] })]));
  if (mergedValues.has(node.type))
    return all(mergeLineages(Object.values(inputs).map(valueLineage)));
  if (node.type === 'CodeFunction') {
    const operation = String(node.params.operation);
    if (['Interpolate1D', 'Interpolate2DNearest', 'ImageNormalize'].includes(operation))
      return all({ ...valueLineage(inputs.in), evidence: 'inferred' });
    if (['ResNet50Backbone', 'NeighborhoodAttention1D'].includes(operation))
      return all(mergeLineages([valueLineage(inputs.in), info({ origins: ['parameter'] })]));
    if (['GpuPhotoMetricDistortion', 'DropPath'].includes(operation))
      return all(
        mergeLineages([
          valueLineage(inputs.in),
          info({
            origins: operation === 'DropPath' && Number(node.params.p) === 0 ? [] : ['random'],
          }),
        ]),
      );
  }
  return all(unknown());
}
