export type Dim = number | string;
export interface TensorAutograd {
  requiresGrad: boolean;
  source: 'manual' | 'runtime' | 'inferred';
}
export type TensorGradState = 'on' | 'off' | 'unknown';
export type TensorPalette = Record<TensorGradState, string>;
export interface RootInputSource {
  /** Stable node identity; the display number follows root interface order, not layout. */
  id: string;
  index: number;
  label: string;
}
export type ValueOrigin = 'parameter' | 'constant' | 'random' | 'state' | 'metadata';
export type InitializationKind = 'random' | 'zeros' | 'ones' | 'checkpoint';
/** Legacy declarations are read for compatibility, never used to assign source IDs. */
export interface InformationSource {
  kind: 'image' | 'ego' | 'input' | 'metadata' | 'target' | ValueOrigin | 'unknown';
  initialization?: InitializationKind;
}
/** Static value dependencies, not gradient flow, tensor contents or feature quality. */
export interface ValueLineage {
  sources: RootInputSource[];
  origins: ValueOrigin[];
  initializations: InitializationKind[];
  unknown: boolean;
  evidence: 'root' | 'inferred';
}
export interface TensorSpec {
  shape: Dim[];
  axes: string[];
  dtype: string;
  autograd?: TensorAutograd;
  lineage?: ValueLineage;
  /** Registered parameter reference, not inherited by derived Tensor operations. */
  parameterRef?: { module: string; name: string };
  /** Registration role of a held module buffer, not a Tensor subclass. */
  bufferRef?: { reference: string; persistent: boolean };
}
/** Static reference to a Python nn.Module; the object itself is not a Tensor. */
export interface ModuleSpec {
  kind: 'module';
  reference: string;
  moduleType: string;
  parameters: Record<string, TensorSpec>;
}
export interface ContainerSpec {
  kind: 'forward' | 'dict' | 'list';
  text: string;
}
export interface DictionarySpec {
  kind: 'dict';
  /** JSON field names stay strings; this preserves the original Python key type. */
  keyType?: 'str' | 'int';
  fields: Record<string, TensorSpec | DictionarySpec | TensorSequenceSpec | NoneSpec>;
}
/** Python values used by control flow; distinct from scalar tensors. */
export interface BooleanSpec {
  kind: 'bool';
  value: boolean;
}
export interface NoneSpec {
  kind: 'none';
}
/** A Python integer size, or an unresolved symbolic integer dimension; never a Tensor. */
export interface IntegerSpec {
  kind: 'int';
  value: Dim;
}
/** Python list of integers, not a Tensor list or a Tensor with shape [length]. */
export interface IntegerListSpec {
  kind: 'int-list';
  values: number[];
}
/** Static control dependency; never an observed Boolean branch decision. */
export interface ControlSpec {
  kind: 'control';
  expression: string;
  branch?: boolean;
}
/** Tuple-like torch.Size metadata, not a Tensor with a shape. */
export interface SizeSpec {
  kind: 'size';
  dims: Dim[];
}
/** A symbolic device read or explicit torch.device literal; not a Tensor or runtime observation. */
export interface DeviceSpec {
  kind: 'device';
  reference: string;
}
/** Static NumPy array description, deliberately not a TensorSpec. */
export interface NumpyArraySpec {
  kind: 'ndarray';
  dims: Dim[];
  axes: string[];
  dtype: string;
}
/** torch.dtype metadata, read from a Tensor or explicitly declared; not tensor contents. */
export interface DtypeSpec {
  kind: 'dtype';
  value: string;
}
/** Ordered Tensor references with independent shapes; not a stacked Tensor. */
export interface TensorSequenceSpec {
  kind: 'tuple' | 'list';
  items: TensorSpec[];
}
export type ValueSpec = (
  | TensorSpec
  | ContainerSpec
  | DictionarySpec
  | IntegerSpec
  | IntegerListSpec
  | ControlSpec
  | SizeSpec
  | DeviceSpec
  | NumpyArraySpec
  | DtypeSpec
  | TensorSequenceSpec
  | ModuleSpec
  | BooleanSpec
  | NoneSpec) & { lineage?: ValueLineage };
export const isTensorSpec = (value: ValueSpec): value is TensorSpec => 'shape' in value;
export const isModuleSpec = (value: ValueSpec): value is ModuleSpec =>
  !isTensorSpec(value) && value.kind === 'module';
export const isIntegerSpec = (value: ValueSpec): value is IntegerSpec =>
  !isTensorSpec(value) && value.kind === 'int';
export const isIntegerListSpec = (value: ValueSpec): value is IntegerListSpec =>
  !isTensorSpec(value) && value.kind === 'int-list';
export const isControlSpec = (value: ValueSpec): value is ControlSpec =>
  !isTensorSpec(value) && value.kind === 'control';
export const isSizeSpec = (value: ValueSpec): value is SizeSpec =>
  !isTensorSpec(value) && value.kind === 'size';
export const isDeviceSpec = (value: ValueSpec): value is DeviceSpec =>
  !isTensorSpec(value) && value.kind === 'device';
export const isNumpyArraySpec = (value: ValueSpec): value is NumpyArraySpec =>
  !isTensorSpec(value) && value.kind === 'ndarray';
export const isDtypeSpec = (value: ValueSpec): value is DtypeSpec =>
  !isTensorSpec(value) && value.kind === 'dtype';
export const isTensorSequenceSpec = (value: ValueSpec): value is TensorSequenceSpec =>
  !isTensorSpec(value) && (value.kind === 'tuple' || value.kind === 'list') && 'items' in value;
export const isDictionarySpec = (value: ValueSpec): value is DictionarySpec =>
  !isTensorSpec(value) && value.kind === 'dict' && 'fields' in value;
export function tensorValue(value: ValueSpec | undefined): TensorSpec {
  if (!value) throw new Error('没有可用的 Tensor。');
  if (!isTensorSpec(value)) throw new Error(`需要 Tensor，当前为${shapeText(value)}。`);
  return value;
}
export type Params = Record<string, string | number | boolean | DictionarySpec>;
export interface Hyperparameter {
  type: 'int' | 'float' | 'bool' | 'string';
  value: number | boolean | string;
  source?: string;
}
export type Hyperparameters = Record<string, Hyperparameter>;
export interface InferenceContext {
  hyperparameters: Hyperparameters;
}
export type PortSide = 'left' | 'right' | 'top' | 'bottom';
export interface PortLayout {
  input?: PortSide;
  output?: PortSide;
}
/** Layout only: these return paths never belong to the executable edge graph. */
export interface LoopFeedbackLayout {
  targetSide: 'left' | 'right';
  waypoints: { x: number; y: number }[];
  labelPosition?: number;
}
export interface Port {
  id: string;
  label: string;
}
export interface Field {
  key: string;
  label: string;
  kind: 'number' | 'text' | 'select' | 'boolean';
  default: string | number | boolean;
  min?: number;
  options?: string[];
  help?: string;
  maxLength?: number;
}
export interface ModelNode {
  id: string;
  type: string;
  definitionVersion: 1;
  label: string;
  params: Params;
  position: { x: number; y: number };
  portLayout?: PortLayout;
  loopFeedbackRoutes?: Record<string, LoopFeedbackLayout>;
  /** Legacy background color; migrated to tensorBorderColor on load. */
  tensorColor?: string;
  tensorBorderColor?: string;
  /** Explicit annotation for the single out Tensor, not an executed backward. */
  tensorAutograd?: TensorAutograd;
  /** Legacy semantic annotation, retained for compatibility but ignored by provenance. */
  informationSource?: InformationSource;
  /** Another placement of the same value, owned by a canonical node in this graph. */
  valueSource?: string;
  /** Called module's forward, a helper function body, or an explicitly merged calculation group. */
  subgraph?: Graph;
  customized?: boolean;
}
export interface ModelEdge {
  id: string;
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
  waypoints?: { x: number; y: number }[];
  /** Label anchor as a fraction of the routed line's length, from source (0) to target (1). */
  labelPosition?: number;
}
export interface Graph {
  nodes: ModelNode[];
  edges: ModelEdge[];
}
export interface Inference<T = TensorSpec> {
  outputs: Record<string, T>;
  notes?: string[];
  warnings?: string[];
}
export interface ModuleDefinition {
  type: string;
  label: string;
  category: string;
  description: string;
  fields: Field[];
  /** Fields relevant to the active mode; inactive settings remain stored for switching back. */
  getFields?: (params: Params) => Field[];
  inputs: Port[];
  outputs: Port[];
  /** Optional parameter-dependent ports. IDs must remain stable while labels change. */
  getPorts?: (params: Params) => { inputs: Port[]; outputs: Port[] };
  composite?: boolean;
  infer?: (
    inputs: Record<string, TensorSpec>,
    params: Params,
    context?: InferenceContext,
  ) => Inference;
  inferValue?: (
    inputs: Record<string, ValueSpec>,
    params: Params,
    context?: InferenceContext,
  ) => Inference<ValueSpec>;
  build?: (params: Params) => Graph;
  validate?: (inputs: Record<string, TensorSpec>, params: Params) => string[];
}
export interface Registry {
  get(type: string): ModuleDefinition | undefined;
}
export interface NodeResult {
  status: 'ok' | 'warning' | 'error' | 'blocked';
  inputs: Record<string, ValueSpec>;
  outputs: Record<string, ValueSpec>;
  notes: string[];
  messages: string[];
  child?: GraphResult;
  /** ForEach body results in input order; child is the first iteration for legacy readers. */
  iterations?: GraphResult[];
}
export interface GraphResult {
  order: string[];
  nodes: Record<string, NodeResult>;
  outputs: Record<string, ValueSpec>;
  issues: string[];
}
export interface Variant {
  id: string;
  name: string;
  graph: Graph;
}
export interface Project {
  schemaVersion: 1;
  name: string;
  hyperparameters?: Hyperparameters;
  tensorPalette?: TensorPalette;
  /** Opt in to conservative information-source propagation; old projects remain unchanged. */
  lineageEnabled?: boolean;
  runtime?: {
    active?: string;
    snapshots: RuntimeReference[];
    /** Only portable exports embed the selected snapshot. Normal projects use references. */
    embedded?: RuntimeSnapshot;
  };
  /** Main model's forward body, displayed directly as the initial/root view. */
  graph: Graph;
  variants: Variant[];
}

export interface RuntimeReference {
  id: string;
  name: string;
  file: string;
}
export interface RuntimeSnapshot {
  format: 'tensor-trail.runtime-snapshot';
  version: 1;
  id: string;
  name: string;
  capturedAt: string;
  context: Record<string, unknown>;
  bindings: {
    nodePath: string[];
    nodeType: 'Input' | 'ForwardInput' | 'DictionaryInput';
    value: TensorSpec | DictionarySpec;
  }[];
}

export const shapeText = (s?: ValueSpec) =>
  !s
    ? '待连接'
    : isTensorSpec(s)
      ? `[${s.shape.join(', ')}]`
      : isControlSpec(s)
        ? `${s.branch === undefined ? '条件' : s.branch ? 'True' : 'False'} · ${s.expression}（未执行）`
      : isIntegerListSpec(s)
        ? `list[int] · [${s.values.join(', ')}]`
      : isDeviceSpec(s)
        ? `Device · ${s.reference}`
      : isNumpyArraySpec(s)
        ? `np.ndarray [${s.dims.join(', ')}] · ${s.dtype === 'unknown' ? 'dtype 未采集' : s.dtype}`
      : isDtypeSpec(s)
        ? `torch.${s.value}`
      : isModuleSpec(s)
        ? `Module · ${s.moduleType}`
        : isTensorSequenceSpec(s)
          ? `${s.kind} · ${s.items.length} 个 Tensor${s.items.length === 0 ? (s.kind === 'list' ? ' []' : ' ()') : ''}`
          : isSizeSpec(s)
            ? `torch.Size([${s.dims.join(', ')}])`
            : isIntegerSpec(s)
              ? `int · ${s.value}${typeof s.value === 'string' ? '（符号）' : ''}`
              : s.kind === 'none'
                ? 'None'
                : s.kind === 'bool'
                  ? `bool · ${s.value ? 'True' : 'False'}`
                  : s.kind === 'forward'
                    ? '网络输入'
                    : s.kind === 'dict'
                      ? isDictionarySpec(s)
                        ? `dict${s.keyType === 'int' ? '[int]' : ''} · ${Object.keys(s.fields).length} 个字段`
                        : '字典 dict'
                      : '列表 list';
export const clone = <T>(value: T): T => structuredClone(value);
export const uid = () => crypto.randomUUID();
