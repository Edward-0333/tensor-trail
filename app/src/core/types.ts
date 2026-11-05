export type Dim = number | string;
export interface TensorSpec {
  shape: Dim[];
  axes: string[];
  dtype: string;
}
export interface ContainerSpec {
  kind: 'forward' | 'dict' | 'list';
  text: string;
}
export interface DictionarySpec {
  kind: 'dict';
  fields: Record<string, TensorSpec | DictionarySpec>;
}
/** A Python integer size, or an unresolved symbolic integer dimension; never a Tensor. */
export interface IntegerSpec {
  kind: 'int';
  value: Dim;
}
export type ValueSpec = TensorSpec | ContainerSpec | DictionarySpec | IntegerSpec;
export const isTensorSpec = (value: ValueSpec): value is TensorSpec => 'shape' in value;
export const isIntegerSpec = (value: ValueSpec): value is IntegerSpec =>
  !isTensorSpec(value) && value.kind === 'int';
export const isDictionarySpec = (value: ValueSpec): value is DictionarySpec =>
  !isTensorSpec(value) && value.kind === 'dict' && 'fields' in value;
export function tensorValue(value: ValueSpec | undefined): TensorSpec {
  if (!value) throw new Error('没有可用的 Tensor。');
  if (!isTensorSpec(value)) throw new Error(`需要 Tensor，当前为${shapeText(value)}。`);
  return value;
}
export type Params = Record<string, string | number | boolean>;
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
  tensorColor?: string;
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
  /** Main model's forward body, displayed directly as the initial/root view. */
  graph: Graph;
  variants: Variant[];
}

export const shapeText = (s?: ValueSpec) =>
  !s
    ? '待连接'
    : isTensorSpec(s)
      ? `[${s.shape.join(', ')}]`
      : isIntegerSpec(s)
        ? `int · ${s.value}${typeof s.value === 'string' ? '（符号）' : ''}`
        : s.kind === 'forward'
          ? '网络输入'
          : s.kind === 'dict'
            ? isDictionarySpec(s)
              ? `dict · ${Object.keys(s.fields).length} 个字段`
              : '字典 dict'
            : '列表 list';
export const clone = <T>(value: T): T => structuredClone(value);
export const uid = () => crypto.randomUUID();
