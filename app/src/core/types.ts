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
export type ValueSpec = TensorSpec | ContainerSpec;
export const isTensorSpec = (value: ValueSpec): value is TensorSpec => 'shape' in value;
export function tensorValue(value: ValueSpec | undefined): TensorSpec {
  if (!value) throw new Error('没有可用的 Tensor。');
  if (!isTensorSpec(value)) throw new Error(`需要 Tensor，当前为${shapeText(value)}。`);
  return value;
}
export type Params = Record<string, string | number | boolean>;
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
  inputs: Port[];
  outputs: Port[];
  composite?: boolean;
  infer?: (inputs: Record<string, TensorSpec>, params: Params) => Inference;
  inferValue?: (inputs: Record<string, ValueSpec>, params: Params) => Inference<ValueSpec>;
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
  graph: Graph;
  variants: Variant[];
}

export const shapeText = (s?: ValueSpec) =>
  !s
    ? '待连接'
    : isTensorSpec(s)
      ? `[${s.shape.join(', ')}]`
      : s.kind === 'forward'
        ? 'Forward输入'
        : s.kind === 'dict'
          ? '字典 dict'
          : '列表 list';
export const clone = <T>(value: T): T => structuredClone(value);
export const uid = () => crypto.randomUUID();
