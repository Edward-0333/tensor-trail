import type { Params, Project, TensorSpec } from '../core/types';

export interface RecipeNode {
  id: string;
  type: string;
  label?: string;
  params?: Params;
  subgraph?: RecipeGraph;
}
export interface RecipeGraph {
  nodes: RecipeNode[];
  connections: [string, string][];
}
export interface Recipe extends RecipeGraph {
  recipeVersion?: 1;
  name: string;
}
export type Operation =
  | { op: 'add_node'; path?: string[]; node: RecipeNode }
  | { op: 'remove_node'; path?: string[]; id: string }
  | { op: 'set_params'; path?: string[]; id: string; params: Params }
  | { op: 'rename_node'; path?: string[]; id: string; label: string }
  | { op: 'connect' | 'disconnect'; path?: string[]; from: string; to: string }
  | { op: 'layout'; path?: string[] }
  | { op: 'rename_project'; name: string };
export interface Patch {
  patchVersion?: 1;
  operations: Operation[];
}
export interface Diagnostic {
  code: string;
  severity: 'error' | 'warning';
  path: string[];
  message: string;
  parameter?: string;
}
export interface DefaultApplied {
  path: string[];
  parameter: string;
  value: string | number | boolean;
}
export interface NodeTrace {
  path: string[];
  type: string;
  status: string;
  inputs: Record<string, TensorSpec>;
  outputs: Record<string, TensorSpec>;
}
export interface ModelResponse {
  apiVersion: 1;
  ok: boolean;
  diagnostics: Diagnostic[];
  nodes: NodeTrace[];
  outputs: Record<string, TensorSpec>;
  defaultsApplied: DefaultApplied[];
  project?: Project;
}
