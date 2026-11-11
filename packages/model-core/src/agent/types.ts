import type { Params, Project, ValueSpec, Hyperparameters, RuntimeSnapshot } from '../core/types';

export type ValidationMode = 'strict' | 'draft';
export interface ValidationOptions {
  mode?: ValidationMode;
  /** Optional loaded sidecar; never embedded into the returned project. */
  runtimeSnapshot?: RuntimeSnapshot;
}
export interface PatchOptions extends ValidationOptions {
  /** Existing geometry is preserved by default. Explicit layout operations still run. */
  layout?: 'preserve' | 'auto';
}

export interface RecipeNode {
  id: string;
  type: string;
  label?: string;
  params?: Params;
  subgraph?: RecipeGraph;
  position?: { x: number; y: number };
}
export interface RecipeGraph {
  nodes: RecipeNode[];
  connections: [string, string][];
}
export interface Recipe extends RecipeGraph {
  recipeVersion?: 1;
  name: string;
  hyperparameters?: Hyperparameters;
}
export type Operation =
  | { op: 'add_node'; path?: string[]; node: RecipeNode }
  | { op: 'replace_node'; path?: string[]; id: string; node: RecipeNode }
  | { op: 'duplicate_linked'; path?: string[]; source: string; id: string; position?: { x: number; y: number } }
  | { op: 'remove_node'; path?: string[]; id: string }
  | { op: 'set_params'; path?: string[]; id: string; params: Params }
  | { op: 'rename_node'; path?: string[]; id: string; label: string }
  | { op: 'set_position'; path?: string[]; id: string; position: { x: number; y: number } }
  | { op: 'set_tensor_border'; path?: string[]; id: string; color: string | null }
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
  inputs: Record<string, ValueSpec>;
  outputs: Record<string, ValueSpec>;
  iterations?: { index: number; outputs: Record<string, ValueSpec> }[];
}
export interface ModelResponse {
  apiVersion: 1;
  mode: ValidationMode;
  ok: boolean;
  diagnostics: Diagnostic[];
  nodes: NodeTrace[];
  outputs: Record<string, ValueSpec>;
  defaultsApplied: DefaultApplied[];
  project?: Project;
}
