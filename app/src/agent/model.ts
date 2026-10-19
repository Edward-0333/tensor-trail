import { registry } from '../modules';
import {
  clone,
  type Graph,
  type GraphResult,
  type ModelNode,
  type Params,
  type Project,
} from '../core/types';
import { graphAt, markAncestors, orderGraph } from '../core/graph';
import { inferGraph } from '../core/engine';
import { parseProject, serializeProject } from '../core/serialization';
import { identifierPattern } from './catalog';
import type { DefaultApplied, Diagnostic, ModelResponse } from './types';

class AgentError extends Error {
  constructor(
    public code: string,
    message: string,
    public path: string[] = [],
    public parameter?: string,
  ) {
    super(message);
  }
}
function fail(code: string, message: string, path: string[] = [], parameter?: string): never {
  throw new AgentError(code, message, path, parameter);
}
function record(v: unknown, label: string): Record<string, unknown> {
  if (!v || typeof v !== 'object' || Array.isArray(v))
    fail('INVALID_ARGUMENT', `${label} 必须是对象。`);
  return v as Record<string, unknown>;
}
function keys(v: Record<string, unknown>, allowed: string[]) {
  for (const k of Object.keys(v)) if (!allowed.includes(k)) fail('UNKNOWN_FIELD', `未知字段：${k}`);
}
function id(v: unknown): string {
  if (typeof v !== 'string' || !new RegExp(identifierPattern).test(v))
    fail('INVALID_ID', `无效标识符：${String(v)}`);
  return v;
}
function name(v: unknown): string {
  if (typeof v !== 'string' || !v.trim() || v.length > 200)
    fail('INVALID_ARGUMENT', '名称必须是 1–200 个字符。');
  return v;
}
function bounded(v: unknown) {
  const text = JSON.stringify(v);
  if (!text || new TextEncoder().encode(text).length > 3_000_000)
    fail('LIMIT_EXCEEDED', 'JSON 不得超过 3 MB。');
}
function params(type: string, value: unknown, path: string[]): Params {
  const d = registry.get(type);
  if (!d) fail('UNKNOWN_MODULE', `未知模块 ${type}；请先调用 describeModules。`, path);
  const p = record(value, 'params');
  for (const [k, v] of Object.entries(p)) {
    const f = d.fields.find((field) => field.key === k);
    if (!f) fail('UNKNOWN_PARAMETER', `${type} 没有参数 ${k}。`, path, k);
    const valid =
      f.kind === 'number'
        ? typeof v === 'number' && Number.isSafeInteger(v) && v >= (f.min ?? 1)
        : f.kind === 'boolean'
          ? typeof v === 'boolean'
          : f.kind === 'select'
            ? typeof v === 'string' && !!f.options?.includes(v)
            : typeof v === 'string' && v.length < 2000;
    if (!valid) fail('INVALID_PARAMETER', `${type}.${k} 的类型、范围或选项无效。`, path, k);
  }
  return p as Params;
}
function countNodes(g: Graph): number {
  return g.nodes.reduce((n, node) => n + 1 + (node.subgraph ? countNodes(node.subgraph) : 0), 0);
}
export function layoutGraph(g: Graph) {
  const order = orderGraph(g);
  const levels = new Map<string, number>();
  const rows = new Map<number, number>();
  for (const key of order) {
    const level = Math.max(
      0,
      ...g.edges.filter((e) => e.target === key).map((e) => (levels.get(e.source) ?? 0) + 1),
    );
    levels.set(key, level);
    const row = rows.get(level) ?? 0;
    rows.set(level, row + 1);
    g.nodes.find((n) => n.id === key)!.position = { x: level * 310, y: row * 230 };
  }
}
function endpoint(v: unknown): [string, string] {
  if (typeof v !== 'string' || v.split('.').length !== 2)
    fail('INVALID_ENDPOINT', '端点格式为 nodeId.portId。');
  const [node, port] = v.split('.');
  return [id(node), id(port)];
}
function edge(from: unknown, to: unknown, edgeId: string) {
  const [source, sourcePort] = endpoint(from),
    [target, targetPort] = endpoint(to);
  return { id: edgeId, source, sourcePort, target, targetPort };
}
function makeNode(value: unknown, path: string[], defaults: DefaultApplied[]): ModelNode {
  if (path.length > 12) fail('LIMIT_EXCEEDED', '子图嵌套不能超过 12 层。', path);
  const v = record(value, 'node');
  keys(v, ['id', 'type', 'label', 'params', 'subgraph']);
  const key = id(v.id),
    type = name(v.type),
    at = [...path, key];
  const p = params(type, v.params ?? {}, at),
    d = registry.get(type)!;
  if (!path.length && ['Inlet', 'Outlet'].includes(type))
    fail('INVALID_MODULE', 'Inlet/Outlet 只能用于子图边界。', at);
  if (v.subgraph !== undefined && !d.composite) fail('INVALID_MODULE', `${type} 不支持子图。`, at);
  for (const f of d.fields)
    if (!(f.key in p)) defaults.push({ path: at, parameter: f.key, value: f.default });
  const node = registry.create(type, p, key);
  if (v.label !== undefined) node.label = name(v.label);
  if (v.subgraph !== undefined) {
    node.subgraph = makeGraph(v.subgraph, at, defaults);
    node.customized = true;
  }
  if (d.composite && !node.subgraph)
    fail('MISSING_SUBGRAPH', `${type} 需要带有 Inlet/Outlet 的 subgraph。`, at);
  return node;
}
function makeGraph(value: unknown, path: string[], defaults: DefaultApplied[]): Graph {
  const v = record(value, 'graph');
  keys(v, ['nodes', 'connections']);
  if (
    !Array.isArray(v.nodes) ||
    v.nodes.length > 2000 ||
    !Array.isArray(v.connections) ||
    v.connections.length > 8000
  )
    fail('LIMIT_EXCEEDED', 'nodes/connections 必须是数组，且不能超过 2000/8000 项。', path);
  const g: Graph = { nodes: [], edges: [] };
  let count = 0;
  for (const raw of v.nodes) {
    const node = makeNode(raw, path, defaults);
    count += 1 + (node.subgraph ? countNodes(node.subgraph) : 0);
    if (count > 2000) fail('LIMIT_EXCEEDED', '展开后的节点数不能超过 2000。', path);
    g.nodes.push(node);
  }
  g.edges = v.connections.map((c, i) => {
    if (!Array.isArray(c) || c.length !== 2)
      fail('INVALID_ENDPOINT', '连接必须是 [source.port, target.port]。', path);
    return edge(c[0], c[1], `edge_${i}`);
  });
  // Structural validation happens before layout, including cycles and duplicate IDs.
  return g;
}
function layoutTree(g: Graph) {
  layoutGraph(g);
  for (const n of g.nodes) if (n.subgraph && n.customized) layoutTree(n.subgraph);
}
const empty = (): ModelResponse => ({
  apiVersion: 1,
  ok: false,
  diagnostics: [],
  nodes: [],
  outputs: {},
  defaultsApplied: [],
});
function errorResponse(e: unknown): ModelResponse {
  const r = empty();
  r.diagnostics.push({
    code: e instanceof AgentError ? e.code : 'STRUCTURE_ERROR',
    severity: 'error',
    path: e instanceof AgentError ? e.path : [],
    message: e instanceof Error ? e.message : String(e),
    ...(e instanceof AgentError && e.parameter ? { parameter: e.parameter } : {}),
  });
  return r;
}
function inspect(project: Project): ModelResponse {
  const response = empty();
  const walk = (g: Graph, result: GraphResult, path: string[]) => {
    result.issues.forEach((message) =>
      response.diagnostics.push({ code: 'STRUCTURE_ERROR', severity: 'error', path, message }),
    );
    for (const n of g.nodes) {
      const at = [...path, n.id],
        r = result.nodes[n.id];
      // Customized composites retain historical generator parameters. Their actual child graph is authoritative.
      if (!(n.customized && n.subgraph)) params(n.type, n.params, at);
      if (!r) continue;
      response.nodes.push({
        path: at,
        type: n.type,
        status: r.status,
        inputs: r.inputs,
        outputs: r.outputs,
      });
      const severity = r.status === 'warning' ? 'warning' : 'error';
      if (r.status !== 'ok')
        for (const message of r.messages)
          response.diagnostics.push({
            code:
              r.status === 'blocked'
                ? 'UPSTREAM_BLOCKED'
                : r.status === 'warning'
                  ? 'SHAPE_WARNING'
                  : 'SHAPE_ERROR',
            severity,
            path: at,
            message,
          });
      if (n.subgraph && r.child) walk(n.subgraph, r.child, at);
      if (!path.length && n.type === 'Output' && r.inputs.in) response.outputs[n.id] = r.inputs.in;
    }
  };
  walk(project.graph, inferGraph(project.graph, registry), []);
  for (const type of ['Input', 'Output'])
    if (!project.graph.nodes.some((n) => n.type === type)) {
      response.diagnostics.push({
        code: `MISSING_${type.toUpperCase()}`,
        severity: 'error',
        path: [],
        message: `顶层模型至少需要一个 ${type} 节点。`,
      });
    }
  response.ok = !response.diagnostics.some((d) => d.severity === 'error');
  response.project = project;
  return response;
}
function checkedProject(value: unknown): Project {
  bounded(value);
  return parseProject(JSON.stringify(value), registry);
}

/** Build a draft; no UI, storage or network side effects. */
export function buildModel(recipe: unknown): ModelResponse {
  try {
    bounded(recipe);
    const v = record(recipe, 'recipe');
    keys(v, ['recipeVersion', 'name', 'nodes', 'connections']);
    if (v.recipeVersion !== undefined && v.recipeVersion !== 1)
      fail('UNSUPPORTED_VERSION', '仅支持 recipeVersion: 1。');
    const defaults: DefaultApplied[] = [];
    const project = checkedProject({
      schemaVersion: 1,
      name: name(v.name),
      graph: makeGraph({ nodes: v.nodes, connections: v.connections }, [], defaults),
      variants: [],
    });
    layoutTree(project.graph);
    return { ...inspect(project), defaultsApplied: defaults };
  } catch (e) {
    return errorResponse(e);
  }
}

/** Accept either a recipe or a full schemaVersion:1 project. */
export function validateModel(value: unknown): ModelResponse {
  try {
    const v = record(value, 'model');
    return 'schemaVersion' in v ? inspect(checkedProject(v)) : buildModel(v);
  } catch (e) {
    return errorResponse(e);
  }
}

/** Apply a whole patch to a clone. A failed response never mutates the source. */
export function patchModel(value: unknown, patch: unknown): ModelResponse {
  try {
    const project = checkedProject(value),
      v = record(patch, 'patch');
    bounded(patch);
    keys(v, ['patchVersion', 'operations']);
    if (v.patchVersion !== undefined && v.patchVersion !== 1)
      fail('UNSUPPORTED_VERSION', '仅支持 patchVersion: 1。');
    if (!Array.isArray(v.operations) || !v.operations.length || v.operations.length > 500)
      fail('LIMIT_EXCEEDED', 'operations 需要 1–500 个操作。');
    const defaults: DefaultApplied[] = [];
    const layouts = new Map<string, string[]>();
    const fields: Record<string, string[]> = {
      add_node: ['node'],
      remove_node: ['id'],
      set_params: ['id', 'params'],
      rename_node: ['id', 'label'],
      connect: ['from', 'to'],
      disconnect: ['from', 'to'],
      layout: [],
      rename_project: ['name'],
    };
    for (const raw of v.operations) {
      const op = record(raw, 'operation');
      if (typeof op.op !== 'string' || !Object.hasOwn(fields, op.op))
        fail('UNKNOWN_OPERATION', `未知操作 ${String(op.op)}`);
      keys(op, ['op', ...(op.op === 'rename_project' ? [] : ['path']), ...fields[op.op]]);
      if (op.op === 'rename_project') {
        project.name = name(op.name);
        continue;
      }
      if (op.path !== undefined && (!Array.isArray(op.path) || op.path.length > 12))
        fail('INVALID_PATH', 'path 必须是最多 12 层的节点 ID 数组。');
      const path = ((op.path ?? []) as unknown[]).map(id);
      const g = graphAt(project.graph, path);
      const find = () => {
        const key = id(op.id),
          node = g.nodes.find((n) => n.id === key);
        if (!node) fail('UNKNOWN_NODE', `节点 ${key} 不存在。`, [...path, key]);
        return node;
      };
      if (op.op === 'add_node') {
        g.nodes.push(makeNode(op.node, path, defaults));
        layouts.set(JSON.stringify(path), path);
      } else if (op.op === 'remove_node') {
        const node = find();
        if (['Inlet', 'Outlet'].includes(node.type))
          fail('PROTECTED_BOUNDARY', '不能删除子图边界节点。', [...path, node.id]);
        g.nodes = g.nodes.filter((n) => n.id !== node.id);
        g.edges = g.edges.filter((e) => e.source !== node.id && e.target !== node.id);
      } else if (op.op === 'set_params') {
        const node = find(),
          at = [...path, node.id];
        if (node.subgraph && node.customized)
          fail('CUSTOMIZED_COMPOSITE', '此组合模块已手动修改，请用 path 修改内部算子。', at);
        Object.assign(node.params, params(node.type, op.params, at));
        const def = registry.get(node.type)!;
        if (def.build) node.subgraph = def.build(node.params);
      } else if (op.op === 'rename_node') find().label = name(op.label);
      else if (op.op === 'connect' || op.op === 'disconnect') {
        let i = 0;
        while (g.edges.some((e) => e.id === `edge_${i}`)) i++;
        const e = edge(op.from, op.to, `edge_${i}`);
        const matches = (old: typeof e) =>
          old.source === e.source &&
          old.sourcePort === e.sourcePort &&
          old.target === e.target &&
          old.targetPort === e.targetPort;
        if (op.op === 'connect') g.edges.push(e);
        else {
          if (!g.edges.some(matches)) fail('UNKNOWN_CONNECTION', '要断开的连接不存在。', path);
          g.edges = g.edges.filter((old) => !matches(old));
        }
      } else if (op.op === 'layout') layouts.set(JSON.stringify(path), path);
      if (!['rename_node', 'layout'].includes(op.op)) markAncestors(project.graph, path);
      if (countNodes(project.graph) > 2000) fail('LIMIT_EXCEEDED', '展开后的节点数不能超过 2000。');
    }
    // Validate only the complete transaction: disconnect + reconnect may be temporarily incomplete.
    const checked = checkedProject(project);
    for (const path of layouts.values()) layoutGraph(graphAt(checked.graph, path));
    return { ...inspect(checked), defaultsApplied: defaults };
  } catch (e) {
    return errorResponse(e);
  }
}

export function firstError(response: ModelResponse) {
  return response.diagnostics.find((d) => d.severity === 'error')?.message ?? '模型校验失败。';
}

export function applicationError(code: string, message: string): Diagnostic {
  return { code, message, path: [], severity: 'error' };
}

export { clone, serializeProject };
