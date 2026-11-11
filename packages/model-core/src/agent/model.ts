import { registry } from '../modules/index';
import { isContainerInput } from '../modules/containers';
import {
  clone,
  type Graph,
  type GraphResult,
  type ModelNode,
  type Params,
  type Project,
} from '../core/types';
import { graphAt, markAncestors, orderGraph } from '../core/graph';
import { inferProject } from '../core/engine';
import { parseProject, serializeProject } from '../core/serialization';
import { isValueNode, syncValueLinks, valueOwner } from '../core/valueLinks';
import { removeValueNodes } from '../editing/valueLinkEditing';
import { identifierPattern } from './catalog';
import { parseDictionarySchema } from '../core/dictionary';
import { activeRuntimeSnapshot, parseRuntimeSnapshot } from '../core/runtimeSnapshot';
import { incompleteNodes } from './draft';
import { placeNewNodes, preserveGeneratedLayout } from './layout';
import { isHexColor } from '../core/tensorAppearance';
import type {
  DefaultApplied,
  Diagnostic,
  ModelResponse,
  ValidationOptions,
  PatchOptions,
} from './types';

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
function options(value: ValidationOptions | PatchOptions, patch = false): PatchOptions {
  const v = record(value, 'options');
  keys(v, ['mode', 'runtimeSnapshot', ...(patch ? ['layout'] : [])]);
  if (v.mode !== undefined && (typeof v.mode !== 'string' || !['strict', 'draft'].includes(v.mode)))
    fail('INVALID_ARGUMENT', 'mode 必须为 strict 或 draft。');
  if (
    v.layout !== undefined &&
    (typeof v.layout !== 'string' || !['preserve', 'auto'].includes(v.layout))
  )
    fail('INVALID_ARGUMENT', 'layout 必须为 preserve 或 auto。');
  return {
    mode: (v.mode ?? 'strict') as PatchOptions['mode'],
    layout: (v.layout ?? 'preserve') as PatchOptions['layout'],
    ...(v.runtimeSnapshot !== undefined
      ? { runtimeSnapshot: parseRuntimeSnapshot(v.runtimeSnapshot) }
      : {}),
  };
}
function position(value: unknown): ModelNode['position'] {
  const p = record(value, 'position');
  keys(p, ['x', 'y']);
  if (
    typeof p.x !== 'number' ||
    !Number.isFinite(p.x) ||
    typeof p.y !== 'number' ||
    !Number.isFinite(p.y)
  )
    fail('INVALID_ARGUMENT', 'position.x/y 必须是有限数值。');
  return { x: p.x, y: p.y };
}
function params(type: string, value: unknown, path: string[]): Params {
  const d = registry.get(type);
  if (!d) fail('UNKNOWN_MODULE', `未知模块 ${type}；请先调用 describeModules。`, path);
  const p = record(value, 'params');
  const effective = {
    ...Object.fromEntries(d.fields.map((f) => [f.key, f.default])),
    ...p,
  } as Params;
  const active = new Set((d.getFields?.(effective) ?? d.fields).map((f) => f.key));
  for (const [k, v] of Object.entries(p)) {
    const f = d.fields.find((field) => field.key === k);
    if (!f) fail('UNKNOWN_PARAMETER', `${type} 没有参数 ${k}。`, path, k);
    if (
      k === 'schema' &&
      ['ForwardInput', 'DictionaryInput'].includes(type) &&
      typeof v === 'object'
    ) {
      try {
        p[k] = parseDictionarySchema(v);
      } catch (error) {
        fail('INVALID_PARAMETER', (error as Error).message, path, k);
      }
      continue;
    }
    const valid =
      f.kind === 'number'
        ? typeof v === 'number' && Number.isSafeInteger(v) && (!active.has(k) || v >= (f.min ?? 1))
        : f.kind === 'boolean'
          ? typeof v === 'boolean'
          : f.kind === 'select'
            ? typeof v === 'string' && !!f.options?.includes(v)
            : typeof v === 'string' && v.length <= (f.maxLength ?? 1999);
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
  keys(v, ['id', 'type', 'label', 'params', 'subgraph', 'position']);
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
  if (v.position !== undefined) node.position = position(v.position);
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
  // orderGraph checks cycles before placement; checkedProject validates the full structure next.
  placeNewNodes(
    g,
    new Set(
      g.nodes
        .filter((_, i) => (v.nodes as Record<string, unknown>[])[i].position === undefined)
        .map((n) => n.id),
    ),
  );
  return g;
}
const empty = (): ModelResponse => ({
  apiVersion: 1,
  mode: 'strict',
  ok: false,
  diagnostics: [],
  nodes: [],
  outputs: {},
  defaultsApplied: [],
});
function errorResponse(e: unknown, mode: ModelResponse['mode'] = 'strict'): ModelResponse {
  const r = empty();
  r.mode = mode;
  r.diagnostics.push({
    code: e instanceof AgentError ? e.code : 'STRUCTURE_ERROR',
    severity: 'error',
    path: e instanceof AgentError ? e.path : [],
    message: e instanceof Error ? e.message : String(e),
    ...(e instanceof AgentError && e.parameter ? { parameter: e.parameter } : {}),
  });
  return r;
}
function inspect(project: Project, settings: ValidationOptions): ModelResponse {
  const response = empty();
  const draft = settings.mode === 'draft';
  response.mode = draft ? 'draft' : 'strict';
  const walk = (g: Graph, result: GraphResult, path: string[]) => {
    const incomplete = draft ? incompleteNodes(g, result) : new Set<string>();
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
        status: incomplete.has(n.id) ? 'incomplete' : r.status,
        inputs: r.inputs,
        outputs: r.outputs,
        ...(r.iterations ? { iterations: r.iterations.map((iteration, index) => ({ index, outputs: iteration.outputs })) } : {}),
      });
      const severity = r.status === 'warning' || incomplete.has(n.id) ? 'warning' : 'error';
      if (r.status !== 'ok')
        for (const message of r.messages)
          response.diagnostics.push({
            code: incomplete.has(n.id)
              ? 'INCOMPLETE_NODE'
              : r.status === 'blocked'
                ? 'UPSTREAM_BLOCKED'
                : r.status === 'warning'
                  ? 'SHAPE_WARNING'
                  : 'SHAPE_ERROR',
            severity,
            path: at,
            message,
          });
      if (n.subgraph && r.child) walk(n.subgraph, r.child, at);
      if (!path.length && n.type === 'Output' && !n.valueSource && r.inputs.in)
        response.outputs[n.id] = r.inputs.in;
    }
  };
  let inferenceProject = project;
  if (
    draft &&
    project.runtime?.active &&
    !activeRuntimeSnapshot(project, settings.runtimeSnapshot)
  ) {
    inferenceProject = clone(project);
    delete inferenceProject.runtime!.active;
    response.diagnostics.push({
      code: 'RUNTIME_NOT_LOADED',
      severity: 'warning',
      path: [],
      message: '运行快照未加载：草稿仅校验静态图，保留原快照引用，不代表运行时结果。',
    });
  }
  walk(project.graph, inferProject(inferenceProject, registry, settings.runtimeSnapshot), []);
  for (const type of ['Input', 'Output'])
    if (
      !project.graph.nodes.some(
        (n) => n.type === type || (type === 'Input' && isContainerInput(n.type)),
      )
    ) {
      response.diagnostics.push({
        code: `MISSING_${type.toUpperCase()}`,
        severity: draft ? 'warning' : 'error',
        path: [],
        message:
          type === 'Input'
            ? '顶层模型至少需要一个输入节点（Input、ForwardInput 或 DictionaryInput）。'
            : `顶层模型至少需要一个 ${type} 节点。`,
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
export function buildModel(recipe: unknown, settings: ValidationOptions = {}): ModelResponse {
  try {
    const opts = options(settings);
    bounded(recipe);
    const v = record(recipe, 'recipe');
    keys(v, ['recipeVersion', 'name', 'nodes', 'connections', 'hyperparameters']);
    if (v.recipeVersion !== undefined && v.recipeVersion !== 1)
      fail('UNSUPPORTED_VERSION', '仅支持 recipeVersion: 1。');
    const defaults: DefaultApplied[] = [];
    const project = checkedProject({
      schemaVersion: 1,
      name: name(v.name),
      ...(v.hyperparameters !== undefined ? { hyperparameters: v.hyperparameters } : {}),
      graph: makeGraph({ nodes: v.nodes, connections: v.connections }, [], defaults),
      variants: [],
    });
    return { ...inspect(project, opts), defaultsApplied: defaults };
  } catch (e) {
    return errorResponse(e, settings?.mode === 'draft' ? 'draft' : 'strict');
  }
}

/** Accept either a recipe or a full schemaVersion:1 project. */
export function validateModel(value: unknown, settings: ValidationOptions = {}): ModelResponse {
  try {
    const opts = options(settings);
    const v = record(value, 'model');
    return 'schemaVersion' in v ? inspect(checkedProject(v), opts) : buildModel(v, settings);
  } catch (e) {
    return errorResponse(e, settings?.mode === 'draft' ? 'draft' : 'strict');
  }
}

/** Apply a whole patch to a clone. A failed response never mutates the source. */
export function patchModel(
  value: unknown,
  patch: unknown,
  settings: PatchOptions = {},
): ModelResponse {
  try {
    const opts = options(settings, true);
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
    const unplaced = new Map<string, { path: string[]; ids: Set<string> }>();
    const fields: Record<string, string[]> = {
      add_node: ['node'],
      replace_node: ['id', 'node'],
      duplicate_linked: ['source', 'id', 'position'],
      remove_node: ['id'],
      set_params: ['id', 'params'],
      rename_node: ['id', 'label'],
      set_position: ['id', 'position'],
      set_tensor_border: ['id', 'color'],
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
        const node = makeNode(op.node, path, defaults);
        g.nodes.push(node);
        const key = JSON.stringify(path);
        if (opts.layout === 'auto') layouts.set(key, path);
        else if (record(op.node, 'node').position === undefined) {
          if (!unplaced.has(key)) unplaced.set(key, { path, ids: new Set() });
          unplaced.get(key)!.ids.add(node.id);
        }
      } else if (op.op === 'replace_node') {
        const previous = find(), replacement = record(op.node, 'node');
        if (replacement.id !== previous.id)
          fail('INVALID_ID', '替换节点必须保留同一 id。', [...path, previous.id]);
        if (previous.valueSource !== undefined || g.nodes.some(n => n.valueSource === previous.id))
          fail('INVALID_MODULE', '有关联副本的节点不能直接替换，请先解除链接或移除副本。', [...path, previous.id]);
        const node = makeNode({ ...replacement, position: replacement.position ?? previous.position }, path, defaults);
        if (previous.portLayout) node.portLayout = clone(previous.portLayout);
        if (previous.tensorBorderColor) node.tensorBorderColor = previous.tensorBorderColor;
        g.nodes[g.nodes.indexOf(previous)] = node;
        if (opts.layout === 'auto') layouts.set(JSON.stringify(path), path);
      } else if (op.op === 'duplicate_linked') {
        const sourceId = id(op.source), newId = id(op.id);
        const source = g.nodes.find(n => n.id === sourceId);
        if (!source) fail('UNKNOWN_NODE', `节点 ${sourceId} 不存在于当前画布。`, [...path, sourceId]);
        if (!isValueNode(source.type)) fail('INVALID_MODULE', '只有值节点可以创建关联副本，算子不能共享执行结果。', [...path, sourceId]);
        if (g.nodes.some(n => n.id === newId)) fail('DUPLICATE_ID', `节点 ${newId} 已存在。`, [...path, newId]);
        const owner = valueOwner(g, source);
        const copy = { ...clone(source), id: newId, valueSource: owner.id,
          position: op.position === undefined ? { x: source.position.x + 48, y: source.position.y + 48 } : position(op.position) };
        g.nodes.push(copy);
        const key = JSON.stringify(path);
        if (opts.layout === 'auto') layouts.set(key, path);
        else if (op.position === undefined) {
          if (!unplaced.has(key)) unplaced.set(key, { path, ids: new Set() });
          unplaced.get(key)!.ids.add(copy.id);
        }
      } else if (op.op === 'remove_node') {
        const node = find();
        if (!node.valueSource && ['Inlet', 'Outlet'].includes(node.type))
          fail('PROTECTED_BOUNDARY', '不能删除子图边界节点。', [...path, node.id]);
        removeValueNodes(project.graph, path, [node.id]);
      } else if (op.op === 'set_params') {
        const requested = find(),
          node = valueOwner(g, requested),
          at = [...path, requested.id];
        if (node.subgraph && node.customized)
          fail('CUSTOMIZED_COMPOSITE', '此组合模块已手动修改，请用 path 修改内部算子。', at);
        Object.assign(node.params, params(node.type, op.params, at));
        const def = registry.get(node.type)!;
        if (def.build) {
          const generated = def.build(node.params);
          if (opts.layout === 'preserve' && node.subgraph)
            preserveGeneratedLayout(node.subgraph, generated);
          node.subgraph = generated;
        }
      } else if (op.op === 'set_tensor_border') {
        const node = find();
        if (op.color !== null && !isHexColor(op.color))
          fail('INVALID_ARGUMENT', '边框颜色需要 #RRGGBB，或使用 null 恢复默认。', [...path, node.id]);
        delete node.tensorColor;
        if (op.color === null) delete node.tensorBorderColor;
        else node.tensorBorderColor = op.color as string;
      } else if (op.op === 'rename_node') valueOwner(g, find()).label = name(op.label);
      else if (op.op === 'set_position') {
        find().position = position(op.position);
        unplaced.get(JSON.stringify(path))?.ids.delete(String(op.id));
      } else if (op.op === 'connect' || op.op === 'disconnect') {
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
      if (!['rename_node', 'layout', 'set_position', 'set_tensor_border'].includes(op.op))
        markAncestors(project.graph, path);
      if (countNodes(project.graph) > 2000) fail('LIMIT_EXCEEDED', '展开后的节点数不能超过 2000。');
    }
    // Validate only the complete transaction: disconnect + reconnect may be temporarily incomplete.
    syncValueLinks(project.graph);
    const checked = checkedProject(project);
    for (const { path, ids } of unplaced.values())
      if (!layouts.has(JSON.stringify(path))) placeNewNodes(graphAt(checked.graph, path), ids);
    for (const path of layouts.values()) layoutGraph(graphAt(checked.graph, path));
    return { ...inspect(checked, opts), defaultsApplied: defaults };
  } catch (e) {
    return errorResponse(e, settings?.mode === 'draft' ? 'draft' : 'strict');
  }
}

export function firstError(response: ModelResponse) {
  return response.diagnostics.find((d) => d.severity === 'error')?.message ?? '模型校验失败。';
}

export function applicationError(code: string, message: string): Diagnostic {
  return { code, message, path: [], severity: 'error' };
}

export { clone, serializeProject };
