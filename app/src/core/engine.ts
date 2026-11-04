import type {
  Graph,
  GraphResult,
  NodeResult,
  Registry,
  TensorSpec,
  ValueSpec,
  Hyperparameters,
  Project,
} from './types';
import { isTensorSpec, shapeText } from './types';
import { graphIssues, orderGraph, ports } from './graph';

export function inferGraph(
  graph: Graph,
  registry: Registry,
  external: Record<string, ValueSpec> = {},
  depth = 0,
  hyperparameters: Hyperparameters = {},
): GraphResult {
  const result: GraphResult = { order: [], nodes: {}, outputs: {}, issues: [] };
  if (depth > 12) {
    result.issues.push('子图嵌套超过 12 层。');
    return result;
  }
  result.issues = graphIssues(graph, registry);
  if (result.issues.length) return result;
  result.order = orderGraph(graph);
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  for (const id of result.order) {
    const node = byId.get(id)!;
    const r: NodeResult = { status: 'ok', inputs: {}, outputs: {}, messages: [], notes: [] };
    result.nodes[id] = r;
    const def = registry.get(node.type);
    if (!def) {
      r.status = 'error';
      r.messages.push(`尚未注册 ${node.type} 的形状规则。`);
      continue;
    }
    for (const edge of graph.edges.filter((e) => e.target === id)) {
      const value = result.nodes[edge.source]?.outputs[edge.sourcePort];
      if (value) r.inputs[edge.targetPort] = value;
      else {
        r.status = 'blocked';
        r.messages.push(
          `上游 ${byId.get(edge.source)?.label} 的 ${edge.sourcePort} 暂无有效输出。`,
        );
      }
    }
    if (r.status === 'blocked') continue;
    const missing = ports(node, registry).inputs.filter((p) => !r.inputs[p.id]);
    if (missing.length) {
      r.status = 'blocked';
      r.messages.push(`请连接输入：${missing.map((p) => p.id).join('、')}。`);
      continue;
    }
    try {
      for (const f of def.fields) {
        // Customized composites derive their semantics solely from their concrete child graph.
        if (node.customized && node.subgraph) break;
        const value = node.params[f.key];
        // Older network inputs have only a name and remarks; their schema is optional.
        if (def.type === 'ForwardInput' && f.key === 'schema' && value === undefined) continue;
        if (
          f.kind === 'number' &&
          (typeof value !== 'number' || !Number.isSafeInteger(value) || value < (f.min ?? 1))
        )
          throw new Error(`${f.label} 必须是 ≥ ${f.min ?? 1} 的整数。`);
        if (f.kind === 'select' && !f.options?.includes(String(value)))
          throw new Error(`${f.label} 的选项无效。`);
        if (f.kind === 'boolean' && typeof value !== 'boolean')
          throw new Error(`${f.label} 必须为开关值。`);
        if (f.kind === 'text' && typeof value !== 'string')
          throw new Error(`${f.label} 必须为文本。`);
      }
      const passthrough = ['Output', 'Outlet', 'Identity'].includes(node.type);
      const acceptsContainers =
        passthrough ||
        node.type === 'Inlet' ||
        !!def.inferValue ||
        (!!node.subgraph && (!def.validate || node.customized));
      const tensorInputs: Record<string, TensorSpec> = {};
      for (const [port, value] of Object.entries(r.inputs)) {
        if (isTensorSpec(value)) tensorInputs[port] = value;
        else if (!acceptsContainers)
          throw new Error(
            `输入 ${port} 是${shapeText(value)}，此模块需要 Tensor。${value.kind === 'dict' ? '字典需先取出 Tensor 字段，请使用“字典字段读取”。' : ''}`,
          );
      }
      if (!node.customized) r.messages.push(...(def.validate?.(tensorInputs, node.params) ?? []));
      if (node.type === 'Inlet') {
        const value = external[String(node.params.port)];
        if (!value) throw new Error(`子图输入 ${node.params.port} 没有外部数据。`);
        r.outputs = { out: value };
        r.notes = ['从上一级模块接收输入，保留其数据类型。'];
      } else if (node.subgraph) {
        r.child = inferGraph(node.subgraph, registry, r.inputs, depth + 1, hyperparameters);
        const failures = Object.entries(r.child.nodes).filter(
          ([, v]) => v.status === 'error' || v.status === 'blocked',
        );
        if (r.child.issues.length || failures.length) {
          r.status = 'error';
          r.messages.push(
            ...r.child.issues,
            ...failures
              .slice(0, 3)
              .map(
                ([key, v]) =>
                  `${node.subgraph!.nodes.find((n) => n.id === key)?.label}：${v.messages[0]}`,
              ),
          );
        } else {
          r.outputs = r.child.outputs;
          const warnings = Object.values(r.child.nodes).flatMap((v) =>
            v.status === 'warning' ? v.messages : [],
          );
          r.messages.push(...new Set(warnings));
          r.notes = ['输入输出来自内部子图。双击模块可逐层查看。'];
          if (!Object.keys(r.outputs).length) throw new Error('子图没有可用的 Outlet 输出。');
        }
      } else if (passthrough) {
        r.outputs = { out: r.inputs.in };
        r.notes = ['保留输入的数据类型与内容。'];
      } else if (def.inferValue || def.infer) {
        const inference = def.inferValue
          ? def.inferValue(r.inputs, node.params, { hyperparameters })
          : def.infer!(tensorInputs, node.params, { hyperparameters });
        r.outputs = inference.outputs;
        r.notes = inference.notes ?? [];
        r.messages.push(...(inference.warnings ?? []));
      } else throw new Error('该模块尚无形状推导实现。');
      if (r.status === 'ok' && r.messages.length) r.status = 'warning';
      if (node.type === 'Outlet' && r.outputs.out)
        result.outputs[String(node.params.port)] = r.outputs.out;
    } catch (error) {
      r.status = 'error';
      r.outputs = {};
      r.messages = [(error as Error).message];
    }
  }
  return result;
}

export function inferProject(project: Project, registry: Registry): GraphResult {
  return inferGraph(project.graph, registry, {}, 0, project.hyperparameters ?? {});
}

export function resultAt(result: GraphResult, path: string[]): GraphResult | undefined {
  let current: GraphResult | undefined = result;
  for (const id of path) current = current?.nodes[id]?.child;
  return current;
}
