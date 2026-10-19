import type { Graph, GraphResult, NodeResult, Registry, TensorSpec } from './types';
import { graphIssues, orderGraph, ports } from './graph';

export function inferGraph(
  graph: Graph,
  registry: Registry,
  external: Record<string, TensorSpec> = {},
  depth = 0,
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
      if (!node.customized) r.messages.push(...(def.validate?.(r.inputs, node.params) ?? []));
      if (node.type === 'Inlet') {
        const value = external[String(node.params.port)];
        if (!value) throw new Error(`子图输入 ${node.params.port} 没有外部 Tensor。`);
        r.outputs = { out: value };
        r.notes = ['从上一级模块接收 Tensor。'];
      } else if (node.subgraph) {
        r.child = inferGraph(node.subgraph, registry, r.inputs, depth + 1);
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
      } else if (def.infer) {
        const inference = def.infer(r.inputs, node.params);
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

export function resultAt(result: GraphResult, path: string[]): GraphResult | undefined {
  let current: GraphResult | undefined = result;
  for (const id of path) current = current?.nodes[id]?.child;
  return current;
}
