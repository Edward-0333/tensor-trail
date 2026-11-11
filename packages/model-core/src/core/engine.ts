import type {
  Graph,
  GraphResult,
  NodeResult,
  Registry,
  TensorSpec,
  ValueSpec,
  Hyperparameters,
  Project,
  RuntimeSnapshot,
} from './types';
import { clone, isTensorSpec, shapeText } from './types';
import { graphIssues, orderGraph, ports } from './graph';
import { normalizeTensorSelection } from './indexingTypes';
import { annotateTensorOutputs } from './tensorAppearance';
import { activeRuntimeSnapshot, checkSnapshotBindings } from './runtimeSnapshot';
import { inferForEach, loopIteration, type LoopSelections } from './forEach';
import { inferWhile } from './whileLoop';
import { planAttributes, type AttributeContext } from './attributes';

export function inferGraph(
  graph: Graph,
  registry: Registry,
  external: Record<string, ValueSpec> = {},
  depth = 0,
  hyperparameters: Hyperparameters = {},
  runtime?: RuntimeSnapshot,
  path: string[] = [],
  budget = { remaining: 20000 },
  attributeContext?: AttributeContext,
): GraphResult {
  const result: GraphResult = { order: [], nodes: {}, outputs: {}, issues: [] };
  if (depth > 12) {
    result.issues.push('子图嵌套超过 12 层。');
    return result;
  }
  budget.remaining -= graph.nodes.length;
  if (budget.remaining < 0) {
    result.issues.push('静态推导超过 20000 次节点访问，请减少嵌套循环或元素数量。');
    return result;
  }
  const attributes = attributeContext ?? { plan: planAttributes(graph), values: new Map<string, TensorSpec>() };
  const implicit = attributes.plan.dependencies.get(JSON.stringify(path)) ?? [];
  result.issues = [...attributes.plan.issues, ...graphIssues(graph, registry, implicit)];
  if (result.issues.length) return result;
  result.order = orderGraph(graph, implicit);
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  for (const id of result.order) {
    const stored = byId.get(id)!;
    // Also accept legacy nodes passed directly to inference, without mutating the graph.
    const node = { ...stored, ...normalizeTensorSelection(stored.type, stored.params) };
    const r: NodeResult = { status: 'ok', inputs: {}, outputs: {}, messages: [], notes: [] };
    result.nodes[id] = r;
    const attributePath = JSON.stringify([...path, id]);
    const attributeError = attributes.plan.errors.get(attributePath);
    if (attributeError) {
      r.status = 'error';
      r.messages.push(attributeError);
      continue;
    }
    const def = registry.get(node.type);
    if (!def) {
      r.status = 'error';
      r.messages.push(`尚未注册 ${node.type} 的形状规则。`);
      continue;
    }
    if (node.valueSource !== undefined) {
      const owner = result.nodes[node.valueSource];
      if (!owner || owner.status === 'error' || owner.status === 'blocked') {
        r.status = 'blocked';
        r.messages = [
          `共享值来源 ${byId.get(node.valueSource)?.label ?? node.valueSource} 暂无有效结果。`,
          ...(owner?.messages ?? []),
        ];
      } else {
        r.status = owner.status;
        r.inputs = clone(owner.inputs);
        r.outputs = clone(owner.outputs);
        r.messages = [...owner.messages];
        r.notes = [...owner.notes, '与来源节点共享同一值；解除链接后可独立修改。'];
      }
      // A boundary alias is another local view, never an extra parent interface.
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
      for (const f of def.getFields?.(node.params) ?? def.fields) {
        // Customized composites derive their semantics solely from their concrete child graph.
        if (node.customized && node.subgraph) break;
        const value = node.params[f.key];
        // Projects predating assignment versions have one unversioned attribute write.
        if (f.key === 'version' && ['SetAttribute', 'GetAttribute'].includes(node.type) && value === undefined) continue;
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
        const objectSchema = f.key === 'schema' && ['ForwardInput','DictionaryInput'].includes(node.type) && value && typeof value === 'object';
        if (f.kind === 'text' && typeof value !== 'string' && !objectSchema)
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
      } else if (node.type === 'GetAttribute') {
        const binding = attributes.plan.bindings.get(attributePath)!;
        const value = attributes.values.get(binding.key);
        if (!value) throw new Error(`属性 ${binding.name} 尚无有效写入结果；请检查对应 SET 的上游数据。`);
        r.outputs = { out: clone(value) };
        r.notes = [`从属性 ${JSON.stringify(binding.owner)}.${binding.name} 的唯一 SET 读取同一个 Tensor 引用。`, '只复制规格以隔离编辑，不代表 Python 中 clone；保留上游形状和梯度标注，不新增 forward 参数。'];
      } else if (node.type === 'WhileLoop' && node.subgraph) {
        Object.assign(r, inferWhile(node, r.inputs, external =>
          inferGraph(node.subgraph!, registry, external, depth + 1, hyperparameters, runtime, [...path, id], budget, attributes)));
      } else if (node.type === 'ForEachTensor' && node.subgraph) {
        Object.assign(r, inferForEach(node.subgraph, r.inputs, external =>
          inferGraph(node.subgraph!, registry, external, depth + 1, hyperparameters, runtime, [...path, id], budget, attributes)));
      } else if (node.subgraph) {
        r.child = inferGraph(node.subgraph, registry, r.inputs, depth + 1, hyperparameters, runtime, [...path,id], budget, attributes);
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
          if (!Object.keys(r.outputs).length) {
            if (node.type === 'Submodule' && !node.subgraph.nodes.some((n) => n.type === 'Outlet'))
              r.messages.push('子模块内部流程待完善，尚未定义输出。');
            else throw new Error('子图没有可用的 Outlet 输出。');
          }
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
      const observed = runtime?.bindings.find(binding => JSON.stringify(binding.nodePath) === JSON.stringify([...path,id]));
      if (observed) {
        r.outputs = { out: clone(observed.value) };
        r.notes.push(`运行快照：${runtime!.name} · ${runtime!.capturedAt}；不修改静态图。`);
      } else r.outputs = annotateTensorOutputs(node, r.inputs, r.outputs);
      if (node.type === 'SetAttribute') {
        const binding = attributes.plan.bindings.get(attributePath)!;
        attributes.values.set(binding.key, clone(tensorInputs.in));
      }
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

export function inferProject(project: Project, registry: Registry, loaded?: RuntimeSnapshot): GraphResult {
  const runtime = activeRuntimeSnapshot(project, loaded);
  if (runtime) {
    try { checkSnapshotBindings(project,runtime); }
    catch(error) { return {order:[],nodes:{},outputs:{},issues:[(error as Error).message]}; }
  }
  const result = inferGraph(project.graph, registry, {}, 0, project.hyperparameters ?? {}, runtime);
  if (project.runtime?.active && !runtime) result.issues.push('选中的运行快照尚未加载；当前仅有静态图信息，请在“运行数据”中加载快照或切换为静态图。');
  return result;
}

export function resultAt(result: GraphResult, path: string[], selections: LoopSelections = {}): GraphResult | undefined {
  let current: GraphResult | undefined = result;
  for (const [index, id] of path.entries()) {
    const node: NodeResult | undefined = current?.nodes[id];
    current = node?.iterations
      ? node.iterations[loopIteration(node, selections[JSON.stringify(path.slice(0, index + 1))])]
      : node?.child;
  }
  return current;
}
