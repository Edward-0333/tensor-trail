import type { Graph, ModelNode, Params, TensorSpec } from './types';

export interface AttributeBinding {
  owner: string[];
  name: string;
  version: string;
  key: string;
}
export interface AttributePlan {
  bindings: Map<string, AttributeBinding>;
  writers: Map<string, string[]>;
  dependencies: Map<string, { source: string; target: string }[]>;
  errors: Map<string, string>;
  issues: string[];
}
export interface AttributeContext {
  plan: AttributePlan;
  values: Map<string, TensorSpec>;
}

const forbidden = new Set(['__proto__', 'constructor', 'prototype']);
export const attributeTypes = new Set(['SetAttribute', 'GetAttribute']);

export function attributeBinding(params: Params): AttributeBinding {
  if (typeof params.owner !== 'string' || params.owner.length > 1999)
    throw new Error('所属模块路径需要 JSON 数组，例如 ["pts_bbox_head"]；[] 表示根模型。');
  let owner: unknown;
  try {
    owner = JSON.parse(params.owner);
  } catch {
    throw new Error('所属模块路径不是合法 JSON 数组。');
  }
  if (
    !Array.isArray(owner) ||
    owner.length > 12 ||
    !owner.every(
      (id) => typeof id === 'string' && /^[A-Za-z0-9_-]{1,100}$/.test(id) && !forbidden.has(id),
    )
  )
    throw new Error('所属模块路径需要至多 12 个有效节点 ID，不是 Python 表达式。');
  if (
    typeof params.name !== 'string' ||
    params.name.length > 128 ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(params.name) ||
    forbidden.has(params.name)
  )
    throw new Error('属性名需要单个安全的 Python 属性名称，例如 points_scale。');
  const version = params.version ?? '';
  if (typeof version !== 'string' || (version !== '' && !/^[A-Za-z][A-Za-z0-9_]{0,99}$/.test(version)))
    throw new Error('赋值版本需为空或以字母开头的安全标识符；不是另一个 Python 属性。');
  return { owner, name: params.name, version, key: JSON.stringify([owner, params.name, version]) };
}

export function attributeOwnerLabel(owner: string[]): string {
  return owner.length ? `self.${owner.join('.')}` : 'self';
}

/** Build scheduling dependencies, not forward ports or a persistent Python object store. */
export function planAttributes(root: Graph): AttributePlan {
  const plan: AttributePlan = {
    bindings: new Map(),
    writers: new Map(),
    dependencies: new Map(),
    errors: new Map(),
    issues: [],
  };
  const entries: { node: ModelNode; path: string[]; inLoop: boolean }[] = [];
  let count = 0;
  const walk = (graph: Graph, scope: string[] = [], inLoop = false) => {
    if (scope.length > 12 || (count += graph.nodes.length) > 2000) {
      plan.issues.push('属性绑定分析超过子图深度或节点数量限制。');
      return;
    }
    for (const node of graph.nodes) {
      const path = [...scope, node.id];
      if (attributeTypes.has(node.type)) entries.push({ node, path, inLoop });
      if (node.subgraph) walk(node.subgraph, path, inLoop || ['ForEachTensor', 'ForLoop', 'WhileLoop'].includes(node.type));
    }
  };
  walk(root);
  const grouped = new Map<string, typeof entries>();
  for (const entry of entries) {
    const key = JSON.stringify(entry.path);
    try {
      const binding = attributeBinding(entry.node.params);
      let graph = root;
      for (const id of binding.owner) {
        const owner = graph.nodes.find((node) => node.id === id);
        if (!owner?.subgraph || ['ForEachTensor', 'ForLoop', 'WhileLoop'].includes(owner.type))
          throw new Error(
            `属性所属模块路径不存在，或目标不是模块子图：${binding.owner.join('/')}。`,
          );
        graph = owner.subgraph;
      }
      if (entry.inLoop && entry.node.type === 'SetAttribute') {
        plan.issues.push('暂不支持 For/ForEach/While 循环体中的模块属性写入。');
        throw new Error('暂不支持在 For/ForEach/While 循环体中写入模块属性；循环内可读取外部已写入的属性。');
      }
      plan.bindings.set(key, binding);
      if (entry.node.type === 'SetAttribute') {
        const writers = grouped.get(binding.key) ?? [];
        writers.push(entry);
        grouped.set(binding.key, writers);
      }
    } catch (error) {
      plan.errors.set(key, (error as Error).message);
    }
  }
  for (const [key, writers] of grouped) {
    if (writers.length === 1) plan.writers.set(key, writers[0].path);
    else
      for (const writer of writers)
        plan.errors.set(
          JSON.stringify(writer.path),
          '同一模块属性存在多个写入点，无法确定读取哪次赋值；请使用独立属性或明确的版本。',
        );
  }
  const propertyWrites = new Map<string, typeof entries>();
  for (const entry of entries.filter(e => e.node.type === 'SetAttribute')) {
    const binding = plan.bindings.get(JSON.stringify(entry.path));
    if (!binding) continue;
    const property = JSON.stringify([binding.owner, binding.name]);
    propertyWrites.set(property, [...(propertyWrites.get(property) ?? []), entry]);
  }
  for (const writes of propertyWrites.values()) {
    if (writes.length < 2) continue;
    for (const entry of writes) {
      const key = JSON.stringify(entry.path);
      if (!plan.bindings.get(key)!.version && !plan.errors.has(key))
        plan.errors.set(key, '同一属性有多次赋值，请为每个写入点设置不同的赋值版本；不能默认选择某次条件写入。');
    }
  }
  for (const reader of entries.filter((entry) => entry.node.type === 'GetAttribute')) {
    const readerKey = JSON.stringify(reader.path),
      binding = plan.bindings.get(readerKey);
    if (!binding) continue;
    const writers = grouped.get(binding.key) ?? [];
    const allWrites = propertyWrites.get(JSON.stringify([binding.owner, binding.name])) ?? [];
    if (!binding.version && allWrites.length > 1 && writers.length < 2) {
      plan.errors.set(readerKey, '该属性有多个赋值版本，请显式选择需要读取的版本；不自动推断条件的最终状态。');
      continue;
    }
    if (writers.length !== 1) {
      plan.errors.set(
        readerKey,
        writers.length
          ? '属性存在多个 SetAttribute 写入点，绑定有歧义。'
          : `属性 ${attributeOwnerLabel(binding.owner)}.${binding.name} 缺少 SetAttribute 写入节点。`,
      );
      continue;
    }
    const source = writers[0].path,
      target = reader.path;
    let common = 0;
    while (source[common] === target[common] && common < Math.min(source.length, target.length))
      common++;
    const scope = JSON.stringify(source.slice(0, common));
    const dependencies = plan.dependencies.get(scope) ?? [];
    if (
      !dependencies.some((edge) => edge.source === source[common] && edge.target === target[common])
    )
      dependencies.push({ source: source[common], target: target[common] });
    plan.dependencies.set(scope, dependencies);
  }
  return plan;
}
