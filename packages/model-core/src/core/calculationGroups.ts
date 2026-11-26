import { shortText } from './limits';
import type { Graph, ModelEdge, ModelNode, Registry } from './types';
import { clone, uid } from './types';
import { orderGraph, ports } from './graph';
import { isCalculation, isGlueCalculation } from '../modules/libraryCategories';
export { isGlueCalculation } from '../modules/libraryCategories';
import { ifRegionMembers, isCodeRegion } from './ifRegions';
import { isLayoutRegion, layoutRegionOutputs } from './layoutRegions';

const boundary = (node: ModelNode) => ['Inlet', 'Outlet'].includes(node.type);
const key = (node: string, port: string) => JSON.stringify([node, port]);

/** A visual contraction must not introduce a cycle through an outside branch. */
export function canMergeCalculations(
  graph: Graph,
  a: string,
  b: string,
  registry: Registry,
): boolean {
  return canMergeCalculationSet(graph, [a, b], registry);
}

export function canMergeCalculationSet(graph: Graph, ids: string[], registry: Registry): boolean {
  const members = new Set(ids);
  if (members.size < 2 || members.size !== ids.length) return false;
  const nodes = graph.nodes.filter((n) => members.has(n.id));
  if (nodes.length !== ids.length || !nodes.every((n) => isCalculation(registry.get(n.type))))
    return false;
  for (const region of graph.nodes.filter((n) => isCodeRegion(n) || isLayoutRegion(n))) {
    const count = ifRegionMembers(region).filter((id) => members.has(id)).length;
    if (count && count !== members.size) return false;
  }

  const neighbors = new Map(ids.map((id) => [id, [] as string[]]));
  const contractedEdges: ModelEdge[] = [];
  for (const edge of graph.edges) {
    const source = members.has(edge.source),
      target = members.has(edge.target);
    if (source && target) {
      neighbors.get(edge.source)!.push(edge.target);
      neighbors.get(edge.target)!.push(edge.source);
    } else
      contractedEdges.push({
        ...edge,
        source: source ? ids[0] : edge.source,
        target: target ? ids[0] : edge.target,
      });
  }
  const connected = new Set([ids[0]]);
  for (const id of connected) for (const neighbor of neighbors.get(id)!) connected.add(neighbor);
  if (connected.size !== members.size) return false;
  try {
    orderGraph(graph);
    orderGraph({
      nodes: graph.nodes.filter((n) => !members.has(n.id) || n.id === ids[0]),
      edges: contractedEdges,
    });
    return true;
  } catch {
    return false;
  }
}

export interface NodeBounds {
  id: string;
  position: { x: number; y: number };
  width: number;
  height: number;
}

/** Flow-space distance (independent of zoom); only a deliberate single-node drag merges. */
export function nearbyCalculation(
  graph: Graph,
  moved: NodeBounds,
  others: NodeBounds[],
  registry: Registry,
): string | undefined {
  const adjacent = new Set(
    graph.edges.flatMap((edge) =>
      edge.source === moved.id ? [edge.target] : edge.target === moved.id ? [edge.source] : [],
    ),
  );
  return others
    .filter((n) => adjacent.has(n.id) && canMergeCalculations(graph, moved.id, n.id, registry))
    .map((n) => ({
      id: n.id,
      distance: Math.hypot(
        Math.max(
          0,
          n.position.x - moved.position.x - moved.width,
          moved.position.x - n.position.x - n.width,
        ),
        Math.max(
          0,
          n.position.y - moved.position.y - moved.height,
          moved.position.y - n.position.y - n.height,
        ),
      ),
    }))
    .filter((n) => n.distance <= 24)
    .sort((a, b) => a.distance - b.distance || a.id.localeCompare(b.id))[0]?.id;
}

/** Expand the exact saved child graph, including fan-out and multiple inputs. */
export function splitCalculationGroup(graph: Graph, id: string): string[] {
  return expandGroup(graph, id).members;
}

function expandGroup(graph: Graph, id: string) {
  const group = graph.nodes.find((n) => n.id === id);
  if (group?.type !== 'CalculationGroup' || !group.subgraph) throw new Error('请选择计算组合。');
  const child = group.subgraph;
  const operations = child.nodes.filter((n) => !boundary(n));
  const childNodes = new Map(child.nodes.map((n) => [n.id, n]));
  const occupied = new Set(graph.nodes.filter((n) => n !== group).map((n) => n.id));
  const ids = new Map(
    operations.map((n) => [n.id, !group.calculationOrigin || occupied.has(n.id) ? uid() : n.id]),
  );
  const origin = group.calculationOrigin ?? {
    x: Math.min(...operations.map((n) => n.position.x), 0),
    y: Math.min(...operations.map((n) => n.position.y), 0),
  };
  const translate = (p: ModelNode['position']) => ({
    x: p.x + (group.position.x - origin.x),
    y: p.y + (group.position.y - origin.y),
  });
  const nodes = operations.map((n) => ({
    ...clone(n),
    id: ids.get(n.id)!,
    position: translate(n.position),
    ...(n.valueSource ? { valueSource: ids.get(n.valueSource) } : {}),
  }));
  for (const node of nodes) {
    if (isCodeRegion(node) || isLayoutRegion(node))
      node.params.members = ifRegionMembers(node)
        .map((id) => ids.get(id)!)
        .join(',');
    if (isLayoutRegion(node) && node.params.outputs !== undefined)
      node.params.outputs = layoutRegionOutputs(node)
        .map((id) => ids.get(id)!)
        .join(',');
  }
  const incoming = new Map(
    graph.edges.filter((e) => e.target === id).map((e) => [e.targetPort, e]),
  );
  function inletEdge(nodeId: string) {
    const node = childNodes.get(nodeId);
    if (node?.type !== 'Inlet') return;
    const owner = childNodes.get(node.valueSource ?? node.id);
    return incoming.get(String(owner?.params.port));
  }
  function upstream(nodeId: string, port: string) {
    const mapped = ids.get(nodeId);
    if (mapped) return { source: mapped, sourcePort: port };
    const edge = inletEdge(nodeId);
    if (edge) return { source: edge.source, sourcePort: edge.sourcePort };
  }
  const edges = graph.edges.filter((e) => e.source !== id && e.target !== id);
  const edgeIds = new Set(graph.edges.filter((e) => e.target !== id).map((e) => e.id));
  for (const edge of child.edges) {
    const target = ids.get(edge.target),
      source = upstream(edge.source, edge.sourcePort);
    if (!target || !source) continue;
    const outerEdge = inletEdge(edge.source);
    const original = outerEdge ?? edge;
    const edgeId = edgeIds.has(original.id) ? uid() : original.id;
    edgeIds.add(edgeId);
    edges.push({
      ...original,
      id: edgeId,
      ...source,
      target,
      targetPort: edge.targetPort,
      ...(!outerEdge && edge.waypoints ? { waypoints: edge.waypoints.map(translate) } : {}),
    });
  }
  for (const edge of graph.edges.filter((e) => e.source === id)) {
    const outlet = child.nodes.find(
      (n) => !n.valueSource && n.type === 'Outlet' && n.params.port === edge.sourcePort,
    );
    const inner = child.edges.find((e) => e.target === outlet?.id);
    const source = inner && upstream(inner.source, inner.sourcePort);
    if (source) edges.push({ ...edge, ...source });
  }
  const outletIds = new Set(child.nodes.filter((n) => n.type === 'Outlet').map((n) => n.id));
  const marked =
    group.calculationRegionOutputs ??
    child.edges.filter((e) => outletIds.has(e.target)).map((e) => e.source);
  const regionOutputs = [
    ...new Set(marked.flatMap((key) => (ids.has(key) ? [ids.get(key)!] : []))),
  ];
  graph.nodes = graph.nodes.flatMap((n) => (n.id === id ? nodes : [n]));
  for (const region of graph.nodes.filter((n) => isCodeRegion(n) || isLayoutRegion(n))) {
    const members = ifRegionMembers(region);
    if (!members.includes(id)) continue;
    region.params.members = members
      .flatMap((member) => (member === id ? nodes.map((n) => n.id) : [member]))
      .join(',');
    if (isLayoutRegion(region) && region.params.outputs !== undefined) {
      region.params.outputs = layoutRegionOutputs(region)
        .flatMap((member) => (member === id ? regionOutputs : [member]))
        .join(',');
    }
  }
  graph.edges = edges;
  return { members: nodes.map((n) => n.id), regionOutputs };
}

/** Group graph operations, never replace them with text or discard branch connections. */
export function mergeCalculations(
  graph: Graph,
  a: string,
  b: string,
  registry: Registry,
  position?: ModelNode['position'],
): string {
  return collapse(
    graph,
    [a, b],
    registry,
    { position: position ?? graph.nodes.find((n) => n.id === b)?.position },
    true,
  );
}

export interface CalculationGroupOptions {
  id?: string;
  label?: string;
  summary?: string;
  formula?: string;
  position?: ModelNode['position'];
}

export function mergeCalculationSet(
  graph: Graph,
  ids: string[],
  registry: Registry,
  options: CalculationGroupOptions = {},
): string {
  return collapse(graph, ids, registry, options, false);
}

function collapse(
  graph: Graph,
  ids: string[],
  registry: Registry,
  options: CalculationGroupOptions,
  legacyLayout: boolean,
): string {
  if (!canMergeCalculationSet(graph, ids, registry))
    throw new Error('只能合并同一范围和分区内相连的计算节点；合并不能跨越外部计算形成环路。');
  const id = options.id ?? uid();
  if (graph.nodes.some((n) => n.id === id)) throw Error(`组合 ID 已存在：${id}`);
  const params: ModelNode['params'] = {};
  for (const key of ['summary', 'formula'] as const) {
    const value = options[key];
    if (value !== undefined) {
      if (!shortText(value, 240)) throw Error(`${key} 最多 240 个字符。`);
      params[key] = value;
    }
  }
  const selected = new Set(ids);
  const original = graph.nodes.filter((n) => selected.has(n.id));
  const anchor = options.position ?? {
    x: original.reduce((sum, n) => sum + n.position.x, 0) / ids.length,
    y: original.reduce((sum, n) => sum + n.position.y, 0) / ids.length,
  };
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const members = orderGraph(graph)
    .filter((id) => selected.has(id))
    .map((id) => byId.get(id)!);
  const internal = graph.edges.filter((e) => selected.has(e.source) && selected.has(e.target));
  const internalInputs = new Set(internal.map((e) => key(e.target, e.targetPort)));
  const externalOutputs = new Set(
    graph.edges
      .filter((e) => selected.has(e.source) && !selected.has(e.target))
      .map((e) => key(e.source, e.sourcePort)),
  );
  const consumedOutputs = new Set(internal.map((e) => key(e.source, e.sourcePort)));
  const subgraph: Graph = {
    nodes: clone(members),
    edges: clone(internal),
  };
  let regionOutputs = graph.nodes
    .filter(isLayoutRegion)
    .flatMap(layoutRegionOutputs)
    .filter((id) => selected.has(id));
  const inputs = new Map<string, string>(),
    outputs = new Map<string, string>();
  for (const node of members) {
    for (const port of ports(node, registry).inputs) {
      if (internalInputs.has(key(node.id, port.id))) continue;
      const name = `in${inputs.size + 1}`,
        id = uid();
      inputs.set(key(node.id, port.id), name);
      subgraph.nodes.push({
        id,
        type: 'Inlet',
        definitionVersion: 1,
        label: Array.from(`${node.label} · ${port.label}`).slice(0, 200).join(''),
        params: { port: name },
        position: { x: 0, y: 0 },
      });
      subgraph.edges.push({
        id: uid(),
        source: id,
        sourcePort: 'out',
        target: node.id,
        targetPort: port.id,
      });
    }
    for (const port of ports(node, registry).outputs) {
      const outputKey = key(node.id, port.id);
      if (
        consumedOutputs.has(outputKey) &&
        !externalOutputs.has(outputKey) &&
        (legacyLayout || !regionOutputs.includes(node.id))
      )
        continue;
      const name = `out${outputs.size + 1}`,
        id = uid();
      outputs.set(key(node.id, port.id), name);
      subgraph.nodes.push({
        id,
        type: 'Outlet',
        definitionVersion: 1,
        label: Array.from(`${node.label} · ${port.label}`).slice(0, 200).join(''),
        params: { port: name },
        position: { x: 0, y: 0 },
      });
      subgraph.edges.push({
        id: uid(),
        source: node.id,
        sourcePort: port.id,
        target: id,
        targetPort: 'in',
      });
    }
  }
  // Flatten inside the new scope. Keeping the boundary in place also preserves
  // identity routes (Inlet → Outlet) and unused ports in authored child graphs.
  const nested = subgraph.nodes.filter((n) => n.type === 'CalculationGroup').map((n) => n.id);
  while (nested.length) {
    const groupId = nested.pop()!;
    const expanded = expandGroup(subgraph, groupId);
    regionOutputs = regionOutputs.flatMap((id) => (id === groupId ? expanded.regionOutputs : [id]));
    const restored = new Set(expanded.members);
    nested.push(
      ...subgraph.nodes
        .filter((n) => restored.has(n.id) && n.type === 'CalculationGroup')
        .map((n) => n.id),
    );
  }
  if (legacyLayout) {
    let step = 0;
    const nodes = new Map(subgraph.nodes.map((n) => [n.id, n]));
    for (const id of orderGraph(subgraph)) {
      const node = nodes.get(id)!;
      if (!boundary(node)) node.position = { x: step++ * 280, y: 100 };
    }
  }
  const inner = subgraph.nodes.filter((n) => !boundary(n));
  const left = inner.length ? Math.min(...inner.map((n) => n.position.x)) : anchor.x;
  const right = inner.length ? Math.max(...inner.map((n) => n.position.x)) : anchor.x;
  const top = legacyLayout
    ? 120
    : inner.length
      ? Math.min(...inner.map((n) => n.position.y))
      : anchor.y;
  let inputRow = 0,
    outputRow = 0;
  for (const node of subgraph.nodes) {
    if (node.type === 'Inlet') node.position = { x: left - 240, y: top + 120 * inputRow++ };
    if (node.type === 'Outlet') node.position = { x: right + 280, y: top + 120 * outputRow++ };
  }
  graph.edges = graph.edges
    .filter((e) => !(selected.has(e.source) && selected.has(e.target)))
    .map((e) => {
      if (!selected.has(e.source) && !selected.has(e.target)) return e;
      return {
        ...e,
        source: selected.has(e.source) ? id : e.source,
        sourcePort: selected.has(e.source)
          ? outputs.get(key(e.source, e.sourcePort))!
          : e.sourcePort,
        target: selected.has(e.target) ? id : e.target,
        targetPort: selected.has(e.target)
          ? inputs.get(key(e.target, e.targetPort))!
          : e.targetPort,
      };
    });
  graph.nodes = [
    ...graph.nodes.filter((n) => !selected.has(n.id)),
    {
      id,
      type: 'CalculationGroup',
      definitionVersion: 1,
      label: options.label || options.summary || '计算组合',
      params,
      ...(!legacyLayout ? { calculationOrigin: { ...anchor } } : {}),
      ...(regionOutputs.length ? { calculationRegionOutputs: regionOutputs } : {}),
      position: { ...anchor },
      subgraph,
      customized: true,
    },
  ];
  for (const region of graph.nodes.filter((n) => isCodeRegion(n) || isLayoutRegion(n))) {
    const contained = ifRegionMembers(region);
    if (contained.some((member) => selected.has(member)))
      region.params.members = [
        ...new Set(contained.map((member) => (selected.has(member) ? id : member))),
      ].join(',');
    if (isLayoutRegion(region) && region.params.outputs !== undefined) {
      const outputs = layoutRegionOutputs(region);
      if (outputs.some((member) => selected.has(member)))
        region.params.outputs = [
          ...new Set(outputs.map((member) => (selected.has(member) ? id : member))),
        ].join(',');
    }
  }
  return id;
}

function describeGlue(steps: ModelNode[], edges: ModelEdge[]) {
  const types = steps.map((n) => n.type);
  const terminal = steps[steps.length - 1];
  let summary: string;
  const whereBranches = terminal.type === 'Where'
    ? edges.filter(e => e.target === terminal.id && ['x', 'y'].includes(e.targetPort)).map(e => e.source)
    : [];
  const fillTypes = steps.filter(n => whereBranches.includes(n.id)).map(n => n.type);
  if (fillTypes.some(type => ['ZerosLike', 'Zeros', 'NewZeros'].includes(type)))
    summary = '按掩码选择数值并清零对应分量';
  else if (fillTypes.includes('OnesLike'))
    summary = '按掩码选择数值并将对应分量置一';
  else if (types.includes('Clamp')) summary = '整理数值并限制取值范围';
  else if (
    types.some((t) => ['DictionaryGet', 'TensorIndex', 'Slice', 'TensorSequenceGet'].includes(t))
  )
    summary = types.some((t) =>
      ['Reshape', 'ReshapeLike', 'View', 'Unsqueeze', 'Squeeze', 'Flatten', 'Permute'].includes(t),
    )
      ? '提取所需分量并调整张量形状'
      : '读取并整理所需分量';
  else if (
    types.every((t) =>
      ['Add', 'Subtract', 'Mul', 'Divide', 'Maximum', 'Function', 'Sin', 'Cos', 'Exp', 'Log', 'Sqrt'].includes(
        t,
      ),
    )
  )
    summary = '逐元素计算' + terminal.label;
  else summary = `${steps[0].label}，再${terminal.label}`;
  const sequence = types.length <= 5 ? types : [...types.slice(0, 2), '…', ...types.slice(-2)];
  return { summary: Array.from(summary).slice(0, 240).join(''), formula: sequence.join(' → ') };
}

export function autoSimplifyCalculations(
  graph: Graph,
  registry: Registry,
  options: { recursive?: boolean; minSize?: number } = {},
) {
  const minSize = options.minSize ?? 2;
  if (!Number.isSafeInteger(minSize) || minSize < 2) throw Error('最少合并步数必须是 ≥ 2 的整数。');
  const mergedGroups: string[] = [];
  let removedNodes = 0;

  function simplify(g: Graph) {
    if (options.recursive) {
      for (const node of g.nodes) {
        if (
          node.subgraph &&
          ['Block', 'Submodule', 'ForLoop', 'ForEachTensor', 'ForEach', 'WhileLoop'].includes(
            node.type,
          )
        ) {
          const before = mergedGroups.length;
          simplify(node.subgraph);
          if (before !== mergedGroups.length) node.customized = true;
        }
      }
    }
    const order = orderGraph(g);
    const byId = new Map(g.nodes.map((n) => [n.id, n]));
    const available = new Set(
      g.nodes.filter((n) => isGlueCalculation(n, registry)).map((n) => n.id),
    );
    if (available.size < minSize) return;
    const producers = new Map(order.map((id) => [id, new Set<string>()]));
    const consumers = new Map(order.map((id) => [id, new Set<string>()]));
    for (const e of g.edges) {
      producers.get(e.target)!.add(e.source);
      consumers.get(e.source)!.add(e.target);
    }
    const code = new Map<string, string>(),
      layout = new Map<string, string>(),
      marked = new Set<string>();
    for (const region of g.nodes) {
      if (!isCodeRegion(region) && !isLayoutRegion(region)) continue;
      const scope = isCodeRegion(region) ? code : layout;
      for (const id of ifRegionMembers(region)) scope.set(id, region.id);
      if (isLayoutRegion(region)) for (const id of layoutRegionOutputs(region)) marked.add(id);
    }

    // Grow upstream from a single exit. A shared producer can join only after its last
    // consumer joins. Use original adjacency throughout, so earlier folds cannot hide fan-out.
    for (let i = order.length - 1; i >= 0; i--) {
      const exit = order[i];
      if (!available.has(exit) || consumers.get(exit)!.size > 1) continue;
      const cluster = new Set([exit]);
      for (const target of cluster) {
        for (const source of producers.get(target)!) {
          if (
            available.has(source) &&
            !marked.has(source) &&
            code.get(source) === code.get(exit) &&
            layout.get(source) === layout.get(exit) &&
            [...consumers.get(source)!].every((id) => cluster.has(id))
          )
            cluster.add(source);
        }
      }
      if (cluster.size < minSize) continue;
      const ids = order.filter((id) => cluster.has(id));
      if (!canMergeCalculationSet(g, ids, registry)) continue;
      const description = describeGlue(
        ids.map((id) => byId.get(id)!),
        g.edges,
      );
      mergedGroups.push(mergeCalculationSet(g, ids, registry, description));
      removedNodes += ids.length - 1;
      for (const id of ids) available.delete(id);
    }
  }
  simplify(graph);
  return { mergedGroups, removedNodes };
}
