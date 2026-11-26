import { registry } from '../modules';
import { scalarComparisonSymbol } from '../modules/forecastMath';
import { tensorIndexLabel } from '../modules/indexing';
import { ports } from './graph';
import { normalizeTensorSelection } from './indexingTypes';
import { ifRegionMembers, isCodeRegion } from './ifRegions';
import { isLayoutRegion, layoutRegionMembers, layoutRegionOutputOwners } from './layoutRegions';
import { supportsEdgePresentation } from './nodePresentation';
import type { GraphTensorTones, Tone } from './tensorGeometry';
import {
  shapeText,
  type Graph,
  type GraphResult,
  type ModelEdge,
  type ModelNode,
  type NodeResult,
} from './types';

export interface EdgeOpStep {
  id: string;
  type: string;
  label: string;
  badge: string;
  detail?: string;
  tone: 'blue' | 'sky' | 'green' | 'orange' | 'purple' | 'pink' | 'teal';
  status?: NodeResult['status'];
  shapeText?: string;
}

export interface ProjectedEdge extends ModelEdge {
  canonicalEdgeIds: string[];
  presentation: NonNullable<ModelEdge['presentation']>;
  edgeOps: EdgeOpStep[];
  sourceTone?: Tone;
  sourceLabel?: string;
}

export interface CanvasProjection {
  visibleNodes: ModelNode[];
  hiddenNodeIds: Set<string>;
  edges: ProjectedEdge[];
}

function shorten(text: string, limit: number) {
  const chars = Array.from(text);
  return chars.length <= limit ? text : chars.slice(0, limit - 1).join('') + '…';
}

function opStep(node: ModelNode, outputPort: string, result?: NodeResult): EdgeOpStep {
  const { id, type, label, params: p } = node;
  const args = (...values: unknown[]) =>
    values.map((v) => String(v ?? '?').replace(/\s*,\s*/g, ',')).join(',');
  let badge = type;
  let detail: string | undefined;
  let tone: EdgeOpStep['tone'] = 'orange';
  switch (type) {
    case 'Cos':
    case 'Sin':
      badge = `${type.toLowerCase()}()`;
      break;
    case 'Function':
      detail = String(p.expression ?? '').trim() || label.trim() || type;
      badge = shorten(detail, 24);
      break;
    case 'TensorCompareScalar':
      badge = `${scalarComparisonSymbol(p.operator)} ${String(p.value ?? '').trim()}`;
      tone = 'pink';
      break;
    case 'OnesLike':
    case 'ZerosLike':
      badge = type === 'OnesLike' ? 'ones_like' : 'zeros_like';
      tone = 'teal';
      break;
    case 'TensorIndex':
    case 'Slice':
      badge = tensorIndexLabel(normalizeTensorSelection(type, p).params).replace(/, \.\.\.\]$/, ']');
      tone = 'sky';
      break;
    case 'Linear':
      badge = `Linear ${p.in_features}→${p.out_features}`;
      tone = 'blue';
      break;
    case 'LayerNorm':
      badge = `LN(${args(p.features)})`;
      tone = 'green';
      break;
    case 'Reshape':
    case 'View':
      badge = `${type}(${args(p.shape)})`;
      tone = 'sky';
      break;
    case 'Permute':
      badge = `permute(${args(p.dims)})`;
      tone = 'sky';
      break;
    case 'Repeat':
      badge = `repeat(${args(p.repeats)})`;
      tone = 'sky';
      break;
    case 'Unsqueeze':
    case 'Squeeze':
      badge = `${type.toLowerCase()}(${args(p.dim)})`;
      tone = 'sky';
      break;
    case 'Flatten':
      badge = `flatten(${args(p.start_dim, p.end_dim)})`;
      tone = 'sky';
      break;
    case 'Unflatten':
      badge = `unflatten(${args(p.dim)},(${args(p.sizes)}))`;
      tone = 'sky';
      break;
    case 'Softmax':
      badge = `softmax(${args(p.dim)})`;
      break;
    case 'SemanticOp':
      badge = shorten(label, 24);
      tone = 'purple';
      break;
    case 'Contiguous':
    case 'Detach':
    case 'Identity':
      tone = 'teal';
      break;
  }
  const value = result?.outputs[outputPort];
  return {
    id,
    type,
    label,
    badge,
    ...(detail && { detail }),
    tone,
    status: result?.status,
    shapeText: value && shapeText(value),
  };
}

// A null entry denotes fan-in or fan-out. Only unique incident edges can be in a chain.
function uniqueEdges(edges: ModelEdge[], endpoint: 'source' | 'target') {
  const unique = new Map<string, ModelEdge | null>();
  for (const edge of edges) unique.set(edge[endpoint], unique.has(edge[endpoint]) ? null : edge);
  return unique;
}

function collapsibleNodes(graph: Graph) {
  const candidates = new Map<string, ModelEdge>();
  if (!graph.nodes.some((n) => n.presentation === 'edge')) return candidates;
  const incoming = uniqueEdges(graph.edges, 'target');
  const outgoing = uniqueEdges(graph.edges, 'source');
  const referenced = new Set(graph.nodes.map((n) => n.valueSource));
  const regionOutputs = layoutRegionOutputOwners(graph);
  const regions = new Map<string, string>();
  for (const region of graph.nodes) {
    const members = isCodeRegion(region)
      ? ifRegionMembers(region)
      : isLayoutRegion(region)
        ? layoutRegionMembers(region)
        : [];
    // A node can belong to both a code scope and a layout region.
    for (const id of members) regions.set(id, (regions.get(id) ?? '') + '/' + region.id);
  }
  for (const n of graph.nodes) {
    if (
      n.presentation !== 'edge' ||
      !supportsEdgePresentation(n) ||
      n.visualMedia ||
      n.tensorVisual?.mode === 'block' ||
      n.subgraph ||
      n.valueSource ||
      referenced.has(n.id) ||
      regionOutputs.has(n.id)
    )
      continue;
    const before = incoming.get(n.id),
      after = outgoing.get(n.id);
    if (
      !before ||
      !after ||
      before.source === n.id ||
      after.target === n.id ||
      before.presentation === 'anchor' ||
      after.presentation === 'anchor' ||
      regions.get(before.source) !== regions.get(n.id) ||
      regions.get(after.target) !== regions.get(n.id)
    )
      continue;
    // Dynamic operators may have additional unconnected ports in an editable draft.
    const portNode = n.type === 'Slice' ? { ...n, ...normalizeTensorSelection(n.type, n.params) } : n;
    const declared = ports(portNode, registry);
    if (
      declared.inputs.length === 1 &&
      declared.outputs.length === 1 &&
      declared.inputs[0].id === before.targetPort &&
      declared.outputs[0].id === after.sourcePort
    )
      candidates.set(n.id, after);
  }
  return candidates;
}

/** Read-only presentation of one graph level. Inference, editing and serialization use the canonical graph. */
export function projectCanvasGraph(
  graph: Graph,
  result: GraphResult | undefined,
  graphTones: GraphTensorTones,
): CanvasProjection {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const candidates = collapsibleNodes(graph);
  const hiddenNodeIds = new Set<string>();
  const pipelines = new Map<string, ProjectedEdge>();
  for (const first of graph.edges) {
    if (candidates.has(first.source) || !candidates.has(first.target)) continue;
    const steps: EdgeOpStep[] = [];
    const ids = [first.id];
    const visited = new Set([first.source]);
    let last = first;
    let slide = first.presentation === 'slide';
    while (candidates.has(last.target) && !visited.has(last.target)) {
      const node = nodes.get(last.target)!;
      visited.add(node.id);
      last = candidates.get(node.id)!;
      steps.push(opStep(node, last.sourcePort, result?.nodes[node.id]));
      ids.push(last.id);
      slide ||= last.presentation === 'slide';
    }
    // A cycle has no distinct visible endpoints. Leave every part available for inspection.
    if (visited.has(last.target) || !nodes.has(last.target) || !nodes.has(first.source)) continue;
    for (const step of steps) hiddenNodeIds.add(step.id);
    pipelines.set(last.id, {
      ...last,
      source: first.source,
      sourcePort: first.sourcePort,
      presentation: slide ? 'slide' : 'default',
      annotation: last.annotation ?? first.annotation,
      canonicalEdgeIds: ids,
      edgeOps: steps,
    });
  }

  const edges: ProjectedEdge[] = [];
  for (const edge of graph.edges) {
    const pipeline = pipelines.get(edge.id);
    if (pipeline) {
      edges.push(pipeline);
      continue;
    }
    if (hiddenNodeIds.has(edge.target)) continue;
    const projected: ProjectedEdge = {
      ...edge,
      presentation: edge.presentation ?? 'default',
      canonicalEdgeIds: [edge.id],
      edgeOps: [],
    };
    if (edge.presentation === 'anchor') {
      const label = nodes.get(edge.source)?.label ?? edge.source;
      const annotation = edge.annotation?.trim();
      projected.sourceLabel =
        annotation && Array.from(annotation).length <= 20
          ? annotation
          : shorten(label.split(/[·(（]/, 1)[0].trim() || label.trim() || edge.source, 20);
      projected.sourceTone = graphTones.get(edge.source)?.tones[0] ?? 'blue';
    }
    edges.push(projected);
  }
  return {
    visibleNodes: graph.nodes.filter((n) => !hiddenNodeIds.has(n.id)),
    hiddenNodeIds,
    edges,
  };
}
