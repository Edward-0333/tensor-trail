import { ports } from './graph';
import { supportsTensorVisual } from './tensorVisual';
import { isValueNode } from './valueLinks';
import {
  isTensorSpec,
  type Dim,
  type Graph,
  type GraphResult,
  type ModelNode,
  type NodeResult,
  type NodeTensorVisual,
  type TensorSpec,
  type TensorVisualTone,
} from './types';
import { registry } from '../modules';

export type Tone = Exclude<TensorVisualTone, 'auto'> | `#${string}`;
function explicitTone(visual?: Partial<NodeTensorVisual>): Tone | undefined {
  if (visual?.color) return visual.color as `#${string}`;
  return visual?.tone === 'auto' ? undefined : visual?.tone;
}
export interface TensorToneIdentity {
  tones: Tone[];
  join?: TensorJoin;
}
interface NodeTones extends TensorToneIdentity {
  inputs: Map<string, TensorToneIdentity>;
}
export type GraphTensorTones = Map<string, NodeTones>;

const palette: Tone[] = ['blue', 'violet', 'teal', 'amber', 'rose'];
const neutral: Tone[] = ['slate'];
const sources = new Set(['Input', 'Inlet', 'LearnedParameter', 'BufferSource']);
const joins = new Set(['Concat', 'Stack']);
const blends = new Set(['Add', 'Mul', 'Subtract']);
const passthroughs = new Set([
  'TensorVariable',
  'ValueVariable',
  'Output',
  'Outlet',
  'Contiguous',
  'Detach',
]);
const isBoundary = (type: string) => type === 'Output' || type === 'Outlet';

export function resolveGraphTensorTones(graph: Graph, result?: GraphResult): GraphTensorTones {
  const resolved: GraphTensorTones = new Map();
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const incoming = new Map<string, Map<string, string>>();
  for (const edge of graph.edges) {
    const inputs = incoming.get(edge.target) ?? new Map<string, string>();
    inputs.set(edge.targetPort, edge.source);
    incoming.set(edge.target, inputs);
  }
  let sourceIndex = 0;
  for (const id of result?.order ?? graph.nodes.map((n) => n.id)) {
    const node = nodes.get(id);
    if (!node) continue;
    const inferred = result?.nodes[id];
    const inputs = new Map<string, TensorToneIdentity>();
    const wired = incoming.get(id);
    const declared = ports(node, registry).inputs;
    const tensorInputs: [string, TensorSpec][] = [];
    for (const { id: port } of declared) {
      const value = inferred?.inputs[port];
      if (value && isTensorSpec(value)) tensorInputs.push([port, value]);
      else if (result) continue;
      const source = wired?.get(port);
      if (source) {
        const identity = resolved.get(source);
        inputs.set(port, { tones: identity?.tones ?? neutral, join: identity?.join });
      }
    }
    const linked = node.valueSource ? resolved.get(node.valueSource) : undefined;
    const upstream = [...inputs.values()];
    const override = explicitTone(node.tensorVisual);
    let tones = upstream[0]?.tones ?? neutral;
    if (node.valueSource) tones = linked?.tones ?? neutral;
    else if (sources.has(node.type)) tones = [palette[sourceIndex++ % palette.length]];
    else if (blends.has(node.type) || joins.has(node.type)) {
      const distinct = [...new Set(upstream.flatMap((input) => input.tones))];
      if (distinct.length) tones = blends.has(node.type) ? distinct.slice(0, 2) : distinct;
    }
    const colors: NodeTones = { inputs, tones: override ? [override] : tones };
    const output =
      inferred?.outputs.out ?? (isBoundary(node.type) ? inferred?.inputs.in : undefined);
    if (output && isTensorSpec(output)) {
      if (
        joins.has(node.type) &&
        tensorInputs.length === declared.length &&
        tensorInputs.length > 0
      ) {
        colors.join = joinGeometry(node, tensorInputs, output, colors, override);
      } else if (!override && (node.valueSource || passthroughs.has(node.type))) {
        const join = node.valueSource ? linked?.join : inputs.get('in')?.join;
        const source = node.valueSource
          ? result?.nodes[node.valueSource]?.outputs.out
          : inferred?.inputs.in;
        if (
          join &&
          source &&
          isTensorSpec(source) &&
          source.shape.length === output.shape.length &&
          source.shape.every((dim, index) => dim === output.shape[index])
        )
          colors.join = join;
      }
    }
    resolved.set(id, colors);
  }
  return resolved;
}

interface BoxAxis {
  dimIndex?: number;
  size: Dim;
  axis: string;
}
export interface TensorBox3D {
  ndim: number;
  leadingChips: { label: string; size: Dim }[];
  outer?: BoxAxis & { count: number; elided: boolean };
  depth: BoxAxis & { sliceCount: number; elided: boolean; solid: boolean };
  height: BoxAxis;
  width: BoxAxis;
}

type ShapeOptions = Pick<NodeTensorVisual, 'depthStyle' | 'axisLabels'> & { join?: TensorJoin };
export function mapShapeToBox3D(spec: TensorSpec, options: ShapeOptions = {}): TensorBox3D {
  const ndim = spec.shape.length;
  const axisName = (i: number) => options.axisLabels?.[i]?.trim() || spec.axes[i] || `axis${i}`;
  const axis = (offset: number): BoxAxis => {
    const dimIndex = ndim - offset;
    return dimIndex < 0
      ? { size: 1, axis: '' }
      : { dimIndex, size: spec.shape[dimIndex], axis: axisName(dimIndex) };
  };
  const depth = axis(3);
  const deep = typeof depth.size === 'string' || depth.size > 4;
  const solid =
    ndim >= 3 &&
    (options.depthStyle === 'solid' ||
      (options.depthStyle !== 'slices' && deep && options.join?.direction !== 'depth'));
  const outer = axis(4);
  const elideOuter = typeof outer.size === 'string' || outer.size > 3;
  return {
    ndim,
    leadingChips: spec.shape.slice(0, -4).map((size, i) => ({ label: axisName(i), size })),
    outer:
      ndim >= 4
        ? { ...outer, elided: elideOuter, count: elideOuter ? 2 : Math.max(1, Number(outer.size)) }
        : undefined,
    depth: {
      ...depth,
      solid,
      sliceCount: solid ? 1 : deep ? 3 : Math.max(1, Number(depth.size)),
      elided: !solid && deep,
    },
    height: axis(2),
    width: axis(1),
  };
}

export interface ScaledDims {
  wPx: number;
  hPx: number;
  depthPx: number;
  sliceGapPx: number;
}
const maxVisualDim = 2048;
const boundedDim = (size: number) => Math.min(maxVisualDim, Math.max(1, size));

export function resolveGraphReferenceMaxDim(graph: Graph, result?: GraphResult): number;
export function resolveGraphReferenceMaxDim(): undefined;
export function resolveGraphReferenceMaxDim(graph?: Graph, result?: GraphResult) {
  if (!graph) return;
  let max = 4;
  for (const { id } of graph.nodes) {
    const node = result?.nodes[id];
    if (!node) continue;
    for (const value of [...Object.values(node.inputs), ...Object.values(node.outputs)]) {
      if (!isTensorSpec(value)) continue;
      for (const dim of value.shape.slice(-3)) {
        if (typeof dim === 'number') max = Math.max(max, boundedDim(dim));
      }
    }
  }
  return max;
}

type DimensionOptions = Pick<NodeTensorVisual, 'boxWidth' | 'boxHeight' | 'boxDepth'>;
export function computeScaledDims(
  boxes: TensorBox3D[],
  referenceMaxDim?: number,
  options: DimensionOptions = {},
): ScaledDims[] {
  let max = Math.max(4, referenceMaxDim ?? 4);
  for (const box of boxes) {
    for (const { size } of [box.width, box.height, box.depth]) {
      if (typeof size === 'number') max = Math.max(max, boundedDim(size));
    }
  }
  const scale = (size: Dim) =>
    typeof size === 'number' ? Math.log2(1 + (3 * boundedDim(size)) / max) / 2 : 0.5;
  return boxes.map(({ width, height, depth, ndim }) => {
    let depthPx = ndim === 0 ? 7 : ndim < 3 ? 4 : 6;
    let sliceGapPx = 8 + 3 * scale(depth.size);
    if (depth.solid) {
      // Keep deep prisms comparable even when Z itself is the largest axis on each graph.
      const fraction =
        typeof depth.size === 'number'
          ? Math.log2(boundedDim(depth.size)) / Math.log2(maxVisualDim)
          : 0.5;
      depthPx = 12 + 26 * fraction;
    }
    if (options.boxDepth !== undefined) {
      if (depth.sliceCount === 1) depthPx = options.boxDepth;
      else sliceGapPx = Math.max(4, (options.boxDepth - depthPx) / (depth.sliceCount - 1));
    }
    return {
      wPx: options.boxWidth ?? (ndim === 0 ? 16 : 12 + 86 * scale(width.size)),
      hPx: options.boxHeight ?? (ndim === 0 ? 16 : ndim === 1 ? 8 : 8 + 54 * scale(height.size)),
      depthPx,
      sliceGapPx,
    };
  });
}

type Direction = 'width' | 'height' | 'depth' | 'outer' | 'leading';
const directionFor = (axis: number, rank: number): Direction =>
  (['width', 'height', 'depth', 'outer'] as const)[rank - axis - 1] ?? 'leading';
const normalizeAxis = (dim: number, rank: number) => (dim < 0 ? rank + dim : dim);
interface Segment {
  tones: Tone[];
  start: number;
  end: number;
  size: Dim;
  port: string;
}
export interface TensorJoin {
  axis: number;
  direction: Direction;
  segments: Segment[];
}
export interface GeometryBox extends TensorBox3D, ScaledDims {
  port: string;
  dtype: string;
  tones: Tone[];
}
export interface TensorGeometryDiagram {
  mode: 'block';
  observeInput?: boolean;
  inputs: [];
  output: GeometryBox;
  badge: string;
  join?: TensorJoin;
  width: number;
  customColor?: string;
  caption?: string;
}

export function canvasTensorVisual(
  node: ModelNode,
  globalTensorBlocks: boolean,
): NodeTensorVisual | undefined {
  if (!supportsTensorVisual(node)) return;
  if (node.tensorVisual) return node.tensorVisual;
  if (globalTensorBlocks && !node.visualMedia && (isValueNode(node.type) || sources.has(node.type)))
    return { mode: 'block', tone: 'auto' };
}

interface DiagramOptions extends NodeTensorVisual {
  outputPort?: string;
  observeInput?: boolean;
  referenceMaxDim?: number;
}

function joinGeometry(
  node: ModelNode,
  inputs: [string, TensorSpec][],
  output: TensorSpec,
  colors: NodeTones,
  override?: Tone,
): TensorJoin | undefined {
  const axis = normalizeAxis(Number(node.params.dim), output.shape.length);
  // Inference accepts all-empty one-dimensional cat inputs regardless of dim.
  if (!Number.isInteger(axis) || axis < 0 || axis >= output.shape.length) return;
  const sizes = inputs.map(([, spec]) => (node.type === 'Stack' ? 1 : (spec.shape[axis] ?? 0)));
  // Symbolic extents have no known proportions; give each nonempty segment an equal share.
  const known = sizes.every((size) => typeof size === 'number');
  const weights = sizes.map((size) => (known ? Number(size) : Number(size !== 0)));
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let offset = 0;
  const segments = inputs.map(([port], i) => {
    const start = offset / total;
    offset += weights[i];
    const tones = override ? [override] : (colors.inputs.get(port)?.tones ?? neutral);
    return { port, size: sizes[i], start, end: offset / total, tones };
  });
  return { axis, direction: directionFor(axis, output.shape.length), segments };
}

function projectAxis(index: number, count: number, size: Dim, tones: Tone[], join?: TensorJoin) {
  const position = count === 1 ? 0 : index / (count - 1);
  const sample =
    typeof size === 'number' && size > 0
      ? (Math.round(position * (size - 1)) + 0.5) / size
      : position;
  const segment = join?.segments.find((s) => s.end > s.start && (sample < s.end || s.end === 1));
  return { index, segment: segment?.port, tones: segment?.tones ?? tones };
}

/** Visible layers in painter's order, sampled from their original depth positions. */
export function projectTensorSlices(box: GeometryBox, join?: TensorJoin) {
  const { sliceCount, size } = box.depth;
  return Array.from({ length: sliceCount }, (_, i) =>
    projectAxis(
      sliceCount - 1 - i,
      sliceCount,
      size,
      box.tones,
      join?.direction === 'depth' ? join : undefined,
    ),
  );
}

/** Left-to-right volumes, including the true endpoints of an elided outer axis. */
export function projectOuterVolumes(box: GeometryBox, join?: TensorJoin) {
  const { count, size } = box.outer ?? { count: 1, size: 1 };
  return Array.from({ length: count }, (_, index) =>
    projectAxis(index, count, size, box.tones, join?.direction === 'outer' ? join : undefined),
  );
}

export const defaultTensorVisualWidth = 180;

export function buildTensorBlockDiagram(
  spec: TensorSpec,
  port = 'out',
  tones: Tone[] = neutral,
  options: Omit<DiagramOptions, 'mode'> & { join?: TensorJoin } = {},
): TensorGeometryDiagram {
  const box = mapShapeToBox3D(spec, options);
  const override = explicitTone(options);
  const fills = override ? [override] : tones;
  return {
    mode: 'block',
    inputs: [],
    observeInput: options.observeInput,
    output: {
      ...box,
      ...computeScaledDims([box], options.referenceMaxDim, options)[0],
      port,
      tones: fills,
      dtype: spec.dtype,
    },
    badge: `${box.ndim}D`,
    join: options.join,
    width: options.width ?? ((box.outer?.count ?? 1) > 1 ? 220 : defaultTensorVisualWidth),
    customColor: fills.length === 1 && fills[0].startsWith('#') ? fills[0] : undefined,
    caption: options.caption,
  };
}

export function buildTensorGeometryDiagram(
  node: ModelNode,
  result: NodeResult | undefined,
  graphTones: GraphTensorTones,
  options: DiagramOptions,
): TensorGeometryDiagram | undefined {
  if (!result || options.mode === 'off' || !supportsTensorVisual(node)) return;
  const colors = graphTones.get(node.id);
  const observeInput =
    options.observeInput || (isBoundary(node.type) && !Object.keys(result.outputs).length);
  const values = observeInput ? result.inputs : result.outputs;
  const port =
    options.outputPort ??
    (values.out && isTensorSpec(values.out)
      ? 'out'
      : Object.keys(values).find((key) => isTensorSpec(values[key])));
  if (port === undefined) return;
  const value = values[port];
  if (!value || !isTensorSpec(value)) return;

  const identity = observeInput && !isBoundary(node.type) ? colors?.inputs.get(port) : colors;
  const visual = { ...node.tensorVisual, ...options };
  const override = explicitTone(visual);
  let join = identity?.join;
  if (join && override) {
    join =
      !observeInput && joins.has(node.type)
        ? { ...join, segments: join.segments.map((s) => ({ ...s, tones: [override] })) }
        : undefined;
  }
  return buildTensorBlockDiagram(value, port, identity?.tones, { ...visual, observeInput, join });
}
