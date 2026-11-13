import { create } from 'zustand';
import type {
  Graph,
  ModelEdge,
  LoopFeedbackLayout,
  Params,
  PortLayout,
  PortSide,
  Project,
  Hyperparameter,
  TensorGradState,
  RuntimeSnapshot,
} from '@tensor-trail/model-core/core/types';
import { clone, uid, isTensorSpec } from '@tensor-trail/model-core/core/types';
import { graphAt, graphIssues, markAncestors, ports } from '@tensor-trail/model-core/core/graph';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { forLoopStates } from '@tensor-trail/model-core/core/forLoop';
import { inferProject, resultAt } from '@tensor-trail/model-core/core/engine';
import { checkHyperparameterName } from '@tensor-trail/model-core/core/hyperparameters';
import { parseProject, validateProjectObject } from '@tensor-trail/model-core/core/serialization';
import {
  assignLayoutRegion, canJoinLayoutRegion, fitLayoutRegion, isLayoutRegion, moveLayoutNodes, resizeLayoutRegion,
  type NodeSizes, type RegionRect,
} from '@tensor-trail/model-core/core/layoutRegions';
import { checkSnapshotBindings, parseRuntimeSnapshot } from '@tensor-trail/model-core/core/runtimeSnapshot';
import { registry } from '@tensor-trail/model-core/modules/index';
import { dimensions } from '@tensor-trail/model-core/modules/math';
import { DEFAULT_TENSOR_PALETTE, validateAutograd } from '@tensor-trail/model-core/core/tensorAppearance';
import { copySelection } from '@tensor-trail/model-core/editing/clipboard';
import {
  isValueNode,
  linkedValueNodes,
  syncValueLinks,
  valueOwner,
} from '@tensor-trail/model-core/core/valueLinks';
import {
  createLinkedPaste,
  removeValueNodes,
  unlinkValueInGraph,
} from '@tensor-trail/model-core/editing/valueLinkEditing';
import type { WireRouteChange } from '../views/sharedWires';
import {
  mergeCalculations,
  splitCalculationGroup,
} from '@tensor-trail/model-core/core/calculationGroups';
const initial = {
  project: {
    schemaVersion: 1,
    name: '未打开项目',
    graph: { nodes: [], edges: [] },
    variants: [],
  } as Project,
  documentId: uid(),
  notice: '请打开项目，或新建一个项目。',
};
// File associations follow a document through edits and undo, but never enter exported JSON.
const documentIds = new WeakMap<Project, string>();
const activeDocuments = new WeakSet<Project>();
function documentId(project: Project, id: string = uid()): string {
  const existing = documentIds.get(project);
  if (existing) return existing;
  documentIds.set(project, id);
  return id;
}
interface EditorState {
  project: Project;
  runtimeSnapshot?: RuntimeSnapshot;
  runtimeError?: string;
  setRuntimeSnapshot: (snapshot?: RuntimeSnapshot) => void;
  selectRuntime: (id?: string) => void;
  hasProject: boolean;
  documentId: string;
  startNewDocument: () => void;
  canvasVersion: number;
  path: string[];
  loopSelections: Record<string, number>;
  selectLoopIteration: (path: string[], index: number) => void;
  selected: string | null;
  selectedIds: string[];
  nodeSizes: NodeSizes;
  setNodeSizes: (sizes: NodeSizes) => void;
  createLayoutRegion: (ids: string[]) => void;
  assignLayoutRegion: (ids: string[], region: string | null) => boolean;
  resizeLayoutRegion: (id: string, rect: RegionRect) => boolean;
  fitLayoutRegion: (id: string) => boolean;
  clipboard: Graph | null;
  clipboardOrigin: { documentId: string; path: string[]; fragment: Graph } | null;
  unlinkValue: (id: string) => void;
  pasteCount: number;
  selectMany: (ids: string[]) => void;
  copy: () => void;
  paste: (mode?: 'local' | 'requested') => void;
  duplicateLinked: (id: string) => boolean;
  newProject: () => void;
  setAxis: (id: string, axis: number, name: string) => void;
  setTensorColor: (id: string, color?: string) => void;
  setTensorBorderColor: (id: string, color?: string) => void;
  setTensorPaletteColor: (state: TensorGradState, color: string) => void;
  resetTensorPalette: () => void;
  setTensorAutograd: (id: string, requiresGrad?: boolean) => void;
  setLineageEnabled: (enabled: boolean) => void;
  past: Project[];
  future: Project[];
  notice: string;
  select: (id: string | null) => void;
  navigate: (path: string[]) => void;
  toast: (message: string) => void;
  edit: (fn: (graph: Graph, project: Project) => void, semantic?: boolean) => boolean;
  setParam: (id: string, key: string, value: Params[string]) => boolean;
  setHyperparameter: (name: string, value?: Hyperparameter) => boolean;
  setPortSide: (id: string, kind: keyof PortLayout, side: PortSide | 'auto') => void;
  renameNode: (id: string, name: string) => void;
  renameProject: (name: string) => void;
  add: (type: string, variantId?: string, position?: { x: number; y: number }) => void;
  connect: (edge: Omit<ModelEdge, 'id'>) => void;
  removeNodes: (ids: string[]) => void;
  removeEdges: (ids: string[]) => void;
  setEdgeWaypoints: (id: string, points?: ModelEdge['waypoints']) => void;
  setSharedWireWaypoints: (changes: WireRouteChange[]) => boolean;
  setEdgeLabelPosition: (id: string, position?: number) => void;
  setLoopFeedbackLayout: (state: string, layout?: LoopFeedbackLayout) => boolean;
  move: (id: string, position: { x: number; y: number }) => void;
  finishNodeDrag: (
    positions: { id: string; position: { x: number; y: number } }[],
    merge?: [string, string],
    routes?: WireRouteChange[],
  ) => void;
  splitCalculation: (id: string) => void;
  replace: (id: string, type: string) => void;
  saveVariant: (name: string) => void;
  undo: () => void;
  redo: () => void;
  load: (project: Project, preserveDocument?: boolean) => void;
  importText: (text: string) => void;
}

export const useEditor = create<EditorState>((set, get) => ({
  ...initial,
  hasProject: false,
  setRuntimeSnapshot: (snapshot) => {
    try {
      const checked = snapshot && parseRuntimeSnapshot(snapshot);
      if (checked) {
        if (checked.id !== get().project.runtime?.active) throw Error('快照与当前选中的运行不匹配。');
        checkSnapshotBindings(get().project, checked);
      }
      set({runtimeSnapshot:checked,runtimeError:undefined});
    } catch(error) { set({runtimeSnapshot:undefined,runtimeError:(error as Error).message,notice:(error as Error).message}); }
  },
  selectRuntime: (id) => {
    if (get().edit((_,project)=>{
      if (!project.runtime) { if (id) throw Error('没有运行快照引用。'); return; }
      if (id && !project.runtime.snapshots.some(ref=>ref.id===id)) throw Error('快照不存在。');
      if (id) project.runtime.active = id; else delete project.runtime.active;
    },false)) set({runtimeSnapshot:undefined,runtimeError:undefined});
  },
  documentId: documentId(initial.project, initial.documentId),
  startNewDocument: () => {
    const id = uid();
    documentIds.set(get().project, id);
    activeDocuments.add(get().project);
    set({ documentId: id, hasProject: true });
  },
  canvasVersion: 0,
  path: [],
  loopSelections: {},
  selectLoopIteration: (path, index) => {
    if (!Number.isSafeInteger(index) || index < 0) return;
    set(s => ({ loopSelections: { ...s.loopSelections, [JSON.stringify(path)]: index } }));
  },
  selected: null,
  selectedIds: [],
  nodeSizes: {},
  setNodeSizes: sizes => {
    if (JSON.stringify(sizes) !== JSON.stringify(get().nodeSizes)) set({ nodeSizes: sizes });
  },
  createLayoutRegion: ids => {
    let id = '';
    if (get().edit(graph => {
      const members = graph.nodes.filter(n => ids.includes(n.id) && canJoinLayoutRegion(n));
      const region = registry.create('LayoutRegion');
      region.position = members.length
        ? { x: Math.min(...members.map(n => n.position.x)) - 24, y: Math.min(...members.map(n => n.position.y)) - 60 }
        : { x: Math.max(0, ...graph.nodes.map(n => n.position.x + 300)), y: 0 };
      graph.nodes.push(region); id = region.id;
      assignLayoutRegion(graph, members.map(n => n.id), id, get().nodeSizes);
    }, false)) set({ selected: id, selectedIds: [id], notice: '已创建布局分区；可在属性中添加成员，拖动标题整体移动。' });
  },
  assignLayoutRegion: (ids, region) => get().edit(graph => assignLayoutRegion(graph, ids, region, get().nodeSizes), false),
  resizeLayoutRegion: (id, rect) => get().edit(graph => resizeLayoutRegion(graph, id, rect, get().nodeSizes), false),
  fitLayoutRegion: id => get().edit(graph => fitLayoutRegion(graph, id, get().nodeSizes), false),
  clipboard: null,
  clipboardOrigin: null,
  pasteCount: 0,
  past: [],
  future: [],
  select: (id) => set({ selected: id, selectedIds: id ? [id] : [] }),
  selectMany: (ids) => {
    if (JSON.stringify(ids) !== JSON.stringify(get().selectedIds))
      set({ selected: ids[0] ?? null, selectedIds: ids });
  },
  navigate: (path) => {
    try {
      resolveForwardView(get().project.graph, path);
      set({ path: [...path], selected: null, selectedIds: [], nodeSizes: {} });
    } catch (e) {
      set({ notice: (e as Error).message });
    }
  },
  newProject: () => {
    get().load({
      schemaVersion: 1,
      name: '未命名项目',
      graph: { nodes: [], edges: [] },
      variants: clone(get().project.variants),
    });
    set({ notice: '已新建空白项目，可拖入模块；撤销可返回上一项目。' });
  },
  copy: () => {
    const s = get(),
      fragment = copySelection(
        graphAt(s.project.graph, s.path),
        s.selectedIds.length ? s.selectedIds : s.selected ? [s.selected] : [],
      );
    if (!fragment.nodes.length) {
      set({ clipboard: null, clipboardOrigin: null, pasteCount: 0, notice: '请先选择模块。' });
      return;
    }
    set({
      clipboard: fragment,
      clipboardOrigin: { documentId: s.documentId, path: [...s.path], fragment },
      pasteCount: 0,
      notice: `已复制 ${fragment.nodes.length} 个模块，Ctrl+V 粘贴。`,
    });
  },
  paste: (mode = 'local') => {
    const { clipboard, clipboardOrigin, pasteCount, documentId, path, project } = get();
    if (!clipboard?.nodes.length) return;
    const sameScope =
      clipboardOrigin?.fragment === clipboard &&
      clipboardOrigin.documentId === documentId &&
      JSON.stringify(clipboardOrigin.path) === JSON.stringify(path);
    const fragment = createLinkedPaste(
      graphAt(project.graph, path),
      clipboard,
      32 * (pasteCount + 1),
      sameScope,
      mode,
    );
    if (
      get().edit((g) => {
        g.nodes.push(...fragment.nodes);
        g.edges.push(...fragment.edges);
      })
    )
      set({
        selected: fragment.nodes[0].id,
        selectedIds: fragment.nodes.map((n) => n.id),
        pasteCount: pasteCount + 1,
        notice: fragment.nodes.some((node) => node.valueSource !== undefined)
          ? `已粘贴 ${fragment.nodes.length} 个模块；关联副本共享原值，可在右侧解除链接。`
          : `已粘贴 ${fragment.nodes.length} 个模块，可一次撤销。`,
      });
  },
  duplicateLinked: (id) => {
    const state = get(),
      graph = graphAt(state.project.graph, state.path);
    const node = graph.nodes.find((candidate) => candidate.id === id);
    if (!node || !isValueNode(node.type)) {
      set({ notice: '只有输入输出类节点可以创建关联副本。' });
      return false;
    }
    const fragment = createLinkedPaste(graph, copySelection(graph, [id]), 48, true, 'requested');
    if (
      !get().edit((g) => {
        g.nodes.push(...fragment.nodes);
        g.edges.push(...fragment.edges);
      })
    )
      return false;
    set({
      selected: fragment.nodes[0].id,
      selectedIds: [fragment.nodes[0].id],
      notice: '已按要求创建关联副本，共享原值；可在右侧解除链接。',
    });
    return true;
  },
  setAxis: (id, axis, name) =>
    get().edit((g) => {
      const node = valueOwner(g, id);
      if (node.type !== 'Input') throw new Error('请在输入 Tensor 编辑轴含义，后续模块自动推导。');
      const shape = dimensions(node.params.shape);
      if (axis < 0 || axis >= shape.length) throw new Error('轴超出范围。');
      const previous = String(node.params.axes)
        .split(',')
        .map((s) => s.trim());
      const axes = shape.map((_, i) => previous[i] || `axis${i}`);
      axes[axis] = name.trim().replaceAll(',', '，').slice(0, 120) || `axis${axis}`;
      node.params.axes = axes.join(', ');
    }),
  setTensorColor: (id, color) => get().setTensorBorderColor(id, color),
  setTensorBorderColor: (id, color) =>
    get().edit((g) => {
      const node = g.nodes.find((n) => n.id === id)!;
      delete node.tensorColor;
      if (color) node.tensorBorderColor = color;
      else delete node.tensorBorderColor;
    }, false),
  setTensorPaletteColor: (state, color) =>
    get().edit((_, project) => {
      project.tensorPalette = { ...DEFAULT_TENSOR_PALETTE, ...project.tensorPalette, [state]: color };
    }, false),
  resetTensorPalette: () => get().edit((_, project) => { delete project.tensorPalette; }, false),
  setLineageEnabled: enabled => get().edit((_, project) => { project.lineageEnabled = enabled; }, false),
  setTensorAutograd: (id, requiresGrad) =>
    get().edit((g, project) => {
      const node = valueOwner(g, id);
      if (requiresGrad === undefined) { delete node.tensorAutograd; return; }
      const spec = resultAt(inferProject(project, registry, get().runtimeSnapshot), get().path, get().loopSelections)?.nodes[node.id]?.outputs.out;
      if (!spec || !isTensorSpec(spec)) throw new Error('请先连接有效的 out Tensor，再标注自动求导状态。');
      const annotation = { requiresGrad, source: 'manual' as const };
      validateAutograd(annotation, spec.dtype);
      node.tensorAutograd = annotation;
    }, false),
  toast: (notice) => set({ notice }),
  edit: (fn, semantic = true) => {
    if (!get().hasProject) {
      set({ notice: '请先打开项目或新建项目。' });
      return false;
    }
    const { project, path, past } = get();
    let draft = clone(project);
    try {
      fn(graphAt(draft.graph, path), draft);
      if (semantic) markAncestors(draft.graph, path);
      syncValueLinks(draft.graph);
      // Keep every committed edit reloadable, including size and nesting limits.
      draft = validateProjectObject(draft, registry);
      documentId(project, get().documentId);
      documentId(draft, get().documentId);
      activeDocuments.add(project);
      activeDocuments.add(draft);
      set({ project: draft, past: [...past.slice(-39), project], future: [] });
      return true;
    } catch (e) {
      set({ notice: (e as Error).message });
      return false;
    }
  },
  setHyperparameter: (name, value) =>
    get().edit((_, project) => {
      checkHyperparameterName(name);
      const parameters = { ...project.hyperparameters };
      if (value) parameters[name] = value;
      else delete parameters[name];
      project.hyperparameters = parameters;
    }, false),
  setParam: (id, key, value) =>
    get().edit((graph) => {
      if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('请输入有限数值。');
      const node = valueOwner(graph, id);
      if (node.subgraph && node.customized)
        throw new Error('此模块已有内部编辑，请进入子图修改各算子。');
      if (
        node.type === 'TensorIndex' &&
        ((key === 'mode' && value !== 'slice') || (key === 'stopFromInput' && value !== true)) &&
        graph.edges.some((edge) => edge.target === id && edge.targetPort === 'stop')
      )
        throw new Error('请先断开 stop 连线，再切换取值方式或结束位置来源。');
      if (
        ['Concat', 'Stack'].includes(node.type) &&
        key === 'input_count' &&
        (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > 16)
      )
        throw new Error('输入数量请选择 1–16。');
      if (
        (['Concat', 'Stack'].includes(node.type) && key === 'input_count') ||
        (['View', 'Reshape', 'Zeros'].includes(node.type) && key === 'shape') ||
        (node.type === 'Repeat' && key === 'repeats') ||
        (node.type === 'DictionaryPack' && key === 'keys') ||
        (node.type === 'PlutoCode' && key === 'operation')
      ) {
        const changed = { ...node, params: { ...node.params, [key]: value } };
        const remaining = new Set(ports(changed, registry).inputs.map((port) => port.id));
        const removed = graph.edges.filter(
          (edge) => edge.target === node.id && !remaining.has(edge.targetPort),
        );
        if (removed.length)
          throw new Error(
            `请先断开输入 ${[...new Set(removed.map((edge) => edge.targetPort))].join('、')} 的连线，再修改输入数量或形状。`,
          );
      }
      node.params[key] = value;
      if (node.type === 'Input' && key === 'shape') {
        try {
          const previous = String(node.params.axes)
            .split(',')
            .map((s) => s.trim());
          node.params.axes = dimensions(value)
            .map((_, i) => previous[i] || `axis${i}`)
            .join(', ');
        } catch {
          /* Keep incomplete shape edits visible for correction. */
        }
      }
      const def = registry.get(node.type)!;
      const valid = def.fields.every(
        (f) =>
          f.kind !== 'number' ||
          (Number.isSafeInteger(node.params[f.key]) && Number(node.params[f.key]) >= (f.min ?? 1)),
      );
      if (def.build && valid) node.subgraph = def.build(node.params);
    }, !graphAt(get().project.graph, get().path).nodes.some(n => n.id === id && isLayoutRegion(n))),
  renameNode: (id, name) => {
    if (name.trim())
      get().edit((g) => {
        valueOwner(g, id).label = name.trim().slice(0, 200);
      }, false);
  },
  setPortSide: (id, kind, side) =>
    get().edit((graph) => {
      const node = graph.nodes.find((n) => n.id === id);
      if (!node) throw new Error('模块已不存在。');
      const layout = { ...node.portLayout };
      if (side === 'auto') delete layout[kind];
      else layout[kind] = side;
      if (Object.keys(layout).length) node.portLayout = layout;
      else delete node.portLayout;
    }, false),
  renameProject: (name) => {
    if (name.trim())
      get().edit((_, p) => {
        p.name = name.trim().slice(0, 200);
      }, false);
  },
  add: (type, variantId, position) => {
    let added = '';
    const committed = get().edit((graph, project) => {
      // A drop places an independent module; clicks keep the automatic connection behavior.
      const source = position ? undefined : graph.nodes.find((n) => n.id === get().selected);
      const variant = project.variants.find((v) => v.id === variantId);
      if (variantId !== undefined && !variant) throw new Error('这个变体已不存在，请重新选择。');
      const node = variant
        ? registry.fromVariant(variant.name, variant.graph)
        : registry.create(type);
      if (node.type === 'Inlet' || node.type === 'Outlet') {
        if (!get().path.length) throw new Error('请进入子模块后添加模块输入或模块输出。');
        const base = node.type === 'Inlet' ? 'in' : 'out';
        const used = new Set(
          graph.nodes.filter((n) => n.type === node.type).map((n) => String(n.params.port)),
        );
        let port = base,
          suffix = 2;
        while (used.has(port)) port = `${base}${suffix++}`;
        node.params.port = port;
        node.label = port;
      }
      const specs = resultAt(inferProject(project, registry, get().runtimeSnapshot), get().path, get().loopSelections)?.nodes[source?.id ?? '']
        ?.outputs;
      const sourcePort = source ? ports(source, registry).outputs[0]?.id : undefined;
      const spec = specs?.[sourcePort ?? ''];
      if (spec && isTensorSpec(spec) && !variant) {
        const last = spec.shape.at(-1);
        const keys = ['in_features', 'input_size', 'd_model', 'features'];
        for (const key of keys)
          if (key in node.params && typeof last === 'number') node.params[key] = last;
        if ('in_channels' in node.params && typeof spec.shape[1] === 'number')
          node.params.in_channels = spec.shape[1];
        const def = registry.get(type);
        if (def?.build) node.subgraph = def.build(node.params);
      }
      node.position = position
        ? { ...position }
        : source
          ? { x: source.position.x + 290, y: source.position.y + 190 }
          : { x: (graph.nodes.length % 4) * 290, y: Math.floor(graph.nodes.length / 4) * 220 };
      graph.nodes.push(node);
      const targetPort = ports(node, registry).inputs[0]?.id;
      if (source && sourcePort && targetPort)
        graph.edges.push({ id: uid(), source: source.id, sourcePort, target: node.id, targetPort });
      added = node.id;
    }, type !== 'LayoutRegion' || variantId !== undefined);
    if (committed && added)
      set({
        selected: added,
        selectedIds: [added],
        notice: position
          ? '已在落点添加模块，拖动端口即可连线。'
          : '已添加模块；可拖动端口调整连接。',
      });
  },
  connect: (edge) =>
    get().edit((graph) => {
      const candidate = { ...graph, edges: [...graph.edges, { id: uid(), ...edge }] };
      const issues = graphIssues(candidate, registry);
      if (issues.length) throw new Error(issues[0]);
      graph.edges = candidate.edges;
    }),
  unlinkValue: (id) => {
    if (get().edit((_, project) => unlinkValueInGraph(project.graph, get().path, id)))
      set({ notice: '已解除链接；当前输入连线保留，之后可单独修改或重新连接。' });
  },
  removeNodes: (ids) => {
    const semantic = graphAt(get().project.graph, get().path).nodes.some(n => ids.includes(n.id) && !isLayoutRegion(n));
    const committed = get().edit((_, project) => removeValueNodes(project.graph, get().path, ids), semantic);
    if (committed) set({ selected: null, selectedIds: [] });
  },
  removeEdges: (ids) =>
    get().edit((graph) => {
      graph.edges = graph.edges.filter((e) => !ids.includes(e.id));
    }),
  setEdgeWaypoints: (id, points) =>
    get().edit((graph) => {
      const edge = graph.edges.find((e) => e.id === id);
      if (!edge) throw new Error('连线已不存在。');
      if (points?.length) edge.waypoints = clone(points);
      else delete edge.waypoints;
    }, false),
  setSharedWireWaypoints: (changes) => {
    if (!changes.length) return false;
    return get().edit((graph) => {
      if (new Set(changes.map((change) => change.id)).size !== changes.length)
        throw new Error('重复的共享连线修改。');
      for (const change of changes) {
        const edge = graph.edges.find((candidate) => candidate.id === change.id);
        if (!edge) throw new Error('连线已不存在。');
        if (change.waypoints.length) edge.waypoints = clone(change.waypoints);
        else delete edge.waypoints;
      }
    }, false);
  },
  setEdgeLabelPosition: (id, position) =>
    get().edit((graph) => {
      const edge = graph.edges.find((e) => e.id === id);
      if (!edge) throw new Error('连线已不存在。');
      if (position === undefined) delete edge.labelPosition;
      else edge.labelPosition = position;
    }, false),
  setLoopFeedbackLayout: (state, layout) =>
    get().edit((graph, project) => {
      const path = get().path;
      const owner = path.length
        ? graphAt(project.graph, path.slice(0, -1)).nodes.find(n => n.id === path.at(-1)) : undefined;
      if (owner?.type !== 'ForLoop' || !forLoopStates(owner).includes(state) ||
          !['Inlet', 'Outlet'].every(type => graph.nodes.filter(n =>
            n.type === type && n.valueSource === undefined && n.params.port === state).length === 1))
        throw new Error('当前循环不存在此状态回传线。');
      if (layout) owner.loopFeedbackRoutes = { ...owner.loopFeedbackRoutes, [state]: clone(layout) };
      else if (owner.loopFeedbackRoutes) {
        delete owner.loopFeedbackRoutes[state];
        if (!Object.keys(owner.loopFeedbackRoutes).length) delete owner.loopFeedbackRoutes;
      }
    }, false),
  move: (id, position) =>
    get().edit((graph) => {
      moveLayoutNodes(graph, [{ id, position }], get().nodeSizes);
    }, false),
  finishNodeDrag: (positions, merge, routes = []) => {
    let merged: string | undefined;
    const committed = get().edit((graph) => {
      moveLayoutNodes(graph, positions, get().nodeSizes);
      if (new Set(routes.map((r) => r.id)).size !== routes.length)
        throw new Error('重复的连线布局修改。');
      for (const route of routes) {
        const edge = graph.edges.find((e) => e.id === route.id);
        if (!edge) throw new Error('连线已不存在。');
        if (route.waypoints.length) edge.waypoints = clone(route.waypoints);
        else delete edge.waypoints;
      }
      if (merge) merged = mergeCalculations(graph, merge[0], merge[1], registry);
    }, !!merge);
    if (committed && merged)
      set({
        selected: merged,
        selectedIds: [merged],
        notice: '已合并计算节点；双击编辑内部操作，Ctrl+Z 撤销，属性面板可拆开组合。',
      });
  },
  splitCalculation: (id) => {
    let ids: string[] = [];
    if (
      get().edit((graph) => {
        ids = splitCalculationGroup(graph, id);
      })
    )
      set({
        selected: ids[0] ?? null,
        selectedIds: ids,
        pasteCount: get().pasteCount + 1,
        notice: '已拆开计算组合，恢复各操作与连接。',
      });
  },
  replace: (id, type) =>
    get().edit((graph) => {
      const index = graph.nodes.findIndex((n) => n.id === id),
        old = graph.nodes[index];
      if (linkedValueNodes(graph, id).length > 1) throw new Error('请先解除链接，再替换模块类型。');
      if (['Inlet', 'Outlet', 'Input', 'Output'].includes(old.type) || old.subgraph)
        throw new Error('请在组合模块内选择普通算子进行替换。');
      const fresh = registry.create(type, {}, id),
        a = ports(old, registry),
        b = ports(fresh, registry);
      if (
        JSON.stringify(a.inputs.map((p) => p.id)) !== JSON.stringify(b.inputs.map((p) => p.id)) ||
        JSON.stringify(a.outputs.map((p) => p.id)) !== JSON.stringify(b.outputs.map((p) => p.id))
      )
        throw new Error('替换模块必须具有相同的端口标识。');
      for (const key of Object.keys(fresh.params))
        if (key in old.params) fresh.params[key] = old.params[key];
      fresh.position = old.position;
      if (old.portLayout) fresh.portLayout = clone(old.portLayout);
      if (old.label !== registry.get(old.type)?.label) fresh.label = old.label;
      graph.nodes[index] = fresh;
    }),
  saveVariant: (name) => {
    if (!name.trim()) {
      set({ notice: '请给变体填写名称。' });
      return;
    }
    const state = get();
    const current = graphAt(state.project.graph, state.path);
    const selected = current.nodes.find((n) => n.id === state.selected);
    const owner = selected?.subgraph
      ? selected
      : state.path.length
        ? graphAt(state.project.graph, state.path.slice(0, -1)).nodes.find(
            (n) => n.id === state.path.at(-1),
          )
        : undefined;
    if (!owner?.subgraph) {
      set({ notice: '请选择组合模块，或先进入一个子图。' });
      return;
    }
    if (owner.type === 'ForEachTensor') {
      set({ notice: 'ForEach 具有循环收集语义，请复制整个循环模块；不能将循环体单独保存为普通模块变体。' });
      return;
    }
    if (['WhileLoop', 'ForLoop'].includes(owner.type)) {
      set({ notice: `${owner.type} 具有状态反馈语义，请复制整个循环模块；不能把循环体保存为普通模块变体。` });
      return;
    }
    if (owner.type === 'Submodule' && !owner.subgraph.nodes.some((n) => n.type === 'Outlet')) {
      set({ notice: '补充模块输出后可保存为变体。' });
      return;
    }
    const graph = clone(owner.subgraph);
    const committed = get().edit((_, project) => {
      project.variants.push({ id: uid(), name: name.trim().slice(0, 200), graph });
    }, false);
    if (committed) set({ notice: `已保存「${name.trim()}」，可从左侧“我的变体”再次添加。` });
  },
  undo: () => {
    const { past, project, future } = get();
    if (past.length)
      set({
        project: past.at(-1)!,
        runtimeSnapshot: documentId(past.at(-1)!) === get().documentId ? get().runtimeSnapshot : undefined,
        runtimeError: undefined,
        hasProject: activeDocuments.has(past.at(-1)!),
        documentId: documentId(past.at(-1)!),
        canvasVersion: get().canvasVersion + 1,
        past: past.slice(0, -1),
        future: [project, ...future],
        path: [],
        selected: null,
        selectedIds: [],
        notice: '已撤销。',
        nodeSizes: {},
      });
  },
  redo: () => {
    const { past, project, future } = get();
    if (future.length)
      set({
        project: future[0],
        runtimeSnapshot: documentId(future[0]) === get().documentId ? get().runtimeSnapshot : undefined,
        runtimeError: undefined,
        hasProject: activeDocuments.has(future[0]),
        documentId: documentId(future[0]),
        canvasVersion: get().canvasVersion + 1,
        past: [...past, project],
        future: future.slice(1),
        path: [],
        selected: null,
        selectedIds: [],
        notice: '已重做。',
        nodeSizes: {},
      });
  },
  load: (project, preserveDocument = false) => {
    const { past, project: previous, documentId: previousId } = get();
    documentId(previous, previousId);
    if (get().hasProject) activeDocuments.add(previous);
    const loaded = validateProjectObject(clone(project), registry);
    activeDocuments.add(loaded);
    set({
      project: loaded,
      loopSelections: {},
      nodeSizes: {},
      runtimeSnapshot: undefined,
      runtimeError: undefined,
      hasProject: true,
      documentId: documentId(loaded, preserveDocument ? previousId : uid()),
      canvasVersion: get().canvasVersion + 1,
      path: [],
      selected: null,
      selectedIds: [],
      past: [...past.slice(-39), previous],
      future: [],
      notice: '项目已载入；可撤销返回上一项目。',
    });
  },
  importText: (text) => {
    try {
      get().load(parseProject(text, registry));
    } catch (e) {
      set({ notice: `导入失败：${(e as Error).message}` });
    }
  },
}));
