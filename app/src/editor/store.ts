import { create } from 'zustand';
import type { Graph, ModelEdge, Params, PortLayout, PortSide, Project } from '../core/types';
import { clone, uid } from '../core/types';
import { graphAt, graphIssues, markAncestors, ports } from '../core/graph';
import { inferGraph, resultAt } from '../core/engine';
import { parseProject, serializeProject } from '../core/serialization';
import { registry } from '../modules';
import { example } from '../modules/examples';

const STORAGE_KEY = 'tensor-trail.project.v1';
function restore(): { project: Project; notice: string } {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return {
      project: saved ? parseProject(saved, registry) : example(),
      notice: saved ? '已恢复本地项目' : '从一个 Transformer 开始，双击模块查看内部。',
    };
  } catch {
    return {
      project: example(),
      notice: '本地项目未能读取，已打开示例。旧记录在你编辑前不会被覆盖。',
    };
  }
}
const initial = restore();
interface EditorState {
  project: Project;
  canvasVersion: number;
  path: string[];
  selected: string | null;
  past: Project[];
  future: Project[];
  notice: string;
  saveState: string;
  select: (id: string | null) => void;
  navigate: (path: string[]) => void;
  toast: (message: string) => void;
  edit: (fn: (graph: Graph, project: Project) => void, semantic?: boolean) => boolean;
  setParam: (id: string, key: string, value: Params[string]) => void;
  setPortSide: (id: string, kind: keyof PortLayout, side: PortSide | 'auto') => void;
  renameNode: (id: string, name: string) => void;
  renameProject: (name: string) => void;
  add: (type: string, variantId?: string, position?: { x: number; y: number }) => void;
  connect: (edge: Omit<ModelEdge, 'id'>) => void;
  removeNodes: (ids: string[]) => void;
  removeEdges: (ids: string[]) => void;
  setEdgeWaypoints: (id: string, points?: ModelEdge['waypoints']) => void;
  move: (id: string, position: { x: number; y: number }) => void;
  replace: (id: string, type: string) => void;
  saveVariant: (name: string) => void;
  undo: () => void;
  redo: () => void;
  load: (project: Project) => void;
  importText: (text: string) => void;
}

export const useEditor = create<EditorState>((set, get) => ({
  ...initial,
  canvasVersion: 0,
  path: [],
  selected: 'transformer',
  past: [],
  future: [],
  saveState: '本地保存',
  select: (id) => set({ selected: id }),
  navigate: (path) => set({ path, selected: null }),
  toast: (notice) => set({ notice }),
  edit: (fn, semantic = true) => {
    const { project, path, past } = get();
    const draft = clone(project);
    try {
      fn(graphAt(draft.graph, path), draft);
      if (semantic) markAncestors(draft.graph, path);
      // Keep every committed edit reloadable, including size and nesting limits.
      parseProject(serializeProject(draft), registry);
      set({ project: draft, past: [...past.slice(-39), project], future: [] });
      return true;
    } catch (e) {
      set({ notice: (e as Error).message });
      return false;
    }
  },
  setParam: (id, key, value) =>
    get().edit((graph) => {
      if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('请输入有限数值。');
      const node = graph.nodes.find((n) => n.id === id)!;
      if (node.subgraph && node.customized)
        throw new Error('此模块已有内部编辑，请进入子图修改各算子。');
      node.params[key] = value;
      const def = registry.get(node.type)!;
      const valid = def.fields.every(
        (f) =>
          f.kind !== 'number' ||
          (Number.isSafeInteger(node.params[f.key]) && Number(node.params[f.key]) >= (f.min ?? 1)),
      );
      if (def.build && valid) node.subgraph = def.build(node.params);
    }),
  renameNode: (id, name) => {
    if (name.trim())
      get().edit((g) => {
        g.nodes.find((n) => n.id === id)!.label = name.trim().slice(0, 200);
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
      const specs = resultAt(inferGraph(project.graph, registry), get().path)?.nodes[
        source?.id ?? ''
      ]?.outputs;
      const sourcePort = source ? ports(source, registry).outputs[0]?.id : undefined;
      const spec = specs?.[sourcePort ?? ''];
      if (spec && !variant) {
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
    });
    if (committed && added)
      set({
        selected: added,
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
  removeNodes: (ids) => {
    get().edit((graph) => {
      const removable = new Set(
        graph.nodes
          .filter((n) => ids.includes(n.id) && !['Inlet', 'Outlet'].includes(n.type))
          .map((n) => n.id),
      );
      graph.nodes = graph.nodes.filter((n) => !removable.has(n.id));
      graph.edges = graph.edges.filter((e) => !removable.has(e.source) && !removable.has(e.target));
    });
    set({ selected: null });
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
  move: (id, position) =>
    get().edit((graph) => {
      const node = graph.nodes.find((n) => n.id === id);
      if (node) node.position = position;
    }, false),
  replace: (id, type) =>
    get().edit((graph) => {
      const index = graph.nodes.findIndex((n) => n.id === id),
        old = graph.nodes[index];
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
        canvasVersion: get().canvasVersion + 1,
        past: past.slice(0, -1),
        future: [project, ...future],
        path: [],
        selected: null,
        notice: '已撤销。',
      });
  },
  redo: () => {
    const { past, project, future } = get();
    if (future.length)
      set({
        project: future[0],
        canvasVersion: get().canvasVersion + 1,
        past: [...past, project],
        future: future.slice(1),
        path: [],
        selected: null,
        notice: '已重做。',
      });
  },
  load: (project) => {
    const { past, project: previous } = get();
    set({
      project: clone(project),
      canvasVersion: get().canvasVersion + 1,
      path: [],
      selected: null,
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

useEditor.subscribe((state, previous) => {
  if (state.project === previous.project) return;
  try {
    localStorage.setItem(STORAGE_KEY, serializeProject(state.project));
    if (state.saveState !== '已保存到本机') useEditor.setState({ saveState: '已保存到本机' });
  } catch {
    useEditor.setState({
      saveState: '本地保存失败，请导出 JSON',
      notice: '浏览器存储不可用或已满，请导出 JSON 保存当前项目。',
    });
  }
});
