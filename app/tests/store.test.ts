import { beforeEach, describe, expect, it, vi } from 'vitest';
import { inferGraph } from '../src/core/engine';
import { parseProject } from '../src/core/serialization';
import { example } from '../src/modules/examples';
import { registry } from '../src/modules';

const memory = new Map<string, string>();
vi.stubGlobal('localStorage', {
  getItem: (key: string) => memory.get(key) ?? null,
  setItem: (key: string, value: string) => memory.set(key, value),
});
const { useEditor } = await import('../src/editor/store');
const { canvasApi } = await import('../src/editor/agentBridge');

beforeEach(() => {
  memory.clear();
  useEditor.setState({
    project: example(),
    canvasVersion: 0,
    path: [],
    selected: 'transformer',
    past: [],
    future: [],
    notice: '',
    saveState: '本地保存',
  });
});

describe('编辑与持久化事务', () => {
  it('手动调整子图连线只保存布局，一次拖动可一次撤销重做', () => {
    const before = useEditor.getState().project;
    useEditor.getState().navigate(['transformer']);
    const edge = before.graph.nodes[1].subgraph!.edges[0];
    const points = [
      { x: 120, y: -140 },
      { x: 380, y: -140 },
    ];
    useEditor.getState().setEdgeWaypoints(edge.id, points);
    const after = useEditor.getState().project;
    expect(after.graph.nodes[1].customized).toBeUndefined();
    expect(after.graph.nodes[1].subgraph!.edges[0].waypoints).toEqual(points);
    expect(after.graph.nodes[1].subgraph!.edges[0].waypoints).not.toBe(points);
    expect(after.graph.nodes[1].subgraph!.nodes).toEqual(before.graph.nodes[1].subgraph!.nodes);
    expect(inferGraph(after.graph, registry)).toEqual(inferGraph(before.graph, registry));
    expect(parseProject(memory.get('tensor-trail.project.v1')!, registry)).toEqual(after);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('恢复自动走线并可撤销恢复原控制点', () => {
    const id = useEditor.getState().project.graph.edges[0].id;
    useEditor.getState().setEdgeWaypoints(id, [{ x: 100, y: -100 }]);
    const manual = useEditor.getState().project;
    useEditor.getState().setEdgeWaypoints(id);
    expect(useEditor.getState().project.graph.edges[0].waypoints).toBeUndefined();
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(manual);
  });
  it('拒绝非法或过多的连线控制点且不覆盖当前项目', () => {
    const before = useEditor.getState().project;
    for (const waypoints of [
      null,
      [{ x: 'bad', y: 0 }],
      [{ x: null, y: 0 }],
      Array.from({ length: 101 }, () => ({ x: 0, y: 0 })),
    ]) {
      const candidate = structuredClone(before);
      Object.assign(candidate.graph.edges[0], { waypoints });
      useEditor.getState().importText(JSON.stringify(candidate));
      expect(useEditor.getState().project).toBe(before);
      expect(useEditor.getState().notice).toContain('连线控制点无效');
    }
  });
  it('分别保存输入输出端口位置，布局编辑不改变子图语义并可撤销', () => {
    const before = useEditor.getState().project;
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().setPortSide('norm1', 'input', 'top');
    useEditor.getState().setPortSide('norm1', 'output', 'bottom');
    const after = useEditor.getState().project;
    const parent = after.graph.nodes[1];
    const node = parent.subgraph!.nodes.find((n) => n.id === 'norm1')!;
    expect(node.portLayout).toEqual({ input: 'top', output: 'bottom' });
    expect(parent.customized).toBeUndefined();
    expect(inferGraph(after.graph, registry)).toEqual(inferGraph(before.graph, registry));
    expect(parseProject(memory.get('tensor-trail.project.v1')!, registry)).toEqual(after);
    useEditor.getState().undo();
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
  });
  it('恢复自动端口位置并在替换兼容算子时保留布局', () => {
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().setPortSide('norm1', 'input', 'left');
    useEditor.getState().setPortSide('norm1', 'output', 'bottom');
    useEditor.getState().replace('norm1', 'RMSNorm');
    let node = useEditor
      .getState()
      .project.graph.nodes[1].subgraph!.nodes.find((n) => n.id === 'norm1')!;
    expect(node.portLayout).toEqual({ input: 'left', output: 'bottom' });
    useEditor.getState().setPortSide('norm1', 'input', 'auto');
    useEditor.getState().setPortSide('norm1', 'output', 'auto');
    node = useEditor
      .getState()
      .project.graph.nodes[1].subgraph!.nodes.find((n) => n.id === 'norm1')!;
    expect(node.portLayout).toBeUndefined();
  });
  it('导入非法端口位置时保留当前项目', () => {
    const before = useEditor.getState().project;
    for (const layout of [null, [], { input: 'diagonal' }, { output: 42 }]) {
      const candidate = structuredClone(before);
      Object.assign(candidate.graph.nodes[0], { portLayout: layout });
      useEditor.getState().importText(JSON.stringify(candidate));
      expect(useEditor.getState().project).toBe(before);
      expect(useEditor.getState().notice).toContain('端口位置无效');
    }
  });
  it('拖入时按落点独立添加，并支持持久化及一次撤销重做', () => {
    const before = useEditor.getState().project;
    const position = { x: -130.5, y: 212.25 };
    useEditor.getState().add('Conv1d', undefined, position);
    const after = useEditor.getState().project;
    const node = after.graph.nodes.at(-1)!;
    expect(node.position).toEqual(position);
    expect(node.params.in_channels).toBe(64);
    expect(after.graph.edges).toEqual(before.graph.edges);
    expect(useEditor.getState().selected).toBe(node.id);
    expect(useEditor.getState().past).toHaveLength(1);
    expect(parseProject(memory.get('tensor-trail.project.v1')!, registry)).toEqual(after);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('程序添加未指定落点时可推导参数并自动连线', () => {
    useEditor.getState().setParam('input', 'shape', '8, 100, 96');
    useEditor.getState().select('input');
    useEditor.getState().add('Linear');
    const graph = useEditor.getState().project.graph;
    const node = graph.nodes.at(-1)!;
    expect(node.params.in_features).toBe(96);
    expect(graph.edges.at(-1)).toMatchObject({ source: 'input', target: node.id });
  });
  it('可以将独立的变体实例拖入当前子图，撤销恢复原结构', () => {
    useEditor.getState().saveVariant('拖入变体');
    const before = useEditor.getState().project;
    const variant = before.variants[0];
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().select('attention');
    useEditor.getState().add('Block', variant.id, { x: 530, y: 720 });
    const after = useEditor.getState().project;
    const parent = after.graph.nodes.find((n) => n.id === 'transformer')!;
    const child = parent.subgraph!.nodes.at(-1)!;
    expect(after.graph.nodes).toHaveLength(before.graph.nodes.length);
    expect(parent.customized).toBe(true);
    expect(child.position).toEqual({ x: 530, y: 720 });
    expect(child.subgraph).toEqual(variant.graph);
    expect(child.subgraph).not.toBe(variant.graph);
    expect(parent.subgraph!.edges).toEqual(before.graph.nodes[1].subgraph!.edges);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
  });
  it('AI 应用保留变体、一次撤销，并自动保存到本机', () => {
    useEditor.getState().saveVariant('保留的变体');
    const before = canvasApi.getModel();
    const prepared = canvasApi.patchModel(before.project, {
      operations: [
        { op: 'rename_project', name: 'AI 修改项目' },
        { op: 'set_params', id: 'transformer', params: { norm: 'RMSNorm' } },
      ],
    });
    const history = useEditor.getState().past.length;
    expect(
      canvasApi.applyModel({ project: prepared.project, expectedRevision: before.revision }).ok,
    ).toBe(true);
    expect(useEditor.getState().past).toHaveLength(history + 1);
    expect(useEditor.getState().project.variants).toEqual(before.project.variants);
    expect(parseProject(memory.get('tensor-trail.project.v1')!, registry).name).toBe('AI 修改项目');
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before.project);
    expect(
      canvasApi.applyModel({ project: prepared.project, expectedRevision: before.revision }).ok,
    ).toBe(false);
  });
  it('内部语义编辑标记父模块并支持完整撤销重做', () => {
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().setParam('norm1', 'features', 32);
    expect(useEditor.getState().project.graph.nodes[1].customized).toBe(true);
    expect(inferGraph(useEditor.getState().project.graph, registry).nodes.transformer.status).toBe(
      'error',
    );
    useEditor.getState().undo();
    expect(useEditor.getState().project.graph.nodes[1].customized).toBeUndefined();
    expect(inferGraph(useEditor.getState().project.graph, registry).nodes.transformer.status).toBe(
      'ok',
    );
    useEditor.getState().redo();
    expect(useEditor.getState().project.graph.nodes[1].customized).toBe(true);
  });
  it('仅移动内部节点不会锁定父模块参数', () => {
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().move('norm1', { x: 10, y: 20 });
    expect(useEditor.getState().project.graph.nodes[1].customized).toBeUndefined();
  });
  it('已经自定义的结构不会被父层参数重建覆盖', () => {
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().replace('norm1', 'RMSNorm');
    useEditor.getState().navigate([]);
    const previous = useEditor.getState().project;
    useEditor.getState().setParam('transformer', 'heads', 4);
    expect(useEditor.getState().project).toBe(previous);
    expect(useEditor.getState().notice).toContain('内部编辑');
  });
  it('保存变体并添加得到独立、可推导的实例', () => {
    useEditor.getState().saveVariant('学习变体');
    const variant = useEditor.getState().project.variants[0];
    useEditor.getState().add('Block', variant.id);
    const project = useEditor.getState().project;
    const instance = project.graph.nodes.at(-1)!;
    expect(instance.type).toBe('Block');
    expect(instance.subgraph).not.toBe(variant.graph);
    expect(inferGraph(project.graph, registry).nodes[instance.id].outputs.out.shape).toEqual([
      8, 100, 64,
    ]);
  });
  it('重复输入连接拒绝提交且保留原项目', () => {
    const previous = useEditor.getState().project;
    useEditor
      .getState()
      .connect({ source: 'input', sourcePort: 'out', target: 'transformer', targetPort: 'in' });
    expect(useEditor.getState().project).toBe(previous);
    expect(useEditor.getState().notice).toContain('已有输入');
  });
  it('非有限数字不会损坏本地 JSON', () => {
    const previous = useEditor.getState().project;
    useEditor.getState().setParam('transformer', 'heads', Infinity);
    expect(useEditor.getState().project).toBe(previous);
    expect(useEditor.getState().notice).toContain('有限数值');
    expect(() => parseProject(memory.get('tensor-trail.project.v1')!, registry)).not.toThrow();
  });
  it('达到变体数量上限时不提交也不报告虚假的成功', () => {
    useEditor.getState().edit((_, project) => {
      const graph = project.graph.nodes[1].subgraph!;
      project.variants = Array.from({ length: 100 }, (_, i) => ({
        id: `v_${i}`,
        name: `变体 ${i}`,
        graph: {
          nodes: graph.nodes.filter((n) => ['Inlet', 'Outlet'].includes(n.type)),
          edges: [],
        },
      }));
    }, false);
    const previous = useEditor.getState().project;
    useEditor.getState().saveVariant('第 101 个');
    expect(useEditor.getState().project).toBe(previous);
    expect(useEditor.getState().notice).not.toContain('已保存');
  });
  it('导入失败不覆盖当前项目，正常编辑自动持久化', () => {
    const previous = useEditor.getState().project;
    useEditor.getState().importText('{"schemaVersion":99}');
    expect(useEditor.getState().project).toBe(previous);
    useEditor.getState().setParam('input', 'shape', '2, 20, 64');
    const restored = parseProject(memory.get('tensor-trail.project.v1')!, registry);
    expect(inferGraph(restored.graph, registry).nodes.transformer.outputs.out.shape).toEqual([
      2, 20, 64,
    ]);
  });
});
