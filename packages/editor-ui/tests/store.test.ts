import { tensorValue } from '@tensor-trail/model-core/core/types';
import { beforeEach, describe, expect, it } from 'vitest';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { example } from '@tensor-trail/model-core/modules/examples';
import { registry } from '@tensor-trail/model-core/modules/index';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';

const { useEditor } = await import('../src/editor/store');
const { canvasApi } = await import('../src/editor/agentBridge');
const projectSnapshot = () =>
  parseProject(serializeProject(useEditor.getState().project), registry);

beforeEach(() => {
  useEditor.setState({
    project: example(),
    hasProject: true,
    canvasVersion: 0,
    path: [],
    selected: 'transformer',
    selectedIds: ['transformer'],
    clipboard: null,
    clipboardOrigin: null,
    pasteCount: 0,
    past: [],
    future: [],
    notice: '',
  });
});

describe('编辑与文件序列化事务', () => {
  it('容器输入支持文字、背景、端口布局及撤销重做，JSON 可恢复', () => {
    useEditor.getState().newProject();
    useEditor.getState().add('DictInput', undefined, { x: 80, y: 100 });
    const id = useEditor.getState().selected!;
    const before = useEditor.getState().project;
    useEditor.getState().setParam(id, 'text', '图像和 mask\n可不填写维度');
    useEditor.getState().setTensorColor(id, '#172033');
    useEditor.getState().setPortSide(id, 'output', 'bottom');
    const after = useEditor.getState().project;
    expect(inferGraph(after.graph, registry).nodes[id].outputs.out).toEqual({
      kind: 'forward',
      text: '图像和 mask\n可不填写维度',
    });
    expect(projectSnapshot()).toEqual(after);
    useEditor.getState().undo();
    useEditor.getState().undo();
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    useEditor.getState().redo();
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('编辑保持文档身份，新建和打开切换身份，撤销重做恢复各自身份', () => {
    const original = useEditor.getState().documentId;
    useEditor.getState().renameProject('当前文件的修改');
    expect(useEditor.getState().documentId).toBe(original);
    useEditor.getState().newProject();
    const fresh = useEditor.getState().documentId;
    expect(fresh).not.toBe(original);
    useEditor.getState().undo();
    expect(useEditor.getState().documentId).toBe(original);
    useEditor.getState().redo();
    expect(useEditor.getState().documentId).toBe(fresh);
    useEditor.getState().load(example());
    expect(useEditor.getState().documentId).not.toBe(fresh);
  });
  it('AI 修改保持文件关联，新建模型可独立保存并可撤销返回原文件', () => {
    const original = useEditor.getState().documentId;
    const snapshot = canvasApi.getModel();
    const prepared = canvasApi.patchModel(snapshot.project, {
      operations: [{ op: 'rename_project', name: 'AI 改名' }],
    });
    expect(
      canvasApi.applyModel({ project: prepared.project, expectedRevision: snapshot.revision }).ok,
    ).toBe(true);
    expect(useEditor.getState().documentId).toBe(original);
    useEditor.getState().startNewDocument();
    expect(useEditor.getState().documentId).not.toBe(original);
    useEditor.getState().undo();
    expect(useEditor.getState().documentId).toBe(original);
  });
  it('新建空白项目保留变体，并可一次撤销恢复', () => {
    useEditor.getState().saveVariant('我的模块');
    const before = useEditor.getState().project;
    useEditor.getState().newProject();
    expect(useEditor.getState().project.graph).toEqual({ nodes: [], edges: [] });
    expect(useEditor.getState().project.variants).toEqual(before.variants);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
  });
  it('复制多个模块和内部连接，粘贴更新 ID 并作为一次撤销保存', () => {
    const s = useEditor.getState();
    s.selectMany(['input', 'transformer']);
    s.copy();
    const before = useEditor.getState().project;
    useEditor.getState().paste();
    const after = useEditor.getState().project;
    expect(after.graph.nodes).toHaveLength(before.graph.nodes.length + 2);
    expect(after.graph.edges).toHaveLength(before.graph.edges.length + 1);
    const copied = after.graph.nodes.slice(-2);
    expect(copied[0].id).not.toBe('input');
    expect(copied[1].subgraph).toEqual(before.graph.nodes[1].subgraph);
    expect(copied[1].subgraph).not.toBe(before.graph.nodes[1].subgraph);
    expect(after.graph.edges.at(-1)!.source).toBe(copied[0].id);
    expect(after.graph.edges.at(-1)!.target).toBe(copied[1].id);
    expect(useEditor.getState().selectedIds).toHaveLength(2);
    expect(projectSnapshot()).toEqual(after);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
  });
  it('子图边界默认独立复制，产生新的父模块接口', () => {
    const before = ports(useEditor.getState().project.graph.nodes[1], registry);
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().selectMany(['input', 'norm1']);
    useEditor.getState().copy();
    expect(useEditor.getState().clipboard!.nodes.map((n) => n.id)).toEqual(['input', 'norm1']);
    useEditor.getState().paste();
    const module = useEditor.getState().project.graph.nodes[1];
    expect(ports(module, registry).inputs).toHaveLength(before.inputs.length + 1);
    expect(ports(module, registry).outputs).toEqual(before.outputs);
    const pasted = module.subgraph!.nodes.slice(-2);
    expect(pasted[0]).toMatchObject({ type: 'Inlet' });
    expect(pasted[0].valueSource).toBeUndefined();
    expect(pasted[1]).toMatchObject({ type: 'LayerNorm' });
    expect(pasted[1].valueSource).toBeUndefined();
    expect(projectSnapshot()).toEqual(useEditor.getState().project);
  });
  it('普通算子仍可跨项目粘贴为独立模块', () => {
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().select('norm1');
    useEditor.getState().copy();
    useEditor.getState().newProject();
    useEditor.getState().paste();
    expect(useEditor.getState().project.graph.nodes[0].type).toBe('LayerNorm');
    expect(useEditor.getState().project.graph.nodes[0].valueSource).toBeUndefined();
  });
  it('按轴编辑含义，修改维数时同步输入框数量，边框色可保存恢复', () => {
    useEditor.getState().setParam('input', 'shape', '2, 6');
    useEditor.getState().setAxis('input', 1, '特征');
    useEditor.getState().setTensorColor('input', '#ede9fe');
    const project = useEditor.getState().project;
    expect(project.graph.nodes[0].params.axes).toBe('batch, 特征');
    expect(projectSnapshot().graph.nodes[0].tensorBorderColor).toBe('#ede9fe');
    useEditor.getState().setTensorColor('input', 'url(unsafe)');
    expect(useEditor.getState().project).toBe(project);
    useEditor.getState().setTensorColor('input');
    expect(useEditor.getState().project.graph.nodes[0].tensorBorderColor).toBeUndefined();
  });
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
    expect(projectSnapshot()).toEqual(after);
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
    expect(projectSnapshot()).toEqual(after);
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
    expect(projectSnapshot()).toEqual(after);
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
  it('AI 应用保留变体、一次撤销，可序列化为项目 JSON', () => {
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
    expect(projectSnapshot().name).toBe('AI 修改项目');
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
    expect(
      tensorValue(inferGraph(project.graph, registry).nodes[instance.id].outputs.out).shape,
    ).toEqual([8, 100, 64]);
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
    expect(() => projectSnapshot()).not.toThrow();
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
  it('导入失败不覆盖当前项目，正常编辑可保存为 JSON', () => {
    const previous = useEditor.getState().project;
    useEditor.getState().importText('{"schemaVersion":99}');
    expect(useEditor.getState().project).toBe(previous);
    useEditor.getState().setParam('input', 'shape', '2, 20, 64');
    const restored = projectSnapshot();
    expect(
      tensorValue(inferGraph(restored.graph, registry).nodes.transformer.outputs.out).shape,
    ).toEqual([2, 20, 64]);
  });
});
