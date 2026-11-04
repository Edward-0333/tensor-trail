import { beforeEach, describe, expect, it } from 'vitest';
import { resolveForwardView } from '../src/core/hierarchy';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { useEditor } from '../src/editor/store';
import { canvasApi } from '../src/editor/agentBridge';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';

beforeEach(() => {
  useEditor.getState().load(example());
});

describe('forward 层级约定', () => {
  it('已有项目打开后直接显示根图，子模块留在调用层，深入时只显示对应子图', () => {
    const project = parseProject(serializeProject(example()), registry);
    const root = resolveForwardView(project.graph);
    expect(root.graph.nodes.map((node) => node.id)).toEqual(['input', 'transformer', 'output']);
    expect(root.levels).toHaveLength(1);
    const transformer = root.graph.nodes[1];
    expect(root.graph).toBe(project.graph);
    const child = resolveForwardView(project.graph, ['transformer']);
    expect(child.graph).toBe(transformer.subgraph);
    expect(child.graph.nodes.some((node) => node.id === 'residual1')).toBe(true);
    const attention = child.graph.nodes.find((node) => node.id === 'attention')!;
    const grandchild = resolveForwardView(project.graph, ['transformer', 'attention']);
    expect(grandchild.graph).toBe(attention.subgraph);
    expect(grandchild.levels.map((level) => level.path)).toEqual([
      [],
      ['transformer'],
      ['transformer', 'attention'],
    ]);
    expect(grandchild.current.title).toBe(`${attention.label} · forward()`);
  });

  it('逐层进入并返回不修改项目、父子端口、推导结果或撤销历史', () => {
    const state = useEditor.getState();
    const saved = serializeProject(state.project);
    const result = inferGraph(state.project.graph, registry);
    const boundary = ports(state.project.graph.nodes[1], registry);
    for (const path of [['transformer'], ['transformer', 'attention'], ['transformer'], []]) {
      useEditor.getState().navigate(path);
      expect(useEditor.getState().path).toEqual(path);
      expect(useEditor.getState().project).toBe(state.project);
      expect(useEditor.getState().past).toBe(state.past);
      expect(useEditor.getState().documentId).toBe(state.documentId);
    }
    expect(serializeProject(useEditor.getState().project)).toBe(saved);
    expect(ports(state.project.graph.nodes[1], registry)).toEqual(boundary);
    expect(inferGraph(state.project.graph, registry)).toEqual(result);
  });

  it('拒绝进入叶节点和失效路径，当前视图及选择保持不变', () => {
    const path = ['transformer'];
    useEditor.getState().navigate(path);
    path.push('outside_mutation');
    useEditor.getState().select('norm1');
    for (const invalid of [['missing'], ['input'], ['transformer', 'norm1']]) {
      useEditor.getState().navigate(invalid);
      expect(useEditor.getState().path).toEqual(['transformer']);
      expect(useEditor.getState().selected).toBe('norm1');
      expect(useEditor.getState().notice).toContain('没有可进入的 forward 子图');
    }
  });

  it.each(['load', 'import', 'new', 'api'])('%s 从深层切换项目后回到主 forward', (entry) => {
    useEditor.getState().navigate(['transformer', 'attention']);
    useEditor.getState().select('input');
    const project = example('mixed');
    if (entry === 'load') useEditor.getState().load(project);
    if (entry === 'import') useEditor.getState().importText(serializeProject(project));
    if (entry === 'new') useEditor.getState().newProject();
    if (entry === 'api') {
      const response = canvasApi.applyModel({
        project,
        expectedRevision: canvasApi.getModel().revision,
      });
      expect(response.ok).toBe(true);
    }
    const state = useEditor.getState();
    expect(state.hasProject).toBe(true);
    expect(state.path).toEqual([]);
    expect(state.selected).toBeNull();
    expect(state.selectedIds).toEqual([]);
    expect(resolveForwardView(state.project.graph, state.path).graph).toBe(state.project.graph);
    expect(state.project.graph.nodes).toHaveLength(entry === 'new' ? 0 : 7);
  });
});
