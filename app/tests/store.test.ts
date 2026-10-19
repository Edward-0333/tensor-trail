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

beforeEach(() => {
  memory.clear();
  useEditor.setState({
    project: example(),
    path: [],
    selected: 'transformer',
    past: [],
    future: [],
    notice: '',
    saveState: '本地保存',
  });
});

describe('编辑与持久化事务', () => {
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
