import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { libraryCategory } from '../src/modules/libraryCategories';
import { ports } from '../src/core/graph';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Project } from '../src/core/types';
import { useEditor } from '../src/editor/store';

const inputSchema = {
  kind: 'dict',
  fields: {
    agent: {
      kind: 'dict',
      fields: {
        position: {
          shape: [2, 49, 101, 2],
          axes: ['batch', 'agent', 'time', 'xy'],
          dtype: 'float32',
        },
      },
    },
  },
};

function project(): Project {
  return {
    schemaVersion: 1,
    name: '子画布接口',
    variants: [],
    graph: {
      nodes: [
        registry.create('ForwardInput', { schema: JSON.stringify(inputSchema) }, 'data'),
        registry.create('Submodule', {}, 'encoder'),
      ],
      edges: [
        {
          id: 'call',
          source: 'data',
          sourcePort: 'out',
          target: 'encoder',
          targetPort: 'data',
        },
      ],
    },
  };
}

const owner = () => useEditor.getState().project.graph.nodes.find((node) => node.id === 'encoder')!;
const child = () => owner().subgraph!;

beforeEach(() => useEditor.getState().load(project()));

describe('子画布的模块输入与模块输出', () => {
  it('边界属于输入输出展示分类，常规目录仍隐藏，内部目录显式包含', () => {
    for (const type of ['Inlet', 'Outlet']) {
      expect(libraryCategory(registry.get(type)!)).toBe('输入输出');
      expect(registry.list().some((def) => def.type === type)).toBe(false);
      expect(registry.list(true).filter((def) => def.type === type)).toHaveLength(1);
    }
  });

  it('重复添加两类接口自动分配唯一端口，父层接口同步且保留既有 data', () => {
    const before = structuredClone(useEditor.getState().project);
    const originalData = structuredClone(child().nodes.find((node) => node.id === 'data'));
    useEditor.getState().navigate(['encoder']);
    for (let i = 0; i < 3; i++)
      useEditor.getState().add('Inlet', undefined, { x: 0, y: 100 + i * 80 });
    for (let i = 0; i < 2; i++)
      useEditor.getState().add('Outlet', undefined, { x: 500, y: 100 + i * 80 });
    expect(
      child()
        .nodes.filter((node) => node.type === 'Inlet')
        .map((node) => node.params.port),
    ).toEqual(['data', 'in', 'in2', 'in3']);
    expect(
      child()
        .nodes.filter((node) => node.type === 'Outlet')
        .map((node) => node.params.port),
    ).toEqual(['out', 'out2']);
    expect(
      child()
        .nodes.filter((node) => node.id !== 'data')
        .map((node) => node.label),
    ).toEqual(['in', 'in2', 'in3', 'out', 'out2']);
    expect(ports(owner(), registry)).toEqual({
      inputs: [
        { id: 'data', label: 'data' },
        { id: 'in', label: 'in' },
        { id: 'in2', label: 'in2' },
        { id: 'in3', label: 'in3' },
      ],
      outputs: [
        { id: 'out', label: 'out' },
        { id: 'out2', label: 'out2' },
      ],
    });
    expect(child().nodes.find((node) => node.id === 'data')).toEqual(originalData);
    expect(useEditor.getState().project.graph.edges).toEqual(before.graph.edges);
    expect(parseProject(serializeProject(useEditor.getState().project), registry)).toEqual(
      useEditor.getState().project,
    );
  });

  it('输入和输出分别判重，不因另一方向的同名端口跳号', () => {
    const p = project(),
      inner = p.graph.nodes[1].subgraph!;
    inner.nodes.push(
      registry.create('Inlet', { port: 'out' }, 'existing_in'),
      registry.create('Outlet', { port: 'in' }, 'existing_out'),
    );
    useEditor.getState().load(p);
    useEditor.getState().navigate(['encoder']);
    useEditor.getState().add('Inlet', undefined, { x: 0, y: 200 });
    expect(child().nodes.at(-1)).toMatchObject({
      type: 'Inlet',
      label: 'in',
      params: { port: 'in' },
    });
    useEditor.getState().add('Outlet', undefined, { x: 400, y: 200 });
    expect(child().nodes.at(-1)).toMatchObject({
      type: 'Outlet',
      label: 'out',
      params: { port: 'out' },
    });
    expect(parseProject(serializeProject(useEditor.getState().project), registry)).toEqual(
      useEditor.getState().project,
    );
  });

  it.each(['Inlet', 'Outlet'])('主层拒绝添加 %s，不产生节点或撤销记录', (type) => {
    const before = useEditor.getState();
    useEditor.getState().add(type, undefined, { x: 100, y: 200 });
    const after = useEditor.getState();
    expect(after.path).toEqual([]);
    expect(after.project).toBe(before.project);
    expect(after.past).toBe(before.past);
    expect(after.notice).toMatch(/子|内部/);
  });

  it('新增接口可撤销重做并保存重开，不损坏父层连接', () => {
    useEditor.getState().navigate(['encoder']);
    useEditor.getState().add('Inlet', undefined, { x: 0, y: 160 });
    const beforeOutlet = useEditor.getState().project;
    useEditor.getState().add('Outlet', undefined, { x: 500, y: 160 });
    const afterOutlet = useEditor.getState().project;
    expect(ports(owner(), registry).outputs).toEqual([{ id: 'out', label: 'out' }]);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(beforeOutlet);
    expect(ports(owner(), registry).outputs).toEqual([]);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(afterOutlet);
    const reopened = parseProject(serializeProject(afterOutlet), registry);
    useEditor.getState().load(reopened);
    expect(useEditor.getState().project).toEqual(afterOutlet);
    expect(ports(owner(), registry).outputs).toEqual([{ id: 'out', label: 'out' }]);
    expect(reopened.graph.edges).toEqual(project().graph.edges);
  });

  it('新 Outlet 连线后把内部数据交还父层，改显示名不改变端口或连线', () => {
    useEditor.getState().navigate(['encoder']);
    useEditor.getState().add('Outlet', undefined, { x: 500, y: 0 });
    const outlet = child().nodes.find((node) => node.type === 'Outlet')!;
    useEditor.getState().connect({
      source: 'data',
      sourcePort: 'out',
      target: outlet.id,
      targetPort: 'in',
    });
    const beforeRename = structuredClone(useEditor.getState().project);
    const completed = inferProject(useEditor.getState().project, registry).nodes.encoder;
    expect(completed.status).toBe('ok');
    expect(completed.outputs.out).toEqual(inputSchema);
    useEditor.getState().renameNode('data', '场景 data');
    useEditor.getState().renameNode(outlet.id, '返回数据');
    const renamed = useEditor.getState().project;
    expect(renamed.graph.edges).toEqual(beforeRename.graph.edges);
    expect(child().edges).toEqual(beforeRename.graph.nodes[1].subgraph!.edges);
    expect(ports(owner(), registry)).toEqual({
      inputs: [{ id: 'data', label: '场景 data' }],
      outputs: [{ id: 'out', label: '返回数据' }],
    });
    expect(inferProject(renamed, registry).nodes.encoder.outputs.out).toEqual(inputSchema);
    expect(parseProject(serializeProject(renamed), registry)).toEqual(renamed);
  });
});
