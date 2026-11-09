import { beforeEach, describe, expect, it } from 'vitest';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import type { Project } from '@tensor-trail/model-core/core/types';
import { useEditor } from '../src/editor/store';
import { registry } from '@tensor-trail/model-core/modules/index';

function joiningProject(type = 'Concat', legacy = false): Project {
  const source = registry.create('Input', { shape: '2, 3', axes: 'batch, feature' }, 'source');
  const join = registry.create(type, { dim: -1, input_count: legacy ? 2 : 3 }, 'join');
  if (legacy) join.params = { dim: -1 };
  return {
    schemaVersion: 1,
    name: '动态接口编辑',
    variants: [],
    graph: {
      nodes: [source, join],
      edges: (legacy ? ['a', 'b'] : ['a', 'b', 'c']).map((port) => ({
        id: `edge_${port}`,
        source: 'source',
        sourcePort: 'out',
        target: 'join',
        targetPort: port,
      })),
    },
  };
}

function viewProject(): Project {
  return {
    schemaVersion: 1,
    name: 'View 动态接口',
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '6, 4', axes: 'batch, feature' }, 'source'),
        registry.create('IntegerParameter', { expression: '2' }, 'size0'),
        registry.create('IntegerParameter', { expression: '3' }, 'size1'),
        registry.create('View', { shape: 'd0, d1, -1' }, 'view'),
      ],
      edges: [
        { id: 'tensor', source: 'source', sourcePort: 'out', target: 'view', targetPort: 'in' },
        { id: 'size0', source: 'size0', sourcePort: 'out', target: 'view', targetPort: 'd0' },
        { id: 'size1', source: 'size1', sourcePort: 'out', target: 'view', targetPort: 'd1' },
      ],
    },
  };
}

beforeEach(() => {
  useEditor.setState({
    project: joiningProject(),
    hasProject: true,
    path: [],
    selected: null,
    selectedIds: [],
    clipboard: null,
    clipboardOrigin: null,
    pasteCount: 0,
    past: [],
    future: [],
    notice: '',
  });
});

const currentNode = (id: string) =>
  useEditor.getState().project.graph.nodes.find((node) => node.id === id)!;

describe('动态输入端口的原子编辑', () => {
  it('旧 Concat 仅含 dim 仍按两路推导，可增加第三个输入且保存恢复', () => {
    useEditor.setState({ project: joiningProject('Concat', true) });
    const before = useEditor.getState().project;
    expect(currentNode('join').params).toEqual({ dim: -1 });
    expect(ports(currentNode('join'), registry).inputs.map((port) => port.id)).toEqual(['a', 'b']);
    expect(inferProject(before, registry).nodes.join).toMatchObject({
      status: 'ok',
      outputs: { out: { shape: [2, 6] } },
    });
    useEditor.getState().setParam('join', 'input_count', 3);
    expect(useEditor.getState().project).not.toBe(before);
    expect(useEditor.getState().past).toHaveLength(1);
    expect(ports(currentNode('join'), registry).inputs.map((port) => port.id)).toEqual([
      'a',
      'b',
      'c',
    ]);
    expect(inferProject(useEditor.getState().project, registry).nodes.join.status).toBe('blocked');
    useEditor
      .getState()
      .connect({ source: 'source', sourcePort: 'out', target: 'join', targetPort: 'c' });
    const saved = parseProject(serializeProject(useEditor.getState().project), registry);
    expect(saved.graph.nodes.find((node) => node.id === 'join')!.params.input_count).toBe(3);
    expect(inferProject(saved, registry).nodes.join).toMatchObject({
      status: 'ok',
      outputs: { out: { shape: [2, 9] } },
    });
  });

  it.each(['Concat', 'Stack'])(
    '%s 已连接 c 时减少输入失败，保留项目、连线和撤销重做历史',
    (type) => {
      useEditor.setState({ project: joiningProject(type) });
      useEditor.getState().renameNode('source', '可重做的名称');
      useEditor.getState().undo();
      const before = useEditor.getState();
      expect(before.future).toHaveLength(1);
      useEditor.getState().setParam('join', 'input_count', 2);
      const rejected = useEditor.getState();
      expect(rejected.project).toBe(before.project);
      expect(rejected.past).toBe(before.past);
      expect(rejected.future).toBe(before.future);
      expect(rejected.documentId).toBe(before.documentId);
      expect(currentNode('join').params.input_count).toBe(3);
      expect(rejected.project.graph.edges.map((edge) => edge.targetPort)).toEqual(['a', 'b', 'c']);
      expect(rejected.notice).toContain('先断开输入 c');

      useEditor.getState().removeEdges(['edge_c']);
      useEditor.getState().setParam('join', 'input_count', 2);
      expect(currentNode('join').params.input_count).toBe(2);
      expect(ports(currentNode('join'), registry).inputs.map((port) => port.id)).toEqual([
        'a',
        'b',
      ]);
      expect(useEditor.getState().project.graph.edges.map((edge) => edge.id)).toEqual([
        'edge_a',
        'edge_b',
      ]);
      expect(inferProject(useEditor.getState().project, registry).nodes.join.status).toBe('ok');
      useEditor.getState().undo();
      expect(currentNode('join').params.input_count).toBe(3);
      useEditor.getState().undo();
      expect(useEditor.getState().project).toEqual(before.project);
    },
  );

  it('View 已连接 d1 时不能从 shape 中移除，断线后才可成功修改', () => {
    useEditor.setState({ project: viewProject() });
    const before = useEditor.getState();
    expect(inferProject(before.project, registry).nodes.view.outputs.out).toMatchObject({
      shape: [2, 3, 4],
    });
    useEditor.getState().setParam('view', 'shape', 'd0, -1');
    expect(useEditor.getState().project).toBe(before.project);
    expect(useEditor.getState().past).toBe(before.past);
    expect(useEditor.getState().future).toBe(before.future);
    expect(currentNode('view').params.shape).toBe('d0, d1, -1');
    expect(useEditor.getState().project.graph.edges).toEqual(before.project.graph.edges);
    expect(useEditor.getState().notice).toContain('先断开输入 d1');

    useEditor.getState().removeEdges(['size1']);
    useEditor.getState().setParam('view', 'shape', 'd0, -1');
    expect(currentNode('view').params.shape).toBe('d0, -1');
    expect(ports(currentNode('view'), registry).inputs.map((port) => port.id)).toEqual([
      'in',
      'd0',
    ]);
    expect(useEditor.getState().project.graph.edges.map((edge) => edge.id)).toEqual([
      'tensor',
      'size0',
    ]);
    expect(inferProject(useEditor.getState().project, registry).nodes.view).toMatchObject({
      status: 'ok',
      outputs: { out: { shape: [2, 12] } },
    });
    useEditor.getState().undo();
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before.project);
  });

  it('View 只调整目标轴顺序时保留 d0/d1 连线，可正常修改与保存', () => {
    useEditor.setState({ project: viewProject() });
    const edges = useEditor.getState().project.graph.edges;
    useEditor.getState().setParam('view', 'shape', 'd1, d0, -1');
    expect(currentNode('view').params.shape).toBe('d1, d0, -1');
    expect(useEditor.getState().project.graph.edges).toEqual(edges);
    expect(useEditor.getState().past).toHaveLength(1);
    const saved = parseProject(serializeProject(useEditor.getState().project), registry);
    expect(inferProject(saved, registry).nodes.view).toMatchObject({
      status: 'ok',
      outputs: { out: { shape: [3, 2, 4] } },
    });
  });
});
