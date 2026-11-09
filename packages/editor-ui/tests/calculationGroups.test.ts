import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules/index';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import {
  canMergeCalculations,
  mergeCalculations,
  nearbyCalculation,
  splitCalculationGroup,
} from '@tensor-trail/model-core/core/calculationGroups';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { Graph, Project } from '@tensor-trail/model-core/core/types';
import { useEditor } from '../src/editor/store';

const edge = (source: string, target: string, targetPort = 'in', sourcePort = 'out') => ({
  id: `${source}-${target}-${targetPort}`,
  source,
  target,
  sourcePort,
  targetPort,
});
function example(): Project {
  return {
    schemaVersion: 1,
    name: '组合测试',
    variants: [],
    hyperparameters: { history_steps: { type: 'int', value: 21 } },
    graph: {
      nodes: [
        registry.create(
          'ForwardInput',
          {
            schema: JSON.stringify({
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
            }),
          },
          'data',
        ),
        registry.create('DictionaryGet', { path: 'agent.position' }, 'field'),
        registry.create('TensorIndex', { dim: '2', index: 'history_steps - 1' }, 'index'),
        registry.create('Slice', { dim: -1, start: '', stop: '1', step: 1 }, 'slice'),
        registry.create('Output', {}, 'output'),
      ],
      edges: [
        edge('data', 'field'),
        edge('field', 'index'),
        edge('index', 'slice'),
        edge('slice', 'output'),
      ],
    },
  };
}
// Bind the existing registry explicitly for inference.
function inferred(project: Project) {
  return inferProject(project, registry);
}

describe('计算节点合并', () => {
  it('只接受计算库中的直接相连节点', () => {
    const p = example();
    expect(canMergeCalculations(p.graph, 'field', 'index', registry)).toBe(true);
    expect(canMergeCalculations(p.graph, 'index', 'field', registry)).toBe(true);
    for (const pair of [
      ['data', 'field'],
      ['field', 'slice'],
      ['slice', 'output'],
      ['field', 'field'],
      ['bad', 'field'],
    ])
      expect(canMergeCalculations(p.graph, pair[0], pair[1], registry)).toBe(false);
  });
  it('合并、继续合并、序列化与拆开均保留字典类型、超参数和最终输出', () => {
    const p = example(),
      before = inferred(p).nodes.output.outputs;
    const id = mergeCalculations(p.graph, 'field', 'index', registry);
    expect(p.graph.nodes).toHaveLength(4);
    expect(inferred(p).nodes.output.outputs).toEqual(before);
    const group = p.graph.nodes.find((n) => n.id === id)!;
    expect(ports(group, registry)).toEqual({
      inputs: [expect.objectContaining({ id: 'in1' })],
      outputs: [expect.objectContaining({ id: 'out1' })],
    });
    const next = mergeCalculations(p.graph, id, 'slice', registry);
    const child = p.graph.nodes.find((n) => n.id === next)!.subgraph!;
    expect(child.nodes.map((n) => n.type)).not.toContain('CalculationGroup');
    expect(child.nodes.filter((n) => !['Inlet', 'Outlet'].includes(n.type))).toHaveLength(3);
    expect(inferred(p).nodes.output.outputs).toEqual(before);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    splitCalculationGroup(p.graph, next);
    expect(inferred(p).nodes.output.outputs).toEqual(before);
    expect(p.graph.nodes).toHaveLength(5);
    expect(p.graph.edges).toHaveLength(4);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
  });
  it('保留中间结果的外部分支以及多输入算子的其他输入', () => {
    const p = example();
    p.graph.nodes.push(
      registry.create('Output', {}, 'branch'),
      registry.create('Add', {}, 'add'),
      registry.create('Input', { shape: '2,49,1', axes: 'batch,agent,x' }, 'other'),
    );
    p.graph.edges = p.graph.edges.filter((e) => e.target !== 'output');
    p.graph.edges.push(
      edge('index', 'branch'),
      edge('slice', 'add', 'a'),
      edge('other', 'add', 'b'),
      edge('add', 'output'),
    );
    const before = inferred(p);
    expect(before.nodes.output.status).toBe('ok');
    const id = mergeCalculations(p.graph, 'index', 'slice', registry);
    expect(ports(p.graph.nodes.find((n) => n.id === id)!, registry).outputs).toHaveLength(2);
    const next = mergeCalculations(p.graph, id, 'add', registry);
    expect(ports(p.graph.nodes.find((n) => n.id === next)!, registry).inputs).toHaveLength(2);
    for (const target of ['output', 'branch'])
      expect(inferred(p).nodes[target].outputs).toEqual(before.nodes[target].outputs);
    splitCalculationGroup(p.graph, next);
    for (const target of ['output', 'branch'])
      expect(inferred(p).nodes[target].outputs).toEqual(before.nodes[target].outputs);
  });
  it('不会合并跨外部路径构成环路的节点', () => {
    const graph: Graph = {
      nodes: [
        registry.create('Slice', {}, 'a'),
        registry.create('Slice', {}, 'middle'),
        registry.create('Add', {}, 'b'),
      ],
      edges: [edge('a', 'middle'), edge('a', 'b', 'a'), edge('middle', 'b', 'b')],
    };
    expect(canMergeCalculations(graph, 'a', 'b', registry)).toBe(false);
    const before = structuredClone(graph);
    expect(() => mergeCalculations(graph, 'a', 'b', registry)).toThrow();
    expect(graph).toEqual(before);
  });
  it('未连接的输入和最终输出仍有外部端口', () => {
    const graph: Graph = {
      nodes: [registry.create('Slice', {}, 'a'), registry.create('Add', {}, 'b')],
      edges: [edge('a', 'b', 'a')],
    };
    const id = mergeCalculations(graph, 'a', 'b', registry);
    const p = ports(graph.nodes.find((n) => n.id === id)!, registry);
    expect(p.inputs).toHaveLength(2);
    expect(p.outputs).toHaveLength(1);
  });
  it('可合并两个已有组合并保持一层内部计算', () => {
    const p = example();
    p.graph.nodes.push(registry.create('Slice', {}, 'extra'));
    p.graph.edges = p.graph.edges.filter((e) => e.target !== 'output');
    p.graph.edges.push(edge('slice', 'extra'), edge('extra', 'output'));
    const before = inferred(p).nodes.output.outputs;
    const a = mergeCalculations(p.graph, 'field', 'index', registry);
    const b = mergeCalculations(p.graph, 'slice', 'extra', registry);
    const id = mergeCalculations(p.graph, a, b, registry);
    expect(inferred(p).nodes.output.outputs).toEqual(before);
    expect(
      p.graph.nodes
        .find((n) => n.id === id)!
        .subgraph!.nodes.filter((n) => n.type === 'CalculationGroup'),
    ).toHaveLength(0);
    splitCalculationGroup(p.graph, id);
    expect(inferred(p).nodes.output.outputs).toEqual(before);
  });
  it('靠近阈值按节点边缘计算，远处或无连接节点不触发', () => {
    const p = example();
    const a = { id: 'field', position: { x: 0, y: 0 }, width: 100, height: 40 };
    const b = { id: 'index', position: { x: 124, y: 0 }, width: 100, height: 40 };
    expect(nearbyCalculation(p.graph, a, [a, b], registry)).toBe('index');
    expect(
      nearbyCalculation(p.graph, a, [{ ...b, position: { x: 125, y: 0 } }], registry),
    ).toBeUndefined();
    expect(
      nearbyCalculation(p.graph, a, [{ ...b, id: 'slice', position: { x: 80, y: 0 } }], registry),
    ).toBeUndefined();
  });
});

describe('拖动合并事务', () => {
  beforeEach(() => {
    useEditor.setState({
      project: example(),
      hasProject: true,
      path: [],
      past: [],
      future: [],
      selected: null,
      selectedIds: [],
    });
  });
  it('拖动与合并一次撤销恢复，重做、内部编辑、拆开可持久保存', () => {
    const before = structuredClone(useEditor.getState().project);
    useEditor
      .getState()
      .finishNodeDrag([{ id: 'index', position: { x: 120, y: 100 } }], ['index', 'field']);
    const after = useEditor.getState().project,
      id = useEditor.getState().selected!;
    expect(useEditor.getState().past).toHaveLength(1);
    expect(inferred(after).nodes.output.outputs).toEqual(inferred(before).nodes.output.outputs);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
    useEditor.getState().navigate([id]);
    useEditor.getState().setParam('index', 'index', '0');
    expect(
      useEditor
        .getState()
        .project.graph.nodes.find((n) => n.id === id)!
        .subgraph!.nodes.find((n) => n.id === 'index')!.params.index,
    ).toBe('0');
    useEditor.getState().navigate([]);
    useEditor.getState().splitCalculation(id);
    const split = useEditor.getState().project;
    expect(split.graph.nodes.find((n) => n.type === 'TensorIndex')!.params.index).toBe('0');
    expect(parseProject(serializeProject(split), registry)).toEqual(split);
    expect(inferred(split).nodes.output.status).toBe('ok');
  });
});
