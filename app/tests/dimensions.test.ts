import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { inferProject } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { isIntegerSpec, isTensorSpec, shapeText, type Project } from '../src/core/types';
import { parseProject, serializeProject } from '../src/core/serialization';
import { mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';
import { buildModel } from '../src/agent';
import { useEditor } from '../src/editor/store';

function project(shape = '2,49,2'): Project {
  return {
    schemaVersion: 1,
    name: 'shape 读取',
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape, axes: '', dtype: 'bool' }, 'x'),
        registry.create('TensorSize', { axes: '0,1', names: 'bs,A' }, 'size'),
        registry.create('Output', {}, 'bs'),
        registry.create('Output', {}, 'A'),
      ],
      edges: [
        { id: 'in', source: 'x', sourcePort: 'out', target: 'size', targetPort: 'in' },
        { id: 'bs', source: 'size', sourcePort: 'dim0', target: 'bs', targetPort: 'in' },
        { id: 'A', source: 'size', sourcePort: 'dim1', target: 'A', targetPort: 'in' },
      ],
    },
  };
}

describe('读取维度 Size', () => {
  it('shape[0:2] 分别产生整数，输出端口随参数生成且不继承输入 dtype', () => {
    const p = project(),
      r = inferProject(p, registry);
    expect(r.nodes.size.status).toBe('ok');
    expect(r.nodes.size.outputs).toEqual({
      dim0: { kind: 'int', value: 2 },
      dim1: { kind: 'int', value: 49 },
    });
    expect(ports(p.graph.nodes[1], registry).outputs).toEqual([
      { id: 'dim0', label: 'bs' },
      { id: 'dim1', label: 'A' },
    ]);
    expect(isIntegerSpec(r.nodes.bs.outputs.out)).toBe(true);
    expect(isTensorSpec(r.nodes.bs.outputs.out)).toBe(false);
    expect(shapeText(r.nodes.bs.outputs.out)).toBe('int · 2');
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
  });
  it('支持负轴、零尺寸和未确定的符号尺寸', () => {
    const result = registry.get('TensorSize')!.inferValue!(
      { in: { shape: ['B', 0, 2], axes: ['batch', 'empty', 'xy'], dtype: 'float32' } },
      { axes: '0,-2', names: 'bs,A' },
    );
    expect(result.outputs).toEqual({
      dim0: { kind: 'int', value: 'B' },
      dim1: { kind: 'int', value: 0 },
    });
    expect(shapeText({ kind: 'int', value: 'B' })).toContain('符号');
  });
  it.each(['0,3', '-4,1', '0,1.5', '0,missing', '', '0,1,2,3,4,5,6,7,8'])(
    '无效轴明确报错，不能伪造整数结果：%s',
    (axes) => {
      const p = project();
      p.graph.nodes[1].params.axes = axes;
      p.graph.edges = [];
      p.graph.edges.push({
        id: 'in',
        source: 'x',
        sourcePort: 'out',
        target: 'size',
        targetPort: 'in',
      });
      const result = inferProject(p, registry).nodes.size;
      expect(result.status).toBe('error');
      expect(result.outputs).toEqual({});
    },
  );
  it('零维 Tensor 不能读取轴 0', () => {
    const rule = registry.get('TensorSize')!.inferValue!;
    expect(() =>
      rule({ in: { shape: [], axes: [], dtype: 'float32' } }, { axes: '0', names: '' }),
    ).toThrow('范围');
  });
  it.each(['bs,bs', 'bs,A,extra', 'bad name,A', 'constructor,A'])(
    '拒绝歧义或无效输出名：%s',
    (names) => {
      const p = project();
      p.graph.nodes[1].params.names = names;
      expect(inferProject(p, registry).nodes.size.status).toBe('error');
    },
  );
  it('输入尺寸改变时动态更新，尺寸整数不能冒充 Tensor 接入张量算子', () => {
    const p = project('4,17,2');
    p.graph.nodes.push(registry.create('Slice', {}, 'slice'));
    p.graph.edges.push({
      id: 'invalid',
      source: 'size',
      sourcePort: 'dim0',
      target: 'slice',
      targetPort: 'in',
    });
    const result = inferProject(p, registry);
    expect(result.nodes.size.outputs.dim1).toEqual({ kind: 'int', value: 17 });
    expect(result.nodes.slice.status).toBe('error');
    expect(result.nodes.slice.messages[0]).toContain('需要 Tensor');
    expect(result.nodes.slice.messages[0]).not.toContain('字典');
  });
  it('与切片合并或拆开仍能传递两个整数端口', () => {
    const p = project();
    p.graph.nodes.push(registry.create('Slice', {}, 'slice'));
    p.graph.edges[0].target = 'slice';
    p.graph.edges.push({
      id: 'read',
      source: 'slice',
      sourcePort: 'out',
      target: 'size',
      targetPort: 'in',
    });
    const before = inferProject(p, registry);
    const id = mergeCalculations(p.graph, 'slice', 'size', registry);
    expect(inferProject(p, registry).nodes.A.outputs).toEqual(before.nodes.A.outputs);
    expect(ports(p.graph.nodes.find((n) => n.id === id)!, registry).outputs).toHaveLength(2);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    splitCalculationGroup(p.graph, id);
    expect(inferProject(p, registry).nodes.bs.outputs).toEqual(before.nodes.bs.outputs);
  });
  it('变量名称编辑不改变连线端口，保存和撤销恢复完整结果', () => {
    const p = project();
    useEditor.setState({ project: p, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().setParam('size', 'names', 'batch,agents');
    const edited = useEditor.getState().project;
    expect(edited.graph.edges).toEqual(p.graph.edges);
    expect(ports(edited.graph.nodes[1], registry).outputs[0]).toEqual({
      id: 'dim0',
      label: 'batch',
    });
    expect(inferProject(edited, registry).nodes.bs.outputs).toEqual({
      out: { kind: 'int', value: 2 },
    });
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(p);
  });
  it('API 配方也可创建动态尺寸端口和整数输出', () => {
    const result = buildModel({
      name: 'Size',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,49,2', axes: '' } },
        { id: 's', type: 'TensorSize', params: { axes: '0,1', names: 'bs,A' } },
        { id: 'bs', type: 'Output' },
        { id: 'A', type: 'Output' },
      ],
      connections: [
        ['x.out', 's.in'],
        ['s.dim0', 'bs.in'],
        ['s.dim1', 'A.in'],
      ],
    });
    expect(result.ok).toBe(true);
    expect(result.outputs.A).toEqual({ kind: 'int', value: 49 });
  });
});
