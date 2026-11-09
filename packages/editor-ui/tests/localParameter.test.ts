import { describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules/index';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { canMergeCalculations } from '@tensor-trail/model-core/core/calculationGroups';
import { buildModel } from '@tensor-trail/model-core/agent/index';
import { useEditor } from '../src/editor/store';
import type { Project, ValueSpec } from '@tensor-trail/model-core/core/types';

function project(): Project {
  return {
    schemaVersion: 1,
    name: '局部参数',
    variants: [],
    hyperparameters: { history_steps: { type: 'int', value: 21 } },
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,49,2', axes: '' }, 'x'),
        registry.create('TensorSize', { axes: '0,1', names: 'bs,A' }, 'size'),
        { ...registry.create('LocalParameter', {}, 'bs'), label: 'bs' },
        { ...registry.create('LocalParameter', {}, 'A'), label: 'A' },
        registry.create('Output', {}, 'result'),
      ],
      edges: [
        { id: 's', source: 'x', sourcePort: 'out', target: 'size', targetPort: 'in' },
        { id: 'b', source: 'size', sourcePort: 'dim0', target: 'bs', targetPort: 'in' },
        { id: 'a', source: 'size', sourcePort: 'dim1', target: 'A', targetPort: 'in' },
        { id: 'r', source: 'A', sourcePort: 'out', target: 'result', targetPort: 'in' },
      ],
    },
  };
}

describe('局部参数', () => {
  it('为派生的整数命名并继续向下游传值，保存不写死结果或新增超参数', () => {
    const p = project(),
      before = structuredClone(p);
    const r = inferProject(p, registry);
    expect(r.nodes.bs.outputs.out).toEqual({ kind: 'int', value: 2 });
    expect(r.nodes.result.outputs.out).toEqual({ kind: 'int', value: 49 });
    expect(ports(p.graph.nodes[3], registry)).toEqual({
      inputs: [{ id: 'in', label: '输入' }],
      outputs: [{ id: 'out', label: '输出' }],
    });
    expect(p).toEqual(before);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    p.graph.nodes[0].params.shape = '4,17,2';
    expect(inferProject(p, registry).nodes.bs.outputs.out).toEqual({ kind: 'int', value: 4 });
    expect(inferProject(p, registry).nodes.result.outputs.out).toEqual({ kind: 'int', value: 17 });
    expect(canMergeCalculations(p.graph, 'size', 'bs', registry)).toBe(false);
  });
  it('保留符号与零尺寸，不与零维 Tensor 混淆', () => {
    const rule = registry.get('LocalParameter')!.inferValue!;
    for (const value of ['B', 0]) {
      const input: ValueSpec = { kind: 'int', value };
      const out = rule({ in: input }, {}).outputs.out;
      expect(out).toEqual(input);
      expect(out).not.toBe(input);
    }
    for (const input of [
      { shape: [], axes: [], dtype: 'int64' },
      { shape: [2], axes: ['batch'], dtype: 'float32' },
      { kind: 'dict', fields: {} },
    ] as ValueSpec[])
      expect(() => rule({ in: input }, {})).toThrow('需要上游整数尺寸');
  });
  it('改名与撤销不改变端口、数值、超参数或连接', () => {
    const p = project();
    useEditor.setState({ project: p, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().renameNode('bs', 'batch');
    const edited = useEditor.getState().project;
    expect(edited.graph.nodes[2].label).toBe('batch');
    expect(edited.hyperparameters).toEqual(p.hyperparameters);
    expect(edited.graph.edges).toEqual(p.graph.edges);
    expect(inferProject(edited, registry).nodes.bs.outputs.out).toEqual({ kind: 'int', value: 2 });
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(p);
  });
  it('断开上游后停止产生旧值', () => {
    const p = project();
    p.graph.edges = p.graph.edges.filter((e) => e.id !== 'a');
    const r = inferProject(p, registry);
    expect(r.nodes.A.status).toBe('blocked');
    expect(r.nodes.A.outputs).toEqual({});
    expect(r.nodes.result.status).toBe('blocked');
  });
  it('API 配方可以创建并检查局部参数链路', () => {
    const result = buildModel({
      name: '局部参数',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,49,2', axes: '' } },
        { id: 's', type: 'TensorSize', params: { axes: '1', names: 'A' } },
        { id: 'A', type: 'LocalParameter', label: 'A' },
        { id: 'result', type: 'Output' },
      ],
      connections: [
        ['x.out', 's.in'],
        ['s.dim0', 'A.in'],
        ['A.out', 'result.in'],
      ],
    });
    expect(result.ok).toBe(true);
    expect(result.outputs.result).toEqual({ kind: 'int', value: 49 });
  });
});
