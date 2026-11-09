import { describe, expect, it } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent/index';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { canMergeCalculations } from '@tensor-trail/model-core/core/calculationGroups';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import type { Project, ValueSpec } from '@tensor-trail/model-core/core/types';
import { useEditor } from '../src/editor/store';
import { registry } from '@tensor-trail/model-core/modules/index';

function project(): Project {
  return {
    schemaVersion: 1,
    name: '中间张量',
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,49,21,2', axes: '' }, 'input'),
        registry.create('TensorIndex', { dim: '2', index: '20' }, 'index'),
        { ...registry.create('TensorVariable', {}, 'agent_pos'), label: 'agent_pos' },
        registry.create('TensorSize', { axes: '0,1', names: 'bs,A' }, 'size'),
        registry.create('Concat', { dim: 1 }, 'cat'),
        registry.create('Output', {}, 'result'),
      ],
      edges: [
        { id: 'a', source: 'input', sourcePort: 'out', target: 'index', targetPort: 'in' },
        { id: 'b', source: 'index', sourcePort: 'out', target: 'agent_pos', targetPort: 'in' },
        { id: 'c', source: 'agent_pos', sourcePort: 'out', target: 'size', targetPort: 'in' },
        { id: 'd', source: 'agent_pos', sourcePort: 'out', target: 'cat', targetPort: 'a' },
        { id: 'e', source: 'agent_pos', sourcePort: 'out', target: 'cat', targetPort: 'b' },
        { id: 'f', source: 'cat', sourcePort: 'out', target: 'result', targetPort: 'in' },
      ],
    },
  };
}

describe('张量变量', () => {
  it('中间 Tensor 可分支到 Cat 与 Size，形状随上游变化，保存不写死推导值', () => {
    const p = project(),
      before = structuredClone(p);
    const r = inferProject(p, registry);
    expect(r.issues).toEqual([]);
    expect(Object.values(r.nodes).every((n) => n.status === 'ok')).toBe(true);
    expect(r.nodes.agent_pos.outputs.out).toEqual(r.nodes.index.outputs.out);
    expect(r.nodes.result.outputs.out).toMatchObject({ shape: [2, 98, 2], dtype: 'float32' });
    expect(r.nodes.size.outputs.dim1).toEqual({ kind: 'int', value: 49 });
    expect(ports(p.graph.nodes[2], registry)).toEqual({
      inputs: [{ id: 'in', label: '输入' }],
      outputs: [{ id: 'out', label: '输出' }],
    });
    expect(ports(p.graph.nodes[5], registry).outputs).toEqual([]);
    expect(p).toEqual(before);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    p.graph.nodes[0].params.shape = '4,17,21,2';
    const changed = inferProject(p, registry);
    expect(changed.nodes.result.outputs.out).toMatchObject({ shape: [4, 34, 2] });
    expect(changed.nodes.size.outputs.dim1).toEqual({ kind: 'int', value: 17 });
    expect(canMergeCalculations(p.graph, 'index', 'agent_pos', registry)).toBe(false);
  });

  it('保留 Tensor 的标量、符号、零尺寸和 dtype，拒绝字典与 Python 整数', () => {
    const rule = registry.get('TensorVariable')!.inferValue!;
    for (const input of [
      { shape: [], axes: [], dtype: 'int64' },
      { shape: ['B', 0, 2], axes: ['batch', 'agent', 'xy'], dtype: 'float32' },
      { shape: [2, 49, 21], axes: ['batch', 'agent', 'time'], dtype: 'bool' },
    ]) {
      const out = rule({ in: input }, {}).outputs.out;
      expect(out).toEqual(input);
      expect(out).not.toBe(input);
    }
    for (const input of [
      { kind: 'int', value: 49 },
      { kind: 'dict', fields: {} },
    ] as ValueSpec[])
      expect(() => rule({ in: input }, {})).toThrow('张量变量需要上游 Tensor');
  });

  it('改名和撤销保持两侧连接，断开输入后所有分支停止使用旧形状', () => {
    const p = project();
    useEditor.setState({ project: p, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().renameNode('agent_pos', 'position');
    expect(useEditor.getState().project.graph.edges).toEqual(p.graph.edges);
    expect(inferProject(useEditor.getState().project, registry).nodes.result.outputs.out).toEqual(
      inferProject(p, registry).nodes.result.outputs.out,
    );
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(p);
    p.graph.edges = p.graph.edges.filter((e) => e.id !== 'b');
    const r = inferProject(p, registry);
    for (const id of ['agent_pos', 'size', 'cat', 'result']) {
      expect(r.nodes[id].status).toBe('blocked');
      expect(r.nodes[id].outputs).toEqual({});
    }
  });

  it('API 区分中间张量与网络最终输出，旧 Output 仍不可向后连接', () => {
    const result = buildModel({
      name: '张量变量',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2,49,2', axes: '' } },
        { id: 'position', type: 'TensorVariable', label: 'position' },
        { id: 'result', type: 'Output' },
      ],
      connections: [
        ['x.out', 'position.in'],
        ['position.out', 'result.in'],
      ],
    });
    expect(result.ok).toBe(true);
    expect(Object.keys(result.outputs)).toEqual(['result']);
    expect(result.outputs.result).toMatchObject({ shape: [2, 49, 2] });
    const p = project();
    p.graph.nodes[2].type = 'Output';
    expect(inferProject(p, registry).issues.join('\n')).toContain('没有输出端口 out');
  });
});
