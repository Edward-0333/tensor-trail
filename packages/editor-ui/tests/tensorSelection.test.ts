import { beforeEach, describe, expect, it } from 'vitest';
import type { Params, Project } from '@tensor-trail/model-core/core/types';
import { tensorValue } from '@tensor-trail/model-core/core/types';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules/index';
import { tensorIndexLabel } from '@tensor-trail/model-core/modules/indexing';
import { useEditor } from '../src/editor/store';
import { buildModel, describeModules, getSchemas, validateModel } from '@tensor-trail/model-core/agent/index';

function project(params: Params = {}): Project {
  return {
    schemaVersion: 1,
    name: '索引与切片',
    variants: [],
    hyperparameters: { history_steps: { type: 'int', value: 21 }, axis: { type: 'int', value: 2 } },
    graph: {
      nodes: [
        registry.create(
          'Input',
          { shape: '2, 49, 101', axes: 'batch, agent, time', dtype: 'bool' },
          'x',
        ),
        {
          ...registry.create(
            'TensorIndex',
            { dim: '-1', index: '20', start: '20', stop: '21', ...params },
            'take',
          ),
          position: { x: 317, y: 85 },
          portLayout: { input: 'top', output: 'right' },
        },
        registry.create('Output', {}, 'y'),
      ],
      edges: [
        {
          id: 'read',
          source: 'x',
          sourcePort: 'out',
          target: 'take',
          targetPort: 'in',
          waypoints: [{ x: 10, y: 15 }],
        },
        { id: 'result', source: 'take', sourcePort: 'out', target: 'y', targetPort: 'in' },
      ],
    },
  };
}
const result = (p: Project) => inferProject(p, registry);

describe('统一索引与切片', () => {
  beforeEach(() => {
    useEditor.getState().load(project());
    useEditor.setState({ past: [], future: [], path: [] });
  });

  it.each([
    [{ mode: 'index', index: '20' }, [2, 49]],
    [{ mode: 'index', index: '-1' }, [2, 49]],
    [{ mode: 'slice', start: '20', stop: '21' }, [2, 49, 1]],
    [{ mode: 'slice', start: '-1', stop: '' }, [2, 49, 1]],
    [{ mode: 'slice', start: '', stop: '' }, [2, 49, 101]],
    [{ mode: 'slice', start: '0', stop: '21', step: 2 }, [2, 49, 11]],
    [{ mode: 'slice', start: '500', stop: '1000' }, [2, 49, 0]],
  ] as [Params, number[]][])('取值 %j 正确保留或移除维度', (params, shape) => {
    const p = project(params),
      before = structuredClone(p);
    const r = result(p);
    expect(r.nodes.y.outputs.out).toEqual({
      shape,
      axes: shape.length === 2 ? ['batch', 'agent'] : ['batch', 'agent', 'time'],
      dtype: 'bool',
    });
    expect(r.nodes.y.status).toBe('ok');
    expect(p).toEqual(before);
  });

  it('两种方式的轴与边界引用同一组超参数，空边界无需解析', () => {
    const p = project({ mode: 'slice', dim: 'axis', start: '', stop: 'self.history_steps' });
    expect(tensorValue(result(p).nodes.y.outputs.out).shape).toEqual([2, 49, 21]);
    p.hyperparameters!.history_steps.value = 10;
    expect(tensorValue(result(p).nodes.y.outputs.out).shape).toEqual([2, 49, 10]);
    p.graph.nodes[1].params.mode = 'index';
    p.graph.nodes[1].params.index = 'history_steps - 1';
    expect(tensorValue(result(p).nodes.y.outputs.out).shape).toEqual([2, 49]);
  });

  it('只校验并计算当前方式的参数，切回时仍能恢复原设置和错误提示', () => {
    const p = project({ mode: 'index', start: 'undefined_bound', step: 0 });
    expect(result(p).nodes.y.status).toBe('ok');
    expect(validateModel(p).ok).toBe(true);
    p.graph.nodes[1].params.mode = 'slice';
    expect(validateModel(p).ok).toBe(false);
    expect(result(p).nodes.take.status).toBe('error');
    expect(result(p).nodes.y.status).toBe('blocked');
    const s = project({ mode: 'slice', index: 'undefined_index' });
    expect(result(s).nodes.y.status).toBe('ok');
    s.graph.nodes[1].params.mode = 'index';
    expect(result(s).nodes.take.messages.join()).toContain('未定义超参数');
  });

  it('整数越界报错，切片裁剪边界；无效模式不能静默切换含义', () => {
    expect(result(project({ index: '101' })).nodes.take.status).toBe('error');
    expect(
      tensorValue(result(project({ mode: 'slice', start: '', stop: '1000' })).nodes.y.outputs.out)
        .shape,
    ).toEqual([2, 49, 101]);
    expect(result(project({ mode: 'other' })).nodes.take.status).toBe('error');
  });

  it('切换可撤销、重做，保持节点身份、端口布局、线路及两种设置', () => {
    const before = structuredClone(useEditor.getState().project);
    useEditor.getState().setParam('take', 'mode', 'slice');
    const after = useEditor.getState().project;
    expect(tensorValue(result(after).nodes.y.outputs.out).shape).toEqual([2, 49, 1]);
    expect(after.graph.edges).toEqual(before.graph.edges);
    expect(after.graph.nodes[1]).toEqual({
      ...before.graph.nodes[1],
      params: { ...before.graph.nodes[1].params, mode: 'slice' },
    });
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
    const loaded = parseProject(serializeProject(after), registry);
    expect(loaded).toEqual(after);
    loaded.graph.nodes[1].params.mode = 'index';
    expect(tensorValue(result(loaded).nodes.y.outputs.out).shape).toEqual([2, 49]);
  });

  it.each(['Slice', 'TensorIndex'])(
    '读取旧 %s 保留原有语义与布局，重复读写不会再次变化',
    (type) => {
      const old = project();
      const n = old.graph.nodes[1];
      n.type = type;
      n.label = '自定义取值';
      n.params =
        type === 'Slice'
          ? { dim: -1, start: '20', stop: 'self.history_steps', step: 1 }
          : { dim: '-1', index: 'self.history_steps - 1' };
      const loaded = parseProject(serializeProject(old), registry);
      expect(loaded.graph.nodes[1]).toMatchObject({
        id: n.id,
        label: n.label,
        position: n.position,
        portLayout: n.portLayout,
        type: 'TensorIndex',
        params: { mode: type === 'Slice' ? 'slice' : 'index', dim: '-1' },
      });
      expect(loaded.graph.edges).toEqual(old.graph.edges);
      expect(tensorValue(result(loaded).nodes.y.outputs.out).shape).toEqual(
        type === 'Slice' ? [2, 49, 1] : [2, 49],
      );
      expect(parseProject(serializeProject(loaded), registry)).toEqual(loaded);
      expect(old.graph.nodes[1].type).toBe(type);
    },
  );

  it('旧子图和保存的变体中的 Slice 也会迁移，外部端口和子图边界不变', () => {
    const p = project();
    const block = registry.create('Block', {}, 'block');
    const oldSlice = {
      ...registry.create('Slice', {}, 'slice'),
      type: 'Slice',
      label: '切片 Slice',
      params: { dim: -1, start: '', stop: '1', step: 1 },
    };
    const child = {
      nodes: [
        registry.create('Inlet', { port: 'in' }, 'inlet'),
        oldSlice,
        registry.create('Outlet', { port: 'out' }, 'outlet'),
      ],
      edges: [
        { id: 'a', source: 'inlet', sourcePort: 'out', target: 'slice', targetPort: 'in' },
        { id: 'b', source: 'slice', sourcePort: 'out', target: 'outlet', targetPort: 'in' },
      ],
    };
    block.subgraph = child;
    p.graph.nodes[1] = { ...block, id: 'take' };
    p.variants = [{ id: 'saved', name: '旧变体', graph: structuredClone(child) }];
    const loaded = parseProject(serializeProject(p), registry);
    for (const g of [loaded.graph.nodes[1].subgraph!, loaded.variants[0].graph]) {
      expect(g.nodes[1]).toMatchObject({
        type: 'TensorIndex',
        label: '索引与切片',
        params: { mode: 'slice', stop: '1' },
      });
      expect(g.edges).toEqual(child.edges);
    }
    expect(tensorValue(result(loaded).nodes.y.outputs.out).shape).toEqual([2, 49, 1]);
  });

  it('配方使用统一入口，并继续接受旧 Slice 配方', () => {
    for (const [type, params] of [
      ['Slice', { dim: -1, stop: '1' }],
      ['TensorIndex', { mode: 'slice', dim: '-1', stop: '1' }],
    ] as [string, Params][]) {
      const r = buildModel({
        name: '兼容配方',
        nodes: [
          { id: 'x', type: 'Input', params: { shape: '2, 3', axes: 'batch, feature' } },
          { id: 'take', type, params },
          { id: 'y', type: 'Output' },
        ],
        connections: [
          ['x.out', 'take.in'],
          ['take.out', 'y.in'],
        ],
      });
      expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
      expect(tensorValue(r.outputs.y).shape).toEqual([2, 1]);
      expect(r.project!.graph.nodes[1].type).toBe('TensorIndex');
    }
    expect(
      describeModules()
        .modules.filter((m) => ['TensorIndex', 'Slice'].includes(m.type))
        .map((m) => m.label),
    ).toEqual(['索引与切片']);
    expect(JSON.stringify(getSchemas())).toContain('"mode"');
  });

  it('表达式显示区分整数与范围，并保留超参数引用', () => {
    expect(tensorIndexLabel({ dim: '-1', mode: 'index', index: '2' })).toBe('[..., 2]');
    expect(tensorIndexLabel({ dim: '-1', mode: 'slice', start: '2', stop: '3', step: 1 })).toBe(
      '[..., 2:3]',
    );
    expect(tensorIndexLabel({ dim: '2', index: 'self.history_steps - 1' })).toBe(
      '[:, :, self.history_steps - 1, ...]',
    );
  });
});
