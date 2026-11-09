import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '@tensor-trail/model-core/modules/index';
import { tensorIndexLabel } from '@tensor-trail/model-core/modules/indexing';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { tensorValue, type Project, type ValueSpec } from '@tensor-trail/model-core/core/types';
import { useEditor } from '../src/editor/store';

function project(): Project {
  return {
    schemaVersion: 1,
    name: 'T 驱动历史切片',
    hyperparameters: { history_steps: { type: 'int', value: 21 } },
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,49,101,2', axes: 'batch,agent,time,xy' }, 'position'),
        registry.create('IntegerParameter', { expression: 'history_steps' }, 'parameter'),
        { ...registry.create('LocalParameter', {}, 'T'), label: 'T' },
        registry.create(
          'TensorIndex',
          {
            mode: 'slice',
            dim: '2',
            start: '',
            stop: '2',
            step: 1,
            stopFromInput: true,
            stopLabel: 'T',
          },
          'slice',
        ),
        registry.create('Output', {}, 'out'),
      ],
      edges: [
        {
          id: 'read',
          source: 'parameter',
          sourcePort: 'out',
          target: 'T',
          targetPort: 'in',
        },
        {
          id: 'tensor',
          source: 'position',
          sourcePort: 'out',
          target: 'slice',
          targetPort: 'in',
        },
        {
          id: 'bound',
          source: 'T',
          sourcePort: 'out',
          target: 'slice',
          targetPort: 'stop',
        },
        {
          id: 'result',
          source: 'slice',
          sourcePort: 'out',
          target: 'out',
          targetPort: 'in',
        },
      ],
    },
  };
}

const sliceNode = (p: Project) => p.graph.nodes.find((node) => node.id === 'slice')!;
const shape = (p: Project) => tensorValue(inferProject(p, registry).nodes.out.outputs.out).shape;

beforeEach(() => {
  useEditor.getState().load(project());
  useEditor.setState({ past: [], future: [] });
});

describe('整数参数与局部 T 的真实切片依赖', () => {
  it('超参数先产生整数，再经 T 连入 stop；形状和 dtype 跟随输入', () => {
    const p = project(),
      before = structuredClone(p);
    const result = inferProject(p, registry);
    expect(result.nodes.parameter.outputs.out).toEqual({
      kind: 'int',
      value: 21,
    });
    expect(result.nodes.T.outputs.out).toEqual({ kind: 'int', value: 21 });
    expect(result.nodes.slice.inputs.stop).toEqual({ kind: 'int', value: 21 });
    expect(result.nodes.out.outputs.out).toEqual({
      shape: [2, 49, 21, 2],
      axes: ['batch', 'agent', 'time', 'xy'],
      dtype: 'float32',
    });
    expect(ports(sliceNode(p), registry).inputs.map((port) => port.id)).toEqual(['in', 'stop']);
    expect(tensorIndexLabel(sliceNode(p).params)).toContain(':T');
    expect(p).toEqual(before);
    expect(p.hyperparameters).not.toHaveProperty('T');
  });

  it('修改既有 history_steps 会更新 T 和切片，保存与撤销重做保留依赖', () => {
    const before = useEditor.getState().project;
    expect(useEditor.getState().setHyperparameter('history_steps', { type: 'int', value: 7 })).toBe(
      true,
    );
    const changed = useEditor.getState().project;
    expect(shape(changed)).toEqual([2, 49, 7, 2]);
    expect(inferProject(changed, registry).nodes.T.outputs.out).toEqual({
      kind: 'int',
      value: 7,
    });
    expect(changed.graph.edges).toEqual(before.graph.edges);
    expect(changed.hyperparameters).not.toHaveProperty('T');
    expect(parseProject(serializeProject(changed), registry)).toEqual(changed);
    useEditor.getState().undo();
    expect(shape(useEditor.getState().project)).toEqual([2, 49, 21, 2]);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(changed);
  });

  it.each([
    [0, 0],
    [-1, 100],
    [-200, 0],
    [500, 101],
  ])('动态 stop=%s 遵循 Python 切片截断，长度为 %s', (stop, length) => {
    const p = project();
    p.hyperparameters!.history_steps.value = stop;
    expect(shape(p)).toEqual([2, 49, length, 2]);
  });

  it('输入模式不解析旧 stop 文本，stopLabel 只改变摘要而不能改变计算', () => {
    const p = project();
    sliceNode(p).params.stop = 'not_a_global_parameter';
    sliceNode(p).params.stopLabel = '999';
    expect(shape(p)).toEqual([2, 49, 21, 2]);
    expect(tensorIndexLabel(sliceNode(p).params)).toContain(':999');
    expect(
      registry.get('TensorIndex')!.getFields!(sliceNode(p).params).map((f) => f.key),
    ).not.toContain('stop');
    expect(
      registry.get('TensorIndex')!.getFields!(sliceNode(p).params).map((f) => f.key),
    ).toContain('stopLabel');
  });

  it('IntegerParameter 支持整数结果算式，包括浮点中间值', () => {
    const p = project();
    p.hyperparameters!.half = { type: 'float', value: 10.5 };
    const parameter = p.graph.nodes.find((n) => n.id === 'parameter')!;
    parameter.params.expression = 'self.history_steps - 1';
    expect(shape(p)).toEqual([2, 49, 20, 2]);
    parameter.params.expression = 'half * 2';
    expect(inferProject(p, registry).nodes.parameter.outputs.out).toEqual({
      kind: 'int',
      value: 21,
    });
    expect(shape(p)).toEqual([2, 49, 21, 2]);
  });

  it.each(['', 'missing', '21.5', '1 / 0', '1e309', '9007199254740992', 'history_steps()'])(
    '无效整数表达式 %s 不伪造 T',
    (expression) => {
      const p = project();
      p.graph.nodes.find((n) => n.id === 'parameter')!.params.expression = expression;
      const result = inferProject(p, registry);
      expect(result.nodes.parameter.status).toBe('error');
      expect(result.nodes.parameter.outputs).toEqual({});
      expect(result.nodes.T.status).toBe('blocked');
      expect(result.nodes.slice.status).toBe('blocked');
    },
  );

  it('stop 必须是具体安全整数，符号尺寸、浮点和 Tensor 不能冒充 Python 整数', () => {
    const rule = registry.get('TensorIndex')!.inferValue!;
    const source = {
      shape: [2, 49, 101, 2],
      axes: ['batch', 'agent', 'time', 'xy'],
      dtype: 'float32',
    };
    const params = sliceNode(project()).params;
    const invalid: ValueSpec[] = [
      { kind: 'int', value: 'T' },
      { kind: 'int', value: 21.5 },
      { kind: 'int', value: Infinity },
      { kind: 'int', value: Number.MAX_SAFE_INTEGER + 1 },
      { shape: [], axes: [], dtype: 'int64' },
      { kind: 'dict', fields: {} },
    ];
    for (const stop of invalid) expect(() => rule({ in: source, stop }, params)).toThrow();
    expect(() =>
      rule({ in: { kind: 'int', value: 101 }, stop: { kind: 'int', value: 21 } }, params),
    ).toThrow();
  });

  it('启用 stop 输入但未连接时等待上游，不退回文本 stop', () => {
    const p = project();
    p.graph.edges = p.graph.edges.filter((edge) => edge.id !== 'bound');
    const result = inferProject(p, registry);
    expect(result.nodes.slice.status).toBe('blocked');
    expect(result.nodes.slice.outputs).toEqual({});
    expect(result.nodes.out.status).toBe('blocked');
  });

  it('旧切片与整数索引默认不要求 stop 连线，原 infer API 继续工作', () => {
    const old = project();
    old.graph.edges = old.graph.edges.filter((edge) => edge.id !== 'bound');
    sliceNode(old).params = {
      mode: 'slice',
      dim: '2',
      start: '',
      stop: 'history_steps',
      step: 1,
    };
    const parsed = parseProject(serializeProject(old), registry);
    expect(sliceNode(parsed).params).toMatchObject({
      stopFromInput: false,
      stopLabel: 'stop',
    });
    expect(ports(sliceNode(parsed), registry).inputs.map((port) => port.id)).toEqual(['in']);
    expect(shape(parsed)).toEqual([2, 49, 21, 2]);
    const source = {
      shape: [2, 49, 101, 2],
      axes: ['batch', 'agent', 'time', 'xy'],
      dtype: 'float32',
    };
    expect(
      registry.get('TensorIndex')!.infer!({ in: source }, sliceNode(parsed).params, {
        hyperparameters: parsed.hyperparameters!,
      }).outputs.out.shape,
    ).toEqual([2, 49, 21, 2]);
    sliceNode(old).params = { dim: '2', index: 'history_steps - 1' };
    const indexed = parseProject(serializeProject(old), registry);
    expect(shape(indexed)).toEqual([2, 49, 2]);
    expect(ports(sliceNode(indexed), registry).inputs.map((port) => port.id)).toEqual(['in']);
  });

  it.each([
    ['stopFromInput', false],
    ['mode', 'index'],
  ] as const)('有连线时修改 %s 不能悄悄删除或忽略 stop 依赖', (key, value) => {
    const before = useEditor.getState();
    useEditor.getState().setParam('slice', key, value);
    const after = useEditor.getState();
    expect(after.project).toBe(before.project);
    expect(after.past).toBe(before.past);
    expect(after.notice).toMatch(/断开.*stop|stop.*断开/);
    expect(shape(after.project)).toEqual([2, 49, 21, 2]);
    useEditor.getState().removeEdges(['bound']);
    useEditor.getState().setParam('slice', key, value);
    expect(sliceNode(useEditor.getState().project).params[key]).toBe(value);
    expect(
      ports(sliceNode(useEditor.getState().project), registry).inputs.map((port) => port.id),
    ).toEqual(['in']);
    expect(inferProject(useEditor.getState().project, registry).nodes.slice.status).toBe('ok');
  });
});
