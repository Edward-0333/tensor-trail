import { describe, expect, it } from 'vitest';
import type { Hyperparameters, Project } from '@tensor-trail/model-core/core/types';
import { inferProject, inferGraph } from '@tensor-trail/model-core/core/engine';
import { resolveIntegerExpression, validateHyperparameters } from '@tensor-trail/model-core/core/hyperparameters';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules/index';
import { useEditor } from '../src/editor/store';
import { buildModel, validateModel, getSchemas } from '@tensor-trail/model-core/agent/index';

const parameters: Hyperparameters = { history_steps: { type: 'int', value: 21 } };
const schema = {
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
        mask: { shape: [2, 49, 101], axes: ['batch', 'agent', 'time'], dtype: 'bool' },
      },
    },
    'key.with.dot': { shape: [], axes: [], dtype: 'int8' },
  },
};
function project(): Project {
  return {
    schemaVersion: 1,
    name: 'PLUTO agent_pos',
    hyperparameters: structuredClone(parameters),
    variants: [],
    graph: {
      nodes: [
        registry.create('ForwardInput', { schema: JSON.stringify(schema) }, 'data'),
        registry.create('DictionaryGet', { path: 'agent.position' }, 'field'),
        registry.create('TensorIndex', { dim: '2', index: 'self.history_steps - 1' }, 'agent_pos'),
        registry.create('Output', {}, 'output'),
      ],
      edges: [
        ['data', 'field'],
        ['field', 'agent_pos'],
        ['agent_pos', 'output'],
      ].map(([source, target], i) => ({
        id: `e${i}`,
        source,
        sourcePort: 'out',
        target,
        targetPort: 'in',
      })),
    },
  };
}

describe('超参数表达式', () => {
  it.each([
    ['history_steps - 1', 20],
    ['self.history_steps - 1', 20],
    ['-1', -1],
    ['(history_steps + 3) // 2', 12],
    ['-5 // 2', -3],
    ['-5 % 2', 1],
    ['2 * 3 + 4', 10],
    ['1e2 / 5', 20],
  ])('%s = %s', (expression, expected) =>
    expect(resolveIntegerExpression(expression, parameters)).toBe(expected),
  );
  it.each([
    'missing - 1',
    'history_steps / 2',
    '3 / 0',
    '2 ** 3',
    'Math.random()',
    'self.history_steps.toString()',
    'constructor',
    '(1 + 2',
    '',
    ' ',
    '1;alert(1)',
    '9007199254740992',
    '2 3',
  ])('拒绝无效表达式 %s', (expression) => {
    expect(() => resolveIntegerExpression(expression, parameters)).toThrow();
  });
  it('布尔与文本不隐式转换为数字', () => {
    for (const parameter of [
      { type: 'bool' as const, value: true },
      { type: 'string' as const, value: '21' },
    ])
      expect(() => resolveIntegerExpression('h - 1', { h: parameter })).toThrow('数值类型');
  });
  it('拒绝错误的超参数名称和类型，旧项目仍可打开', () => {
    for (const invalid of [
      null,
      [],
      { self: { type: 'int', value: 1 } },
      { h: { type: 'int', value: 1.5 } },
      { h: { type: 'float', value: Infinity } },
    ])
      expect(() => validateHyperparameters(invalid)).toThrow();
    const p = project();
    delete p.hyperparameters;
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    expect(inferProject(p, registry).nodes.agent_pos.messages.join()).toContain('未定义超参数');
  });
});

describe('字典字段读取与整数索引', () => {
  it('真实 PLUTO 尺寸链路正确，读取和索引均不修改输入', () => {
    const p = project();
    const before = structuredClone(p);
    const result = inferProject(p, registry);
    expect(result.nodes.field.outputs.out).toEqual(schema.fields.agent.fields.position);
    expect(result.nodes.agent_pos.outputs.out).toEqual({
      shape: [2, 49, 2],
      axes: ['batch', 'agent', 'xy'],
      dtype: 'float32',
    });
    expect(result.nodes.agent_pos.notes.join()).toContain('= 20');
    expect(validateModel(p).ok).toBe(true);
    expect(p).toEqual(before);
  });
  it('支持逐级取子字典以及含点号的原始键名', () => {
    const p = project();
    p.graph.nodes[1].params.path = 'agent';
    p.graph.nodes[2] = registry.create('DictionaryGet', { path: 'mask' }, 'agent_pos');
    expect(inferProject(p, registry).nodes.agent_pos.outputs.out).toEqual(
      schema.fields.agent.fields.mask,
    );
    p.graph.nodes[1].params.path = '["key.with.dot"]';
    expect(inferProject(p, registry).nodes.field.outputs.out).toEqual(
      schema.fields['key.with.dot'],
    );
    p.graph.nodes[1].params.path = 'agent.not_present';
    expect(inferProject(p, registry).nodes.field.messages.join()).toContain('不存在字段');
    p.graph.nodes[1].params.path = 'agent.position.shape';
    expect(inferProject(p, registry).nodes.field.messages.join()).toContain('需要字典');
  });
  it('负索引和负轴符合张量规则，边界错误会阻塞下游', () => {
    const p = project();
    p.graph.nodes[2].params = { dim: '-2', index: '-101' };
    expect(inferProject(p, registry).nodes.agent_pos.outputs.out).toEqual({
      shape: [2, 49, 2],
      axes: ['batch', 'agent', 'xy'],
      dtype: 'float32',
    });
    for (const index of ['101', '-102']) {
      p.graph.nodes[2].params.index = index;
      const r = inferProject(p, registry);
      expect(r.nodes.agent_pos.status).toBe('error');
      expect(r.nodes.output.status).toBe('blocked');
    }
    p.graph.nodes[2].params = { dim: '4', index: '0' };
    expect(inferProject(p, registry).nodes.agent_pos.messages.join()).toContain('轴 4');
  });
  it('一维索引产生标量；空轴和标量不能索引，符号维度提示待验证', () => {
    for (const [shape, status, output] of [
      [[1], 'ok', []],
      [[0], 'error', undefined],
      [[], 'error', undefined],
      [['T'], 'warning', []],
    ] as const) {
      const input = registry.create('DictionaryGet', { path: 'x' }, 'input');
      const data = registry.create(
        'ForwardInput',
        { schema: JSON.stringify({ kind: 'dict', fields: { x: { shape, dtype: 'bool' } } }) },
        'data',
      );
      const index = registry.create('TensorIndex', { dim: '0', index: '0' }, 'index');
      const r = inferGraph(
        {
          nodes: [data, input, index],
          edges: [
            { id: 'a', source: 'data', sourcePort: 'out', target: 'input', targetPort: 'in' },
            { id: 'b', source: 'input', sourcePort: 'out', target: 'index', targetPort: 'in' },
          ],
        },
        registry,
      );
      expect(r.nodes.index.status).toBe(status);
      if (output)
        expect(r.nodes.index.outputs.out).toEqual({ shape: output, axes: [], dtype: 'bool' });
    }
  });
  it('项目参数可编辑、撤销、删除并重新打开，data 快照不变', () => {
    useEditor.getState().load(project());
    const before = useEditor.getState().project;
    expect(
      useEditor.getState().setHyperparameter('history_steps', { type: 'int', value: 10 }),
    ).toBe(true);
    const after = useEditor.getState().project;
    expect(inferProject(after, registry).nodes.agent_pos.notes.join()).toContain('= 9');
    expect(after.graph).toEqual(before.graph);
    expect(inferProject(parseProject(serializeProject(after), registry), registry)).toEqual(
      inferProject(after, registry),
    );
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
    useEditor.getState().setHyperparameter('history_steps');
    expect(inferProject(useEditor.getState().project, registry).nodes.agent_pos.status).toBe(
      'error',
    );
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('子图继承项目超参数，编辑公共配置不把子图标记为自定义', () => {
    const p = project();
    const block = registry.fromVariant('index', {
      nodes: [
        registry.create('Inlet', {}, 'in'),
        p.graph.nodes[2],
        registry.create('Outlet', {}, 'out'),
      ],
      edges: [
        { id: 'a', source: 'in', sourcePort: 'out', target: 'agent_pos', targetPort: 'in' },
        { id: 'b', source: 'agent_pos', sourcePort: 'out', target: 'out', targetPort: 'in' },
      ],
    });
    block.id = 'agent_pos';
    p.graph.nodes[2] = block;
    expect(inferProject(p, registry).nodes.agent_pos.outputs.out).toEqual({
      shape: [2, 49, 2],
      axes: ['batch', 'agent', 'xy'],
      dtype: 'float32',
    });
    useEditor.getState().load(p);
    useEditor.getState().navigate(['agent_pos']);
    useEditor.getState().setHyperparameter('history_steps', { type: 'int', value: 3 });
    expect(useEditor.getState().project.graph).toEqual(p.graph);
    expect(
      inferProject(
        useEditor.getState().project,
        registry,
      ).nodes.agent_pos.child!.nodes.agent_pos.notes.join(),
    ).toContain('= 2');
  });
  it('AI 配方与项目校验同样使用超参数', () => {
    const p = project();
    const built = buildModel({
      name: p.name,
      hyperparameters: p.hyperparameters,
      nodes: p.graph.nodes.map(({ id, type, params }) => ({ id, type, params })),
      connections: p.graph.edges.map((e) => [
        `${e.source}.${e.sourcePort}`,
        `${e.target}.${e.targetPort}`,
      ]),
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(built.outputs.output).toEqual({
      shape: [2, 49, 2],
      axes: ['batch', 'agent', 'xy'],
      dtype: 'float32',
    });
    expect(JSON.stringify(getSchemas())).toContain('hyperparameters');
  });
});
