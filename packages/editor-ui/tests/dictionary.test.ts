import { describe, expect, it } from 'vitest';
import {
  DICTIONARY_SCHEMA_LIMIT,
  dictionaryShape,
  emptyDictionary,
  parseDictionarySchema,
} from '@tensor-trail/model-core/core/dictionary';
import type { DictionarySpec, Graph } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, tensorValue } from '@tensor-trail/model-core/core/types';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules/index';
import { plutoForwardInputExample, plutoInputSchema } from '@tensor-trail/model-core/modules/plutoInputExample';
import { buildModel, describeModules, getSchemas, validateModel } from '@tensor-trail/model-core/agent/index';
import { copySelection, duplicateSelection } from '@tensor-trail/model-core/editing/clipboard';
import { useEditor } from '../src/editor/store';

function projectWithOutput() {
  const project = plutoForwardInputExample();
  project.graph.nodes.push(registry.create('Output', {}, 'output'));
  project.graph.edges.push({
    id: 'edge',
    source: 'data',
    sourcePort: 'out',
    target: 'output',
    targetPort: 'in',
  });
  return project;
}

describe('嵌套字典输入', () => {
  it('网络输入保存真实字典类型、尺寸和备注，沿连线传递并支持重新打开', () => {
    const schema = JSON.stringify(plutoInputSchema());
    expect(schema.length).toBeGreaterThan(1999);
    const built = buildModel({
      name: 'data 网络输入',
      nodes: [
        { id: 'data', type: 'ForwardInput', label: 'data', params: { schema, text: '训练输入' } },
        { id: 'output', type: 'Output' },
      ],
      connections: [['data.out', 'output.in']],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const restored = parseProject(serializeProject(built.project!), registry);
    expect(restored.graph.nodes[0]).toMatchObject({
      type: 'ForwardInput',
      label: 'data',
      params: { schema: plutoInputSchema(), text: '训练输入' },
    });
    expect(inferGraph(restored.graph, registry).nodes.output.outputs.out).toEqual(
      plutoInputSchema(),
    );
    const catalog = describeModules().modules.find((module) => module.type === 'ForwardInput')!;
    expect(catalog.fields.find((field) => field.key === 'schema')!.maxLength).toBe(
      DICTIONARY_SCHEMA_LIMIT,
    );
  });

  it('没有 schema 的旧网络输入仍可打开，非法 schema 会报错而非退回文字标记', () => {
    const input = registry.create('ForwardInput', { text: '原来的备注' }, 'data');
    delete input.params.schema;
    const project = {
      schemaVersion: 1 as const,
      name: '旧输入',
      variants: [],
      graph: { nodes: [input], edges: [] },
    };
    const restored = parseProject(serializeProject(project), registry);
    expect(inferGraph(restored.graph, registry).nodes.data.outputs.out).toEqual({
      kind: 'forward',
      text: '原来的备注',
    });
    restored.graph.nodes[0].params.schema = '{bad schema}';
    const invalid = inferGraph(restored.graph, registry).nodes.data;
    expect(invalid.status).toBe('error');
    expect(invalid.outputs).toEqual({});
  });

  it('描述 PLUTO 字段，区分整份字典、Tensor、布尔掩码和输入总时间维度', () => {
    const project = plutoForwardInputExample();
    const result = inferGraph(project.graph, registry).nodes.data;
    expect(result.status).toBe('ok');
    const data = result.outputs.out;
    expect(isDictionarySpec(data)).toBe(true);
    if (!isDictionarySpec(data)) throw new Error('expected dict');
    expect(Object.keys(data.fields)).toEqual([
      'agent',
      'map',
      'static_objects',
      'reference_line',
      'current_state',
    ]);
    const agent = data.fields.agent as DictionarySpec;
    expect(tensorValue(agent.fields.position).shape).toEqual(['B', 'A', 'T', 2]);
    expect(tensorValue(agent.fields.valid_mask).dtype).toBe('bool');
    expect(tensorValue(agent.fields.category).dtype).toBe('int8');
    expect(() => tensorValue(data)).toThrow('需要 Tensor');
    expect(data).not.toHaveProperty('shape');
  });

  it('空字典、标量、空 Tensor 和省略的轴名均可明确表示', () => {
    expect(parseDictionarySchema(JSON.stringify(emptyDictionary()))).toEqual(emptyDictionary());
    const value = parseDictionarySchema(
      JSON.stringify({
        kind: 'dict',
        fields: {
          scalar: { shape: [], dtype: 'bool' },
          empty: { shape: ['B', 0, 2], dtype: 'float32' },
        },
      }),
    );
    expect(value.fields.scalar).toEqual({ shape: [], axes: [], dtype: 'bool' });
    expect(value.fields.empty).toEqual({
      shape: ['B', 0, 2],
      axes: ['axis0', 'axis1', 'axis2'],
      dtype: 'float32',
    });
    expect(dictionaryShape('[]')).toEqual([]);
    expect(dictionaryShape(' B, 0, 2 ')).toEqual(['B', 0, 2]);
    expect(() => dictionaryShape('')).toThrow();
    expect(() => dictionaryShape('B, -1')).toThrow();
  });

  it('同一字典经过自定义子模块的 Inlet、Identity、Outlet 后完整返回', () => {
    const input = registry.create(
      'DictionaryInput',
      { schema: JSON.stringify(plutoInputSchema()) },
      'data',
    );
    const block = registry.fromVariant('接收 data 的子模块', {
      nodes: [
        registry.create('Inlet', {}, 'in'),
        registry.create('Identity', {}, 'identity'),
        registry.create('Outlet', {}, 'out'),
      ],
      edges: [
        { id: 'a', source: 'in', sourcePort: 'out', target: 'identity', targetPort: 'in' },
        { id: 'b', source: 'identity', sourcePort: 'out', target: 'out', targetPort: 'in' },
      ],
    });
    block.id = 'block';
    const graph: Graph = {
      nodes: [input, block],
      edges: [{ id: 'edge', source: 'data', sourcePort: 'out', target: 'block', targetPort: 'in' }],
    };
    const result = inferGraph(graph, registry);
    expect(result.nodes.block.status).toBe('ok');
    expect(result.nodes.block.child!.nodes.identity.outputs.out).toEqual(
      result.nodes.data.outputs.out,
    );
    expect(result.nodes.block.outputs.out).toEqual(result.nodes.data.outputs.out);
  });

  it('字典不能直接参与 Tensor 计算，下游给出字段读取提示', () => {
    const project = projectWithOutput();
    project.graph.nodes[1] = registry.create('Linear', {}, 'output');
    const result = inferGraph(project.graph, registry);
    expect(result.nodes.data.status).toBe('ok');
    expect(result.nodes.output.status).toBe('error');
    expect(result.nodes.output.messages[0]).toContain('需先取出 Tensor 字段');
    expect(result.nodes.output.outputs).toEqual({});
  });

  it('大于旧短文本限制的结构支持项目保存、复制、API 构建和 Schema', () => {
    const project = projectWithOutput();
    const schema = project.graph.nodes[0].params.schema;
    expect(JSON.stringify(schema).length).toBeGreaterThan(1999);
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
    expect(validateModel(project).ok).toBe(true);
    const copied = duplicateSelection(copySelection(project.graph, ['data', 'output']), 32);
    expect(copied.nodes[0].params.schema).toEqual(schema);
    expect(copied.nodes[0].id).not.toBe('data');
    expect(inferGraph(copied, registry).nodes[copied.nodes[1].id].outputs.out).toEqual(
      plutoInputSchema(),
    );
    const built = buildModel({
      name: '字典接口',
      nodes: [
        { id: 'data', type: 'DictionaryInput', params: { schema } },
        { id: 'output', type: 'Output' },
      ],
      connections: [['data.out', 'output.in']],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(built.outputs.output).toEqual(plutoInputSchema());
    const catalog = describeModules().modules;
    expect(catalog.some((module) => module.type === 'DictionaryInput')).toBe(false);
    expect(catalog.some((module) => module.type === 'ForwardInput')).toBe(true);
    expect(JSON.stringify(getSchemas())).not.toContain('DictionaryInput');
    expect(JSON.stringify(getSchemas())).toContain('ForwardInput');
  });

  it('编辑字段可撤销、重做，序列化保留修改', () => {
    useEditor.getState().load(plutoForwardInputExample());
    const before = useEditor.getState().project;
    const schema = plutoInputSchema();
    schema.fields.extra = { shape: ['B', 16], axes: ['batch', 'feature'], dtype: 'float32' };
    useEditor.getState().setParam('data', 'schema', JSON.stringify(schema));
    expect(inferGraph(useEditor.getState().project.graph, registry).nodes.data.outputs.out).toEqual(
      schema,
    );
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    const saved = serializeProject(useEditor.getState().project);
    expect(
      inferGraph(parseProject(saved, registry).graph, registry).nodes.data.outputs.out,
    ).toEqual(schema);
  });

  it.each<unknown>([
    null,
    { kind: 'dict', fields: [] },
    { kind: 'dict', fields: { x: { shape: [-1], dtype: 'float32' } } },
    { kind: 'dict', fields: { x: { shape: [2], dtype: 'made_up' } } },
    { kind: 'dict', fields: { x: { shape: [2], dtype: 'bool', axes: [] } } },
    { kind: 'dict', fields: { x: { shape: [1.5], dtype: 'float32' } } },
    { kind: 'dict', fields: { '': { shape: [], dtype: 'bool' } } },
    { kind: 'dict', fields: { constructor: { shape: [], dtype: 'bool' } } },
  ])('无效结构不会伪装成可用 Tensor：%j', (schema) => {
    const node = registry.create('DictionaryInput', { schema: JSON.stringify(schema) }, 'data');
    const result = inferGraph({ nodes: [node], edges: [] }, registry).nodes.data;
    expect(result.status).toBe('error');
    expect(result.outputs).toEqual({});
  });

  it('限制结构大小和递归深度，旧备注参数仍保留原长度限制', () => {
    let deep = emptyDictionary();
    for (let i = 0; i < 9; i++) deep = { kind: 'dict', fields: { child: deep } };
    expect(() => parseDictionarySchema(JSON.stringify(deep))).toThrow('8 层');
    const many = {
      kind: 'dict',
      fields: Object.fromEntries(
        Array.from({ length: 256 }, (_, i) => [`key${i}`, { shape: [], dtype: 'bool' }]),
      ),
    };
    expect(() => parseDictionarySchema(JSON.stringify(many))).toThrow('256');
    const project = projectWithOutput();
    project.graph.nodes[0].params.schema = ' '.repeat(DICTIONARY_SCHEMA_LIMIT + 1);
    expect(() => parseProject(serializeProject(project), registry)).toThrow();
    project.graph.nodes[0] = registry.create('ForwardInput', { text: 'x'.repeat(2000) }, 'data');
    expect(() => parseProject(serializeProject(project), registry)).toThrow();
  });
});
