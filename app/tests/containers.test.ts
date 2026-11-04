import { describe, expect, it } from 'vitest';
import type { Graph, Project } from '../src/core/types';
import { isTensorSpec, shapeText, tensorValue } from '../src/core/types';
import { inferGraph } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { buildModel, describeModules, getSchemas, validateModel } from '../src/agent';
import { copySelection, duplicateSelection } from '../src/editor/clipboard';

function connected(
  type = 'ForwardInput',
  target = 'Output',
  text = '自由描述，不要求是 JSON',
): Graph {
  return {
    nodes: [registry.create(type, { text }, 'data'), registry.create(target, {}, 'next')],
    edges: [{ id: 'link', source: 'data', sourcePort: 'out', target: 'next', targetPort: 'in' }],
  };
}

describe('网络输入与旧输入兼容', () => {
  it('文字型网络输入仍使用 ForwardInput 类型，默认节点只有名称和可选备注', () => {
    expect(
      registry
        .list()
        .filter((def) => ['ForwardInput', 'DictInput', 'ListInput'].includes(def.type))
        .map((def) => def.type),
    ).toEqual(['ForwardInput']);
    const node = registry.create('ForwardInput');
    expect(node.label).toBe('网络输入');
    expect(node.params).toEqual({ schema: '', text: '' });
  });
  it('打开旧项目会合并根图、子图和变体中的输入，保留自定义名称、备注和连线', () => {
    const graph = connected();
    Object.assign(graph.nodes[0], {
      type: 'DictInput',
      label: '字典输入 Dict',
      tensorColor: '#172033',
      portLayout: { output: 'bottom' },
    });
    const child: Graph = {
      nodes: [
        registry.create('Inlet', {}, 'in'),
        registry.create('Outlet', {}, 'out'),
        {
          ...registry.create('ForwardInput', { text: '原来的列表说明' }, 'list'),
          type: 'ListInput',
          label: '自定义输入名',
        },
      ],
      edges: [{ id: 'pass', source: 'in', sourcePort: 'out', target: 'out', targetPort: 'in' }],
    };
    graph.nodes.push({
      ...registry.create('Block', {}, 'block'),
      customized: true,
      subgraph: child,
    });
    graph.edges.push({
      id: 'block-input',
      source: 'data',
      sourcePort: 'out',
      target: 'block',
      targetPort: 'in',
    });
    const project: Project = {
      schemaVersion: 1,
      name: '旧项目',
      graph,
      variants: [{ id: 'variant', name: '旧变体', graph: child }],
    };
    const migrated = parseProject(serializeProject(project), registry);
    expect(migrated.graph.nodes[0]).toMatchObject({
      id: 'data',
      type: 'ForwardInput',
      label: '网络输入',
      tensorColor: '#172033',
      params: { text: '自由描述，不要求是 JSON' },
      portLayout: { output: 'bottom' },
    });
    expect(migrated.graph.edges).toEqual(project.graph.edges);
    for (const nested of [migrated.graph.nodes[2].subgraph!, migrated.variants[0].graph])
      expect(nested.nodes[2]).toMatchObject({
        type: 'ForwardInput',
        label: '自定义输入名',
        params: { text: '原来的列表说明' },
      });
    expect(validateModel(migrated).ok).toBe(true);
    expect(project.graph.nodes[0].type).toBe('DictInput');
  });
  it.each([
    ['ForwardInput', 'forward', '网络输入'],
    ['DictInput', 'forward', '网络输入'],
    ['ListInput', 'forward', '网络输入'],
  ])('%s 无需任何形状参数，文本可以留空或自由填写', (type, kind, label) => {
    for (const text of ['', 'forward(batch)\n图像、掩码和其他信息', '<b>作为普通文字显示</b>']) {
      const node = registry.create(type, { text }, 'data');
      expect(node.params.schema).toBe('');
      const result = inferGraph({ nodes: [node], edges: [] }, registry).nodes.data;
      expect(result.status).toBe('ok');
      expect(result.outputs.out).toEqual({ kind, text });
      expect(isTensorSpec(result.outputs.out)).toBe(false);
      expect(shapeText(result.outputs.out)).toBe(label);
      expect(result.outputs.out).not.toHaveProperty('shape');
      expect(result.outputs.out).not.toHaveProperty('dtype');
      expect(() => tensorValue(result.outputs.out)).toThrow('需要 Tensor');
    }
  });
  it.each(['DictInput', 'ListInput'])('%s 可通过 Identity 和输出节点保留结构', (type) => {
    const graph = connected(type, 'Identity');
    graph.nodes.push(registry.create('Output', {}, 'output'));
    graph.edges.push({
      id: 'final',
      source: 'next',
      sourcePort: 'out',
      target: 'output',
      targetPort: 'in',
    });
    const result = inferGraph(graph, registry);
    expect(result.nodes.output.status).toBe('ok');
    expect(result.nodes.next.outputs.out).toEqual(result.nodes.data.outputs.out);
    expect(result.nodes.output.inputs.in).toEqual(result.nodes.data.outputs.out);
    expect(result.nodes.output.outputs.out).toEqual(result.nodes.data.outputs.out);
  });
  it.each(['Linear', 'Transformer', 'Compute'])(
    '容器直接接入 %s 给出类型提示，后续节点阻塞',
    (type) => {
      const graph = connected('DictInput', type);
      graph.nodes.push(registry.create('Output', {}, 'output'));
      graph.edges.push({
        id: 'final',
        source: 'next',
        sourcePort: 'out',
        target: 'output',
        targetPort: 'in',
      });
      const result = inferGraph(graph, registry);
      expect(result.nodes.data.status).toBe('ok');
      expect(result.nodes.next.status).toBe('error');
      expect(result.nodes.next.messages[0]).toContain('网络输入');
      expect(result.nodes.next.messages[0]).toContain('需要 Tensor');
      expect(result.nodes.next.outputs).toEqual({});
      expect(result.nodes.output.status).toBe('blocked');
    },
  );
  it('自定义子图边界可以传递列表，不将其当作标量 Tensor', () => {
    const graph = connected('ListInput', 'Block');
    graph.nodes[1].customized = true;
    graph.nodes[1].subgraph = {
      nodes: [registry.create('Inlet', {}, 'in'), registry.create('Outlet', {}, 'out')],
      edges: [{ id: 'inner', source: 'in', sourcePort: 'out', target: 'out', targetPort: 'in' }],
    };
    const result = inferGraph(graph, registry);
    expect(result.nodes.next.status).toBe('ok');
    expect(result.nodes.next.outputs.out).toEqual(result.nodes.data.outputs.out);
    expect(result.nodes.next.child!.outputs.out).not.toHaveProperty('shape');
  });
  it('子图内张量算子同样拒绝容器，其他 Tensor 分支照常推导', () => {
    const graph = connected('ListInput', 'Block');
    graph.nodes[1].customized = true;
    graph.nodes[1].subgraph = {
      nodes: [
        registry.create('Inlet', {}, 'in'),
        registry.create('Linear', {}, 'linear'),
        registry.create('Outlet', {}, 'out'),
      ],
      edges: [
        { id: 'inner', source: 'in', sourcePort: 'out', target: 'linear', targetPort: 'in' },
        { id: 'final', source: 'linear', sourcePort: 'out', target: 'out', targetPort: 'in' },
      ],
    };
    graph.nodes.push(
      registry.create('Input', {}, 'tensor'),
      registry.create('Linear', {}, 'linear'),
    );
    graph.edges.push({
      id: 'tensor-edge',
      source: 'tensor',
      sourcePort: 'out',
      target: 'linear',
      targetPort: 'in',
    });
    const result = inferGraph(graph, registry);
    expect(result.nodes.next.status).toBe('error');
    expect(result.nodes.next.child!.nodes.linear.messages[0]).toContain('网络输入');
    expect(tensorValue(result.nodes.linear.outputs.out).shape).toEqual([8, 100, 64]);
  });
  it('内容说明、名称、背景、端口方向及位置可保存、重开和复制', () => {
    const graph = connected('DictInput');
    Object.assign(graph.nodes[0], {
      label: 'forward 的 batch',
      tensorColor: '#172033',
      portLayout: { output: 'bottom' },
      position: { x: 120, y: -80 },
    });
    const project: Project = { schemaVersion: 1, name: '字典输入', graph, variants: [] };
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    const copied = duplicateSelection(copySelection(restored.graph, ['data', 'next']), 32);
    expect(copied.nodes[0]).toMatchObject({
      type: 'ForwardInput',
      label: 'forward 的 batch',
      tensorColor: '#172033',
      params: { text: '自由描述，不要求是 JSON' },
      portLayout: { output: 'bottom' },
      position: { x: 152, y: -48 },
    });
    expect(copied.nodes[0].id).not.toBe('data');
    expect(inferGraph(copied, registry).nodes[copied.nodes[1].id].status).toBe('ok');
  });
  it('AI 配方、模块目录、Schema 和输出报告均保留容器类型', () => {
    const result = buildModel({
      name: 'forward 参数',
      nodes: [
        { id: 'data', type: 'DictInput', params: { text: 'batch["image"], batch["mask"]' } },
        { id: 'output', type: 'Output' },
      ],
      connections: [['data.out', 'output.in']],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.outputs.output).toEqual({
      kind: 'forward',
      text: 'batch["image"], batch["mask"]',
    });
    expect(validateModel(result.project!).ok).toBe(true);
    expect(describeModules().modules.filter((m) => m.type === 'ForwardInput')).toHaveLength(1);
    expect(JSON.stringify(getSchemas())).toContain('ForwardInput');
    expect(JSON.stringify(getSchemas())).not.toContain('DictInput');
    expect(JSON.stringify(getSchemas())).not.toContain('ListInput');
  });
});
