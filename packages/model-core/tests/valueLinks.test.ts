import { describe, expect, it } from 'vitest';
import { clone, type Graph, type ModelNode, type Project } from '../src/core/types';
import { inferGraph } from '../src/core/engine';
import { graphIssues, orderGraph, ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import {
  effectiveValueNode,
  isValueNode,
  linkedValueNodes,
  syncValueLinks,
  valueLinkIssues,
  valueOwner,
} from '../src/core/valueLinks';
import { registry } from '../src/modules/index';
import { example } from '../src/modules/examples';

function alias(source: ModelNode, id = 'alias'): ModelNode {
  return {
    ...clone(source),
    id,
    valueSource: source.id,
    position: { x: 560, y: 240 },
  };
}

function project(graph: Graph): Project {
  return { schemaVersion: 1, name: '共享值', graph, variants: [] };
}

function derivedGraph(): Graph {
  const input = registry.create('Input', { shape: '2, 3', axes: 'batch, channel' }, 'input');
  const owner = registry.create('TensorVariable', {}, 'owner');
  const output = registry.create('Output', {}, 'output');
  return {
    // The alias intentionally precedes its owner: implicit dependency must reorder them.
    nodes: [alias(owner), output, owner, input],
    edges: [
      { id: 'input_owner', source: 'input', sourcePort: 'out', target: 'owner', targetPort: 'in' },
      {
        id: 'alias_output',
        source: 'alias',
        sourcePort: 'out',
        target: 'output',
        targetPort: 'in',
      },
    ],
  };
}

describe('输入输出节点的共享值', () => {
  it('支持输入输出值类型，并排除计算算子与子模块', () => {
    for (const type of [
      'Input',
      'Output',
      'ForwardInput',
      'DictionaryInput',
      'LocalParameter',
      'TensorVariable',
      'IntegerParameter',
      'Inlet',
      'Outlet',
    ])
      expect(isValueNode(type)).toBe(true);
    for (const type of ['Linear', 'TensorIndex', 'Submodule', 'unknown'])
      expect(isValueNode(type)).toBe(false);
  });

  it('依赖上游的副本不需要独立输入，排序后向下游传递源结果', () => {
    const graph = derivedGraph();
    expect(graphIssues(graph, registry)).toEqual([]);
    const order = orderGraph(graph);
    expect(order.indexOf('input')).toBeLessThan(order.indexOf('owner'));
    expect(order.indexOf('owner')).toBeLessThan(order.indexOf('alias'));
    expect(order.indexOf('alias')).toBeLessThan(order.indexOf('output'));
    expect(ports(graph.nodes[0], registry)).toEqual({
      inputs: [],
      outputs: [{ id: 'out', label: '输出' }],
    });
    const result = inferGraph(graph, registry);
    expect(result.nodes.alias.status).toBe('ok');
    expect(result.nodes.alias.inputs).toEqual(result.nodes.owner.inputs);
    expect(result.nodes.alias.outputs).toEqual(result.nodes.owner.outputs);
    expect(result.nodes.output.outputs.out).toEqual(result.nodes.owner.outputs.out);
    expect(result.nodes.output.outputs.out).toMatchObject({ shape: [2, 3], dtype: 'float32' });
  });

  it('字典与整数来源也保持完整数据类型、层级和数值', () => {
    const dictionary = {
      kind: 'dict',
      fields: {
        agent: {
          kind: 'dict',
          fields: {
            position: {
              shape: [2, 49, 21, 2],
              dtype: 'float32',
              axes: ['batch', 'agent', 'time', 'xy'],
            },
          },
        },
      },
    };
    const data = registry.create('ForwardInput', { schema: JSON.stringify(dictionary) }, 'data');
    const count = registry.create('IntegerParameter', { expression: 'history_steps' }, 'count');
    const graph = {
      nodes: [alias(data, 'data_alias'), alias(count, 'count_alias'), data, count],
      edges: [],
    };
    const result = inferGraph(graph, registry, {}, 0, {
      history_steps: { type: 'int', value: 21 },
    });
    expect(result.issues).toEqual([]);
    expect(result.nodes.data_alias.outputs).toEqual(result.nodes.data.outputs);
    expect(result.nodes.data_alias.outputs.out).toEqual(dictionary);
    expect(result.nodes.count_alias.outputs.out).toEqual({ kind: 'int', value: 21 });
  });

  it('源等待连接时副本及其下游也阻塞，不生成伪造结果', () => {
    const graph = derivedGraph();
    graph.edges = graph.edges.filter((edge) => edge.target !== 'owner');
    const result = inferGraph(graph, registry);
    expect(result.nodes.owner.status).toBe('blocked');
    expect(result.nodes.alias.status).toBe('blocked');
    expect(result.nodes.alias.outputs).toEqual({});
    expect(result.nodes.alias.messages.join(' ')).toContain('共享值来源');
    expect(result.nodes.output.status).toBe('blocked');
  });

  it('源自身参数报错时副本阻塞，并保留可定位的错误原因', () => {
    const owner = registry.create('Input', { shape: '2, 3', axes: 'one' }, 'owner');
    const result = inferGraph({ nodes: [alias(owner), owner], edges: [] }, registry);
    expect(result.nodes.owner.status).toBe('error');
    expect(result.nodes.alias.status).toBe('blocked');
    expect(result.nodes.alias.outputs).toEqual({});
    expect(result.nodes.alias.messages).toContain(result.nodes.owner.messages[0]);
  });

  it('共享值隐式依赖同样参与反馈环检测', () => {
    const owner = registry.create('TensorVariable', {}, 'owner');
    const graph = {
      nodes: [owner, alias(owner)],
      edges: [
        { id: 'feedback', source: 'alias', sourcePort: 'out', target: 'owner', targetPort: 'in' },
      ],
    };
    expect(() => orderGraph(graph)).toThrow('环路');
    expect(graphIssues(graph, registry).join(' ')).toContain('环路');
    expect(() => parseProject(serializeProject(project(graph)), registry)).toThrow('环路');
  });

  it('不能给已链接副本接入另一个独立值', () => {
    const graph = derivedGraph();
    graph.edges.push({
      id: 'independent',
      source: 'input',
      sourcePort: 'out',
      target: 'alias',
      targetPort: 'in',
    });
    expect(valueLinkIssues(graph).join(' ')).toContain('独立输入连线');
    expect(() => parseProject(serializeProject(project(graph)), registry)).toThrow('独立输入连线');
  });

  it('解析相同源组并共享配置，保留每个显示位置的外观', () => {
    const graph = derivedGraph();
    const owner = valueOwner(graph, 'alias');
    const copy = graph.nodes[0];
    owner.label = 'agent_pos';
    owner.params = { description: '新内容' };
    copy.tensorColor = '#123456';
    copy.portLayout = { output: 'bottom' };
    const effective = effectiveValueNode(graph, copy);
    expect(valueOwner(graph, copy)).toBe(owner);
    expect(
      linkedValueNodes(graph, 'alias')
        .map((node) => node.id)
        .sort(),
    ).toEqual(['alias', 'owner']);
    expect(effective).toMatchObject({
      id: 'alias',
      label: 'agent_pos',
      params: { description: '新内容' },
      position: copy.position,
      tensorColor: '#123456',
      portLayout: { output: 'bottom' },
    });
    expect(copy.label).not.toBe('agent_pos');
    syncValueLinks(graph);
    expect(copy.params).toEqual(owner.params);
    expect(copy.params).not.toBe(owner.params);
    expect(copy.label).toBe(owner.label);
    expect(copy.tensorColor).toBe('#123456');
    expect(() => valueOwner(graph, 'missing')).toThrow('不存在');
  });

  it('递归同步内部副本，序列化往返将源配置作为唯一依据', () => {
    const child = registry.create('Submodule', {}, 'child');
    const inlet = child.subgraph!.nodes[0];
    const copy = alias(inlet, 'data_alias');
    child.subgraph!.nodes.push(copy);
    inlet.label = 'data';
    copy.label = '过时名称';
    copy.params.port = 'obsolete';
    const graph = { nodes: [child], edges: [] };
    syncValueLinks(graph);
    expect(copy.label).toBe('data');
    expect(copy.params.port).toBe('data');
    copy.label = '再次过时';
    const loaded = parseProject(serializeProject(project(graph)), registry);
    expect(loaded.graph.nodes[0].subgraph!.nodes[1].label).toBe('data');
    expect(loaded.graph.nodes[0].subgraph!.nodes[1].valueSource).toBe('data');
  });

  it('模块输入输出副本不增加父接口，也不重复登记父输出', () => {
    const child = registry.create('Submodule', {}, 'child');
    const input = child.subgraph!.nodes[0];
    const inletCopy = alias(input, 'data_alias');
    const output = registry.create('Outlet', { port: 'out' }, 'result');
    const outletCopy = alias(output, 'result_alias');
    // Even stale alias metadata must not introduce a second output during direct inference.
    outletCopy.params.port = 'phantom_output';
    child.subgraph!.nodes.push(inletCopy, output, outletCopy);
    child.subgraph!.edges.push({
      id: 'return',
      source: 'data_alias',
      sourcePort: 'out',
      target: 'result',
      targetPort: 'in',
    });
    const boundary = ports(child, registry);
    expect(boundary.inputs.map((port) => port.id)).toEqual(['data']);
    expect(boundary.outputs.map((port) => port.id)).toEqual(['out']);
    const value = { shape: [2, 3], axes: ['batch', 'feature'], dtype: 'float32' };
    const result = inferGraph(child.subgraph!, registry, { data: value });
    expect(result.issues).toEqual([]);
    expect(result.outputs).toEqual({ out: value });
    expect(result.nodes.result_alias.outputs).toEqual(result.nodes.result.outputs);
    expect(result.nodes.data_alias.outputs).toEqual(result.nodes.data.outputs);
    expect(
      parseProject(serializeProject(project({ nodes: [child], edges: [] })), registry),
    ).toBeDefined();
  });

  it.each([
    [
      '悬空引用',
      (graph: Graph) => {
        graph.nodes[0].valueSource = 'missing';
      },
      /不存在/,
    ],
    [
      '自身引用',
      (graph: Graph) => {
        graph.nodes[0].valueSource = 'alias';
      },
      /自身/,
    ],
    [
      '间接引用',
      (graph: Graph) => {
        const copy = alias(graph.nodes[0], 'chain');
        graph.nodes.push(copy);
      },
      /串联/,
    ],
    [
      '类型不符',
      (graph: Graph) => {
        graph.nodes[0].type = 'LocalParameter';
      },
      /类型必须一致/,
    ],
    [
      '计算节点引用',
      (graph: Graph) => {
        graph.nodes[0].type = 'Linear';
      },
      /输入输出类/,
    ],
    [
      '非法标识符',
      (graph: Graph) => {
        graph.nodes[0].valueSource = '__proto__';
      },
      /有效节点 ID/,
    ],
  ] as const)('拒绝不合法的链接：%s', (_, mutate, message) => {
    const graph = derivedGraph();
    mutate(graph);
    expect(() => parseProject(serializeProject(project(graph)), registry)).toThrow(message);
  });

  it('子图链接无法跨越当前forward引用根层节点', () => {
    const root = registry.create('Input', {}, 'root_value');
    const child = registry.create('Submodule', {}, 'child');
    child.subgraph!.nodes.push(alias(root, 'outside_alias'));
    expect(() =>
      parseProject(serializeProject(project({ nodes: [root, child], edges: [] })), registry),
    ).toThrow('不存在于当前画布');
  });

  it('原有边界限制不放松：根层禁止边界、实际端口不重复、组合仍须输出', () => {
    const inlet = registry.create('Inlet', { port: 'data' }, 'data');
    expect(() =>
      parseProject(
        serializeProject(project({ nodes: [inlet, alias(inlet)], edges: [] })),
        registry,
      ),
    ).toThrow('根图不能');
    const child = registry.create('Submodule', {}, 'child');
    child.subgraph!.nodes.push(registry.create('Inlet', { port: 'data' }, 'duplicate'));
    expect(() =>
      parseProject(serializeProject(project({ nodes: [child], edges: [] })), registry),
    ).toThrow('边界端口重复');
    child.subgraph!.nodes.pop();
    child.type = 'Block';
    expect(() =>
      parseProject(serializeProject(project({ nodes: [child], edges: [] })), registry),
    ).toThrow('至少需要一个 Inlet 和一个 Outlet');
  });

  it('没有共享引用的旧项目语义与序列化保持不变', () => {
    const old = example();
    const serialized = serializeProject(old);
    const loaded = parseProject(serialized, registry);
    expect(serializeProject(loaded)).toBe(serialized);
    expect(inferGraph(loaded.graph, registry)).toEqual(inferGraph(old.graph, registry));
  });
});
