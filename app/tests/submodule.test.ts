import { beforeEach, describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { isCalculation } from '../src/modules/libraryCategories';
import { inferProject, resultAt } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { resolveForwardView } from '../src/core/hierarchy';
import { parseProject, serializeProject } from '../src/core/serialization';
import { canMergeCalculations } from '../src/core/calculationGroups';
import type { DictionarySpec, Graph, Project } from '../src/core/types';
import { useEditor } from '../src/editor/store';

const draftMessage = '子模块内部流程待完善，尚未定义输出。';
const schema: DictionarySpec = {
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
        valid_mask: {
          shape: [2, 49, 101],
          axes: ['batch', 'agent', 'time'],
          dtype: 'bool',
        },
      },
    },
    map: {
      kind: 'dict',
      fields: {
        polygon_center: {
          shape: [2, 58, 3],
          axes: ['batch', 'polygon', 'pose'],
          dtype: 'float32',
        },
      },
    },
  },
};

function project(tensorInput = false): Project {
  const input = tensorInput
    ? registry.create('Input', { shape: '2,49,128', axes: 'batch,agent,feature' }, 'data')
    : registry.create('ForwardInput', { schema: JSON.stringify(schema) }, 'data');
  input.label = 'data';
  const encoder = registry.create('Submodule', {}, 'encoder');
  encoder.label = 'agent_encoder';
  return {
    schemaVersion: 1,
    name: 'PLUTO 子模块',
    variants: [],
    graph: {
      nodes: [input, encoder],
      edges: [
        {
          id: 'call',
          source: 'data',
          sourcePort: 'out',
          target: 'encoder',
          targetPort: 'data',
        },
      ],
    },
  };
}

function completeChild(child: Graph): void {
  child.nodes.push(
    registry.create('DictionaryGet', { path: '["agent","position"]' }, 'position'),
    registry.create('Outlet', { port: 'out' }, 'return'),
  );
  child.edges.push(
    {
      id: 'read',
      source: 'data',
      sourcePort: 'out',
      target: 'position',
      targetPort: 'in',
    },
    {
      id: 'return',
      source: 'position',
      sourcePort: 'out',
      target: 'return',
      targetPort: 'in',
    },
  );
}

beforeEach(() => {
  useEditor.getState().load(project());
});

describe('子模块草稿与输入边界', () => {
  it('默认是可进入的网络模块，仅声明 data 输入，不加入计算合并', () => {
    const node = registry.create('Submodule');
    const def = registry.get('Submodule')!;
    expect(def.label).toBe('子模块');
    expect(def.category).toBe('网络结构');
    expect(def.composite).toBe(true);
    expect(def.fields).toEqual([]);
    expect(registry.list().some((entry) => entry.type === 'Submodule')).toBe(true);
    expect(node.subgraph?.nodes).toHaveLength(1);
    expect(node.subgraph?.nodes[0]).toMatchObject({
      id: 'data',
      type: 'Inlet',
      label: 'data',
      params: { port: 'data' },
    });
    expect(node.subgraph?.edges).toEqual([]);
    expect(ports(node, registry)).toEqual({
      inputs: [{ id: 'data', label: 'data' }],
      outputs: [],
    });
    expect(isCalculation(def)).toBe(false);
    const p = project();
    expect(canMergeCalculations(p.graph, 'data', 'encoder', registry)).toBe(false);
  });

  it('传入整份嵌套字典和字段尺寸，草稿只给出待完善提示而不伪造输出', () => {
    const p = project(),
      before = structuredClone(p);
    const result = inferProject(p, registry);
    const encoder = result.nodes.encoder;
    expect(encoder.status).toBe('warning');
    expect(encoder.messages).toContain(draftMessage);
    expect(encoder.inputs.data).toEqual(schema);
    expect(encoder.outputs).toEqual({});
    expect(encoder.child?.nodes.data.status).toBe('ok');
    expect(encoder.child?.nodes.data.outputs.out).toEqual(schema);
    expect(encoder.child?.outputs).toEqual({});
    expect(p).toEqual(before);
  });

  it('同一输入边界也能传递 Tensor，未连接输入仍等待上游', () => {
    const p = project(true);
    const result = inferProject(p, registry);
    expect(result.nodes.encoder.child?.nodes.data.outputs.out).toEqual({
      shape: [2, 49, 128],
      axes: ['batch', 'agent', 'feature'],
      dtype: 'float32',
    });
    p.graph.edges = [];
    const disconnected = inferProject(p, registry).nodes.encoder;
    expect(disconnected.status).toBe('blocked');
    expect(disconnected.outputs).toEqual({});
  });

  it('草稿保存重开及逐层进入返回不修改模型或撤销历史', () => {
    const p = project();
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const initial = useEditor.getState();
    const originalProject = initial.project,
      past = initial.past;
    useEditor.getState().navigate(['encoder']);
    const view = resolveForwardView(originalProject.graph, useEditor.getState().path);
    expect(view.current.title).toBe('agent_encoder · forward()');
    expect(view.graph.nodes.map((n) => n.id)).toEqual(['data']);
    const inner = resultAt(inferProject(originalProject, registry), ['encoder']);
    expect(inner?.nodes.data.outputs.out).toEqual(schema);
    useEditor.getState().navigate([]);
    expect(useEditor.getState().project).toBe(originalProject);
    expect(useEditor.getState().past).toBe(past);
  });

  it('内部添加计算可保存和撤销；补好 Outlet 后再提供真实推导输出', () => {
    const before = useEditor.getState().project;
    useEditor.getState().navigate(['encoder']);
    expect(useEditor.getState().edit(completeChild)).toBe(true);
    const completed = useEditor.getState().project;
    expect(completed.graph.nodes[1].customized).toBe(true);
    expect(parseProject(serializeProject(completed), registry)).toEqual(completed);
    const result = inferProject(completed, registry).nodes.encoder;
    expect(result.status).toBe('ok');
    expect(result.outputs.out).toEqual((schema.fields.agent as DictionarySpec).fields.position);
    expect(ports(completed.graph.nodes[1], registry).outputs.map((p) => p.id)).toEqual(['out']);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    expect(inferProject(useEditor.getState().project, registry).nodes.encoder.outputs).toEqual({});
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(completed);
    expect(inferProject(useEditor.getState().project, registry).nodes.encoder.status).toBe('ok');
  });

  it('子模块可继续嵌套，输入字典穿过两层且不冒充输出', () => {
    const p = project(),
      child = p.graph.nodes[1].subgraph!;
    child.nodes.push(registry.create('Submodule', {}, 'inner'));
    child.edges.push({
      id: 'nested',
      source: 'data',
      sourcePort: 'out',
      target: 'inner',
      targetPort: 'data',
    });
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
    const result = inferProject(p, registry);
    expect(result.nodes.encoder.status).toBe('warning');
    const inner = resultAt(result, ['encoder', 'inner']);
    expect(inner?.nodes.data.outputs.out).toEqual(schema);
    expect(result.nodes.encoder.outputs).toEqual({});
    expect(result.nodes.encoder.child?.nodes.inner.outputs).toEqual({});
  });
});

describe('草稿例外保持严格范围', () => {
  it('已有 Outlet 但未连接时仍报错，不作为空草稿忽略', () => {
    const p = project();
    p.graph.nodes[1].subgraph!.nodes.push(registry.create('Outlet', { port: 'out' }, 'return'));
    const result = inferProject(p, registry).nodes.encoder;
    expect(result.status).toBe('error');
    expect(result.child?.nodes.return.status).toBe('blocked');
    expect(result.outputs).toEqual({});
  });

  it('草稿内真实计算错误仍报错，并保留内部诊断', () => {
    const p = project(),
      child = p.graph.nodes[1].subgraph!;
    child.nodes.push(registry.create('DictionaryGet', { path: '["missing"]' }, 'bad'));
    child.edges.push({
      id: 'bad',
      source: 'data',
      sourcePort: 'out',
      target: 'bad',
      targetPort: 'in',
    });
    const result = inferProject(p, registry).nodes.encoder;
    expect(result.status).toBe('error');
    expect(result.child?.nodes.bad.status).toBe('error');
    expect(result.outputs).toEqual({});
  });

  it.each(['Block', 'CalculationGroup'])('旧 %s 子图缺少 Outlet 仍拒绝', (type) => {
    const p = project();
    p.graph.nodes[1].type = type;
    expect(() => parseProject(serializeProject(p), registry)).toThrow();
    expect(inferProject(p, registry).nodes.encoder.status).toBe('error');
  });

  it('Submodule 仍必须有 Inlet，孤立草稿变体不能绕过边界检查', () => {
    const p = project();
    p.graph.nodes[1].subgraph = { nodes: [], edges: [] };
    p.graph.edges = [];
    expect(() => parseProject(serializeProject(p), registry)).toThrow();
    const withVariant = project();
    withVariant.variants.push({
      id: 'draft',
      name: '草稿',
      graph: structuredClone(withVariant.graph.nodes[1].subgraph!),
    });
    expect(() => parseProject(serializeProject(withVariant), registry)).toThrow();
  });

  it('草稿拒绝保存为变体，完成后的子图仍可正常保存和重用', () => {
    useEditor.getState().select('encoder');
    const before = useEditor.getState().project;
    useEditor.getState().saveVariant('尚未完成');
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().notice).toBe('补充模块输出后可保存为变体。');
    useEditor.getState().navigate(['encoder']);
    expect(useEditor.getState().edit(completeChild)).toBe(true);
    useEditor.getState().saveVariant('完成的字段读取模块');
    const saved = useEditor.getState().project;
    expect(saved.variants).toHaveLength(1);
    expect(parseProject(serializeProject(saved), registry)).toEqual(saved);
    const instance = registry.fromVariant(saved.variants[0].name, saved.variants[0].graph);
    expect(instance.type).toBe('Block');
    expect(ports(instance, registry).inputs.map((p) => p.id)).toEqual(['data']);
    expect(ports(instance, registry).outputs.map((p) => p.id)).toEqual(['out']);
  });
});
