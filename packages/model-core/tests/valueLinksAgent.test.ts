import { describe, expect, it } from 'vitest';
import { patchModel, validateModel } from '../src/agent/index';
import { graphAt } from '../src/core/graph';
import { clone, tensorValue, type ModelNode, type Project } from '../src/core/types';
import { registry } from '../src/modules/index';

const alias = (source: ModelNode, id: string): ModelNode => ({
  ...clone(source),
  id,
  valueSource: source.id,
  position: { x: 400, y: 240 },
  tensorBorderColor: '#dbeafe',
  portLayout: { output: 'bottom' },
});

function project(): Project {
  const input = registry.create(
    'Input',
    { shape: '2,3', axes: 'batch,feature', dtype: 'float32' },
    'input',
  );
  const output = registry.create('Output', {}, 'output');
  return {
    schemaVersion: 1,
    name: '关联值代理测试',
    variants: [],
    graph: {
      nodes: [input, alias(input, 'input_alias'), output, alias(output, 'output_alias')],
      edges: [
        {
          id: 'value',
          source: 'input_alias',
          sourcePort: 'out',
          target: 'output',
          targetPort: 'in',
        },
      ],
    },
  };
}

function boundaryProject(): Project {
  const p = project();
  const inlet = registry.create('Inlet', { port: 'in' }, 'inlet'),
    outlet = registry.create('Outlet', { port: 'out' }, 'outlet');
  const block = registry.create('Submodule', {}, 'block');
  block.customized = true;
  block.subgraph = {
    nodes: [inlet, alias(inlet, 'inlet_alias'), outlet, alias(outlet, 'outlet_alias')],
    edges: [
      {
        id: 'inside',
        source: 'inlet',
        sourcePort: 'out',
        target: 'outlet',
        targetPort: 'in',
      },
    ],
  };
  p.graph.nodes.push(block);
  p.graph.edges = [
    {
      id: 'enter',
      source: 'input_alias',
      sourcePort: 'out',
      target: 'block',
      targetPort: 'in',
    },
    {
      id: 'leave',
      source: 'block',
      sourcePort: 'out',
      target: 'output',
      targetPort: 'in',
    },
  ];
  return p;
}

describe('关联值的代理编辑', () => {
  it('Output 副本仍可查看结果，但不会重复出现在模型输出列表', () => {
    const result = validateModel(project());
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(Object.keys(result.outputs)).toEqual(['output']);
    expect(result.nodes.find((node) => node.path[0] === 'output_alias')!.outputs.out).toEqual(
      result.outputs.output,
    );
  });

  it.each(['input', 'input_alias'])('编辑 %s 的参数与名称同步整个关联组，外观保持独立', (id) => {
    const original = project(),
      before = clone(original);
    const result = patchModel(original, {
      operations: [
        {
          op: 'set_params',
          id,
          params: { shape: '4,5', axes: 'sample,feature', dtype: 'float64' },
        },
        { op: 'rename_node', id, label: '共享数据' },
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(original).toEqual(before);
    for (const member of result.project!.graph.nodes.filter((node) => node.type === 'Input')) {
      expect(member.params).toEqual({
        shape: '4,5',
        axes: 'sample,feature',
        dtype: 'float64',
      });
      expect(member.label).toBe('共享数据');
      const stored = before.graph.nodes.find((node) => node.id === member.id)!;
      expect(member.position).toEqual(stored.position);
      expect(member.tensorBorderColor).toEqual(stored.tensorBorderColor);
      expect(member.portLayout).toEqual(stored.portLayout);
    }
    expect(tensorValue(result.outputs.output)).toEqual({
      shape: [4, 5],
      axes: ['sample', 'feature'],
      dtype: 'float64',
    });
  });

  it('删除主 Input 和 Output 时由剩余副本接管数据与连接', () => {
    const original = project(),
      before = clone(original);
    const result = patchModel(original, {
      operations: [
        { op: 'remove_node', id: 'input' },
        { op: 'remove_node', id: 'output' },
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(original).toEqual(before);
    expect(result.project!.graph.nodes.map((node) => node.id)).toEqual([
      'input_alias',
      'output_alias',
    ]);
    expect(result.project!.graph.nodes.every((node) => !node.valueSource)).toBe(true);
    expect(result.project!.graph.edges).toEqual([
      {
        id: 'value',
        source: 'input_alias',
        sourcePort: 'out',
        target: 'output_alias',
        targetPort: 'in',
      },
    ]);
    expect(Object.keys(result.outputs)).toEqual(['output_alias']);
    expect(tensorValue(result.outputs.output_alias).shape).toEqual([2, 3]);
  });

  it.each(['inlet_alias', 'outlet_alias'])('允许删除边界副本 %s，保留父层接口与连线', (id) => {
    const original = boundaryProject(),
      before = clone(original);
    const result = patchModel(original, {
      operations: [{ op: 'remove_node', path: ['block'], id }],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(original).toEqual(before);
    expect(graphAt(result.project!.graph, ['block']).nodes.some((node) => node.id === id)).toBe(
      false,
    );
    expect(result.project!.graph.edges).toEqual(before.graph.edges);
    expect(tensorValue(result.outputs.output).shape).toEqual([2, 3]);
  });

  it.each(['inlet', 'outlet'])('仍保护原始边界 %s', (id) => {
    const result = patchModel(boundaryProject(), {
      operations: [{ op: 'remove_node', path: ['block'], id }],
    });
    expect(result.ok).toBe(false);
    expect(result.diagnostics[0]).toMatchObject({
      code: 'PROTECTED_BOUNDARY',
      path: ['block', id],
    });
  });

  it('关联组修改后遇到失败操作，整个事务不会修改输入项目', () => {
    const original = project(),
      before = clone(original);
    const result = patchModel(original, {
      operations: [
        { op: 'rename_node', id: 'input_alias', label: '临时名称' },
        { op: 'set_params', id: 'input_alias', params: { shape: 7 } },
      ],
    });
    expect(result.ok).toBe(false);
    expect(result.diagnostics[0]).toMatchObject({
      code: 'INVALID_PARAMETER',
      path: ['input_alias'],
      parameter: 'shape',
    });
    expect(result.project).toBeUndefined();
    expect(original).toEqual(before);
  });
});
