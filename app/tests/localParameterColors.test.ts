import { describe, expect, it } from 'vitest';
import type { Graph, ModelNode } from '../src/core/types';
import { registry } from '../src/modules';
import { parseProject, serializeProject } from '../src/core/serialization';
import { localParameterColors } from '../src/views/localParameterColors';
const local = (id: string, label = 'T', source?: string): ModelNode => ({
  ...registry.create('LocalParameter', {}, id),
  label,
  ...(source ? { valueSource: source } : {}),
});
const fixture = (): Graph => ({
  nodes: [
    local('history'),
    local('history_copy', 'T', 'history'),
    local('feature'),
    local('feature_copy', 'T', 'feature'),
    local('batch', 'bs'),
  ],
  edges: [],
});
describe('同名局部参数的定义颜色', () => {
  it('两个独立T使用不同颜色，各自副本使用来源颜色，别名不占用新的颜色', () => {
    const colors = localParameterColors(fixture());
    expect(colors.get('history')).toEqual({
      ownerId: 'history',
      index: 0,
      count: 2,
      color: 'var(--category-orange)',
    });
    expect(colors.get('feature')).toEqual({
      ownerId: 'feature',
      index: 1,
      count: 2,
      color: 'var(--category-blue)',
    });
    expect(colors.get('history_copy')).toEqual(colors.get('history'));
    expect(colors.get('feature_copy')).toEqual(colors.get('feature'));
    expect(colors.get('batch')!.count).toBe(1);
  });
  it('独立定义即使值未知或相同仍然区分，不依赖某次运行的数值', () => {
    const graph = fixture();
    // Both definitions are initially unconnected and unknown; identity still distinguishes them.
    const colors = localParameterColors(graph);
    expect(colors.get('history')!.color).not.toBe(colors.get('feature')!.color);
    graph.nodes.unshift(registry.create('IntegerParameter', { expression: '21' }, 'value'));
    graph.edges.push(
      ...['history', 'feature'].map((target) => ({
        id: target,
        source: 'value',
        sourcePort: 'out',
        target,
        targetPort: 'in',
      })),
    );
    expect(localParameterColors(graph)).toEqual(colors);
    graph.nodes[0].params.expression = '12';
    expect(localParameterColors(graph)).toEqual(colors);
  });
  it('移动节点、增删关联副本、切换端口方向不改变已有组的颜色', () => {
    const graph = fixture(),
      before = localParameterColors(graph);
    graph.nodes.forEach((n) => {
      n.position = { x: 170, y: 25 };
      n.portLayout = { output: 'bottom' };
    });
    graph.nodes.unshift(local('extra', 'T', 'history'));
    for (const [id, color] of before) expect(localParameterColors(graph).get(id)).toEqual(color);
    graph.nodes = graph.nodes.filter((n) => !n.valueSource);
    for (const n of graph.nodes)
      expect(localParameterColors(graph).get(n.id)).toEqual(before.get(n.id));
  });
  it('只在当前子图按名称分组，其他类型和其他名字不消耗T的颜色', () => {
    const graph = fixture();
    graph.nodes.push({ ...registry.create('TensorVariable', {}, 'tensor'), label: 'T' });
    expect(localParameterColors(graph).has('tensor')).toBe(false);
    expect(localParameterColors(graph).get('history')!.count).toBe(2);
    expect(
      localParameterColors({ nodes: [local('child_T')], edges: [] }).get('child_T')!.count,
    ).toBe(1);
    graph.nodes.find((n) => n.id === 'feature')!.label = 'T_feature';
    const colors = localParameterColors(graph);
    expect(colors.get('history')!.count).toBe(1);
    // Stale alias labels never create another definition or pick the old name's color.
    expect(colors.get('feature_copy')).toEqual(colors.get('feature'));
  });
  it('保存重开可还原颜色，计算显示不写入节点或改变连线', () => {
    const project = { schemaVersion: 1 as const, name: 'colors', graph: fixture(), variants: [] };
    const before = structuredClone(project);
    const colors = localParameterColors(project.graph);
    expect(project).toEqual(before);
    const reopened = parseProject(serializeProject(project), registry);
    expect(localParameterColors(reopened.graph)).toEqual(colors);
    expect(reopened).toEqual(before);
  });
  it('超过基础配色数量时继续分配新颜色，不循环复用前8个颜色', () => {
    const graph = { nodes: Array.from({ length: 16 }, (_, i) => local('T' + i)), edges: [] };
    const colors = localParameterColors(graph);
    expect(new Set([...colors.values()].map((c) => c.color)).size).toBe(16);
  });
});
