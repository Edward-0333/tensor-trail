import { describe, expect, it } from 'vitest';
import { projectLabelPosition, routeLabelAnchor, type Point } from '../src/views/edgeRouting';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { copySelection, duplicateSelection } from '@tensor-trail/model-core/editing/clipboard';
import { useEditor } from '../src/editor/store';
import { registry } from '@tensor-trail/model-core/modules/index';
import { example } from '@tensor-trail/model-core/modules/examples';

const route: Point[] = [
  { x: 0, y: 0 },
  { x: 200, y: 0 },
  { x: 200, y: 300 },
  { x: 500, y: 300 },
];

describe('连线文字沿线路定位', () => {
  it('旧项目默认放在最长直线段旁，自定义位置沿完整折线测量', () => {
    expect(routeLabelAnchor(route)).toEqual({
      x: 200,
      y: 150,
      horizontal: false,
      position: 350 / 800,
    });
    expect(routeLabelAnchor(route, 0.125)).toEqual({
      x: 100,
      y: 0,
      horizontal: true,
      position: 0.125,
    });
    expect(routeLabelAnchor(route, 0.5)).toEqual({
      x: 200,
      y: 200,
      horizontal: false,
      position: 0.5,
    });
    expect(routeLabelAnchor(route, 0.875)).toEqual({
      x: 400,
      y: 300,
      horizontal: true,
      position: 0.875,
    });
  });
  it.each([
    [{ x: 100, y: -75 }, 0.125],
    [{ x: 260, y: 150 }, 350 / 800],
    [{ x: 400, y: 360 }, 0.875],
    [{ x: -500, y: -100 }, 0],
    [{ x: 800, y: 400 }, 1],
  ] as [Point, number][])('鼠标 %j 投影到当前连线，不会离开线路', (point, expected) => {
    expect(projectLabelPosition(route, point)).toBeCloseTo(expected);
  });
  it('同一位置多次经过时，优先保留接近当前位置的路线段', () => {
    const loop = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
      { x: 0, y: 0 },
      { x: 100, y: 0 },
    ];
    expect(projectLabelPosition(loop, { x: 50, y: -10 }, 0.15)).toBeCloseTo(0.1);
    expect(projectLabelPosition(loop, { x: 50, y: -10 }, 0.85)).toBeCloseTo(0.9);
  });
  it('正反向、重复点和退化路线均有有限结果，拖动不会修改路线', () => {
    const reversed = [...route].reverse();
    expect(projectLabelPosition(reversed, { x: 400, y: 360 })).toBeCloseTo(0.125);
    const repeated = [route[0], ...route];
    expect(projectLabelPosition(repeated, { x: 400, y: 360 })).toBeCloseTo(0.875);
    for (const points of [[], [route[0]], [route[0], route[0]]]) {
      expect(projectLabelPosition(points, { x: 100, y: 100 })).toBe(0.5);
      expect(routeLabelAnchor(points).x).toBe(0);
    }
    expect(route).toEqual([
      { x: 0, y: 0 },
      { x: 200, y: 0 },
      { x: 200, y: 300 },
      { x: 500, y: 300 },
    ]);
  });
  it('位置保存为比例，节点平移或线路长度变化后仍在新路径上', () => {
    const shifted = route.map((p) => ({ x: p.x * 2 + 13, y: p.y * 2 - 40 }));
    expect(routeLabelAnchor(shifted, 0.875)).toEqual({
      x: 813,
      y: 560,
      horizontal: true,
      position: 0.875,
    });
    expect(routeLabelAnchor(route, 0)).toMatchObject({ x: 0, y: 0 });
    expect(routeLabelAnchor(route, 1)).toMatchObject({ x: 500, y: 300 });
  });
});

describe('连线文字布局保存', () => {
  it('子图文字位置可撤销、重做、保存重开，且不更改节点、线路和计算', () => {
    useEditor.getState().load(example());
    useEditor.setState({ past: [], future: [] });
    useEditor.getState().navigate(['transformer']);
    const before = structuredClone(useEditor.getState().project);
    const edge = before.graph.nodes.find((n) => n.id === 'transformer')!.subgraph!.edges[0];
    useEditor.getState().setEdgeLabelPosition(edge.id, 0.73);
    const after = useEditor.getState().project;
    const parent = after.graph.nodes.find((n) => n.id === 'transformer')!;
    expect(parent.customized).toBeUndefined();
    expect(parent.subgraph!.edges[0]).toEqual({ ...edge, labelPosition: 0.73 });
    expect(parent.subgraph!.nodes).toEqual(
      before.graph.nodes.find((n) => n.id === 'transformer')!.subgraph!.nodes,
    );
    expect(inferProject(after, registry)).toEqual(inferProject(before, registry));
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('非法位置不会覆盖项目；恢复默认位置也可撤销', () => {
    useEditor.getState().load(example());
    const before = useEditor.getState().project,
      id = before.graph.edges[0].id;
    for (const value of [-0.1, 1.1, NaN, Infinity]) {
      useEditor.getState().setEdgeLabelPosition(id, value);
      expect(useEditor.getState().project).toBe(before);
    }
    for (const value of [null, '0.5', {}, []]) {
      const invalid = structuredClone(before);
      Object.assign(invalid.graph.edges[0], { labelPosition: value });
      expect(() => parseProject(JSON.stringify(invalid), registry)).toThrow('连线文字位置');
    }
    useEditor.getState().setEdgeLabelPosition(id, 0);
    const positioned = useEditor.getState().project;
    expect(positioned.graph.edges[0].labelPosition).toBe(0);
    useEditor.getState().setEdgeLabelPosition(id);
    expect(useEditor.getState().project.graph.edges[0].labelPosition).toBeUndefined();
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(positioned);
  });
  it('复制节点与连线时保留相对文字位置', () => {
    const graph = example().graph;
    graph.edges[0].labelPosition = 0.8;
    const fragment = copySelection(graph, ['input', 'transformer']);
    const copied = duplicateSelection(fragment, 100);
    expect(copied.edges[0].labelPosition).toBe(0.8);
  });
});
