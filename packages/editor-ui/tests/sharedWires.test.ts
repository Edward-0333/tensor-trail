import { describe, it, expect, beforeEach } from 'vitest';
import {
  type RoutedEdge,
  type Point,
  automaticRoute,
  manualRoute,
  branchJunctions,
  routePath,
} from '../src/views/edgeRouting';
import {
  sharedPrefixLength,
  wireDrawing,
  sharedSections,
  sharedDragTargets,
  moveSharedTargets,
} from '../src/views/sharedWires';
import { useEditor } from '../src/editor/store';
import { registry } from '@tensor-trail/model-core/modules/index';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
const p = (x: number, y: number): Point => ({ x, y });
const edge = (id: string, points: Point[], extra: Partial<RoutedEdge> = {}): RoutedEdge => ({
  id,
  source: 'x',
  sourcePort: 'out',
  points,
  sourceSide: 'right',
  targetSide: 'left',
  ...extra,
});
const a = () => edge('a', [p(0, 0), p(100, 0), p(100, 100), p(300, 100)]);
const b = () => edge('b', [p(0, 0), p(100, 0), p(100, 200), p(300, 200)]);
const parse = (path: string) => {
  const nums = (path.match(/[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  return Array.from({ length: nums.length / 2 }, (_, i) => p(nums[2 * i], nums[2 * i + 1]));
};
const total = (path: string) =>
  parse(path)
    .slice(1)
    .reduce(
      (sum, point, i) =>
        sum + Math.abs(point.x - parse(path)[i].x) + Math.abs(point.y - parse(path)[i].y),
      0,
    );
describe('共享线段的单次绘制', () => {
  it('同源公共前缀只绘制一次，保留每条目的分支', () => {
    const edges = [a(), b()];
    expect(sharedPrefixLength(edges[0].points, edges[1].points)).toBe(200);
    expect(wireDrawing(edges[0], edges).path).toBe(routePath(edges[0].points));
    expect(wireDrawing(edges[1], edges).path).toBe('M 100 100 L 100 200 L 300 200');
    expect(edges.reduce((sum, e) => sum + total(wireDrawing(e, edges).path), 0)).toBe(700);
  });
  it('选中分支接管公共段高亮，仍然只有一次描边', () => {
    const edges = [a(), { ...b(), selected: true }];
    expect(wireDrawing(edges[1], edges).path).toBe(routePath(edges[1].points));
    expect(wireDrawing(edges[0], edges).path).toBe('M 100 100 L 300 100');
    expect(edges.reduce((sum, e) => sum + total(wireDrawing(e, edges).path), 0)).toBe(700);
  });
  it('三路自动折线及分段长度不同的路线也能去重，几何和输入不变', () => {
    const edges = [-80, 0, 80].map((y, i) =>
      edge(String(i), automaticRoute(p(0, 0), p(300, y), 'right', 'left')),
    );
    const original = structuredClone(edges);
    const lengths = edges.map((e) => total(wireDrawing(e, edges).path));
    expect(lengths.reduce((a, b) => a + b, 0)).toBe(760);
    expect(edges).toEqual(original);
    expect(sharedPrefixLength([p(0, 0), p(40, 0), p(100, 0)], [p(0, 0), p(100, 0)])).toBe(100);
  });
  it('不同数据源或端口不能因重合而合并；分岔后的交叉不吞掉线段', () => {
    for (const extra of [{ source: 'y' }, { sourcePort: 'other' }]) {
      const peers = [a(), edge('b', a().points, extra)];
      expect(wireDrawing(peers[1], peers).path).toBe(routePath(peers[1].points));
    }
    const peers = [
      edge('a', [p(0, 0), p(50, 0), p(50, 50), p(100, 50), p(200, 50)]),
      edge('b', [p(0, 0), p(100, 0), p(100, 50), p(200, 50)]),
    ];
    expect(wireDrawing(peers[1], peers).path).toBe('M 50 0 L 100 0 L 100 50 L 200 50');
  });
  it('相同路径和浮点坐标无多余笔画；端口处立即分开则各自完整', () => {
    const first = edge('a', [p(-50.1, 10.2), p(100.3, 10.2), p(100.3, 60.5)]),
      second = { ...first, id: 'b' };
    expect(wireDrawing(second, [first, second]).path).toBe('');
    expect(sharedPrefixLength([p(0, 0), p(100, 0)], [p(0, 0), p(0, 100)])).toBe(0);
  });
  it('公共段上完全重合的同值文字只显示一份，并跟随选中线高亮', () => {
    const label = { text: '[2,49,21]', x: 50, y: 0, horizontal: true };
    const first = { ...a(), label },
      second = { ...b(), label };
    expect(wireDrawing(first, [first, second]).showLabel).toBe(true);
    expect(wireDrawing(second, [first, second]).showLabel).toBe(false);
    second.selected = true;
    expect(wireDrawing(first, [first, second]).showLabel).toBe(false);
    expect(wireDrawing(second, [first, second]).showLabel).toBe(true);
  });
});
describe('公共线段整体拖动', () => {
  it('在实际分岔位置拆分命中区域，仅捕获经过该段的同源分支', () => {
    const peers = [a(), b()];
    const sections = sharedSections(peers[1], peers);
    expect(sections).toEqual([p(0, 0), p(100, 0), p(100, 100), p(100, 200), p(300, 200)]);
    expect(sharedDragTargets(peers[1], sections, 1, peers).map((t) => t.id)).toEqual(['b', 'a']);
    expect(sharedDragTargets(peers[1], sections, 2, peers).map((t) => t.id)).toEqual(['b']);
    expect(sharedDragTargets(peers[1], sections, 1, peers, true).map((t) => t.id)).toEqual(['b']);
  });
  it('同步移动多个分支，端点保持原位，分岔点更新且公共段持续合并', () => {
    const peers = [a(), b()];
    const targets = sharedDragTargets(peers[0], sharedSections(peers[0], peers), 1, peers);
    const changes = moveSharedTargets(targets, p(30, 0), 130);
    const moved = peers.map((e) => ({
      ...e,
      points: manualRoute(
        e.points[0],
        e.points.at(-1)!,
        'right',
        'left',
        changes.find((c) => c.id === e.id)!.waypoints,
      ),
    }));
    expect(branchJunctions(moved)[0]).toMatchObject(p(130, 100));
    for (const [e, i] of moved.map((e, i) => [e, i] as const)) {
      expect(e.points[0]).toMatchObject(peers[i].points[0]);
      expect(e.points.at(-1)).toMatchObject(peers[i].points.at(-1)!);
    }
    expect(sharedPrefixLength(moved[0].points, moved[1].points)).toBeGreaterThan(0);
    expect(wireDrawing(moved[1], moved).sharedLength).toBeGreaterThan(0);
  });
  it.each([0, 1, 2, 3])('支持四种端口方向，旋转%s次', (turns) => {
    const rotate = (point: Point) => {
      let r = point;
      for (let i = 0; i < turns; i++) r = p(-r.y, r.x);
      return r;
    };
    const sides = ['right', 'bottom', 'left', 'top'] as const;
    const peers = [a(), b()].map((e) => ({
      ...e,
      points: e.points.map(rotate),
      sourceSide: sides[turns],
      targetSide: sides[(turns + 2) % 4],
    }));
    const targets = sharedDragTargets(peers[0], sharedSections(peers[0], peers), 0, peers);
    const changes = moveSharedTargets(targets, rotate(p(0, 30)));
    const routes = peers.map((e) =>
      manualRoute(
        e.points[0],
        e.points.at(-1)!,
        e.sourceSide,
        e.targetSide,
        changes.find((c) => c.id === e.id)!.waypoints,
      ),
    );
    expect(sharedPrefixLength(routes[0], routes[1])).toBeGreaterThan(0);
    routes.forEach((route, i) => {
      expect(route[0]).toMatchObject(peers[i].points[0]);
      expect(route.at(-1)).toMatchObject(peers[i].points.at(-1)!);
    });
  });
});
describe('共享段编辑事务', () => {
  beforeEach(() => {
    useEditor.getState().load({
      schemaVersion: 1,
      name: 'shared',
      variants: [],
      graph: {
        nodes: [
          registry.create('Input', {}, 'x'),
          registry.create('Output', {}, 'a'),
          registry.create('Output', {}, 'b'),
        ],
        edges: ['a', 'b'].map((target) => ({
          id: target,
          source: 'x',
          sourcePort: 'out',
          target,
          targetPort: 'in',
        })),
      },
    });
    useEditor.setState({ past: [], future: [] });
  });
  it('多个分支一次提交、撤销重做和保存；节点与数据连接保持不变', () => {
    const original = useEditor.getState().project;
    expect(
      useEditor.getState().setSharedWireWaypoints([
        { id: 'a', waypoints: [p(20, 30), p(120, 30)] },
        { id: 'b', waypoints: [p(20, 30), p(120, 30)] },
      ]),
    ).toBe(true);
    const after = useEditor.getState().project;
    expect(after.graph.nodes).toEqual(original.graph.nodes);
    expect(useEditor.getState().past).toHaveLength(1);
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(original);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
  it('任一分支无效时整次拒绝，没有部分修改或额外历史', () => {
    const original = useEditor.getState().project;
    expect(
      useEditor.getState().setSharedWireWaypoints([
        { id: 'a', waypoints: [p(20, 30)] },
        { id: 'missing', waypoints: [] },
      ]),
    ).toBe(false);
    expect(useEditor.getState().project).toBe(original);
    expect(
      useEditor.getState().setSharedWireWaypoints([
        { id: 'a', waypoints: [p(20, 30)] },
        { id: 'b', waypoints: [p(Infinity, 0)] },
      ]),
    ).toBe(false);
    expect(useEditor.getState().project).toBe(original);
    expect(useEditor.getState().past).toHaveLength(0);
  });
});
