import { describe, expect, it } from 'vitest';
import type { PortSide } from '../src/core/types';
import {
  automaticRoute,
  branchJunctions,
  manualRoute,
  moveRouteSegment,
  simplifyRoute,
  routeInsertIndex,
  routeMidpoint,
  routePath,
  type Point,
  type RoutedEdge,
} from '../src/views/edgeRouting';
import { useEditor } from '../src/editor/store';
import { registry } from '../src/modules';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Project } from '../src/core/types';

const sides: PortSide[] = ['left', 'right', 'top', 'bottom'];
const direction = {
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
  top: { x: 0, y: -1 },
  bottom: { x: 0, y: 1 },
};

describe('手动连线几何', () => {
  it.each(sides.flatMap((source) => sides.map((target) => [source, target] as const)))(
    '%s → %s 保留端点、出口方向和正交路径',
    (sourceSide, targetSide) => {
      const source = { x: 25, y: 70 },
        target = { x: 470, y: 260 };
      const pins = [
        { x: 150, y: -120 },
        { x: 360, y: -120 },
      ];
      const route = manualRoute(source, target, sourceSide, targetSide, pins);
      expect(route[0]).toMatchObject(source);
      expect(route.at(-1)).toMatchObject(target);
      expect(route[1]).toMatchObject({
        x: source.x + direction[sourceSide].x * 24,
        y: source.y + direction[sourceSide].y * 24,
      });
      expect(route.at(-2)).toMatchObject({
        x: target.x + direction[targetSide].x * 24,
        y: target.y + direction[targetSide].y * 24,
      });
      for (const pin of pins)
        expect(route.some((point) => point.x === pin.x && point.y === pin.y)).toBe(true);
      route
        .slice(1)
        .forEach((point, i) => expect(point.x === route[i].x || point.y === route[i].y).toBe(true));
      expect(routePath(route)).not.toMatch(/NaN|Infinity/);
    },
  );
  it('在点击的线段旁插入控制点，保留已有控制点顺序', () => {
    const route = manualRoute({ x: 0, y: 0 }, { x: 300, y: 100 }, 'right', 'left', [
      { x: 100, y: -60 },
      { x: 200, y: -60 },
    ]);
    expect(routeInsertIndex(route, { x: 10, y: 0 })).toBe(0);
    expect(routeInsertIndex(route, { x: 150, y: -60 })).toBe(1);
    expect(routeInsertIndex(route, { x: 290, y: 100 })).toBe(2);
  });
  it('重合控制点不会生成非法路径，维度标签沿线路居中', () => {
    const route = manualRoute({ x: 0, y: 0 }, { x: 100, y: 0 }, 'right', 'left', [
      { x: 24, y: 0 },
      { x: 24, y: 0 },
    ]);
    expect(routePath(route)).not.toMatch(/NaN|Infinity/);
    expect(routeMidpoint(route)).toEqual({ x: 50, y: 0 });
  });
});

function expectOrthogonal(route: Point[]) {
  expect(route.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y))).toBe(true);
  route.slice(1).forEach((p, i) => expect(p.x === route[i].x || p.y === route[i].y).toBe(true));
}

describe('整段连线拖动', () => {
  it.each(sides.flatMap((source) => sides.map((target) => [source, target] as const)))(
    '%s → %s 自动线路各段可拖动，端点和端口方向保持固定',
    (sourceSide, targetSide) => {
      const source = { x: 25.5, y: 70.25 },
        target = { x: 470.5, y: 260.25 };
      const route = automaticRoute(source, target, sourceSide, targetSide);
      expectOrthogonal(route);
      expect(route[0]).toEqual(source);
      expect(route.at(-1)).toEqual(target);
      for (let i = 0; i < route.length - 1; i++) {
        const original = structuredClone(route);
        const pins = moveRouteSegment(route, i, { x: -57, y: 43 }, sourceSide, targetSide);
        const changed = manualRoute(source, target, sourceSide, targetSide, pins);
        expectOrthogonal(changed);
        expect(changed[0]).toMatchObject(source);
        expect(changed.at(-1)).toMatchObject(target);
        expect(changed[1]).toMatchObject({
          x: source.x + direction[sourceSide].x * 24,
          y: source.y + direction[sourceSide].y * 24,
        });
        expect(changed.at(-2)).toMatchObject({
          x: target.x + direction[targetSide].x * 24,
          y: target.y + direction[targetSide].y * 24,
        });
        expect(route).toEqual(original);
        expect(routePath(changed)).not.toMatch(/NaN|Infinity/);
      }
    },
  );
  it('直线第一次拖动产生两个拐角，水平段只采用纵向位移', () => {
    const source = { x: 0, y: 0 },
      target = { x: 300, y: 0 };
    const route = automaticRoute(source, target, 'right', 'left');
    expect(route).toEqual([source, target]);
    const pins = moveRouteSegment(route, 0, { x: 200, y: 60 }, 'right', 'left');
    expect(pins).toEqual([
      { x: 24, y: 60 },
      { x: 276, y: 60 },
    ]);
    expect(simplifyRoute(manualRoute(source, target, 'right', 'left', pins))).toEqual([
      source,
      { x: 24, y: 0 },
      { x: 24, y: 60 },
      { x: 276, y: 60 },
      { x: 276, y: 0 },
      target,
    ]);
  });
  it('竖直段左右平移，两端相邻水平段跟随，重复拖动可保持简洁', () => {
    const source = { x: 0, y: 0 },
      target = { x: 300, y: 100 };
    let route = automaticRoute(source, target, 'right', 'left');
    for (let i = 0; i < 20; i++) {
      const pins = moveRouteSegment(route, 1, { x: 3, y: 90 }, 'right', 'left');
      route = simplifyRoute(manualRoute(source, target, 'right', 'left', pins));
      expect(route).toHaveLength(4);
    }
    expect(route).toEqual([source, { x: 210, y: 0 }, { x: 210, y: 100 }, target]);
  });
  it('旧控制点路线也可拖段，更新只占一步撤销并可保存重开', () => {
    const p: Project = {
      schemaVersion: 1,
      name: 'route',
      variants: [],
      graph: {
        nodes: [registry.create('Input', {}, 'x'), registry.create('Output', {}, 'y')],
        edges: [
          {
            id: 'xy',
            source: 'x',
            sourcePort: 'out',
            target: 'y',
            targetPort: 'in',
            waypoints: [
              { x: 100, y: -60 },
              { x: 200, y: -60 },
            ],
          },
        ],
      },
    };
    const source = { x: 0, y: 0 },
      target = { x: 300, y: 100 };
    const route = simplifyRoute(
      manualRoute(source, target, 'right', 'left', p.graph.edges[0].waypoints!),
    );
    const pins = moveRouteSegment(route, 2, { x: 0, y: -30 }, 'right', 'left');
    useEditor.setState({ project: p, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().setEdgeWaypoints('xy', pins);
    const changed = useEditor.getState().project;
    expect(changed.graph.nodes).toEqual(p.graph.nodes);
    expect(changed.graph.edges[0]).toMatchObject({
      source: 'x',
      target: 'y',
      sourcePort: 'out',
      targetPort: 'in',
      waypoints: pins,
    });
    expect(useEditor.getState().past).toHaveLength(1);
    expect(parseProject(serializeProject(changed), registry)).toEqual(changed);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(p);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(changed);
  });
});

const edge = (id: string, points: Point[], source = 'x', sourcePort = 'out'): RoutedEdge => ({
  id,
  source,
  sourcePort,
  points,
});
const p = (x: number, y: number): Point => ({ x, y });
describe('分支连接点', () => {
  it('分数坐标的舍入残差不破坏共享拐角后的分岔判断', () => {
    const a = edge('a', [p(-50.1, 10.2), p(100.3, 10.2), p(100.3, 60.5), p(200.4, 60.5)]);
    const b = edge('b', [p(-50.1, 10.2), p(100.3, 10.2), p(100.3, 90.7), p(200.4, 90.7)]);
    const junctions = branchJunctions([a, b]);
    expect(junctions).toHaveLength(1);
    expect(junctions[0].x).toBeCloseTo(100.3);
    expect(junctions[0].y).toBeCloseTo(60.5);
  });
  it('同一输出共享干线的实际分岔位置显示一个点', () => {
    const paths = [
      edge('a', [p(0, 0), p(100, 0), p(100, -80), p(300, -80)]),
      edge('b', [p(0, 0), p(40, 0), p(100, 0), p(100, 80), p(300, 80)]),
    ];
    expect(branchJunctions(paths).map(({ x, y }) => ({ x, y }))).toEqual([p(100, 0)]);
  });
  it('两条自动线路同样有分支点', () => {
    const paths = [
      edge('a', automaticRoute(p(0, 0), p(300, -80), 'right', 'left')),
      edge('b', automaticRoute(p(0, 0), p(300, 80), 'right', 'left')),
    ];
    expect(branchJunctions(paths)).toHaveLength(1);
    expect(branchJunctions(paths)[0]).toMatchObject(p(150, 0));
  });
  it('三路分支可有两个不同分岔点，重合分支点去重', () => {
    const paths = [
      edge('a', [p(0, 0), p(100, 0), p(100, -50)]),
      edge('b', [p(0, 0), p(200, 0), p(200, -50)]),
      edge('c', [p(0, 0), p(200, 0), p(200, 50)]),
    ];
    expect(branchJunctions(paths).map(({ x, y }) => ({ x, y }))).toEqual([p(100, 0), p(200, 0)]);
  });
  it('不同来源、不同输出口以及仅仅交叉的线均不产生连接点', () => {
    const points = [p(0, 0), p(100, 0), p(100, 80)];
    const other = [p(0, 0), p(100, 0), p(100, -80)];
    expect(branchJunctions([edge('a', points), edge('b', other, 'other')])).toEqual([]);
    expect(branchJunctions([edge('a', points), edge('b', other, 'x', 'other')])).toEqual([]);
    expect(
      branchJunctions([
        edge('a', [p(0, 0), p(300, 0)]),
        edge('b', [p(150, -50), p(150, 50)], 'other'),
      ]),
    ).toEqual([]);
  });
  it('已经分开的同源线路后来交叉不增加连接点；删除一条分支后点消失', () => {
    const a = edge('a', [p(0, 0), p(50, 0), p(50, 100), p(200, 100)]);
    const b = edge('b', [p(0, 0), p(150, 0), p(150, 150)]);
    expect(branchJunctions([a, b]).map(({ x, y }) => ({ x, y }))).toEqual([p(50, 0)]);
    expect(branchJunctions([a])).toEqual([]);
  });
  it('完全重合、零长点及在端口即分开的线不产生虚假分支点', () => {
    const points = [p(0, 0), p(0, 0), p(100, 0)];
    expect(branchJunctions([edge('a', points), edge('b', points)])).toEqual([]);
    expect(
      branchJunctions([edge('a', [p(0, 0), p(100, 0)]), edge('b', [p(0, 0), p(0, 100)])]),
    ).toEqual([]);
    expect(branchJunctions([edge('a', []), edge('b', [p(0, 0)])])).toEqual([]);
  });
});
