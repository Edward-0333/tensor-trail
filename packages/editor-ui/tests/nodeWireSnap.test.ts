import { describe, expect, it } from 'vitest';
import type { Graph, PortSide, Project } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { automaticRoute, manualRoute, simplifyRoute, type Point } from '../src/views/edgeRouting';
import { automaticWireRoutes, type WireGeometry } from '../src/views/automaticWires';
import { groupWirePreview, nodeDragRoutes, snapNodesToWires } from '../src/editor/nodeWireSnap';
import { useEditor } from '../src/editor/store';

const p = (x: number, y: number) => ({ x, y });
const model = (id: string, position = p(0, 0)) => ({
  ...registry.create(id === 's' ? 'Input' : 'Output', {}, id),
  position,
});
const connection = (id: string, target: string, waypoints?: Point[]) => ({
  id,
  source: 's',
  sourcePort: 'out',
  target,
  targetPort: 'in',
  ...(waypoints ? { waypoints } : {}),
});
const wire = (id: string, source: Point, target: Point, pins?: Point[]): WireGeometry => ({
  id,
  source: 's',
  sourcePort: 'out',
  sourceSide: 'right',
  targetSide: 'left',
  automatic: !pins?.length,
  points: pins
    ? simplifyRoute(manualRoute(source, target, 'right', 'left', pins))
    : automaticRoute(source, target, 'right', 'left'),
});
const rotate = (point: Point, turns: number): Point => {
  let q = point;
  for (let i = 0; i < turns; i++) q = p(-q.y, q.x);
  return q;
};
const sides: PortSide[] = ['right', 'bottom', 'left', 'top'];

describe('同一输出自动共用干线', () => {
  it.each([0, 1, 2, 3])('不同距离的目标也共用一条干线，旋转 %i 个直角', (turns) => {
    const edges = [wire('a', p(0, 0), p(300, 100)), wire('b', p(0, 0), p(600, 200))].map((e) => ({
      ...e,
      sourceSide: sides[turns],
      targetSide: sides[(turns + 2) % 4],
      points: e.points.map((q) => rotate(q, turns)),
    }));
    const before = structuredClone(edges);
    const routes = automaticWireRoutes(edges);
    expect(routes.get('a')).toEqual(
      [p(0, 0), p(150, 0), p(150, 100), p(300, 100)].map((q) => rotate(q, turns)),
    );
    expect(routes.get('b')).toEqual(
      [p(0, 0), p(150, 0), p(150, 200), p(600, 200)].map((q) => rotate(q, turns)),
    );
    expect(edges).toEqual(before);
    expect(automaticWireRoutes([...edges].reverse())).toEqual(routes);
  });
  it('不同输出、端口以及回折方向不共用，手动控制点不被自动改写', () => {
    const a = wire('a', p(0, 0), p(300, 100)),
      b = wire('b', p(0, 0), p(600, 200));
    expect(automaticWireRoutes([a, { ...b, source: 'other' }]).size).toBe(0);
    expect(automaticWireRoutes([a, { ...b, sourcePort: 'other' }]).size).toBe(0);
    expect(automaticWireRoutes([a, { ...b, targetSide: 'right' }]).size).toBe(0);
    expect(automaticWireRoutes([a, wire('back', p(0, 0), p(-200, 200))]).size).toBe(0);
    const manual = { ...a, automatic: false };
    expect(automaticWireRoutes([manual, b]).has('a')).toBe(false);
    expect(automaticWireRoutes([manual, b]).get('b')?.[1]).toEqual(p(150, 0));
  });
});

describe('模块按连线吸附', () => {
  const graph: Graph = {
    nodes: [model('s'), model('a', p(300, 100))],
    edges: [connection('a', 'a', [p(100, 0), p(100, 100)])],
  };
  const routes = [wire('a', p(0, 0), p(300, 100), graph.edges[0].waypoints)];
  const initial = graph.nodes.map((n) => ({ id: n.id, position: n.position }));
  it.each([0, 1, 2, 3])('端口短段对齐自身手动线路并锁定模块，旋转 %i 个直角', (turns) => {
    const rotatePoint = (q: Point) => rotate(q, turns);
    const rotated: Graph = {
      nodes: graph.nodes.map((n) => ({ ...n, position: rotatePoint(n.position) })),
      edges: graph.edges.map((e) => ({ ...e, waypoints: e.waypoints?.map(rotatePoint) })),
    };
    const rendered = routes.map((r) => ({
      ...r,
      sourceSide: sides[turns],
      targetSide: sides[(turns + 2) % 4],
      points: r.points.map(rotatePoint),
    }));
    const result = snapNodesToWires(rotated, rendered, rotated.nodes, [
      { id: 'a', position: rotatePoint(p(350, 108)) },
    ]);
    expect(result.positions[0].position).toEqual(rotatePoint(p(350, 100)));
    expect(result.guides).toHaveLength(1);
    expect(result.guides[0].coordinate).toBe(rotatePoint(p(350, 100))[result.guides[0].axis]);
  });
  it('按屏幕像素判断，Alt 关闭吸附，并有释放滞回范围', () => {
    const moving = [{ id: 'a', position: p(350, 108) }];
    expect(snapNodesToWires(graph, routes, initial, moving, 2).guides).toHaveLength(0);
    expect(snapNodesToWires(graph, routes, initial, moving, 0.5).positions[0].position.y).toBe(100);
    expect(snapNodesToWires(graph, routes, initial, moving, 1, [], true).positions).toEqual(moving);
    const locked = snapNodesToWires(graph, routes, initial, moving);
    expect(
      snapNodesToWires(
        graph,
        routes,
        initial,
        [{ id: 'a', position: p(350, 116) }],
        1,
        locked.guides,
      ).positions[0].position.y,
    ).toBe(100);
    expect(
      snapNodesToWires(
        graph,
        routes,
        initial,
        [{ id: 'a', position: p(350, 119) }],
        1,
        locked.guides,
      ).guides,
    ).toHaveLength(0);
  });
  it('自动线路也可与连接的另一端端口对齐', () => {
    const automatic = wire('a', p(0, 0), p(300, 100));
    const result = snapNodesToWires(
      { ...graph, edges: [connection('a', 'a')] },
      [automatic],
      initial,
      [{ id: 'a', position: p(350, 7) }],
    );
    expect(result.positions[0].position).toEqual(p(350, 0));
    expect(result.guides[0].key).toContain(':port');
  });
  it('输出端同样可对齐手动起始线段', () => {
    const result = snapNodesToWires(graph, routes, initial, [{ id: 's', position: p(30, 8) }]);
    expect(result.positions[0].position).toEqual(p(30, 0));
  });
  it('多选模块统一平移，内部线路不成为自身锁定参考', () => {
    const group: Graph = {
      nodes: [...graph.nodes, model('b', p(600, 300))],
      edges: [...graph.edges, { ...connection('ab', 'b'), source: 'a' }],
    };
    const allRoutes = [...routes, { ...wire('ab', p(300, 100), p(600, 300)), source: 'a' }];
    const snapped = snapNodesToWires(group, allRoutes, group.nodes, [
      { id: 'a', position: p(340, 108) },
      { id: 'b', position: p(640, 308) },
    ]);
    expect(snapped.positions.map((n) => n.position)).toEqual([p(340, 100), p(640, 300)]);
    expect(snapped.guides.every((g) => !g.key.startsWith('ab:'))).toBe(true);
    expect(
      snapNodesToWires(graph, routes, initial, [
        { id: 's', position: p(30, 8) },
        { id: 'a', position: p(330, 108) },
      ]).guides,
    ).toHaveLength(0);
  });
  it('可对齐同源分支的实际线段；几何相同的不同信号不会误吸附', () => {
    const branch = wire('b', p(0, 0), p(600, 200), [p(100, 0), p(100, 170), p(500, 170)]);
    const g: Graph = {
      nodes: [...graph.nodes, model('b', p(600, 200))],
      edges: [connection('a', 'a'), connection('b', 'b', [p(100, 170), p(500, 170)])],
    };
    const moving = [{ id: 'a', position: p(350, 176) }];
    expect(
      snapNodesToWires(g, [wire('a', p(0, 0), p(300, 100)), branch], initial, moving).positions[0]
        .position.y,
    ).toBe(170);
    expect(
      snapNodesToWires(
        g,
        [wire('a', p(0, 0), p(300, 100)), { ...branch, source: 'other' }],
        initial,
        moving,
      ).guides,
    ).toHaveLength(0);
  });
  it('组内手动折点随模块平移，节点与路线一次撤销并可保存重开', () => {
    const project: Project = { schemaVersion: 1, name: 'snap', variants: [], graph };
    const moving = [
      { id: 's', position: p(30, 40) },
      { id: 'a', position: p(330, 140) },
    ];
    const changes = nodeDragRoutes(graph, routes, initial, moving);
    expect(groupWirePreview(graph, initial, moving)).toEqual(changes);
    expect(graph.edges[0].waypoints).toEqual([p(100, 0), p(100, 100)]);
    expect(changes[0].waypoints).toEqual([p(130, 40), p(130, 140)]);
    useEditor.setState({ project, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().finishNodeDrag(moving, undefined, changes);
    const after = useEditor.getState().project;
    expect(after.graph.nodes.map((n) => n.position)).toEqual(moving.map((n) => n.position));
    expect(after.graph.edges[0].waypoints).toEqual(changes[0].waypoints);
    expect(useEditor.getState().past).toHaveLength(1);
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(project);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
});
