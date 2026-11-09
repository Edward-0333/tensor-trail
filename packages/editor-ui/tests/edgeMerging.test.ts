import { describe, expect, it } from 'vitest';
import { mergeNearbyWire, mergeSharedWireChanges, movedSection } from '../src/views/edgeMerging';
import { moveSharedTargets, type WireDragTarget } from '../src/views/sharedWires';
import {
  branchJunctions,
  manualRoute,
  moveRouteSegment,
  routePath,
  simplifyRoute,
  type Point,
  type RoutedEdge,
} from '../src/views/edgeRouting';
import type { PortSide, Project } from '@tensor-trail/model-core/core/types';
import { useEditor } from '../src/editor/store';
import { registry } from '@tensor-trail/model-core/modules/index';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';

const p = (x: number, y: number): Point => ({ x, y });
const edge = (id: string, points: Point[], source = 'x', sourcePort = 'out'): RoutedEdge => ({
  id,
  source,
  sourcePort,
  points,
});
const peer = edge('peer', [p(0, 0), p(200, 0), p(200, 100), p(400, 100)]);
const original = [p(0, 0), p(100, 0), p(100, 200), p(400, 200)];
const moving = edge('moving', [p(0, 0), p(191, 0), p(191, 200), p(400, 200)]);
const orthogonal = (points: Point[]) =>
  points.slice(1).every((q, i) => q.x === points[i].x || q.y === points[i].y);

describe('直角折线与同源干线合并', () => {
  it('拖动公共竖线时整组分支也能吸附另一干线，不被已重合起始段抢走候选', () => {
    const a = edge('a', [p(0, 0), p(100, 0), p(100, 100), p(400, 100)]);
    const b = edge('b', [p(0, 0), p(100, 0), p(100, 200), p(400, 200)]);
    const anchor = edge('anchor', [p(0, 0), p(200, 0), p(200, 300), p(500, 300)]);
    const targets: WireDragTarget[] = [a, b].map((e) => ({
      id: e.id,
      route: e.points,
      index: 1,
      sourceSide: 'right',
      targetSide: 'left',
    }));
    const changes = moveSharedTargets(targets, p(91, 0));
    const result = mergeSharedWireChanges(changes, targets, [a, b, anchor], p(91, 0), 12, 8);
    expect(result).not.toEqual(changes);
    for (const change of result) expect(change.waypoints[0].x).toBe(200);
    expect(result.map((c) => c.id)).toEqual(['a', 'b']);
    expect(
      mergeSharedWireChanges(
        changes,
        targets,
        [a, b, { ...anchor, source: 'other' }],
        p(91, 0),
        12,
        8,
      ),
    ).toEqual(changes);
  });
  it('折线只含直线指令，原有路线与拐角坐标保持不变', () => {
    expect(routePath(original)).toBe('M 0 0 L 100 0 L 100 200 L 400 200');
    expect(routePath([])).toBe('');
    expect(
      routePath(manualRoute(p(0, 0), p(400, 200), 'right', 'left', [p(100, 0), p(100, 200)])),
    ).not.toMatch(/[QCA]/);
  });
  it('吸附近处同源竖线并共用其前缀，分岔点随之移动', () => {
    const before = structuredClone([moving, peer]);
    const result = mergeNearbyWire(moving, [1], [peer], 'right', 'left')!;
    expect(result.targetEdgeId).toBe('peer');
    expect(result.distance).toBe(9);
    expect(result.route).toEqual([p(0, 0), p(200, 0), p(200, 200), p(400, 200)]);
    expect(branchJunctions([edge('moving', original), peer])[0]).toMatchObject(p(100, 0));
    expect(branchJunctions([edge('moving', result.route), peer])[0]).toMatchObject(p(200, 100));
    expect([moving, peer]).toEqual(before);
    expect(orthogonal(result.route)).toBe(true);
  });
  it.each([0, 1, 2, 3])('旋转 %i 个直角后仍可合并，覆盖水平、竖直及反向端口', (turns) => {
    const rotate = (point: Point) => {
      let q = { ...point };
      for (let i = 0; i < turns; i++) q = { x: -q.y, y: q.x };
      return q;
    };
    const sides: PortSide[] = ['right', 'bottom', 'left', 'top'];
    const m = { ...moving, points: moving.points.map(rotate) },
      other = { ...peer, points: peer.points.map(rotate) };
    const result = mergeNearbyWire(m, [1], [other], sides[turns], sides[(turns + 2) % 4])!;
    expect(result.route).toEqual([p(0, 0), p(200, 0), p(200, 200), p(400, 200)].map(rotate));
    expect(orthogonal(result.route)).toBe(true);
  });
  it('替换整个上游绕行部分，不留下先分开后重新并回的回路', () => {
    const detour = edge('moving', [
      p(0, 0),
      p(100, 0),
      p(100, 40),
      p(191, 40),
      p(191, 200),
      p(400, 200),
    ]);
    const result = mergeNearbyWire(detour, [3], [peer], 'right', 'left')!;
    expect(result.route).toEqual([p(0, 0), p(200, 0), p(200, 200), p(400, 200)]);
    expect(branchJunctions([edge('moving', result.route), peer])).toHaveLength(1);
  });
  it('相同形状不等于同一数据：不同来源、不同输出口和自身均不吸附', () => {
    for (const other of [
      { ...peer, source: 'other' },
      { ...peer, sourcePort: 'other' },
      { ...peer, id: 'moving' },
    ])
      expect(mergeNearbyWire(moving, [1], [other], 'right', 'left')).toBeUndefined();
  });
  it('超过距离、只有交叉、没有投影重叠或方向相反均不合并', () => {
    expect(mergeNearbyWire(moving, [1], [peer], 'right', 'left', 8)).toBeUndefined();
    const noOverlap = edge('peer', [p(0, 0), p(50, 0), p(50, 300), p(200, 300), p(200, 400)]);
    const crossing = edge('peer', [p(0, 0), p(24, 0), p(24, 50), p(400, 50)]);
    const opposite = edge('peer', [
      p(0, 0),
      p(50, 0),
      p(50, 300),
      p(200, 300),
      p(200, 50),
      p(400, 50),
    ]);
    for (const other of [noOverlap, crossing, opposite])
      expect(mergeNearbyWire(moving, [1], [other], 'right', 'left')).toBeUndefined();
  });
  it('只判断正在拖动的线段，按最近距离选择候选', () => {
    expect(mergeNearbyWire(moving, [2], [peer], 'right', 'left')).toBeUndefined();
    const nearer = edge('nearer', [p(0, 0), p(195, 0), p(195, 100), p(400, 100)]);
    const result = mergeNearbyWire(moving, [1], [peer, nearer], 'right', 'left')!;
    expect(result.targetEdgeId).toBe('nearer');
  });
  it('缩放后的屏幕阈值一致，分数坐标不会使路线偏离直角', () => {
    expect(mergeNearbyWire(moving, [1], [peer], 'right', 'left', 12 / 2, 8 / 2)).toBeUndefined();
    expect(mergeNearbyWire(moving, [1], [peer], 'right', 'left', 12 / 0.5, 8 / 0.5)).toBeDefined();
    const translate = (q: Point) => p(q.x - 50.1, q.y + 10.2);
    const result = mergeNearbyWire(
      { ...moving, points: moving.points.map(translate) },
      [1],
      [{ ...peer, points: peer.points.map(translate) }],
      'right',
      'left',
    )!;
    expect(orthogonal(result.route)).toBe(true);
    expect(result.route[1].x).toBeCloseTo(149.9);
  });
  it('新拖出的端部折线能够吸附回干线，仍保留目标端口', () => {
    const line = edge('moving', [p(0, 0), p(24, 0), p(24, 8), p(276, 8), p(276, 0), p(300, 0)]);
    const other = edge('peer', [p(0, 0), p(350, 0), p(350, 100), p(400, 100)]);
    const result = mergeNearbyWire(line, [2], [other], 'right', 'left')!;
    expect(result.route).toEqual([p(0, 0), p(300, 0)]);
  });
  it('查找移动后的活动段，不吸附固定端口的短线段', () => {
    const delta = p(91, 0),
      pins = moveRouteSegment(original, 1, delta, 'right', 'left');
    const route = simplifyRoute(manualRoute(original[0], original.at(-1)!, 'right', 'left', pins));
    expect(movedSection(route, original, 1, delta)).toEqual([1]);
    const straight = [p(0, 0), p(300, 0)];
    const bent = simplifyRoute(
      manualRoute(
        straight[0],
        straight[1],
        'right',
        'left',
        moveRouteSegment(straight, 0, p(0, 50), 'right', 'left'),
      ),
    );
    expect(movedSection(bent, straight, 0, p(0, 50))).toEqual([2]);
  });
  it('只改变选中分支的布局；模块、端口关系和其他分支保持不变，保存及撤销恢复几何', () => {
    const project: Project = {
      schemaVersion: 1,
      name: 'merge',
      variants: [],
      graph: {
        nodes: [
          registry.create('Input', {}, 'x'),
          registry.create('Output', {}, 'a'),
          registry.create('Output', {}, 'b'),
        ],
        edges: [
          {
            id: 'moving',
            source: 'x',
            sourcePort: 'out',
            target: 'a',
            targetPort: 'in',
            waypoints: original.slice(1, -1),
          },
          {
            id: 'peer',
            source: 'x',
            sourcePort: 'out',
            target: 'b',
            targetPort: 'in',
            waypoints: peer.points.slice(1, -1),
          },
        ],
      },
    };
    useEditor.setState({ project, hasProject: true, path: [], past: [], future: [] });
    const result = mergeNearbyWire(moving, [1], [peer], 'right', 'left')!;
    useEditor.getState().setEdgeWaypoints('moving', result.waypoints);
    const after = useEditor.getState().project;
    expect(after.graph.nodes).toEqual(project.graph.nodes);
    expect(after.graph.edges[1]).toEqual(project.graph.edges[1]);
    expect(after.graph.edges[0]).toMatchObject({
      id: 'moving',
      source: 'x',
      sourcePort: 'out',
      target: 'a',
      targetPort: 'in',
    });
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(project);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
  });
});
