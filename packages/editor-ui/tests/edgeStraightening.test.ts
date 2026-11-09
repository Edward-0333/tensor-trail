import { describe, expect, it } from 'vitest';
import type { Point } from '../src/views/edgeRouting';
import {
  automaticRoute,
  manualRoute,
  moveRouteSegment,
  simplifyRoute,
  snapSegmentMovement,
} from '../src/views/edgeRouting';
import type { PortSide, Project } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules/index';
import { useEditor } from '../src/editor/store';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';

const p = (x: number, y: number): Point => ({ x, y });
const z = [p(0, 0), p(100, 0), p(100, 100), p(300, 100)];
const bump = [p(0, 0), p(100, 0), p(100, 80), p(200, 80), p(200, 0), p(400, 0)];

function drag(
  route: Point[],
  index: number,
  delta: Point,
  threshold = 12,
  sourceSide: PortSide = 'right',
  targetSide: PortSide = 'left',
) {
  const snap = snapSegmentMovement(route, index, delta, threshold);
  const pins = moveRouteSegment(route, index, snap.delta, sourceSide, targetSide, snap.coordinate);
  const raw = manualRoute(route[0], route.at(-1)!, sourceSide, targetSide, pins);
  return { snap, pins, raw, route: simplifyRoute(raw) };
}

function orthogonal(route: Point[]) {
  expect(route.every((q) => Number.isFinite(q.x) && Number.isFinite(q.y))).toBe(true);
  for (let i = 1; i < route.length; i++)
    expect(route[i].x === route[i - 1].x || route[i].y === route[i - 1].y).toBe(true);
}

describe('同一连线相邻段吸附拉直', () => {
  it('Z 下横段向上靠近会消除中间短台阶，目标端口保持固定', () => {
    const before = structuredClone(z);
    const result = drag(z, 2, p(37, -95));
    expect(result.snap.snapped).toBe(true);
    expect(result.snap.coordinate).toBe(0);
    expect(result.route).toEqual([p(0, 0), p(276, 0), p(276, 100), p(300, 100)]);
    expect(result.raw[0]).toMatchObject(z[0]);
    expect(result.raw.at(-1)).toMatchObject(z.at(-1)!);
    expect(result.raw.at(-2)).toMatchObject(p(276, 100));
    expect(z).toEqual(before);
    orthogonal(result.route);
  });

  it('Z 上横段向下靠近会消除中间短台阶，源端口出口保持固定', () => {
    const result = drag(z, 0, p(-31, 95));
    expect(result.snap.snapped).toBe(true);
    expect(result.snap.coordinate).toBe(100);
    expect(result.route).toEqual([p(0, 0), p(24, 0), p(24, 100), p(300, 100)]);
    expect(result.raw[1]).toMatchObject(p(24, 0));
    orthogonal(result.route);
  });

  it.each([0, 1, 2, 3])('旋转 %i 次，内部凸起靠近原轴后压成完整直线', (turns) => {
    const rotate = (point: Point) => {
      let q = { ...point };
      for (let i = 0; i < turns; i++) q = p(-q.y, q.x);
      return p(q.x === 0 ? 0 : q.x, q.y === 0 ? 0 : q.y);
    };
    const sides: PortSide[] = ['right', 'bottom', 'left', 'top'];
    const route = bump.map(rotate);
    const result = drag(route, 2, rotate(p(0, -75)), 12, sides[turns], sides[(turns + 2) % 4]);
    expect(result.snap.snapped).toBe(true);
    expect(result.route).toEqual([route[0], route.at(-1)!]);
    orthogonal(result.route);
    // An empty set of pins must reconstruct the same visible line when saved as automatic.
    if (result.pins.length === 0)
      expect(
        simplifyRoute(
          automaticRoute(route[0], route.at(-1)!, sides[turns], sides[(turns + 2) % 4]),
        ),
      ).toEqual(result.route);
  });

  it('压合反向相邻横段时只去掉重合回折，不留下上下双线', () => {
    const original = [p(0, 0), p(200, 0), p(200, 80), p(100, 80), p(100, 160), p(400, 160)];
    const result = drag(original, 2, p(0, -75));
    expect(result.snap.snapped).toBe(true);
    expect(result.route).toEqual([p(0, 0), p(100, 0), p(100, 160), p(400, 160)]);
    orthogonal(result.route);
  });

  it('反向末端横段压合后清理局部折返，但保留最后24单位出口', () => {
    const original = [p(0, 0), p(400, 0), p(400, 100), p(300, 100)];
    const result = drag(original, 2, p(0, -96));
    expect(result.snap.snapped).toBe(true);
    expect(result.route).toEqual([p(0, 0), p(276, 0), p(276, 100), p(300, 100)]);
    expect(result.raw.at(-2)).toMatchObject(p(276, 100));
  });

  it.each([0.5, 1, 2])('zoom=%s 按屏幕距离吸附，超阈值和 Alt 等效零阈值不吸附', (zoom) => {
    const near = p(0, -80 + 6 / zoom);
    expect(snapSegmentMovement(bump, 2, near, 12 / zoom).snapped).toBe(true);
    expect(snapSegmentMovement(bump, 2, p(0, -80 + 13 / zoom), 12 / zoom).snapped).toBe(false);
    const disabled = snapSegmentMovement(bump, 2, near, 0);
    expect(disabled.snapped).toBe(false);
    expect(disabled.coordinate).toBeCloseTo(6 / zoom);
    expect(disabled.delta).toEqual(near);
  });

  it('两侧邻段都在阈值内时选择距离最近者，不受遍历顺序影响', () => {
    const route = [p(0, 0), p(100, 0), p(100, 50), p(200, 50), p(200, 20), p(400, 20)];
    const result = snapSegmentMovement(route, 2, p(0, -38));
    expect(result.snapped).toBe(true);
    expect(result.coordinate).toBe(20);
    expect(result.delta.y).toBe(-30);
  });

  it('恰好12单位纳入吸附，略超过12单位不吸附', () => {
    const boundary = snapSegmentMovement(bump, 2, p(0, -68));
    expect(boundary.snapped).toBe(true);
    expect(boundary.coordinate).toBe(0);
    expect(snapSegmentMovement(bump, 2, p(0, -67.999999)).snapped).toBe(false);
  });

  it('小数坐标使用目标轴绝对坐标，吸附后每段严格正交且无微小折角', () => {
    const offset = p(-567.9823655302941, 253.28001887175606);
    const route = bump.map((q) => p(q.x + offset.x, q.y + offset.y));
    const result = drag(route, 2, p(0, -75));
    expect(result.snap.snapped).toBe(true);
    expect(result.snap.coordinate).toBe(offset.y);
    expect(result.route).toEqual([route[0], route.at(-1)!]);
    orthogonal(result.route);
  });

  it('只比较隔一个转弯的邻段；远处同轴线和没有邻段的直线不抢吸附', () => {
    const route = [
      p(0, 0),
      p(100, 0),
      p(100, 100),
      p(200, 100),
      p(200, 40),
      p(300, 40),
      p(300, 200),
      p(500, 200),
    ];
    const distant = snapSegmentMovement(route, 4, p(0, -35));
    expect(distant.snapped).toBe(false);
    expect(distant.coordinate).toBe(5);
    const straight = snapSegmentMovement([p(0, 0), p(300, 0)], 0, p(7, 9));
    expect(straight.snapped).toBe(false);
    expect(straight.coordinate).toBe(9);
  });

  it('吸附只清理本次坍缩区域，保留远处已存在的 U-turn', () => {
    const original = [
      p(0, 0),
      p(100, 0),
      p(100, 80),
      p(200, 80),
      p(200, 0),
      p(300, 0),
      p(260, 0),
      p(260, 160),
      p(500, 160),
    ];
    const result = drag(original, 2, p(0, -75));
    expect(result.route).toEqual([p(0, 0), p(300, 0), p(260, 0), p(260, 160), p(500, 160)]);
    orthogonal(result.route);
  });

  it('保存和撤销恢复线路，模块位置、逻辑连接及其他分支保持不变', () => {
    const project: Project = {
      schemaVersion: 1,
      name: 'self straightening',
      variants: [],
      graph: {
        nodes: [
          registry.create('Input', {}, 'source'),
          registry.create('Output', {}, 'target'),
          registry.create('Output', {}, 'other'),
        ],
        edges: [
          {
            id: 'moving',
            source: 'source',
            sourcePort: 'out',
            target: 'target',
            targetPort: 'in',
            waypoints: z.slice(1, -1),
            labelPosition: 0.3,
          },
          {
            id: 'untouched',
            source: 'source',
            sourcePort: 'out',
            target: 'other',
            targetPort: 'in',
            waypoints: [p(180, 0), p(180, 200)],
          },
        ],
      },
    };
    useEditor.setState({ project, hasProject: true, path: [], past: [], future: [] });
    const result = drag(z, 2, p(0, -95));
    useEditor.getState().setEdgeWaypoints('moving', result.pins);
    const saved = useEditor.getState().project;
    expect(saved.graph.nodes).toEqual(project.graph.nodes);
    expect(saved.graph.edges[1]).toEqual(project.graph.edges[1]);
    expect(saved.graph.edges[0]).toMatchObject({
      id: 'moving',
      source: 'source',
      sourcePort: 'out',
      target: 'target',
      targetPort: 'in',
      labelPosition: 0.3,
    });
    expect(parseProject(serializeProject(saved), registry)).toEqual(saved);
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(project);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(saved);
  });
});
