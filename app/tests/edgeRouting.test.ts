import { describe, expect, it } from 'vitest';
import type { PortSide } from '../src/core/types';
import { manualRoute, routeInsertIndex, routeMidpoint, routePath } from '../src/views/edgeRouting';

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
