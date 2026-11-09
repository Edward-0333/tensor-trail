import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { branchJunctions, type RoutedEdge } from '../src/views/edgeRouting';
import { normalizeWireRoutes, ROUTE_NOISE } from '../src/views/routeNormalization';
import {
  sharedDragTargets,
  sharedPrefixLength,
  sharedSections,
  wireDrawing,
} from '../src/views/sharedWires';
import { selectWireSection, wireHighlightStarts } from '../src/views/wireHighlight';
import { createWirePlanner } from '../src/views/automaticWires';

// Five physical mask branches, reduced to coordinates only from the reported project.
const noisy: RoutedEdge[] = JSON.parse(
  readFileSync(new URL('./fixtures/noisyMaskRoutes.json', import.meta.url), 'utf8'),
);
const normalize = (edges: RoutedEdge[]) => {
  const paths = normalizeWireRoutes(edges);
  return edges.map((e) => ({ ...e, points: paths.get(e.id)! }));
};

describe('重叠线路的亚像素误差', () => {
  it('一个输出变化时，其余信号保留几何引用；删除旧组不会留下线路', () => {
    const plan = createWirePlanner();
    const inputs = noisy.map((e) => ({ ...e, automatic: false }));
    const unrelated = { ...inputs[0], id: 'unrelated', source: 'other' };
    const first = plan([...inputs, unrelated]);
    const changed = {
      ...inputs[0],
      points: inputs[0].points.map((p) => ({ x: p.x + 10, y: p.y })),
    };
    const second = plan([changed, ...inputs.slice(1), unrelated]);
    expect(second.get('unrelated')).toBe(first.get('unrelated'));
    expect(second.get(inputs[0].id)).not.toBe(first.get(inputs[0].id));
    expect(plan([unrelated]).has(inputs[0].id)).toBe(false);
    expect(plan([unrelated]).get('unrelated')).toBe(first.get('unrelated'));
  });
  it('复现原始微小折返造成的近源头伪分岔，恢复四个实际分岔点', () => {
    const original = structuredClone(noisy);
    expect(branchJunctions(noisy)).toHaveLength(3);
    const clean = normalize(noisy);
    const points = branchJunctions(clean);
    expect(points).toHaveLength(4);
    for (const x of [2888.1713609964636, 3285.5860193345115, 3455.585766531033])
      expect(
        points.some(
          (p) => Math.abs(p.x - x) < ROUTE_NOISE && Math.abs(p.y - 367.59375) < ROUTE_NOISE,
        ),
      ).toBe(true);
    expect(
      points.some(
        (p) =>
          Math.abs(p.x - 2888.1713609964636) < ROUTE_NOISE &&
          Math.abs(p.y - 412.04105717304844) < ROUTE_NOISE,
      ),
    ).toBe(true);
    expect(points.some((p) => p.x > 3500)).toBe(false); // Last corner is not a fork.
    expect(noisy).toEqual(original);
  });
  it('去重、高亮和拖动使用同一条共享前缀；中间公共段覆盖正确的两条分支', () => {
    const clean = normalize(noisy);
    expect(sharedPrefixLength(clean[1].points, clean[2].points)).toBeGreaterThan(400);
    expect(wireDrawing(clean[2], clean).sharedLength).toBe(
      sharedPrefixLength(clean[1].points, clean[2].points),
    );
    clean[2].selected = true;
    const sections = sharedSections(clean[2], clean);
    const index = sections.findIndex(
      (p, i) =>
        sections[i + 1] &&
        p.x > 3200 &&
        p.x < 3400 &&
        sections[i + 1].x > 3400 &&
        p.y === sections[i + 1].y,
    );
    expect(index).toBeGreaterThanOrEqual(0);
    const targets = sharedDragTargets(clean[2], sections, index, clean);
    expect(targets.map((t) => t.id).sort()).toEqual(['mask_2', 'mask_3']);
    const selection = selectWireSection(clean[2], clean, index);
    expect(selection.members.sort()).toEqual(['mask_2', 'mask_3']);
    expect([...wireHighlightStarts(clean, selection).keys()].sort()).toEqual(['mask_2', 'mask_3']);
  });
  it('端口坐标保持精确、路线保持直角；顺序变化与重复归一化均稳定', () => {
    const clean = normalize(noisy);
    for (let i = 0; i < clean.length; i++) {
      expect(clean[i].points[0]).toEqual(noisy[i].points[0]);
      expect(clean[i].points.at(-1)).toEqual(noisy[i].points.at(-1));
      expect(clean[i].points.length).toBeLessThan(noisy[i].points.length);
      expect(
        clean[i].points
          .slice(1)
          .every((p, j) => p.x === clean[i].points[j].x || p.y === clean[i].points[j].y),
      ).toBe(true);
    }
    expect(normalize(clean)).toEqual(clean);
    expect(normalize([...noisy].reverse()).reverse()).toEqual(clean);
  });
  it('不同来源和不同端口的重叠不生成连接点，单条线的折角也不生成点', () => {
    const clean = normalize(noisy);
    expect(branchJunctions([clean[0]])).toEqual([]);
    const separate = normalize(noisy.map((e, i) => ({ ...e, source: 'source_' + i })));
    expect(branchJunctions(separate)).toEqual([]);
    const ports = normalize(noisy.map((e, i) => ({ ...e, sourcePort: 'port_' + i })));
    expect(branchJunctions(ports)).toEqual([]);
  });
  it('只处理不可见坐标残差；有意错开的平行线和两个不同端口坐标不会被抹平', () => {
    const a: RoutedEdge = {
      id: 'a',
      source: 's',
      sourcePort: 'out',
      points: [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 300.003, y: 100 },
      ],
    };
    const b: RoutedEdge = {
      ...a,
      id: 'b',
      points: [
        { x: 0, y: 0 },
        { x: 24, y: 0 },
        { x: 24, y: 0.03 },
        { x: 150, y: 0.03 },
        { x: 150, y: 100.008 },
        { x: 400, y: 100.008 },
      ],
    };
    const clean = normalize([a, b]);
    expect(clean[1].points.some((p) => p.y === 0.03)).toBe(true);
    expect(clean[0].points.at(-1)).toEqual(a.points.at(-1));
    expect(clean[1].points.at(-1)).toEqual(b.points.at(-1));
    expect(sharedPrefixLength(clean[0].points, clean[1].points)).toBe(24);
  });
});
