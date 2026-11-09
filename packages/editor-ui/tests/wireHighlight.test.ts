import { describe, it, expect } from 'vitest';
import { branchJunctions, type Point, type RoutedEdge } from '../src/views/edgeRouting';
import { sharedSections, wireDrawing } from '../src/views/sharedWires';
import {
  selectWireSection,
  wireHighlightStarts,
  highlightedDrawing,
  nearestSection,
  pointHighlighted,
  junctionHighlighted,
} from '../src/views/wireHighlight';

const p = (x: number, y: number): Point => ({ x, y });
const edge = (id: string, points: Point[], extra: Partial<RoutedEdge> = {}): RoutedEdge => ({
  id,
  source: 'data',
  sourcePort: 'out',
  points,
  ...extra,
});
function tree() {
  return [
    edge('a', [p(0, 0), p(100, 0), p(100, 100), p(300, 100)]),
    edge('b', [p(0, 0), p(100, 0), p(100, 200), p(200, 200), p(200, 260), p(300, 260)]),
    edge('c', [p(0, 0), p(100, 0), p(100, 200), p(200, 200), p(200, 340), p(300, 340)], {
      selected: true,
    }),
    edge('d', [p(0, 0), p(100, 0), p(100, -100), p(300, -100)]),
  ];
}
const pick = (edges: RoutedEdge[], at: Point) => {
  const owner = edges.find((e) => e.selected)!;
  return selectWireSection(owner, edges, nearestSection(sharedSections(owner, edges), at));
};

describe('以分岔为界选择连线子树', () => {
  it('点源头干线高亮同一数据的全部分支，包括端口处分开的路线', () => {
    const edges = [...tree(), edge('another', [p(0, 0), p(0, 200), p(300, 200)])];
    const selected = pick(edges, p(40, 0));
    expect(new Set(selected.members)).toEqual(new Set(edges.map((e) => e.id)));
    expect([...wireHighlightStarts(edges, selected).values()]).toEqual([0, 0, 0, 0, 0]);
  });
  it('点中间干线只选中后续分支，之前分出的兄弟和上游保持原色', () => {
    const edges = tree(),
      selected = pick(edges, p(100, 150));
    expect(new Set(selected.members)).toEqual(new Set(['b', 'c']));
    const starts = wireHighlightStarts(edges, selected);
    expect([...starts]).toEqual([
      ['b', 200],
      ['c', 200],
    ]);
    expect(pointHighlighted(edges[2], p(100, 50), starts)).toBe(false);
    expect(pointHighlighted(edges[2], p(100, 170), starts)).toBe(true);
    expect(pointHighlighted(edges[1], p(270, 260), starts)).toBe(true);
    expect(pointHighlighted(edges[0], p(270, 100), starts)).toBe(false);
  });
  it('点末端分支只高亮它，从最近一个分岔点起算', () => {
    const edges = tree(),
      selected = pick(edges, p(270, 340));
    expect(selected.members).toEqual(['c']);
    expect([...wireHighlightStarts(edges, selected)]).toEqual([['c', 460]]);
  });
  it('明确关联的副本算同一数据；同名、同尺寸、不同端口都不能混选', () => {
    const edges = tree();
    edges.forEach((e) => {
      e.valueKey = '["original-T","out"]';
    });
    edges.push(
      edge('alias', [p(500, 0), p(600, 0)], {
        source: 'alias-T',
        valueKey: '["original-T","out"]',
      }),
    );
    edges.push(
      edge('different-T', [p(500, 50), p(600, 50)], {
        source: 'independent-T',
        valueKey: '["independent-T","out"]',
      }),
    );
    edges.push(
      edge('other-port', edges[2].points, {
        sourcePort: 'other',
        valueKey: '["original-T","other"]',
      }),
    );
    expect(pick(edges, p(40, 0)).members).toContain('alias');
    expect(pick(edges, p(40, 0)).members).not.toContain('different-T');
    expect(pick(edges, p(40, 0)).members).not.toContain('other-port');
    expect(pick(edges, p(100, 150)).members).not.toContain('alias');
  });
  it('无关数据即使路径完全重叠也不高亮，且不跨越计算节点继续传播', () => {
    const edges = tree();
    edges.push(edge('unrelated', edges[2].points, { source: 'another' }));
    edges.push(edge('calculated', [p(300, 340), p(600, 340)], { source: 'consumer' }));
    const starts = wireHighlightStarts(edges, pick(edges, p(40, 0)));
    expect(starts.has('unrelated')).toBe(false);
    expect(starts.has('calculated')).toBe(false);
  });
  it('移动分岔后保持同一组分支，并从新的分岔点开始高亮', () => {
    const edges = tree(),
      selected = pick(edges, p(100, 150));
    const moved = edges.map((e) => ({
      ...e,
      points: e.points.map((p) => ({ x: p.x === 100 ? 130 : p.x, y: p.y })),
    }));
    expect([...wireHighlightStarts(moved, selected)]).toEqual([
      ['b', 230],
      ['c', 230],
    ]);
  });
  it('取消选择或删除主连接后清除联动高亮；键盘选择保留单线高亮', () => {
    const edges = tree(),
      selected = pick(edges, p(40, 0));
    expect(
      wireHighlightStarts(
        edges.map((e) => ({ ...e, selected: false })),
        selected,
      ).size,
    ).toBe(0);
    expect(
      wireHighlightStarts(
        edges.filter((e) => e.id !== 'c'),
        selected,
      ).size,
    ).toBe(0);
    expect([...wireHighlightStarts(edges, null)]).toEqual([['c', 0]]);
  });
  it('分岔点与下游文字同步高亮，非下游分岔保持原色', () => {
    const edges = tree(),
      starts = wireHighlightStarts(edges, pick(edges, p(100, 150)));
    const junctions = branchJunctions(edges);
    expect(junctions.map((j) => [j.x, j.y, junctionHighlighted(j, edges, starts)])).toEqual(
      expect.arrayContaining([
        [100, 0, false],
        [100, 100, true],
        [200, 260, true],
      ]),
    );
  });
});

describe('公共线段仍只绘制一次', () => {
  it('主连接上游保留普通颜色，选中分支和后续区间使用高亮色', () => {
    const edges = tree(),
      starts = wireHighlightStarts(edges, pick(edges, p(100, 150)));
    const strokes = edges.map((e) =>
      highlightedDrawing(e.points, wireDrawing(e, edges).sharedLength, starts.get(e.id)),
    );
    expect(strokes[2]).toEqual({
      normal: 'M 0 0 L 100 0 L 100 100',
      highlighted: 'M 100 100 L 100 200 L 200 200 L 200 340 L 300 340',
    });
    expect(strokes[1]).toEqual({
      normal: '',
      highlighted: 'M 200 260 L 300 260',
    });
    expect(strokes[0].highlighted).toBe('');
    expect(strokes[3].highlighted).toBe('');
  });
  it('所有线段都属于公共区间时，不重复画高亮笔画', () => {
    expect(highlightedDrawing([p(0, 0), p(100, 0)], 100, 0)).toEqual({
      normal: '',
      highlighted: '',
    });
  });
  it('分岔坐标位于长线段中间时准确切开，不丢失上游笔画', () => {
    expect(highlightedDrawing([p(0, 0), p(100, 0), p(100, 200)], 40, 170)).toEqual({
      normal: 'M 40 0 L 100 0 L 100 70',
      highlighted: 'M 100 70 L 100 200',
    });
  });
  it('鼠标靠近折角或分岔时选最近的下游段，兼容负坐标和逆向路线', () => {
    const points = [p(100, 100), p(-50, 100), p(-50, -100)];
    expect(nearestSection(points, p(0, 104))).toBe(0);
    expect(nearestSection(points, p(-52, -40))).toBe(1);
    expect(nearestSection(points, p(-50, 100))).toBe(1);
  });
});
