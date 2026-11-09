import { describe, expect, it } from 'vitest';
import type { Graph, PortSide, Project } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { manualRoute, simplifyRoute, type Point, type RoutedEdge } from '../src/views/edgeRouting';
import { reconnectMovedPorts } from '../src/views/endpointRouting';
import { nodeDragRoutes, nodeWirePreview, snapNodesToWires } from '../src/editor/nodeWireSnap';
import { useEditor } from '../src/editor/store';

const p = (x: number, y: number): Point => ({ x, y });
const rotate = (point: Point, turns: number) => {
  let q = point;
  for (let i = 0; i < turns; i++) q = p(-q.y, q.x);
  return q;
};
const sides: PortSide[] = ['right', 'bottom', 'left', 'top'];
const orthogonal = (points: Point[]) =>
  points.slice(1).every((q, i) => q.x === points[i].x || q.y === points[i].y);

describe('移动模块时重新接入手动线路', () => {
  it.each([0, 1, 2, 3])('末端冗余控制点不超出向内移动的输入口，旋转 %i 个直角', (turns) => {
    const r = (q: Point) => rotate(q, turns);
    const route = [p(0, 0), p(100, 0), p(100, 200), p(500, 200), p(276, 200), p(300, 200)].map(r);
    const result = reconnectMovedPorts(
      route,
      sides[turns],
      sides[(turns + 2) % 4],
      undefined,
      r(p(-150, 0)),
    );
    expect(result).toEqual([p(0, 0), p(100, 0), p(100, 200), p(150, 200)].map(r));
    expect(orthogonal(result)).toBe(true);
    expect(route.at(-1)).toEqual(r(p(300, 200)));
  });
  it('目标越过原有末端折角时，最近的竖段移到端口外，远端转弯保持不变', () => {
    const route = [p(0, 0), p(50, 0), p(50, -80), p(180, -80), p(180, 200), p(300, 200)];
    const result = reconnectMovedPorts(route, 'right', 'left', undefined, p(-150, 20));
    expect(result).toEqual([p(0, 0), p(50, 0), p(50, -80), p(126, -80), p(126, 220), p(150, 220)]);
    expect(orthogonal(result)).toBe(true);
  });
  it('源端向右移时也收缩接出段；保留远处有意设置的绕行', () => {
    const route = [p(0, 0), p(80, 0), p(80, -100), p(180, -100), p(180, 60), p(300, 60)];
    const result = reconnectMovedPorts(route, 'right', 'left', p(100, 10));
    expect(result).toEqual([
      p(100, 10),
      p(124, 10),
      p(124, -100),
      p(180, -100),
      p(180, 60),
      p(300, 60),
    ]);
  });
  it('无实际位移时不整理旧折点，不修改传入路线', () => {
    const route = [p(0, 0), p(100, 0), p(100, 200), p(500, 200), p(300, 200)];
    const before = structuredClone(route);
    expect(reconnectMovedPorts(route, 'right', 'left', p(0, 0), p(0, 0))).toEqual(before);
    expect(route).toEqual(before);
  });
  it('原本整条为直线时，缩短后直接到新端点，不保留旧终点附近的固定控制点', () => {
    const result = reconnectMovedPorts(
      [p(0, 0), p(400, 0)],
      'right',
      'left',
      undefined,
      p(-250, 0),
    );
    expect(result).toEqual([p(0, 0), p(150, 0)]);
  });
  it('截图中的三控制点输入线在预览与提交时一致缩短，原图只占一步撤销', () => {
    const source = p(-444.982581215998, -79.71998112824392),
      target = p(301.37116020907195, 1093.2884715238315);
    const pins = [
      p(-420.982581215998, source.y),
      p(-420.982581215998, target.y),
      p(242.8711602090719, target.y),
    ];
    const input = { ...registry.create('Input', {}, 's'), position: source };
    const output = { ...registry.create('Output', {}, 'a'), position: target };
    const graph: Graph = {
      nodes: [input, output],
      edges: [
        {
          id: 'line',
          source: 's',
          sourcePort: 'out',
          target: 'a',
          targetPort: 'in',
          waypoints: pins,
          labelPosition: 0.7,
        },
      ],
    };
    const rendered: RoutedEdge[] = [
      {
        id: 'line',
        source: 's',
        sourcePort: 'out',
        sourceSide: 'right',
        targetSide: 'left',
        points: simplifyRoute(manualRoute(source, target, 'right', 'left', pins)),
      },
    ];
    const moving = [{ id: 'a', position: p(target.x - 230, target.y) }];
    const preview = nodeWirePreview(graph, rendered, graph.nodes, moving);
    const changes = nodeDragRoutes(graph, rendered, graph.nodes, moving, 1, true);
    expect(changes).toEqual(preview);
    expect(preview[0].waypoints).toEqual(pins.slice(0, 2));
    const live = simplifyRoute(
      manualRoute(source, moving[0].position, 'right', 'left', preview[0].waypoints),
    );
    expect(Math.max(...live.map((q) => q.x))).toBe(moving[0].position.x);
    const guides = snapNodesToWires(graph, rendered, graph.nodes, moving).guides;
    expect(guides).toHaveLength(1);
    expect(guides[0].from.x).toBeLessThan(moving[0].position.x);
    const project: Project = { schemaVersion: 1, name: 'endpoint', variants: [], graph };
    useEditor.setState({ project, hasProject: true, path: [], past: [], future: [] });
    useEditor.getState().finishNodeDrag(moving, undefined, changes);
    const after = useEditor.getState().project;
    expect(after.graph.edges[0].labelPosition).toBe(0.7);
    expect(parseProject(serializeProject(after), registry)).toEqual(after);
    expect(useEditor.getState().past).toHaveLength(1);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(project);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(after);
    expect(graph.edges[0].waypoints).toEqual(pins);
  });
});
