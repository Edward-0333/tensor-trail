import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, patchModel, validateModel } from '../src/agent';
import { assignLayoutRegion, fitLayoutRegion, moveLayoutNodes, resizeLayoutRegion } from '../src/core/layoutRegions';

const fixture = () => buildModel({ name: 'Fractional regions', nodes: [
  { id: 'input', type: 'Input', params: { shape: '2,3', axes: 'B,C' } },
  { id: 'output', type: 'Output' },
  { id: 'region', type: 'LayoutRegion', position: { x: -5.537276193791968, y: 790.5466601464753 },
    params: { width: 672.5, height: 534.0000000000001 } },
], connections: [['input.out', 'output.in']] });

describe('Fractional layout geometry', () => {
  it('accepts and preserves existing subpixel region sizes through validation and API patches', () => {
    const built = fixture(); expect(built.ok).toBe(true);
    const region = built.project!.graph.nodes.find(n => n.id === 'region')!;
    const checked = validateModel(built.project!); expect(checked.ok).toBe(true);
    const changed = patchModel(built.project!, { operations: [
      { op: 'rename_node', id: 'region', label: 'Region' },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(changed.ok).toBe(true);
    expect(changed.project!.graph.nodes.find(n => n.id === 'region')!.params).toEqual(region.params);
    expect(changed.outputs).toEqual(built.outputs);
  });

  it('keeps auto-fit, fractional measured members and manual resize editable by the API', () => {
    const project = fixture().project!, sizes = { input: { width: 250.4, height: 160.7 } };
    assignLayoutRegion(project.graph, ['input'], 'region', sizes);
    expect(validateModel(project).ok).toBe(true);
    fitLayoutRegion(project.graph, 'region', sizes);
    expect(validateModel(project).ok).toBe(true);
    resizeLayoutRegion(project.graph, 'region', { x: -5.5, y: 790.5, width: 800.3, height: 600.8 }, sizes);
    expect(validateModel(project).ok).toBe(true);
    expect(patchModel(project, { operations: [{ op: 'set_params', id: 'region', params: { width: 900.25 } }] }).ok).toBe(true);
  });

  it('preserves manual routes and label placement on fractional rigid group translations', () => {
    const project = fixture().project!, graph = project.graph;
    const region = graph.nodes.find(n => n.id === 'region')!;
    region.params.members = 'input,output';
    region.position = { x: 490, y: 1070 };
    graph.nodes.find(n => n.id === 'input')!.position = { x: 1424.6639320328163, y: 1250 };
    graph.nodes.find(n => n.id === 'output')!.position = { x: 2206.6258121139153, y: 1425.0703706367901 };
    graph.edges[0].waypoints = [{ x: 2179.62584195681, y: 1232.429768329142 }];
    graph.edges[0].labelPosition = .386;
    const changed = patchModel(project, { operations: [
      { op: 'set_position', id: 'region', position: { x: 0, y: 1700 } },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(changed.ok).toBe(true);
    const edge = changed.project!.graph.edges[0];
    expect(edge.waypoints).toHaveLength(1);
    expect(edge.waypoints![0].x).toBeCloseTo(2179.62584195681 - 490, 8);
    expect(edge.waypoints![0].y).toBeCloseTo(1232.429768329142 + 630, 8);
    expect(edge.labelPosition).toBe(.386);
    expect(changed.outputs).toEqual(validateModel(project).outputs);
  });

  it('still clears routes when two endpoints actually move different distances', () => {
    const graph = fixture().project!.graph, region = graph.nodes.find(n => n.id === 'region')!;
    region.params.members = 'input,output';
    graph.edges[0].waypoints = [{ x: 150, y: 200 }];
    graph.edges[0].labelPosition = .4;
    const moves = graph.nodes.filter(n => ['input', 'output'].includes(n.id)).map((n, i) => ({
      id: n.id, position: { x: n.position.x + 100 + i * .001, y: n.position.y + 50 },
    }));
    moveLayoutNodes(graph, moves);
    expect(graph.edges[0].waypoints).toBeUndefined();
    expect(graph.edges[0].labelPosition).toBeUndefined();
  });

  it.each([0, -1, 199.5, 1000000.5, NaN, Infinity])('still rejects persisted invalid region height %s', height => {
    const built = fixture(); expect(built.ok).toBe(true);
    built.project!.graph.nodes.find(n => n.id === 'region')!.params.height = height;
    expect(validateModel(built.project!).ok).toBe(false);
  });

  it('only opts visual fields into fractional values; network dimensions remain integer-only', () => {
    const modules = describeModules().modules;
    expect(modules.find(m => m.type === 'LayoutRegion')!.paramsSchema.properties.width).toMatchObject({ type: 'number' });
    expect(modules.find(m => m.type === 'Linear')!.paramsSchema.properties.out_features).toMatchObject({ type: 'integer' });
    expect(buildModel({ name: 'Invalid dimensions', nodes: [{ id: 'linear', type: 'Linear', params: { out_features: 3.5 } }], connections: [] }, { mode: 'draft' }).ok).toBe(false);
  });
});
