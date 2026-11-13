import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import type { Project } from '../src/core/types';
import { patchModel, validateModel } from '../src/agent';
import { parseProject, serializeProject } from '../src/core/serialization';
import { layoutRegionOutputs, layoutRegionOutputOwners, layoutRegionIssues, assignLayoutRegion } from '../src/core/layoutRegions';
import { removeValueNodes } from '../src/editing/valueLinkEditing';
import { copySelection, duplicateSelection } from '../src/editing/clipboard';
import { mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';

function fixture(): Project {
  const nodes = [registry.create('Input', { shape: '2,3', axes: 'B,C' }, 'img'),
    registry.create('Detach', {}, 'a'), registry.create('Detach', {}, 'b'), registry.create('Output', {}, 'out'),
    registry.create('LayoutRegion', { members: 'a,b', outputs: 'b', note: '提取视觉信息', sizing: 'manual', width: 360, height: 200 }, 'region')];
  nodes.forEach((n, i) => { n.position = { x: i * 500, y: 100 }; });
  return { schemaVersion: 1, name: 'Regions', variants: [], graph: { nodes, edges: [
    { id: 'e0', source: 'img', sourcePort: 'out', target: 'a', targetPort: 'in' },
    { id: 'e1', source: 'a', sourcePort: 'out', target: 'b', targetPort: 'in' },
    { id: 'e2', source: 'b', sourcePort: 'out', target: 'out', targetPort: 'in', waypoints: [{ x: 1400, y: 80 }] },
  ] } };
}

describe('Layout region annotations', () => {
  it('edits name, note and outputs without changing any geometry, wires or calculated outputs', () => {
    const p = fixture();
    const result = patchModel(p, { operations: [
      { op: 'rename_node', id: 'region', label: '视觉感知' },
      { op: 'set_params', id: 'region', params: { note: '预测占用点及类别', outputs: 'a,b' } },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const expected = structuredClone(p); expected.graph.nodes[4].label = '视觉感知';
    Object.assign(expected.graph.nodes[4].params, { note: '预测占用点及类别', outputs: 'a,b' });
    expect(result.project).toEqual(expected);
    expect(result.outputs).toEqual(validateModel(p).outputs);
    expect(parseProject(serializeProject(result.project!), registry)).toEqual(expected);
  });

  it('accepts legacy regions without annotation fields and rejects invalid outputs or notes', () => {
    const p = fixture(), region = p.graph.nodes[4];
    delete region.params.note; delete region.params.outputs;
    expect(validateModel(p).ok).toBe(true);
    expect(layoutRegionOutputs(region)).toEqual([]);
    for (const outputs of ['missing', 'img', 'b,b', 'region']) {
      region.params.outputs = outputs; expect(layoutRegionIssues(p.graph).length).toBeGreaterThan(0);
    }
    region.params.outputs = 'b'; region.params.note = 'x'.repeat(241);
    expect(layoutRegionIssues(p.graph).length).toBeGreaterThan(0);
  });

  it('only marks explicitly designated members, not aliases elsewhere or all outgoing edges', () => {
    const p = fixture();
    const input = p.graph.nodes[0]; p.graph.nodes.push({ ...structuredClone(input), id: 'alias', valueSource: input.id });
    expect([...layoutRegionOutputOwners(p.graph).keys()]).toEqual(['b']);
  });

  it('prunes output roles on release, transfer and deletion', () => {
    for (const operation of ['release', 'transfer', 'delete']) {
      const p = fixture(), region = p.graph.nodes[4];
      p.graph.nodes.push(registry.create('LayoutRegion', {}, 'other'));
      if (operation === 'delete') removeValueNodes(p.graph, [], ['b']);
      else assignLayoutRegion(p.graph, ['b'], operation === 'transfer' ? 'other' : null);
      expect(layoutRegionOutputs(region)).toEqual([]);
      expect(layoutRegionIssues(p.graph)).toEqual([]);
    }
  });

  it('copies output references with a region and does not inherit them when only a node is copied', () => {
    const p = fixture(), fragment = copySelection(p.graph, ['region']);
    const copy = duplicateSelection(fragment, 50), region = copy.nodes.find(n => n.type === 'LayoutRegion')!;
    expect(region.params.note).toBe('提取视觉信息');
    expect(layoutRegionOutputs(region)).toHaveLength(1);
    expect(layoutRegionOutputs(region)[0]).not.toBe('b');
    expect(copy.nodes.some(n => n.id === layoutRegionOutputs(region)[0])).toBe(true);
    expect(layoutRegionIssues(copy)).toEqual([]);
    expect(layoutRegionOutputOwners(copySelection(p.graph, ['b'])).size).toBe(0);
  });

  it('updates explicit output roles when merging and splitting calculations', () => {
    const p = fixture(), region = p.graph.nodes[4];
    const merged = mergeCalculations(p.graph, 'a', 'b', registry);
    expect(layoutRegionOutputs(region)).toEqual([merged]);
    const nodes = splitCalculationGroup(p.graph, merged);
    const output = p.graph.edges.find(e => e.target === 'out')!.source;
    expect(nodes).toContain(output);
    expect(layoutRegionOutputs(region)).toEqual([output]);
    expect(layoutRegionIssues(p.graph)).toEqual([]);
  });

  it('supports declaring new members and outputs together without losing the new marker', () => {
    const p = fixture();
    const result = patchModel(p, { operations: [{ op: 'set_params', id: 'region', params: { members: 'a,out', outputs: 'out' } }] });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.project!.graph.nodes[4].params.outputs).toBe('out');
  });
});
