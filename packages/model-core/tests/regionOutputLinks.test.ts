import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import type { Graph, Project } from '../src/core/types';
import { validateModel } from '../src/agent';
import { parseProject, serializeProject } from '../src/core/serialization';
import { assignLayoutRegion, layoutRegionOf, layoutRegionOutputOwners, layoutRegionOutputRegions } from '../src/core/layoutRegions';
import { removeValueNodes, unlinkValueInGraph } from '../src/editing/valueLinkEditing';

function fixture(): Project {
  const value = registry.create('TensorVariable', {}, 'value');
  const alias = { ...structuredClone(value), id: 'alias', valueSource: value.id, position: { x: 800, y: 100 } };
  const alias2 = { ...structuredClone(alias), id: 'alias2', position: { x: 1200, y: 100 } };
  const region = registry.create('LayoutRegion', { members: 'value', outputs: 'value' }, 'region');
  region.label = '视觉感知';
  return { schemaVersion: 1, name: 'Linked region outputs', variants: [], graph: {
    nodes: [registry.create('Input', { shape: '2,3', axes: 'B,C' }, 'img'), value, alias, alias2,
      registry.create('Output', {}, 'out'), region],
    edges: [
      { id: 'input', source: 'img', sourcePort: 'out', target: 'value', targetPort: 'in' },
      { id: 'output', source: 'alias', sourcePort: 'out', target: 'out', targetPort: 'in' },
    ],
  } };
}

const roles = (graph: Graph, id: string) => layoutRegionOutputRegions(graph).get(id)?.map(r => r.id) ?? [];

describe('Linked region output markers', () => {
  it('shows the owner role on every alias without writing metadata, adding membership or changing computation', () => {
    const p = fixture(), original = structuredClone(p), before = validateModel(p);
    expect(before.ok, JSON.stringify(before.diagnostics)).toBe(true);
    for (const id of ['value', 'alias', 'alias2']) expect(roles(p.graph, id)).toEqual(['region']);
    expect(roles(p.graph, 'out')).toEqual([]); // Wires are not value links.
    expect(layoutRegionOf(p.graph, 'alias')).toBeUndefined();
    expect([...layoutRegionOutputOwners(p.graph).keys()]).toEqual(['value']);
    expect(p).toEqual(original);
    expect(validateModel(p).outputs).toEqual(before.outputs);
    const restored = parseProject(serializeProject(p), registry);
    expect(roles(restored.graph, 'alias')).toEqual(['region']);
    expect(restored.graph.nodes.find(n => n.id === 'alias')).not.toHaveProperty('regionOutputs');
  });

  it('tracks name, color and output toggles from the current region definition', () => {
    const p = fixture(), region = p.graph.nodes.find(n => n.id === 'region')!;
    region.label = '自车状态编码'; region.params.color = '#60a5fa';
    expect(layoutRegionOutputRegions(p.graph).get('alias')![0]).toMatchObject({ label: region.label, params: { color: '#60a5fa' } });
    region.params.outputs = '';
    expect(roles(p.graph, 'alias')).toEqual([]);
    region.params.outputs = 'value';
    expect(roles(p.graph, 'alias')).toEqual(['region']);
  });

  it('retains a copy\'s own output role alongside the inherited role, without duplication', () => {
    const p = fixture();
    p.graph.nodes.push(registry.create('LayoutRegion', { members: 'alias', outputs: 'alias' }, 'other'));
    expect(roles(p.graph, 'alias')).toEqual(['other', 'region']);
    expect(roles(p.graph, 'value')).toEqual(['region']);
    expect(roles(p.graph, 'alias2')).toEqual(['region']);
    expect(layoutRegionOf(p.graph, 'alias')!.id).toBe('other');
    assignLayoutRegion(p.graph, ['alias'], 'region');
    p.graph.nodes.find(n => n.id === 'region')!.params.outputs = 'value,alias';
    expect(roles(p.graph, 'alias')).toEqual(['region']);
  });

  it('does not propagate a copy-only role backwards, to siblings, or to an unlinked namesake', () => {
    const p = fixture(), region = p.graph.nodes.find(n => n.id === 'region')!;
    region.params.members = 'value,alias'; region.params.outputs = 'alias';
    p.graph.nodes.push({ ...structuredClone(p.graph.nodes[1]), id: 'namesake' });
    expect(roles(p.graph, 'alias')).toEqual(['region']);
    for (const id of ['value', 'alias2', 'namesake']) expect(roles(p.graph, id)).toEqual([]);
  });

  it.each(['release', 'transfer', 'delete-owner', 'delete-region'])('clears inherited roles after %s', action => {
    const p = fixture();
    if (action === 'delete-owner') removeValueNodes(p.graph, [], ['value']);
    else if (action === 'delete-region') removeValueNodes(p.graph, [], ['region']);
    else {
      p.graph.nodes.push(registry.create('LayoutRegion', {}, 'other'));
      assignLayoutRegion(p.graph, ['value'], action === 'transfer' ? 'other' : null);
    }
    expect(roles(p.graph, 'alias')).toEqual([]);
    expect(roles(p.graph, 'alias2')).toEqual([]);
  });

  it('clears inherited markers after unlinking without affecting other views of the owner', () => {
    const p = fixture();
    unlinkValueInGraph(p.graph, [], 'alias');
    expect(roles(p.graph, 'alias')).toEqual([]);
    expect(roles(p.graph, 'alias2')).toEqual(['region']);
    expect(validateModel(p).ok).toBe(true);
  });

  it('only inherits through valid direct, same-graph and same-type value links', () => {
    const p = fixture(), alias = p.graph.nodes.find(n => n.id === 'alias')!;
    for (const source of ['missing', 'alias2', 'alias', 'img']) {
      alias.valueSource = source;
      expect(roles(p.graph, 'alias')).toEqual([]);
    }
    alias.valueSource = 'value';
    const inner: Graph = { nodes: [structuredClone(alias)], edges: [] };
    expect(roles(inner, 'alias')).toEqual([]);
    expect(roles(p.graph, 'alias')).toEqual(['region']);
  });
});
