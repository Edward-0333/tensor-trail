import { describe, expect, it } from 'vitest';
import { getSchemas, patchModel, validateModel } from '../src/agent';
import { registry } from '../src/modules';
import type { Project } from '../src/core/types';
import { rootInputSources } from '../src/core/rootInputs';

function fixture(): Project {
  const input = registry.create('ForwardInput', {}, 'img_metas');
  input.label = 'img_metas';
  const copy = { ...structuredClone(input), id: 'near_head', position: { x: 400, y: 500 },
    portLayout: { output: 'bottom' as const }, tensorBorderColor: '#ffcc00' };
  const region = registry.create('LayoutRegion', { members: 'img_metas,near_head' }, 'region');
  return { schemaVersion: 1, name: 'Repair input', variants: [], graph: { nodes: [input, copy, region,
    registry.create('Output', {}, 'output')], edges: [{ id: 'existing', source: 'near_head', sourcePort: 'out',
      target: 'output', targetPort: 'in', waypoints: [{ x: 710, y: 540 }], labelPosition: .3 }] } };
}

describe('Link an existing value through the API', () => {
  it('restores one input identity without moving nodes, dropping wires or changing region membership', () => {
    const before = fixture();
    expect(rootInputSources(before.graph)).toHaveLength(2);
    const result = patchModel(before, { operations: [{ op: 'link_value', id: 'near_head', source: 'img_metas' }] },
      { mode: 'draft', layout: 'preserve' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const expected = structuredClone(before); expected.graph.nodes[1].valueSource = 'img_metas';
    expect(result.project).toEqual(expected);
    expect(before.graph.nodes[1].valueSource).toBeUndefined();
    expect(rootInputSources(result.project!.graph)).toHaveLength(1);
    expect(validateModel(result.project!, { mode: 'draft' }).ok).toBe(true);
    expect(JSON.stringify(getSchemas().patch)).toContain('link_value');
  });

  it('resolves source aliases directly instead of creating a chain', () => {
    const project = fixture();
    project.graph.nodes.push({ ...structuredClone(project.graph.nodes[0]), id: 'reference', valueSource: 'img_metas' });
    const result = patchModel(project, { operations: [{ op: 'link_value', id: 'near_head', source: 'reference' }] }, { mode: 'draft' });
    expect(result.ok).toBe(true);
    expect(result.project!.graph.nodes[1].valueSource).toBe('img_metas');
  });

  it.each([
    ['near_head', 'missing'], ['missing', 'img_metas'], ['img_metas', 'img_metas'],
    ['region', 'img_metas'], ['output', 'img_metas'],
  ])('rejects invalid bindings atomically: %s -> %s', (id, source) => {
    const project = fixture(), before = structuredClone(project);
    expect(patchModel(project, { operations: [{ op: 'link_value', id, source }] }, { mode: 'draft' }).ok).toBe(false);
    expect(project).toEqual(before);
  });

  it('refuses to discard incoming data or strand existing aliases', () => {
    const p = fixture();
    p.graph.nodes.push(registry.create('Output', {}, 'other_output'));
    expect(patchModel(p, { operations: [{ op: 'link_value', id: 'output', source: 'other_output' }] }, { mode: 'draft' }).ok).toBe(false);
    p.graph.nodes.push({ ...structuredClone(p.graph.nodes[1]), id: 'dependent', valueSource: 'near_head' });
    expect(patchModel(p, { operations: [{ op: 'link_value', id: 'near_head', source: 'img_metas' }] }, { mode: 'draft' }).ok).toBe(false);
  });

  it('does not silently merge existing child interfaces', () => {
    const parent = registry.create('Transformer', {}, 'parent');
    const input = parent.subgraph!.nodes.find(n => n.type === 'Inlet')!;
    parent.subgraph!.nodes.push(registry.create('Inlet', { port: 'extra' }, 'extra'));
    const p: Project = { schemaVersion: 1, name: 'nested', variants: [], graph: { nodes: [parent], edges: [] } };
    expect(patchModel(p, { operations: [{ op: 'link_value', path: ['parent'], id: 'extra', source: input.id }] },
      { mode: 'draft' }).ok).toBe(false);
  });
});
