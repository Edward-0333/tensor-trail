import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';
import { describeModules, patchModel, validateModel } from '../src/agent';
import { layoutRegionImportance, layoutRegionIssues } from '../src/core/layoutRegions';
import { parseProject, serializeProject } from '../src/core/serialization';
import { copySelection, duplicateSelection } from '../src/editing/clipboard';

function fixture() {
  const project = example();
  project.graph.nodes.push(registry.create('LayoutRegion', { members: 'transformer', outputs: 'transformer' }, 'region'));
  const block = project.graph.nodes.find(n => n.id === 'transformer')!;
  block.subgraph!.nodes.push(registry.create('LayoutRegion', {}, 'region'));
  return project;
}

describe('Level-local region importance', () => {
  it('accepts ungraded legacy regions without injecting scores or moving geometry', () => {
    const p = fixture(), region = p.graph.nodes.at(-1)!;
    delete region.params.importance; delete region.params.importance_note;
    expect(layoutRegionImportance(region)).toBe(0);
    expect(validateModel(p).ok).toBe(true);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);
  });

  it.each([0, 1, 2, 3, 4, 5])('persists %i stars and a reason through the API without semantic changes', importance => {
    const p = fixture(), before = structuredClone(p), original = validateModel(p);
    const result = patchModel(p, { operations: [{ op: 'set_params', id: 'region',
      params: { importance, importance_note: '本层核心特征计算' } }] }, { mode: 'draft', layout: 'preserve' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const expected = structuredClone(p);
    Object.assign(expected.graph.nodes.at(-1)!.params, { importance, importance_note: '本层核心特征计算' });
    expect(result.project).toEqual(expected);
    expect(result.outputs).toEqual(original.outputs);
    expect(result.nodes.map(n => n.outputs)).toEqual(original.nodes.map(n => n.outputs));
    expect(parseProject(serializeProject(result.project!), registry)).toEqual(expected);
    expect(p).toEqual(before);
  });

  it('treats same IDs in different graphs independently and does not customize the parent', () => {
    const p = fixture();
    const result = patchModel(p, { operations: [
      { op: 'set_params', id: 'region', params: { importance: 5 } },
      { op: 'set_params', path: ['transformer'], id: 'region', params: { importance: 1 } },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const root = result.project!.graph, block = root.nodes.find(n => n.id === 'transformer')!;
    expect(layoutRegionImportance(root.nodes.at(-1)!)).toBe(5);
    expect(layoutRegionImportance(block.subgraph!.nodes.at(-1)!)).toBe(1);
    expect(block.customized).toBeFalsy();
    expect(block.params).toEqual(p.graph.nodes.find(n => n.id === block.id)!.params);
  });

  it.each([-1, 6, 1.5, '3', true])('rejects invalid scores %s in imports and API patches atomically', importance => {
    const p = fixture(), before = structuredClone(p);
    expect(patchModel(p, { operations: [{ op: 'set_params', id: 'region', params: { importance } }] }).ok).toBe(false);
    expect(p).toEqual(before);
    p.graph.nodes.at(-1)!.params.importance = importance;
    expect(layoutRegionIssues(p.graph).join()).toContain('重要性');
    expect(() => parseProject(serializeProject(p), registry)).toThrow();
  });

  it('bounds reasons and advertises the score range in the API schema', () => {
    const p = fixture();
    for (const importance_note of [true, 'x'.repeat(241)])
      expect(patchModel(p, { operations: [{ op: 'set_params', id: 'region', params: { importance_note } }] }).ok).toBe(false);
    const schema = describeModules().modules.find(m => m.type === 'LayoutRegion')!.paramsSchema;
    expect(schema.properties.importance).toMatchObject({ type: 'integer', minimum: 0, maximum: 5, default: 0 });
  });

  it('copies annotations with a region but does not attach a rating to its member nodes', () => {
    const p = fixture(), region = p.graph.nodes.at(-1)!;
    region.params.importance = 4; region.params.importance_note = '特征提取';
    const fragment = duplicateSelection(copySelection(p.graph, ['region']), 20);
    const copied = fragment.nodes.find(n => n.type === 'LayoutRegion')!;
    expect(copied.params.importance).toBe(4); expect(copied.params.importance_note).toBe('特征提取');
    expect(fragment.nodes.filter(n => n.type !== 'LayoutRegion').every(n => n.params.importance === undefined)).toBe(true);
  });
});
