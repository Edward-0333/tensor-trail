import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject, validateProjectObject } from '../src/core/serialization';
import { setNodeTensorVisual, validateTensorVisual } from '../src/core/tensorVisual';
import { createLinkedPaste } from '../src/editing/valueLinkEditing';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';

const visual = {
  mode: 'block' as const,
  tone: 'amber' as const,
  color: '#3b82F6',
  depthStyle: 'solid' as const,
  width: 420,
  boxWidth: 100.5,
  boxHeight: 29.5,
  boxDepth: 32,
  axisLabels: ['queue', 'query', 'head', 'dim'],
  caption: '输入特征',
};
const recipe = {
  name: 'Geometry',
  nodes: [
    { id: 'x', type: 'Input' },
    { id: 'y', type: 'Output' },
  ],
  connections: [['x.out', 'y.in']],
};
const op = { op: 'set_node_tensor_visual', id: 'x', visual };

describe('tensorVisual metadata', () => {
  it('round trips recipe, patch and clear without a version bump or inferred/layout changes', () => {
    const original = buildModel(recipe).project!;
    const edited = patchModel(original, { operations: [op] }).project!;
    const built = buildModel({
      ...recipe,
      nodes: [{ ...recipe.nodes[0], tensorVisual: visual }, recipe.nodes[1]],
    }).project!;
    expect(built).toEqual(edited);
    expect(parseProject(serializeProject(edited), registry)).toEqual(edited);
    expect(inferProject(original, registry)).toEqual(inferProject(edited, registry));
    expect(patchModel(edited, { operations: [{ ...op, visual: null }] }).project).toEqual(original);
    expect(serializeProject(original)).not.toContain('tensorVisual');
  });

  it.each([
    {},
    [],
    null,
    { mode: 'on' },
    { ...visual, tone: 'red' },
    { ...visual, caption: 42 },
    { ...visual, caption: '字'.repeat(121) },
    { ...visual, width: '150' },
    { ...visual, width: 139.9 },
    { ...visual, width: 421 },
    { ...visual, width: Infinity },
    { ...visual, width: NaN },
    { ...visual, url: 'img' },
    ...[123, 'red', '#abc', '#abcdeg', '#abcdef00'].map((color) => ({ ...visual, color })),
    { ...visual, depthStyle: 'stacked' },
    ...[15.9, 160.1, NaN].map((boxWidth) => ({ ...visual, boxWidth })),
    ...[11.9, 120.1, Infinity].map((boxHeight) => ({ ...visual, boxHeight })),
    ...[5.9, 72.1, '24'].map((boxDepth) => ({ ...visual, boxDepth })),
    ...['axis', [''], ['  '], [3], ['🧊'.repeat(33)], Array(9).fill('x')].map((axisLabels) => ({
      ...visual,
      axisLabels,
    })),
  ])('rejects malformed stored and recipe configs %#', (invalid) => {
    const project = buildModel(recipe).project!;
    Object.assign(project.graph.nodes[0], { tensorVisual: invalid });
    expect(() => validateProjectObject(project, registry)).toThrow();
    expect(
      buildModel({
        ...recipe,
        nodes: [{ ...recipe.nodes[0], tensorVisual: invalid }, recipe.nodes[1]],
      }).ok,
    ).toBe(false);
    if (invalid !== null)
      expect(
        patchModel(buildModel(recipe).project!, { operations: [{ ...op, visual: invalid }] }).ok,
      ).toBe(false);
  });

  it('reads legacy project previews as blocks and rejects obsolete settings in new edits', () => {
    const project = buildModel(recipe).project!;
    for (const mode of ['operation', 'block', 'off']) {
      const legacy = { ...visual, mode, width: 360 };
      Object.assign(project.graph.nodes[0], { tensorVisual: legacy });
      expect(parseProject(JSON.stringify(project), registry).graph.nodes[0].tensorVisual).toEqual({
        ...visual,
        mode: mode === 'operation' ? 'block' : mode,
        width: 360,
      });
    }
    Object.assign(project.graph.nodes[0], { tensorVisual: { mode: 'operation', width: 500 } });
    expect(parseProject(JSON.stringify(project), registry).graph.nodes[0].tensorVisual).toEqual({
      mode: 'block',
      width: 420,
    });
    for (const invalid of [
      { ...visual, mode: 'operation' },
      { ...visual, width: 421 },
    ]) {
      expect(
        buildModel({
          ...recipe,
          nodes: [{ ...recipe.nodes[0], tensorVisual: invalid }, recipe.nodes[1]],
        }).ok,
      ).toBe(false);
      expect(
        patchModel(buildModel(recipe).project!, { operations: [{ ...op, visual: invalid }] }).ok,
      ).toBe(false);
    }
    expect(getSchemas().recipe.$defs.tensorVisual.properties.mode).toMatchObject({
      enum: ['off', 'block'],
    });
  });

  it('clones settings, accepts inclusive bounds and 120 Unicode characters', () => {
    const node = registry.create('Input');
    const wrap = { ...visual, width: 140, caption: '🧊'.repeat(120) };
    setNodeTensorVisual(node, wrap);
    wrap.width = 420;
    validateTensorVisual(node, wrap);
    expect(node.tensorVisual?.width).toBe(140);
    validateTensorVisual(node, {
      ...wrap,
      boxWidth: 16,
      boxHeight: 12,
      boxDepth: 6,
      axisLabels: [],
    });
    validateTensorVisual(node, {
      ...wrap,
      boxWidth: 160,
      boxHeight: 120,
      boxDepth: 72,
      axisLabels: Array(8).fill('🧊'.repeat(32)),
    });
    setNodeTensorVisual(node, null);
    expect(node).not.toHaveProperty('tensorVisual');
  });

  it('validates recipes and metadata patches against the published schemas', () => {
    const schemas = getSchemas();
    const ajv = new Ajv2020({ strict: true, inlineRefs: false });
    const validVisual = ajv.compile(schemas.recipe.$defs.tensorVisual);
    expect(validVisual({ ...visual, axisLabels: ['🧊'.repeat(32)] })).toBe(true);
    for (const fields of [
      { color: '#abc' },
      { depthStyle: 'plates' },
      { boxWidth: 161 },
      { boxHeight: 11 },
      { boxDepth: 73 },
      { axisLabels: ['🧊'.repeat(33)] },
      { axisLabels: ['  '] },
      { axisLabels: Array(9).fill('x') },
    ]) {
      expect(validVisual({ ...visual, ...fields })).toBe(false);
    }
    const recipeSchema = ajv.compile(schemas.recipe),
      patchSchema = ajv.compile(schemas.patch);
    expect(
      recipeSchema({
        ...recipe,
        nodes: [{ ...recipe.nodes[0], tensorVisual: visual }, recipe.nodes[1]],
      }),
    ).toBe(true);
    expect(
      recipeSchema({ ...recipe, nodes: [{ id: 'r', type: 'LoopRegion', tensorVisual: visual }] }),
    ).toBe(false);
    expect(patchSchema({ operations: [op, { ...op, visual: null }] })).toBe(true);
    expect(patchSchema({ operations: [{ ...op, visual: { mode: 'block', width: 421 } }] })).toBe(
      false,
    );
  });

  it.each(['IfRegion', 'LoopRegion', 'LayoutRegion'])(
    'refuses %s in the schema, reader and setter',
    (type) => {
      const node = registry.create(type);
      expect(() => setNodeTensorVisual(node, visual)).toThrow();
      expect(() => setNodeTensorVisual(node, null)).toThrow();
      const project = buildModel(recipe).project!;
      project.graph.nodes.push({ ...node, tensorVisual: visual });
      expect(() => parseProject(JSON.stringify(project), registry)).toThrow(/积木/);
      const defs = getSchemas().recipe.$defs;
      expect(defs.tensorVisual.required).toContain('mode');
      expect(
        defs.node.oneOf.find((n) => (n.properties.type as { const: string }).const === type)
          ?.properties,
      ).not.toHaveProperty('tensorVisual');
      expect(
        defs.node.oneOf.find((n) => (n.properties.type as { const: string }).const === 'Input')
          ?.properties,
      ).toHaveProperty('tensorVisual');
    },
  );

  it('linked paste and duplicate_linked start with independent placement settings', () => {
    const edited = patchModel(buildModel(recipe).project!, { operations: [op] }).project!;
    const graph = edited.graph;
    expect(
      createLinkedPaste(graph, { nodes: [graph.nodes[0]], edges: [] }, 32, true, 'requested')
        .nodes[0].tensorVisual,
    ).toBeUndefined();
    const copy = patchModel(edited, {
      operations: [{ op: 'duplicate_linked', source: 'x', id: 'copy' }],
    }).project!;
    expect(copy.graph.nodes.at(-1)).not.toHaveProperty('tensorVisual');
    expect(copy.graph.nodes[0].tensorVisual).toEqual(visual);
  });

  it('preserves generated module artwork after a nonstructural patch and rebuild', () => {
    const project = example();
    const child = project.graph.nodes.find((n) => n.id === 'transformer')!.subgraph!.nodes[0];
    const edited = patchModel(project, {
      operations: [{ ...op, path: ['transformer'], id: child.id }],
    }).project!;
    expect(edited.graph.nodes.find((n) => n.id === 'transformer')?.customized).toBeUndefined();
    const rebuilt = patchModel(edited, {
      operations: [{ op: 'set_params', id: 'transformer', params: { heads: 4 } }],
    }).project!;
    expect(
      rebuilt.graph.nodes
        .find((n) => n.id === 'transformer')
        ?.subgraph?.nodes.find((n) => n.id === child.id)?.tensorVisual,
    ).toEqual(visual);
  });
});
