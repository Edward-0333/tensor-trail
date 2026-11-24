import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel, validateModel } from '../src/agent';
import type { Operation } from '../src/agent';
import {
  autoSimplifyCalculations,
  canMergeCalculationSet,
  isGlueCalculation,
  mergeCalculationSet,
  splitCalculationGroup,
} from '../src/core/calculationGroups';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Graph } from '../src/core/types';
import { registry } from '../src/modules';

const chain = ['field', 'index', 'expand', 'zero', 'where'];
function fixture() {
  const built = buildModel({
    name: '掩码位置',
    nodes: [
      {
        id: 'arg',
        type: 'ForwardInput',
        params: {
          schema: {
            kind: 'dict',
            fields: {
              xy: { shape: [2, 4, 2], axes: ['batch', 'time', 'xy'], dtype: 'float32' },
            },
          },
        },
      },
      {
        id: 'mask',
        type: 'Input',
        params: { shape: '2,2,1', axes: 'batch,xy,singleton', dtype: 'bool' },
      },
      { id: 'field', type: 'DictionaryGet', params: { path: 'xy' } },
      { id: 'index', type: 'TensorIndex', params: { dim: '1', index: '0' } },
      { id: 'expand', type: 'Unsqueeze' },
      { id: 'zero', type: 'ZerosLike' },
      { id: 'where', type: 'Where' },
      { id: 'y', type: 'Output' },
    ],
    connections: [
      ['arg.out', 'field.in'],
      ['field.out', 'index.in'],
      ['index.out', 'expand.in'],
      ['expand.out', 'zero.in'],
      ['expand.out', 'where.x'],
      ['zero.out', 'where.y'],
      ['mask.out', 'where.condition'],
      ['where.out', 'y.in'],
    ],
  });
  expect(built).toMatchObject({ ok: true, diagnostics: [] });
  return built.project!;
}
function node(g: Graph, id: string) {
  return g.nodes.find((n) => n.id === id)!;
}

describe('batch calculation groups', () => {
  it.each([
    [],
    ['field'],
    ['field', 'field'],
    ['field', 'missing'],
    ['mask', 'where'],
    ['field', 'where'],
    ['expand', 'where'],
  ])('rejects an invalid set without a partial edit (%j)', (...ids) => {
    const g = fixture().graph,
      before = structuredClone(g);
    // Vitest spreads each parameter row.
    expect(canMergeCalculationSet(g, ids, registry)).toBe(false);
    expect(() => mergeCalculationSet(g, ids, registry)).toThrow();
    expect(g).toEqual(before);
  });
  it.each(['IfRegion', 'LoopRegion', 'LayoutRegion'])('rejects cross-%s membership', (type) => {
    const g = fixture().graph;
    g.nodes.push(registry.create(type, { members: 'field,index' }, 'region'));
    expect(canMergeCalculationSet(g, ['index', 'field'], registry)).toBe(true);
    expect(canMergeCalculationSet(g, chain, registry)).toBe(false);
  });
  it('contracts a convex selection with internal branches, flattens groups, and restores exact wiring, layout and declared roles', () => {
    const p = fixture(),
      g = p.graph;
    for (const e of g.edges) {
      e.waypoints = [{ x: -10, y: 22 }];
      e.labelPosition = 0.7;
    }
    g.nodes.push(
      registry.create(
        'LayoutRegion',
        { members: chain.join(','), outputs: 'index,where' },
        'region',
      ),
      registry.create('IfRegion', { members: chain.join(',') }, 'scope'),
    );
    const original = structuredClone(g);
    const group = mergeCalculationSet(g, ['index', 'field'], registry, { id: 'first' });
    const ids = [group, 'expand', 'zero', 'where'];
    const centers = ids.map((id) => node(g, id).position);
    expect(canMergeCalculationSet(g, ids, registry)).toBe(true);
    const summary = '掩码选择',
      formula = 'where(mask, x, 0)';
    mergeCalculationSet(g, ids, registry, { id: 'block', summary, formula });
    const block = node(g, 'block');
    expect(block).toMatchObject({
      label: summary,
      params: { summary, formula },
      position: {
        x: centers.reduce((sum, p) => sum + p.x, 0) / 4,
        y: centers.reduce((sum, p) => sum + p.y, 0) / 4,
      },
    });
    expect(ports(block, registry).outputs).toHaveLength(2);
    expect(block.subgraph!.nodes.map((n) => n.type)).not.toContain('CalculationGroup');
    expect(node(g, 'region').params).toMatchObject({ members: 'block', outputs: 'block' });
    expect(node(g, 'scope').params.members).toBe('block');
    const restored = parseProject(serializeProject(p), registry);
    splitCalculationGroup(restored.graph, 'block');
    const sorted = (graph: Graph) => ({
      nodes: [...graph.nodes].sort((a, b) => a.id.localeCompare(b.id)),
      edges: [...graph.edges].sort((a, b) => a.id.localeCompare(b.id)),
    });
    expect(sorted(restored.graph)).toEqual(sorted(original));
    expect(validateModel(restored)).toMatchObject({ ok: true, diagnostics: [] });
  });
  it('flattens authored identity groups even when the only data path runs directly between boundaries', () => {
    const group = (id: string) => ({
      type: 'CalculationGroup',
      id,
      subgraph: {
        nodes: [
          { id: 'entry', type: 'Inlet', params: { port: 'x' } },
          { id: 'exit', type: 'Outlet', params: { port: 'y' } },
        ],
        connections: [['entry.out', 'exit.in']],
      },
    });
    const built = buildModel({
      name: '透传',
      nodes: [{ id: 'x', type: 'Input' }, group('a'), group('b'), { id: 'y', type: 'Output' }],
      connections: [
        ['x.out', 'a.x'],
        ['a.y', 'b.x'],
        ['b.y', 'y.in'],
      ],
    });
    const p = built.project!;
    const id = mergeCalculationSet(p.graph, ['a', 'b'], registry);
    expect(node(p.graph, id).subgraph!.nodes.map((n) => n.type)).toEqual(['Inlet', 'Outlet']);
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [], outputs: built.outputs });
    splitCalculationGroup(p.graph, id);
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [], outputs: built.outputs });
  });
  it('accepts explicit position, gives label precedence and rejects conflicting IDs and excessive metadata atomically', () => {
    const g = fixture().graph;
    expect(() => mergeCalculationSet(g, chain, registry, { id: 'mask' })).toThrow(/ID/);
    expect(() => mergeCalculationSet(g, chain, registry, { formula: '长'.repeat(241) })).toThrow(
      /240/,
    );
    const position = { x: -52, y: 16 };
    const group = mergeCalculationSet(g, chain, registry, {
      position,
      label: '分量',
      summary: '按掩码选择',
    });
    expect(node(g, group)).toMatchObject({ label: '分量', position });
    g.nodes.push(registry.create('Input', {}, 'index'));
    const ids = splitCalculationGroup(g, group);
    expect(new Set(g.nodes.map((n) => n.id)).size).toBe(g.nodes.length);
    expect(ids).not.toContain('index');
  });
});

describe('automatic glue simplification', () => {
  it('collapses the whole masked diamond, preserves its output role, and round-trips through strict inference', () => {
    const p = fixture(),
      g = p.graph;
    g.nodes.push(
      registry.create('LayoutRegion', { members: chain.join(','), outputs: 'where' }, 'region'),
    );
    const before = validateModel(p);
    const { mergedGroups, removedNodes } = autoSimplifyCalculations(g, registry);
    expect(removedNodes).toBe(4);
    expect(mergedGroups).toHaveLength(1);
    const block = node(g, mergedGroups[0]);
    expect(block.params.summary).toMatch(/掩码/);
    expect(block.params.formula).toContain('Where');
    expect(ports(block, registry).outputs).toHaveLength(1);
    expect(node(g, 'region').params.outputs).toBe(block.id);
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [], outputs: before.outputs });
    expect(autoSimplifyCalculations(g, registry, { recursive: true }).removedNodes).toBe(0);
    splitCalculationGroup(g, block.id);
    expect(node(g, 'region').params.outputs).toBe('where');
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [], outputs: before.outputs });
  });
  it.each([
    'Linear',
    'LayerNorm',
    'Embedding',
    'MultiheadAttention',
    'SiLU',
    'GELU',
    'Sigmoid',
    'TopK',
    'Softmax',
    'Concat',
    'Stack',
    'SemanticOp',
    'Block',
    'Submodule',
    'ForLoop',
    'ValueVariable',
    'Input',
    'Output',
    'Inlet',
    'Outlet',
  ])('keeps %s visible as a boundary', (type) => {
    const g = fixture().graph;
    const boundary = registry.create(type, {}, 'protected');
    expect(isGlueCalculation(boundary, registry)).toBe(false);
    g.nodes.push(boundary);
    g.edges.find((e) => e.target === 'y')!.target = 'protected';
    autoSimplifyCalculations(g, registry);
    expect(node(g, 'protected')).toBe(boundary);
  });
  it('preserves a strictly wired neural and selection pipeline, including both TopK outputs', () => {
    const stages = [
      'Input',
      'Contiguous',
      'Function',
      'Linear',
      'LayerNorm',
      'Softmax',
      'TopK',
      'Concat',
      'SemanticOp',
      'Output',
    ];
    const built = buildModel({
      name: 'TopK pipeline',
      nodes: [...stages.map((type) => ({ id: type, type })), { id: 'indices', type: 'Output' }],
      connections: [
        ...stages.slice(1, 7).map((type, i) => [stages[i] + '.out', type + '.in']),
        ['TopK.indices', 'indices.in'],
        ['TopK.values', 'Concat.a'],
        ['TopK.values', 'Concat.b'],
        ['Concat.out', 'SemanticOp.x'],
        ['SemanticOp.out', 'Output.in'],
      ],
    });
    expect(built).toMatchObject({ ok: true, diagnostics: [] });
    const project = built.project!;
    expect(autoSimplifyCalculations(project.graph, registry).removedNodes).toBe(1);
    expect(project.graph.nodes.map((n) => n.type)).toEqual(expect.arrayContaining(stages.slice(3)));
    expect(validateModel(project)).toMatchObject({
      ok: true,
      diagnostics: [],
      outputs: built.outputs,
    });
  });

  it('preserves shared producers and never absorbs a marked layout output as an intermediate', () => {
    const p = fixture(),
      g = p.graph;
    g.nodes.push(
      registry.create('LayoutRegion', { members: chain.join(','), outputs: 'index' }, 'region'),
    );
    for (const id of ['a', 'b']) {
      g.nodes.push(registry.create('Output', {}, id));
      g.edges.push({ id, source: 'expand', target: id, sourcePort: 'out', targetPort: 'in' });
    }
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [] });
    const { mergedGroups } = autoSimplifyCalculations(g, registry);
    expect(mergedGroups).toHaveLength(2);
    expect(node(g, 'expand')).toBeDefined();
    expect(mergedGroups.map((id) => ports(node(g, id), registry).outputs.length)).toEqual([1, 1]);
    const marked = node(g, String(node(g, 'region').params.outputs));
    expect(marked.subgraph!.nodes.some((n) => n.id === 'index')).toBe(true);
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [] });
  });
  it('honors minSize and recursive scope, and marks modified ancestors for generator protection', () => {
    const p = fixture(),
      outer = p.graph;
    const id = mergeCalculationSet(outer, chain, registry);
    const owner = node(outer, id);
    Object.assign(owner, registry.create('Block', {}, id), { subgraph: owner.subgraph });
    delete owner.calculationOrigin;
    delete owner.customized;
    expect(autoSimplifyCalculations(outer, registry).removedNodes).toBe(0);
    expect(
      autoSimplifyCalculations(outer, registry, { recursive: true, minSize: 6 }).removedNodes,
    ).toBe(0);
    expect(owner.customized).toBeUndefined();
    expect(autoSimplifyCalculations(outer, registry, { recursive: true }).removedNodes).toBe(4);
    expect(owner.customized).toBe(true);
    expect(validateModel(p)).toMatchObject({ ok: true, diagnostics: [] });
  });
  it('preserves factory metadata inputs when folding a constant with its consumers', () => {
    const p = buildModel({
      name: '常量',
      nodes: [
        { id: 'x', type: 'Input' },
        { id: 'zeros', type: 'Zeros', params: { shape: '2' } },
        { id: 'expand', type: 'Unsqueeze' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['x.out', 'zeros.device_like'],
        ['zeros.out', 'expand.in'],
        ['expand.out', 'out.in'],
      ],
    });
    expect(p.diagnostics).toEqual([]);
    const graph = p.project!.graph;
    const { mergedGroups } = autoSimplifyCalculations(graph, registry);
    expect(ports(node(graph, mergedGroups[0]), registry).inputs).toHaveLength(1);
    expect(validateModel(p.project)).toMatchObject({
      ok: true,
      diagnostics: [],
      outputs: p.outputs,
    });
  });
});

describe('semantic patch API', () => {
  it('publishes all three schemas and patches nested and customized blocks as strict, atomic transactions', () => {
    const p = fixture(),
      before = validateModel(p);
    const operations: Operation[] = [
      { op: 'collapse_calculations', ids: chain, id: 'group', summary: '组'.repeat(240) },
      {
        op: 'set_params',
        id: 'group',
        params: { summary: '掩码清零', formula: 'where(mask,x,0)' },
      },
    ];
    const folded = patchModel(p, { operations });
    expect(folded).toMatchObject({ ok: true, diagnostics: [], outputs: before.outputs });
    expect(node(folded.project!.graph, 'group').params).toEqual({
      summary: '掩码清零',
      formula: 'where(mask,x,0)',
    });
    const expanded = patchModel(folded.project, {
      operations: [{ op: 'expand_calculation', id: 'group' }],
    });
    expect(expanded).toMatchObject({ ok: true, diagnostics: [], outputs: before.outputs });
    const nested = folded.project!;
    const owner = node(nested.graph, 'group');
    Object.assign(owner, registry.create('Block', {}, 'group'), { subgraph: owner.subgraph });
    delete owner.calculationOrigin;
    const result = patchModel(nested, {
      operations: [{ op: 'auto_simplify', path: ['group'], min_size: 3 }],
    });
    expect(result).toMatchObject({ ok: true, diagnostics: [], outputs: before.outputs });
    expect(
      node(result.project!.graph, 'group').subgraph!.nodes.filter(
        (n) => n.type === 'CalculationGroup',
      ),
    ).toHaveLength(1);
    const schema = JSON.stringify(getSchemas().patch);
    for (const op of ['collapse_calculations', 'expand_calculation', 'auto_simplify'])
      expect(schema).toContain(`"const":"${op}"`);
    expect(p).toEqual(before.project);
  });
  it.each([
    { op: 'auto_simplify', min_size: 1 },
    { op: 'auto_simplify', recursive: 'yes' },
    { op: 'expand_calculation', id: 'index' },
    { op: 'collapse_calculations', ids: chain, id: 'bad.id' },
    { op: 'collapse_calculations', ids: chain, formula: 9 },
  ])('rejects invalid patch %j without changing the project', (op) => {
    const p = fixture(),
      before = structuredClone(p);
    expect(patchModel(p, { operations: [op] }).ok).toBe(false);
    expect(p).toEqual(before);
  });
});
