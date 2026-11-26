import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import {
  resolveSubmoduleVisual,
  setNodeSubmoduleVisual,
  validateSubmoduleVisual,
} from '../src/core/submoduleVisual';
import { preserveGeneratedMedia } from '../src/core/visualMedia';
import { registry } from '../src/modules';

const child = {
  nodes: [
    { id: 'x', type: 'Inlet' },
    { id: 'view', type: 'Identity' },
    { id: 'y', type: 'Outlet' },
  ],
  connections: [
    ['x.out', 'view.in'],
    ['view.out', 'y.in'],
  ],
};
const recipe = {
  name: 'Nested module',
  nodes: [
    { id: 'x', type: 'Input' },
    {
      id: 'outer',
      type: 'Submodule',
      subgraph: {
        nodes: [
          { id: 'x', type: 'Inlet' },
          {
            id: 'inner',
            type: 'Submodule',
            subgraph: child,
            submoduleVisual: { icon: 'mlp', summary: '  简介  ' },
          },
          { id: 'y', type: 'Outlet' },
        ],
        connections: [
          ['x.out', 'inner.in'],
          ['inner.out', 'y.in'],
        ],
      },
    },
    { id: 'y', type: 'Output' },
  ],
  connections: [
    ['x.out', 'outer.in'],
    ['outer.out', 'y.in'],
  ],
};
const make = () => registry.create('Submodule', {}, 'unit');

describe('Submodule appearance', () => {
  it.each([
    'auto',
    'backbone',
    'attention',
    'cross_attention',
    'mlp',
    'projection',
    'encoder',
    'decoder',
    'head',
    'module',
  ])('accepts icon %s and derives its color from the final icon', (icon) => {
    const node = make();
    node.label = 'CrossAttention';
    setNodeSubmoduleVisual(node, { icon });
    const final = icon === 'auto' ? 'cross_attention' : icon;
    expect(resolveSubmoduleVisual(node).icon).toBe(final);
    const reference = make();
    reference.label = final;
    expect(resolveSubmoduleVisual(node).tone).toBe(resolveSubmoduleVisual(reference).tone);
  });
  it.each(['auto', 'blue', 'violet', 'teal', 'amber', 'rose', 'slate'])(
    'accepts tone %s',
    (tone) => {
      const node = make();
      setNodeSubmoduleVisual(node, { tone });
      expect(resolveSubmoduleVisual(node).tone).toBe(tone === 'auto' ? 'slate' : tone);
    },
  );
  it.each([
    ['SpatialCrossAttention + LN', 'cross_attention'],
    ['CrossAttention', 'cross_attention'],
    ['cross_attn', 'cross_attention'],
    ['spatial-cross-attn', 'cross_attention'],
    ['TemporalSelfAttention + LN', 'attention'],
    ['SelfAttention', 'attention'],
    ['Attention', 'attention'],
    ['ViT', 'backbone'],
    ['Backbone', 'backbone'],
    ['V-JEPA', 'backbone'],
    ['ResNet50', 'backbone'],
    ['MLP', 'mlp'],
    ['FFN', 'mlp'],
    ['投影', 'projection'],
    ['Projection', 'projection'],
    ['lidar2img', 'projection'],
    ['Sampling', 'projection'],
    ['Decoder', 'decoder'],
    ['Encoder', 'encoder'],
    ['Head', 'head'],
    ['Scorer', 'head'],
    ['Planner', 'head'],
  ])('recognizes %s before inspecting layers', (name, icon) => {
    const node = make();
    node.label = name;
    node.subgraph!.nodes.push(registry.create('Linear'));
    const visual = resolveSubmoduleVisual(node);
    expect(visual.icon).toBe(icon);
    if (icon === 'cross_attention') expect(visual.tone).toBe('teal');
  });
  it('counts local operations before canvas folding and distinguishes contracts from ordinary semantic steps', () => {
    const node = make(),
      graph = node.subgraph!;
    const read = () => resolveSubmoduleVisual(node);
    expect(read()).toMatchObject({ icon: 'module', statusLabel: '待定义输出', steps: 0 });
    graph.nodes.push(
      ...[
        'Outlet',
        'Output',
        'Input',
        'ForwardInput',
        'DictionaryInput',
        'DeviceVariable',
        'TensorVariable',
        'ValueVariable',
        'IfRegion',
        'LoopRegion',
        'LayoutRegion',
      ].map((t) => registry.create(t)),
    );
    graph.nodes.push({ ...registry.create('LocalParameter'), valueSource: 'data' });
    const semantic = registry.create('SemanticOp', { summary: '特征归一化' });
    graph.nodes.push(semantic);
    expect(read()).toMatchObject({ summary: '特征归一化', statusLabel: '1 个语义步骤', steps: 1 });
    node.label = '接口契约 · Placeholder';
    semantic.label = 'Contract the last dimension';
    expect(read().statusLabel).toBe('1 个语义步骤');
    semantic.label = '前向契约';
    expect(read().statusLabel).toBe('待展开 · 契约');
    semantic.label = '接口';
    semantic.params.summary = '待展开的采样';
    expect(read()).toMatchObject({ summary: '待展开的采样', statusLabel: '待展开 · 契约' });
    for (const marker of ['未展开', 'AttentionPlaceholder', 'unexpanded_module', 'pending expansion']) {
      semantic.params.summary = marker;
      expect(read().statusLabel).toBe('待展开 · 契约');
    }
    graph.nodes.pop();
    graph.nodes.push(registry.create('Linear'), {
      ...registry.create('Reshape'),
      presentation: 'edge',
    });
    const original = structuredClone(node);
    expect(read()).toMatchObject({
      icon: 'projection',
      summary: 'Linear · Reshape',
      statusLabel: '已展开 · 2 步',
    });
    expect(node).toEqual(original);
  });
  it('counts GetDevice through a device variable as one operation', () => {
    const node = make();
    node.subgraph!.nodes.push(...['GetDevice', 'DeviceVariable', 'Outlet'].map(type => registry.create(type)));
    expect(resolveSubmoduleVisual(node)).toMatchObject({ steps: 1, summary: 'GetDevice', statusLabel: '已展开 · 1 步' });
  });
  it('counts only TensorSize when its result passes through a LocalParameter', () => {
    const built = buildModel({
      ...recipe,
      nodes: recipe.nodes.map(node => node.id !== 'outer' ? node : {
        ...node,
        subgraph: {
          nodes: [
            { id: 'x', type: 'Inlet' },
            { id: 'size', type: 'TensorSize' },
            { id: 'batch', type: 'LocalParameter' },
            { id: 'y', type: 'Outlet' },
          ],
          connections: [['x.out', 'size.in'], ['size.dim0', 'batch.in'], ['batch.out', 'y.in']],
        },
      }),
    });
    expect(built.ok).toBe(true);
    expect(resolveSubmoduleVisual(built.project!.graph.nodes[1])).toMatchObject({
      steps: 1, summary: 'TensorSize', statusLabel: '已展开 · 1 步',
    });
  });
  it('replaces, trims, and clears all-default styles', () => {
    const node = make();
    setNodeSubmoduleVisual(node, { icon: 'head', tone: 'blue', summary: '  摘要  ' });
    expect(node.submoduleVisual).toEqual({ icon: 'head', tone: 'blue', summary: '摘要' });
    const partial = { icon: 'auto' as const, tone: 'blue' as const, summary: ' ' };
    setNodeSubmoduleVisual(node, partial);
    expect(node.submoduleVisual).toEqual({ ...partial, summary: '' });
    const summary = '🧊'.repeat(120);
    setNodeSubmoduleVisual(node, { summary });
    expect(node.submoduleVisual).toEqual({ summary });
    for (const reset of [{ icon: 'auto', tone: 'auto', summary: '  ' }, {}, null]) {
      setNodeSubmoduleVisual(node, { icon: 'mlp' });
      setNodeSubmoduleVisual(node, reset);
      expect(node).not.toHaveProperty('submoduleVisual');
    }
  });
  it('normalizes the 120-code-point boundary in recipes, patches and stored projects', () => {
    const summary = '🧊'.repeat(118);
    const raw = { summary: ' ' + summary + '\t' };
    const built = buildModel({
      ...recipe, nodes: recipe.nodes.map(n => n.id === 'outer' ? { ...n, submoduleVisual: raw } : n),
    });
    expect(built.ok).toBe(true);
    expect(built.project!.graph.nodes[1].submoduleVisual).toEqual({ summary });
    const project = buildModel(recipe).project!;
    const patched = patchModel(project, { operations: [{ op: 'set_node_submodule_visual', id: 'outer', visual: raw }] });
    expect(patched.ok).toBe(true);
    expect(patched.project!.graph.nodes[1].submoduleVisual).toEqual({ summary });
    project.graph.nodes[1].submoduleVisual = raw;
    expect(parseProject(JSON.stringify(project), registry).graph.nodes[1].submoduleVisual).toEqual({ summary });
  });
  it('matches Draft 2020-12 maxLength on the raw summary before trimming', () => {
    const accepts = new Ajv2020().compile(getSchemas().recipe.$defs.submoduleVisual);
    const node = make();
    for (const raw of ['🧊'.repeat(120), ' ' + '🧊'.repeat(119)]) {
      expect(accepts({ summary: raw })).toBe(true);
      setNodeSubmoduleVisual(node, { summary: raw });
      expect(node.submoduleVisual?.summary).toBe(raw.trim());
    }
    const overlong = { summary: ' ' + '🧊'.repeat(120) };
    expect(accepts(overlong)).toBe(false);
    expect(() => setNodeSubmoduleVisual(node, overlong)).toThrow(/120/);
  });
  it.each([
    null,
    [],
    false,
    { extra: 1 },
    { tone: null },
    { icon: 'Attention' },
    { tone: 'red' },
    { summary: 3 },
    { summary: '🧊'.repeat(121) },
    { summary: ' ' + '🧊'.repeat(120) },
  ])('rejects invalid stored and recipe values %#', (invalid) => {
    expect(() => validateSubmoduleVisual(make(), invalid)).toThrow();
    const original = buildModel(recipe).project!;
    expect(
      buildModel({
        ...recipe,
        nodes: recipe.nodes.map((n) => (n.id === 'outer' ? { ...n, submoduleVisual: invalid } : n)),
      }).ok,
    ).toBe(false);
    const stored = structuredClone(original);
    Object.assign(stored.graph.nodes[1], { submoduleVisual: invalid });
    expect(() => parseProject(JSON.stringify(stored), registry)).toThrow();
    if (invalid !== null)
      expect(
        patchModel(original, {
          operations: [{ op: 'set_node_submodule_visual', id: 'outer', visual: invalid }],
        }).ok,
      ).toBe(false);
  });
  it('rejects use on other node types including other composite cards', () => {
    for (const type of ['Identity', 'CalculationGroup', 'FFN']) {
      expect(() => setNodeSubmoduleVisual(registry.create(type), null)).toThrow(/Submodule/);
      expect(buildModel({ ...recipe, nodes: [{ id: 'n', type, submoduleVisual: {} }] }).ok).toBe(
        false,
      );
    }
    const project = buildModel(recipe).project!;
    Object.assign(project.graph.nodes[0], { submoduleVisual: {} });
    expect(() => parseProject(JSON.stringify(project), registry)).toThrow(/Submodule/);
  });
  it.each([true, false])(
    'preserves nested graphs, manual routes and ancestor customized=%s',
    (customized) => {
      const built = buildModel(recipe);
      expect(built.ok).toBe(true);
      const original = built.project!,
        outer = original.graph.nodes[1],
        inner = outer.subgraph!.nodes[1];
      expect(outer.customized).toBe(true);
      expect(inner.customized).toBe(true);
      expect(inner.submoduleVisual).toEqual({ icon: 'mlp', summary: '简介' });
      outer.customized = customized;
      for (const graph of [original.graph, outer.subgraph!, inner.subgraph!]) {
        graph.nodes[0].position = { x: 71.5, y: -38 };
        graph.edges[0].waypoints = [
          { x: 88.5, y: 14 },
          { x: 120, y: 14 },
        ];
      }
      const before = structuredClone(original);
      const visual = { icon: 'decoder' as const, tone: 'amber' as const };
      const edited = patchModel(original, {
        operations: [{ op: 'set_node_submodule_visual', path: ['outer'], id: 'inner', visual }],
      });
      expect(edited.ok).toBe(true);
      const expected = structuredClone(original);
      expected.graph.nodes[1].subgraph!.nodes[1].submoduleVisual = visual;
      expect(edited.project).toEqual(expected);
      expect(parseProject(serializeProject(edited.project!), registry)).toEqual(expected);
      expect(inferProject(expected, registry)).toEqual(inferProject(before, registry));
      expect(original).toEqual(before);
    },
  );
  it('retains generated styles only on surviving identities', () => {
    const before = buildModel(recipe).project!.graph;
    const after = structuredClone(before),
      inner = after.nodes[1].subgraph!.nodes[1];
    before.nodes[1].subgraph!.nodes[1].submoduleVisual!.summary = '  简介  ';
    before.nodes[0].submoduleVisual = { icon: 'head' };
    before.nodes[1].submoduleVisual = { icon: 'auto', summary: ' ' };
    after.nodes[1].submoduleVisual = { icon: 'decoder' };
    delete inner.submoduleVisual;
    preserveGeneratedMedia(before, after);
    expect(inner.submoduleVisual).toEqual({ icon: 'mlp', summary: '简介' });
    expect(inner.submoduleVisual).not.toBe(before.nodes[1].subgraph!.nodes[1].submoduleVisual);
    expect(after.nodes[0]).not.toHaveProperty('submoduleVisual');
    expect(after.nodes[1]).not.toHaveProperty('submoduleVisual');
    Object.assign(before.nodes[1], { submoduleVisual: { icon: 'invalid' } });
    expect(() => preserveGeneratedMedia(before, after)).toThrow(/图标原型/);
  });
});
