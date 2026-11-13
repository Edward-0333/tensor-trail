import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { planAttributes } from '../src/core/attributes';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Graph, Params } from '../src/core/types';
import { registry } from '../src/modules';
import { attributeCode } from '../src/modules/attributes';
import { loopRecipe } from './fixtures/forEach';

const binding = { owner: '["head"]', name: 'points_scale' };
function graph(): Graph {
  const head = registry.create('Submodule', {}, 'head');
  head.subgraph!.nodes.push(registry.create('GetAttribute', binding, 'get'));
  head.subgraph!.nodes.push(registry.create('Unsqueeze', { dim: 1 }, 'unsqueeze'));
  head.subgraph!.nodes.push(registry.create('Outlet', { port: 'out' }, 'out'));
  head.subgraph!.edges.push(
    { id: 'expand', source: 'get', sourcePort: 'out', target: 'unsqueeze', targetPort: 'in' },
    { id: 'return', source: 'unsqueeze', sourcePort: 'out', target: 'out', targetPort: 'in' },
  );
  const input = registry.create('Input', { shape: '2,1,3', axes: 'batch,ego,xyz' }, 'value');
  input.tensorAutograd = { requiresGrad: true, source: 'manual' };
  return {
    nodes: [head, input, registry.create('SetAttribute', binding, 'set')],
    edges: [
      { id: 'input', source: 'value', sourcePort: 'out', target: 'head', targetPort: 'data' },
      { id: 'assign', source: 'value', sourcePort: 'out', target: 'set', targetPort: 'in' },
    ],
  };
}
const project = () => ({
  schemaVersion: 1 as const,
  name: 'attributes',
  variants: [],
  graph: graph(),
});

describe('Scoped Tensor attributes', () => {
  it('preserves runtime-observed metadata rather than treating GET as a fresh input', () => {
    const observed = { shape: [4, 1, 3], axes: ['batch', 'ego', 'xyz'], dtype: 'float32', autograd: { requiresGrad: false, source: 'runtime' as const } };
    const r = inferGraph(graph(), registry, {}, 0, {}, {
      format: 'tensor-trail.runtime-snapshot', version: 1, id: 'runtime', name: 'observed',
      capturedAt: '2026-11-11T00:00:00Z', context: {},
      bindings: [{ nodePath: ['value'], nodeType: 'Input', value: observed }],
    });
    expect(r.nodes.head.child!.nodes.get.outputs.out).toEqual(observed);
  });

  it('allows loop iterations to read a module attribute written outside the loop', () => {
    const recipe = loopRecipe(), loop = recipe.nodes.find(n => n.id === 'loop')!;
    loop.subgraph!.nodes.push({ id: 'get_attribute', type: 'GetAttribute', params: { owner: '[]', name: 'scale' } });
    recipe.nodes.push(
      { id: 'attribute_value', type: 'Input', params: { shape: '2,1,3' } },
      { id: 'attribute_set', type: 'SetAttribute', params: { owner: '[]', name: 'scale' } },
    );
    recipe.connections.push(['attribute_value.out', 'attribute_set.in']);
    const built = buildModel(recipe);
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const result = inferGraph(built.project!.graph, registry);
    expect(result.order.indexOf('attribute_set')).toBeLessThan(result.order.indexOf('loop'));
    expect(result.nodes.loop.iterations).toHaveLength(4);
    for (const iteration of result.nodes.loop.iterations!)
      expect(iteration.nodes.get_attribute.outputs.out).toMatchObject({ shape: [2,1,3] });
  });

  it('schedules SET before a child GET without exposing an attribute as a forward port', () => {
    const g = graph(),
      result = inferGraph(g, registry);
    expect(result.issues).toEqual([]);
    expect(result.nodes.head.status, result.nodes.head.messages.join('\n')).toBe('ok');
    expect(result.order.indexOf('set')).toBeLessThan(result.order.indexOf('head'));
    expect(ports(g.nodes[0], registry).inputs.map((p) => p.id)).toEqual(['data']);
    expect(ports(g.nodes[2], registry).outputs).toEqual([]);
    expect(ports(g.nodes[0].subgraph!.nodes.find((n) => n.id === 'get')!, registry).inputs).toEqual(
      [],
    );
    expect(result.nodes.set.outputs).toEqual({});
    expect(result.nodes.head.child!.nodes.get.outputs.out).toEqual(result.nodes.set.inputs.in);
    expect(result.nodes.head.child!.nodes.get.outputs.out).toMatchObject({
      shape: [2, 1, 3],
      autograd: { requiresGrad: true, source: 'manual' },
    });
    expect(result.nodes.head.outputs.out).toMatchObject({ shape: [2, 1, 1, 3] });
    expect(result.nodes.head.inputs).toHaveProperty('data');
    expect(result.nodes.head.inputs).not.toHaveProperty('points_scale');
  });

  it('updates every dependent value from upstream edits and never reuses previous-run state', () => {
    const g = graph();
    inferGraph(g, registry);
    g.nodes.find((n) => n.id === 'value')!.params.shape = '5,1,3';
    expect(inferGraph(g, registry).nodes.head.outputs.out).toMatchObject({ shape: [5, 1, 1, 3] });
    g.edges = g.edges.filter((e) => e.id !== 'assign');
    const failed = inferGraph(g, registry);
    expect(failed.nodes.set.status).toBe('blocked');
    expect(failed.nodes.head.child!.nodes.get.status).toBe('error');
    expect(failed.nodes.head.child!.nodes.get.outputs).toEqual({});
  });

  it('keeps same-named attributes on different modules separate', () => {
    const g = graph();
    const other = structuredClone(g.nodes[0]);
    other.id = 'other';
    other.subgraph!.nodes.find((n) => n.id === 'get')!.params.owner = '["other"]';
    g.nodes.push(
      other,
      registry.create('Input', { shape: '7,1,3' }, 'other_value'),
      registry.create('SetAttribute', { ...binding, owner: '["other"]' }, 'other_set'),
    );
    g.edges.push(
      {
        id: 'other_input',
        source: 'other_value',
        sourcePort: 'out',
        target: 'other',
        targetPort: 'data',
      },
      {
        id: 'other_write',
        source: 'other_value',
        sourcePort: 'out',
        target: 'other_set',
        targetPort: 'in',
      },
    );
    const r = inferGraph(g, registry);
    expect(r.nodes.head.outputs.out).toMatchObject({ shape: [2, 1, 1, 3] });
    expect(r.nodes.other.outputs.out).toMatchObject({ shape: [7, 1, 1, 3] });
  });

  it('supports the root model and an attribute read outside its owner', () => {
    const input = registry.create('Input', { shape: '2,3' }, 'input');
    const set = registry.create('SetAttribute', { owner: '[]', name: 'state' }, 'set');
    const get = registry.create('GetAttribute', { owner: '[]', name: 'state' }, 'get');
    const r = inferGraph(
      {
        nodes: [get, input, set],
        edges: [
          { id: 'write', source: 'input', sourcePort: 'out', target: 'set', targetPort: 'in' },
        ],
      },
      registry,
    );
    expect(r.nodes.get.outputs).toEqual(r.nodes.input.outputs);
  });

  it('rejects missing or ambiguous writers instead of inventing a Tensor input', () => {
    const g = graph();
    g.nodes = g.nodes.filter((n) => n.id !== 'set');
    g.edges = g.edges.filter((e) => e.id !== 'assign');
    expect(inferGraph(g, registry).nodes.head.child!.nodes.get.messages[0]).toContain(
      '缺少 SetAttribute',
    );
    const duplicate = graph();
    duplicate.nodes.push(registry.create('SetAttribute', binding, 'set2'));
    duplicate.edges.push({
      id: 'second',
      source: 'value',
      sourcePort: 'out',
      target: 'set2',
      targetPort: 'in',
    });
    const result = inferGraph(duplicate, registry);
    expect(result.nodes.set.status).toBe('error');
    expect(result.nodes.set2.status).toBe('error');
    expect(result.nodes.head.child!.nodes.get.messages[0]).toContain('有歧义');
  });

  it.each<Params>([
    { owner: 'self.head' },
    { owner: '["missing"]' },
    { owner: '["value"]' },
    { owner: '["__proto__"]' },
    { owner: '{}' },
    { name: '__proto__' },
    { name: 'x.y' },
  ])('reports an invalid binding %j without executing an expression', (params) => {
    const g = graph();
    Object.assign(g.nodes.find((n) => n.id === 'set')!.params, params);
    expect(inferGraph(g, registry).nodes.set.status).toBe('error');
    expect(() => attributeCode(g.nodes.find((n) => n.id === 'set')!)).not.toThrow();
  });

  it('detects feedback through hidden attribute dependencies', () => {
    const g = graph();
    g.edges.find((e) => e.target === 'set')!.source = 'head';
    expect(inferGraph(g, registry).issues.join(' ')).toContain('环路');
    expect(validateModel({ ...project(), graph: g }, { mode: 'draft' }).ok).toBe(false);
    expect(() => parseProject(serializeProject({ ...project(), graph: g }), registry)).toThrow(
      /环路/,
    );
  });

  it('detects cycles wholly inside a subgraph', () => {
    const g = graph(),
      head = g.nodes[0];
    const set = g.nodes.find((n) => n.id === 'set')!;
    g.nodes = g.nodes.filter((n) => n.id !== 'set');
    g.edges = g.edges.filter((e) => e.target !== 'set');
    head.subgraph!.nodes.push(set);
    head.subgraph!.edges.push({
      id: 'feedback',
      source: 'get',
      sourcePort: 'out',
      target: 'set',
      targetPort: 'in',
    });
    expect(inferGraph(g, registry).nodes.head.status).toBe('error');
    expect(validateModel({ ...project(), graph: g }, { mode: 'draft' }).ok).toBe(false);
  });

  it('rejects Parameter registration masquerading as a Tensor attribute', () => {
    const g = graph();
    g.nodes = g.nodes.map((n) =>
      n.id === 'value' ? registry.create('GetParameter', {}, 'value') : n,
    );
    g.nodes.push(registry.create('EmbeddingModule', {}, 'embedding'));
    g.edges.push({
      id: 'parameter',
      source: 'embedding',
      sourcePort: 'out',
      target: 'value',
      targetPort: 'in',
    });
    expect(inferGraph(g, registry).nodes.set.messages[0]).toContain('nn.Parameter');
  });

  it.each(['ForEachTensor', 'ForLoop', 'WhileLoop'])('rejects even an unexecuted SET inside a %s body', type => {
    const g = graph(),
      body = registry.create('Submodule', {}, 'loop');
    body.type = type;
    body.subgraph!.nodes.push(registry.create('SetAttribute', binding, 'inside_set'));
    g.nodes.push(body);
    expect(planAttributes(g).issues.join(' ')).toContain('循环体');
  });

  it('supports writes inside a module and reads outside it', () => {
    const g = graph(),
      head = g.nodes[0],
      set = g.nodes.find((n) => n.id === 'set')!;
    g.nodes = g.nodes.filter((n) => n.id !== 'set');
    g.edges = g.edges.filter((e) => e.target !== 'set');
    head.subgraph!.nodes.push(set);
    head.subgraph!.edges.push({
      id: 'inside_write',
      source: 'data',
      sourcePort: 'out',
      target: 'set',
      targetPort: 'in',
    });
    g.nodes.unshift(registry.create('GetAttribute', binding, 'outside_get'));
    const result = inferGraph(g, registry);
    expect(result.nodes.head.status).toBe('ok');
    expect(result.nodes.outside_get.outputs.out).toEqual(result.nodes.value.outputs.out);
    expect(result.order.indexOf('head')).toBeLessThan(result.order.indexOf('outside_get'));
  });

  it('round trips the binding and exposes the nodes through the public API', () => {
    const p = project(),
      result = validateModel(p, { mode: 'draft' });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const reopened = validateModel(parseProject(serializeProject(p), registry), { mode: 'draft' });
    expect(reopened.nodes).toEqual(result.nodes);
    const edited = patchModel(
      p,
      { operations: [{ op: 'set_params', id: 'value', params: { shape: '3,1,3' } }] },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(edited.ok, JSON.stringify(edited.diagnostics)).toBe(true);
    expect(edited.nodes.find((n) => n.path.join('/') === 'head/get')!.outputs.out).toMatchObject({
      shape: [3, 1, 3],
    });
    expect(JSON.stringify(getSchemas())).toContain('SetAttribute');
    expect(JSON.stringify(getSchemas().patch)).toContain('replace_node');
  });
});

describe('Atomic node replacement', () => {
  it('converts an auxiliary Inlet to GET only when its obsolete parent edge is removed', () => {
    const p = project(),
      head = p.graph.nodes[0];
    const get = head.subgraph!.nodes.find((n) => n.id === 'get')!;
    Object.assign(get, registry.create('Inlet', { port: 'attribute' }, 'get'));
    p.graph.edges.push({
      id: 'capture',
      source: 'value',
      sourcePort: 'out',
      target: 'head',
      targetPort: 'attribute',
    });
    const operation = {
      op: 'replace_node',
      path: ['head'],
      id: 'get',
      node: { id: 'get', type: 'GetAttribute', params: binding },
    };
    expect(patchModel(p, { operations: [operation] }, { mode: 'draft' }).ok).toBe(false);
    const r = patchModel(
      p,
      { operations: [{ op: 'disconnect', from: 'value.out', to: 'head.attribute' }, operation] },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    expect(r.project!.graph.nodes[0].subgraph!.nodes.find((n) => n.id === 'get')!.position).toEqual(
      get.position,
    );
    expect(ports(r.project!.graph.nodes[0], registry).inputs.map((p) => p.id)).toEqual(['data']);
    expect(serializeProject(p)).toContain('capture');
    expect(
      patchModel(
        p,
        { operations: [{ op: 'remove_node', path: ['head'], id: 'get' }] },
        { mode: 'draft' },
      ).ok,
    ).toBe(false);
  });

  it('rejects a changed ID and leaves the original project untouched', () => {
    const p = project(),
      before = serializeProject(p);
    const r = patchModel(
      p,
      { operations: [{ op: 'replace_node', id: 'value', node: { id: 'other', type: 'Input' } }] },
      { mode: 'draft' },
    );
    expect(r.ok).toBe(false);
    expect(serializeProject(p)).toBe(before);
  });
});
