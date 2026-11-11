import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel, validateModel } from '../src/agent';
import { inferProject, resultAt } from '../src/core/engine';
import { graphAt, ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { loopRecipe, reshapedFeatures } from './fixtures/forEach';

const original = () => buildModel(loopRecipe()).project!;

describe('Linked value copies through the patch API', () => {
  it('creates aliases without extra input edges and links copies directly to their canonical owner', () => {
    const p = original(),
      before = serializeProject(p);
    const r = patchModel(
      p,
      {
        operations: [
          { op: 'duplicate_linked', source: 'B', id: 'B_copy', position: { x: 650, y: 100 } },
          { op: 'duplicate_linked', source: 'B_copy', id: 'B_copy2', position: { x: 750, y: 100 } },
          { op: 'disconnect', from: 'B.out', to: 'loop.B' },
          { op: 'connect', from: 'B_copy2.out', to: 'loop.B' },
        ],
      },
      { layout: 'preserve' },
    );
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    for (const id of ['B_copy', 'B_copy2']) {
      const alias = r.project!.graph.nodes.find((n) => n.id === id)!;
      expect(alias.valueSource).toBe('B');
      expect(ports(alias, registry).inputs).toEqual([]);
      expect(r.project!.graph.edges.some((e) => e.target === id)).toBe(false);
    }
    expect(r.project!.graph.nodes.find((n) => n.id === 'B_copy')!.position).toEqual({
      x: 650,
      y: 100,
    });
    for (const n of p.graph.nodes)
      expect(r.project!.graph.nodes.find((node) => node.id === n.id)).toEqual(n);
    expect(r.outputs).toEqual(validateModel(p).outputs);
    expect(serializeProject(p)).toBe(before);
    expect(validateModel(parseProject(serializeProject(r.project!), registry)).ok).toBe(true);
    const changed = patchModel(r.project!, {
      operations: [{ op: 'set_params', id: 'B_copy2', params: { expression: '3' } }],
    });
    expect(changed.ok).toBe(true);
    for (const id of ['B', 'B_copy', 'B_copy2'])
      expect(changed.nodes.find((n) => n.path.join('/') === id)!.outputs.out).toEqual({
        kind: 'int',
        value: 3,
      });
  });

  it('allows LocalParameter and Inlet aliases inside ForEach without adding capture ports; values follow each iteration', () => {
    const p = original();
    const r = patchModel(
      p,
      {
        operations: [
          { op: 'add_node', path: ['loop'], node: { id: 'H', type: 'LocalParameter' } },
          { op: 'connect', path: ['loop'], from: 'size.dim2', to: 'H.in' },
          {
            op: 'duplicate_linked',
            path: ['loop'],
            source: 'H',
            id: 'H_copy',
            position: { x: 600, y: 200 },
          },
          { op: 'duplicate_linked', path: ['loop'], source: 'B', id: 'B_copy' },
          { op: 'disconnect', path: ['loop'], from: 'size.dim2', to: 'view.d3' },
          { op: 'connect', path: ['loop'], from: 'H_copy.out', to: 'view.d3' },
          { op: 'disconnect', path: ['loop'], from: 'B.out', to: 'view.d0' },
          { op: 'connect', path: ['loop'], from: 'B_copy.out', to: 'view.d0' },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    expect(
      ports(r.project!.graph.nodes.find((n) => n.id === 'loop')!, registry).inputs.map((p) => p.id),
    ).toEqual(['items', 'initial', 'B']);
    const result = inferProject(r.project!, registry);
    for (let i = 0; i < 4; i++) {
      const body = resultAt(result, ['loop'], { '["loop"]': i })!;
      expect(body.nodes.H_copy.outputs.out).toEqual(body.nodes.H.outputs.out);
      expect(body.nodes.H_copy.outputs.out).toEqual({ kind: 'int', value: 64 / 2 ** i });
      expect(body.outputs.out).toMatchObject({ shape: reshapedFeatures[i] });
    }
    expect(graphAt(r.project!.graph, ['loop']).edges.some((e) => e.target.endsWith('_copy'))).toBe(
      false,
    );
  });

  it.each([
    { op: 'duplicate_linked', source: 'missing', id: 'copy' },
    { op: 'duplicate_linked', source: 'pack', id: 'copy' },
    { op: 'duplicate_linked', source: 'B', id: 'B' },
    { op: 'duplicate_linked', source: 'B', id: 'copy', position: { x: Infinity, y: 0 } },
    { op: 'duplicate_linked', path: ['loop'], source: 'input0', id: 'copy' },
  ])('rejects invalid requests atomically: %j', (operation) => {
    const p = original(),
      before = serializeProject(p);
    const r = patchModel(p, { operations: [operation] }, { mode: 'draft' });
    expect(r.ok).toBe(false);
    expect(serializeProject(p)).toBe(before);
  });

  it('rejects an independent input or a hidden alias cycle, and exposes the operation schema', () => {
    const p = original();
    const independent = patchModel(
      p,
      {
        operations: [
          { op: 'duplicate_linked', path: ['loop'], source: 'B', id: 'copy' },
          { op: 'connect', path: ['loop'], from: 'size.dim0', to: 'copy.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(independent.ok).toBe(false);
    const cycle = patchModel(
      p,
      {
        operations: [
          { op: 'add_node', path: ['loop'], node: { id: 'n', type: 'LocalParameter' } },
          { op: 'duplicate_linked', path: ['loop'], source: 'n', id: 'copy' },
          { op: 'connect', path: ['loop'], from: 'copy.out', to: 'n.in' },
        ],
      },
      { mode: 'draft' },
    );
    expect(cycle.ok).toBe(false);
    expect(JSON.stringify(getSchemas().patch)).toContain('duplicate_linked');
  });
});
