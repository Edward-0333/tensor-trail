import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel } from '../src/agent';

const base = () =>
  buildModel(
    {
      name: 'border',
      nodes: [
        {
          id: 'input',
          type: 'Input',
          params: { shape: '2,3', axes: '' },
          position: { x: 12, y: 34 },
        },
        { id: 'init', type: 'EmbeddingXYZInit', position: { x: 300, y: 50 } },
      ],
      connections: [],
    },
    { mode: 'draft' },
  ).project!;

describe('Tensor border Patch API', () => {
  it('changes only the border and supports restoring the default', () => {
    const original = base(),
      before = structuredClone(original);
    const result = patchModel(
      original,
      { operations: [{ op: 'set_tensor_border', id: 'input', color: '#60a5fa' }] },
      { mode: 'draft' },
    );
    expect(result.ok).toBe(true);
    expect(original).toEqual(before);
    expect(result.project!.graph.nodes[0]).toEqual({
      ...before.graph.nodes[0],
      tensorBorderColor: '#60a5fa',
    });
    expect(result.project!.tensorPalette).toEqual(before.tensorPalette);
    const reset = patchModel(
      result.project!,
      { operations: [{ op: 'set_tensor_border', id: 'input', color: null }] },
      { mode: 'draft' },
    );
    expect(reset.project).toEqual(before);
  });

  it('does not mark generated composites as customized', () => {
    const result = patchModel(
      base(),
      {
        operations: [
          { op: 'set_tensor_border', path: ['init'], id: 'registered_weight', color: '#ff0000' },
        ],
      },
      { mode: 'draft' },
    );
    expect(result.ok).toBe(true);
    expect(result.project!.graph.nodes.find((n) => n.id === 'init')!.customized).toBeUndefined();
  });

  it('sets alias-local color without changing its source', () => {
    const result = patchModel(
      base(),
      {
        operations: [
          { op: 'duplicate_linked', source: 'input', id: 'copy', position: { x: 500, y: 80 } },
          { op: 'set_tensor_border', id: 'copy', color: '#60a5fa' },
        ],
      },
      { mode: 'draft' },
    );
    expect(result.ok).toBe(true);
    expect(
      result.project!.graph.nodes.find((n) => n.id === 'input')!.tensorBorderColor,
    ).toBeUndefined();
    expect(result.project!.graph.nodes.find((n) => n.id === 'copy')!.tensorBorderColor).toBe(
      '#60a5fa',
    );
  });

  it.each(['red', '#fff', '#11223344', 'url(bad)', 42, undefined])(
    'rejects invalid color %s atomically',
    (color) => {
      const project = base(),
        before = structuredClone(project);
      const result = patchModel(
        project,
        { operations: [{ op: 'set_tensor_border', id: 'input', color }] },
        { mode: 'draft' },
      );
      expect(result.ok).toBe(false);
      expect(project).toEqual(before);
    },
  );

  it('advertises the operation in the schema', () => {
    expect(JSON.stringify(getSchemas().patch)).toContain('set_tensor_border');
  });
});
