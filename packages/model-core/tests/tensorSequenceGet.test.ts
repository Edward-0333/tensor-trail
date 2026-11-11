import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { tensorSequenceCode } from '../src/modules/tensorSequence';

const items = [64, 32, 16, 8].map((h, i) => ({
  shape: [2, 30, 256, h, 176 / 2 ** i],
  axes: ['batch', 'view', 'channel', 'height', 'width'],
  dtype: i === 1 ? 'float16' : 'float32',
  ...(i < 2 ? { autograd: { requiresGrad: i === 0, source: 'runtime' as const } } : {}),
}));
const def = registry.get('TensorSequenceGet')!;
const read = (value: ValueSpec, index: string) => def.inferValue!({ in: value }, { index });

describe('TensorSequenceGet', () => {
  it.each(['list', 'tuple'] as const)(
    'reads an existing %s element, keeping all axes and metadata',
    (kind) => {
      const input = { kind, items: structuredClone(items) };
      const before = structuredClone(input);
      for (const i of [0, 1, 2, 3, -1, -4]) {
        const out = read(input, String(i)).outputs.out;
        const expected = items[i < 0 ? i + 4 : i];
        expect(out).toEqual(expected);
        tensorValue(out).shape[0] = 99;
        expect(input).toEqual(before);
      }
    },
  );

  it.each(['4', '-5', '0.5', '', 'missing_name', 'Number(0)'])(
    'rejects invalid index %s',
    (index) => {
      expect(() => read({ kind: 'list', items }, index)).toThrow();
    },
  );

  it.each<ValueSpec>([
    { kind: 'list', items: [] },
    { kind: 'tuple', items: [] },
    { kind: 'forward', text: 'list[dict]' },
    { kind: 'dict', fields: {} },
    items[0],
    { kind: 'int', value: 0 },
  ])('rejects empty or non-Tensor-sequence inputs %j', (value) => {
    expect(() => read(value, '0')).toThrow();
  });

  it('supports safe integer expressions and shows list access, not tensor slicing', () => {
    expect(read({ kind: 'list', items }, '1 + 1').outputs.out).toEqual(items[2]);
    expect(tensorSequenceCode(registry.create('TensorSequenceGet', { index: '-1' }))).toContain(
      'out = items[-1]',
    );
  });

  it.each([true, false])(
    'preserves requiresGrad=%s through the engine, unlike a new tensor operation',
    (requiresGrad) => {
      const result = buildModel({
        name: 'list read',
        nodes: [
          { id: 'x', type: 'Input', params: { shape: '2,30,256,64,176', axes: '' } },
          { id: 'pack', type: 'TensorSequencePack', params: { kind: 'list', input_count: 1 } },
          { id: 'get', type: 'TensorSequenceGet' },
          { id: 'out', type: 'Output' },
        ],
        connections: [
          ['x.out', 'pack.item0'],
          ['pack.out', 'get.in'],
          ['get.out', 'out.in'],
        ],
      });
      expect(result.ok).toBe(true);
      const project = result.project!;
      project.graph.nodes[0].tensorAutograd = { requiresGrad, source: 'runtime' };
      const inferred = inferProject(project, registry);
      expect(inferred.nodes.get.outputs.out).toEqual(inferred.nodes.x.outputs.out);
      expect(tensorValue(inferred.nodes.get.outputs.out).autograd).toEqual({
        requiresGrad,
        source: 'runtime',
      });
      expect(validateModel(parseProject(serializeProject(project), registry)).ok).toBe(true);
    },
  );

  it('supports incremental API edits without moving existing nodes', () => {
    const base = buildModel(
      {
        name: 'incremental',
        nodes: [
          {
            id: 'items',
            type: 'TensorSequencePack',
            params: { input_count: 0 },
            position: { x: 70, y: 80 },
          },
        ],
        connections: [],
      },
      { mode: 'draft' },
    );
    const next = patchModel(
      base.project!,
      {
        operations: [
          {
            op: 'add_node',
            node: { id: 'get', type: 'TensorSequenceGet', position: { x: 400, y: 80 } },
          },
        ],
      },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(next.ok).toBe(true);
    expect(next.project!.graph.nodes[0]).toEqual(base.project!.graph.nodes[0]);
    const invalid = patchModel(
      next.project!,
      { operations: [{ op: 'connect', from: 'items.out', to: 'get.in' }] },
      { mode: 'draft' },
    );
    expect(invalid.ok).toBe(false);
    expect(next.project!.graph.edges).toEqual([]);
  });
});
