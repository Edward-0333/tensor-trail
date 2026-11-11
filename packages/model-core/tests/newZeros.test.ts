import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { newZerosCode } from '../src/modules/newZeros';

const reference: TensorSpec = {
  shape: [2, 1040, 1, 3],
  axes: ['batch', 'query', 'point', 'xyz'],
  dtype: 'float32',
  autograd: { requiresGrad: true, source: 'runtime' },
  parameterRef: { module: 'self.head', name: 'weight' },
};
function infer(params: Params = {}, values: Record<string, ValueSpec> = {}) {
  const node = registry.create('NewZeros', params);
  return registry.get('NewZeros')!.inferValue!(
    {
      in: reference,
      d0: { kind: 'int', value: 2 },
      d1: { kind: 'int', value: 1040 },
      d2: { kind: 'int', value: 256 },
      ...values,
    },
    node.params,
  ).outputs.out;
}

describe('Tensor.new_zeros', () => {
  it('creates a new explicit shape, inheriting dtype but not values, axes, Parameter identity or requires_grad', () => {
    const before = structuredClone(reference);
    expect(infer({ axes: 'batch,query,embedding' })).toEqual({
      shape: [2, 1040, 256],
      axes: ['batch', 'query', 'embedding'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(reference).toEqual(before);
  });

  it.each(['float16', 'bfloat16', 'float32', 'float64', 'complex64', 'int64', 'uint8', 'bool'])(
    'inherits reference dtype=%s independently of the default dtype',
    (dtype) => {
      expect(infer({}, { in: { ...reference, dtype } })).toMatchObject({
        dtype,
        autograd: { requiresGrad: false },
      });
    },
  );

  it('supports an explicit factory requires_grad=True without making a Parameter', () => {
    const out = infer({ requires_grad: true });
    expect(out).toMatchObject({ autograd: { requiresGrad: true, source: 'inferred' } });
    expect(out).not.toHaveProperty('parameterRef');
    expect(() => infer({ requires_grad: true }, { in: { ...reference, dtype: 'int64' } })).toThrow(
      /不能设置/,
    );
  });

  it('accepts zero, repeated and symbolic integer sizes, not inferred reshape dimensions', () => {
    expect(infer({ shape: 'd0,0,d0' }, { d0: { kind: 'int', value: 'B' } })).toMatchObject({
      shape: ['B', 0, 'B'],
    });
    expect(infer({ shape: '2,+3' })).toMatchObject({ shape: [2, 3] });
    expect(
      ports(registry.create('NewZeros', { shape: 'd1,d1,3' }), registry).inputs.map((p) => p.id),
    ).toEqual(['in', 'd1']);
  });

  it.each<Params>([
    { shape: '-1,3' },
    { shape: 'B,Q,256' },
    { shape: 'd8,3' },
    { shape: 'd0,,3' },
    { shape: '' },
    { shape: '1,2,3,4,5,6,7,8,9' },
    { shape: '1.5,3' },
    { shape: '9007199254740992,3' },
    { axes: 'batch,query' },
    { axes: 'batch,,feature' },
    { requires_grad: 1 },
  ])('rejects malformed factory arguments %j', (params) => {
    expect(() => infer(params)).toThrow();
  });

  it('rejects Tensor sizes and missing or non-Tensor references', () => {
    expect(() => infer({}, { d0: reference })).toThrow(/整数尺寸/);
    expect(() => infer({}, { d0: { kind: 'int', value: -2 } })).toThrow(/非负/);
    expect(() => infer({}, { in: { kind: 'int', value: 2 } })).toThrow(/需要 Tensor/);
  });

  it('keeps the declared grad:off through engine inference and readout', () => {
    const built = buildModel({
      name: 'query feature initialization',
      nodes: [
        { id: 'reference', type: 'Input', params: { shape: '2,1040,1,3', axes: '' } },
        { id: 'B', type: 'IntegerParameter', params: { expression: '2' } },
        { id: 'Q', type: 'IntegerParameter', params: { expression: '1040' } },
        { id: 'D', type: 'IntegerParameter', params: { expression: '256' } },
        { id: 'zeros', type: 'NewZeros', params: { axes: 'batch,query,embedding' } },
        { id: 'query_feat', type: 'TensorVariable' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['reference.out', 'zeros.in'],
        ['B.out', 'zeros.d0'],
        ['Q.out', 'zeros.d1'],
        ['D.out', 'zeros.d2'],
        ['zeros.out', 'query_feat.in'],
        ['query_feat.out', 'out.in'],
      ],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const p = built.project!;
    p.graph.nodes[0].tensorAutograd = { requiresGrad: true, source: 'manual' };
    const result = inferGraph(p.graph, registry);
    expect(result.nodes.query_feat.outputs.out).toMatchObject({
      shape: [2, 1040, 256],
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    const updated = patchModel(p, {
      operations: [{ op: 'set_params', id: 'Q', params: { expression: '720' } }],
    });
    expect(updated.ok).toBe(true);
    expect(updated.outputs.out).toMatchObject({ shape: [2, 720, 256] });
    expect(validateModel(parseProject(serializeProject(updated.project!), registry)).nodes).toEqual(
      updated.nodes,
    );
    expect(newZerosCode(p.graph.nodes.find((n) => n.id === 'zeros')!)).toContain(
      'requires_grad=False',
    );
    expect(newZerosCode(p.graph.nodes.find((n) => n.id === 'zeros')!)).toContain('不继承形状');
  });
});
