import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { bufferSourceCode } from '../src/modules/bufferSource';

function source(params: Params = {}) {
  return registry.create(
    'BufferSource',
    { shape: '1040,7', axes: 'query,time', ...params },
    'buffer',
  );
}
function normalization() {
  const nodes = [
    source(),
    registry.create('Cast', { dtype: 'float32' }, 'float'),
    registry.create('ReduceSum', { dim: -1, keepdim: true }, 'sum'),
    registry.create('Divide', {}, 'divide'),
  ];
  return {
    nodes,
    edges: [
      { id: 'read', source: 'buffer', sourcePort: 'out', target: 'float', targetPort: 'in' },
      { id: 'sum', source: 'float', sourcePort: 'out', target: 'sum', targetPort: 'in' },
      { id: 'numerator', source: 'float', sourcePort: 'out', target: 'divide', targetPort: 'a' },
      { id: 'denominator', source: 'sum', sourcePort: 'out', target: 'divide', targetPort: 'b' },
    ],
  };
}

describe('Registered buffer and stamp row normalization', () => {
  it('reads a module buffer, not a dataset input or Parameter, without resetting it each forward', () => {
    const node = source(),
      r = inferGraph({ nodes: [node], edges: [] }, registry).nodes.buffer;
    expect(r.status).toBe('ok');
    expect(r.outputs.out).toEqual({
      shape: [1040, 7],
      axes: ['query', 'time'],
      dtype: 'int64',
      autograd: { requiresGrad: false, source: 'inferred' },
      bufferRef: { reference: 'self.num_stamps_all', persistent: true },
    });
    expect(r.outputs.out).not.toHaveProperty('parameterRef');
    expect(bufferSourceCode(node)).toContain('不是 register_buffer 的返回值');
    expect(bufferSourceCode(node)).toContain('checkpoint');
    expect(bufferSourceCode(node)).toContain('torch.ones');
  });

  it('keeps the original buffer unchanged, drops registration on float conversion, and normalizes via [Q,1]', () => {
    const g = normalization(),
      before = structuredClone(g),
      result = inferGraph(g, registry);
    expect(result.issues).toEqual([]);
    expect(Object.values(result.nodes).every((r) => r.status === 'ok')).toBe(true);
    expect(result.nodes.float.outputs.out).toEqual({
      shape: [1040, 7],
      axes: ['query', 'time'],
      dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(result.nodes.sum.outputs.out).toMatchObject({
      shape: [1040, 1],
      dtype: 'float32',
      autograd: { requiresGrad: false },
    });
    expect(result.nodes.divide.outputs.out).toEqual(result.nodes.float.outputs.out);
    expect(result.nodes.buffer.outputs.out).toMatchObject({
      dtype: 'int64',
      bufferRef: { persistent: true },
    });
    expect(g).toEqual(before);
    expect(result.nodes.divide.notes.join(' ')).toContain('不添加');
  });

  it('retains registration on a no-op cast and never assumes True through arithmetic', () => {
    const g = normalization();
    g.nodes[0] = source({ dtype: 'float32', requires_grad: true, persistent: false });
    const r = inferGraph(g, registry);
    expect(r.nodes.float.outputs.out).toMatchObject({
      bufferRef: { persistent: false },
      autograd: { requiresGrad: true },
    });
    expect(r.nodes.sum.outputs.out).not.toHaveProperty('bufferRef');
    expect(r.nodes.sum.outputs.out).not.toHaveProperty('autograd');
  });

  it.each<Params>([
    { reference: 'self.x()' },
    { reference: 'self.__proto__' },
    { shape: '-1,7' },
    { axes: 'query' },
    { dtype: 'invalid' },
    { persistent: 1 },
    { requires_grad: true },
  ])('rejects invalid buffer declarations %j', (params) => {
    const node = source(params);
    expect(() => registry.get('BufferSource')!.inferValue!({}, node.params)).toThrow();
  });

  it('supports dynamic Q/time integer sizes, persistence, and API round trips', () => {
    const built = buildModel({
      name: 'buffer',
      nodes: [
        { id: 'Q', type: 'IntegerParameter', params: { expression: '1040' } },
        { id: 'T', type: 'IntegerParameter', params: { expression: '6+1' } },
        { id: 'buffer', type: 'BufferSource', params: { shape: 'd0,d1', axes: 'query,time' } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['Q.out', 'buffer.d0'],
        ['T.out', 'buffer.d1'],
        ['buffer.out', 'out.in'],
      ],
    }, { mode: 'draft' });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(ports(built.project!.graph.nodes[2], registry).inputs.map((p) => p.id)).toEqual([
      'd0',
      'd1',
    ]);
    const r = patchModel(
      built.project!,
      { operations: [{ op: 'set_params', id: 'T', params: { expression: '4' } }] },
      { mode: 'draft', layout: 'preserve' },
    );
    expect(r.ok).toBe(true);
    expect(r.outputs.out).toMatchObject({ shape: [1040, 4], bufferRef: { persistent: true } });
    expect(validateModel(parseProject(serializeProject(r.project!), registry), { mode: 'draft' }).nodes).toEqual(
      r.nodes,
    );
    const alias = patchModel(r.project!, { operations: [{ op: 'duplicate_linked', source: 'buffer', id: 'buffer_read_copy' }] }, { mode: 'draft', layout: 'preserve' });
    expect(alias.ok).toBe(true);
    expect(alias.nodes.find(n => n.path.join('/') === 'buffer_read_copy')!.outputs).toEqual(alias.nodes.find(n => n.path.join('/') === 'buffer')!.outputs);
    expect(ports(alias.project!.graph.nodes.find(n => n.id === 'buffer_read_copy')!, registry).inputs).toEqual([]);
  });
});

describe('Broadcast true division', () => {
  const spec = (shape: number[], dtype = 'float32'): TensorSpec => ({
    shape,
    axes: shape.map((_, i) => `axis${i}`),
    dtype,
  });
  const divide = (a: TensorSpec, b: TensorSpec, params: Params = {}) => {
    const node = registry.create('Divide', params);
    return registry.get('Divide')!.infer!({ a, b }, node.params).outputs.out;
  };
  it('broadcasts rows and promotes real dtypes', () => {
    expect(divide(spec([1040, 7]), spec([1040, 1]))).toMatchObject({
      shape: [1040, 7],
      dtype: 'float32',
    });
    expect(divide(spec([2, 3], 'int64'), spec([1, 3], 'int32'))).toMatchObject({
      shape: [2, 3],
      dtype: 'float32',
    });
    expect(
      divide(spec([2, 3], 'int64'), spec([1, 3], 'int32'), { default_dtype: 'float64' }).dtype,
    ).toBe('float64');
    expect(divide(spec([2, 3], 'float16'), spec([1, 3], 'int64')).dtype).toBe('float16');
  });
  it('rejects mismatched broadcasting and unsupported dtypes', () => {
    expect(() => divide(spec([1040, 7]), spec([1040]))).toThrow(/不匹配/);
    expect(() => divide(spec([2, 3], 'complex64'), spec([2, 3]))).toThrow(/不支持/);
  });
});
