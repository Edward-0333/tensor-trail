import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';

const x = { shape: [2, 1040, 48, 3], axes: ['batch', 'query', 'point', 'xyz'], dtype: 'float32' };
const mask = { shape: [1040], axes: ['query'], dtype: 'bool' };
const integer = (value: number | string): ValueSpec => ({ kind: 'int', value });
function infer(type: string, values: Record<string, ValueSpec>, params: Params = {}) {
  return registry.get(type)!.inferValue!(values, registry.create(type, params).params);
}
describe('BooleanIndexAxis', () => {
  it('uses a symbolic count by default and preserves batch and all other axes', () => {
    expect(infer('BooleanIndexAxis', { in: x, mask }).outputs.out).toEqual({ ...x, shape: [2, 'N_selected', 48, 3] });
  });
  it('uses an explicit quota contract without claiming an observed mask count', () => {
    const values = { in: x, mask, count: integer(720) }, before = structuredClone(values);
    const result = infer('BooleanIndexAxis', values, { declared_count: true });
    expect(result.outputs.out).toEqual({ ...x, shape: [2, 720, 48, 3] });
    expect(result.notes?.join(' ')).toContain('尚未执行或采集掩码计数');
    expect(values).toEqual(before);
    expect(ports(registry.create('BooleanIndexAxis'), registry).inputs.map(p => p.id)).toEqual(['in','mask']);
    expect(ports(registry.create('BooleanIndexAxis', { declared_count: true }), registry).inputs.map(p => p.id)).toEqual(['in','mask','count']);
  });
  it.each([0, 720, 1040, 'Q_current'])('accepts declared count %s', count => {
    expect(infer('BooleanIndexAxis', { in: x, mask, count: integer(count) }, { declared_count: true }).outputs.out)
      .toMatchObject({ shape: [2, count, 48, 3] });
  });
  it('supports negative axes and a genuinely empty axis', () => {
    expect(infer('BooleanIndexAxis', { in: x, mask }, { dim: '-3' }).outputs.out)
      .toMatchObject({ shape: [2, 'N_selected', 48, 3] });
    expect(infer('BooleanIndexAxis', { in: { ...x, shape: [2, 0, 48, 3] }, mask: { ...mask, shape: [0] } }).outputs.out)
      .toMatchObject({ shape: [2, 0, 48, 3] });
  });
  it('rejects mismatched masks, broadcasting, malformed dimensions and invalid counts', () => {
    for (const invalid of [{ ...mask, shape: [2, 1040] }, { ...mask, shape: [1] }, { ...mask, dtype: 'float32' }])
      expect(() => infer('BooleanIndexAxis', { in: x, mask: invalid })).toThrow();
    for (const dim of ['4', '-5', '1.5', 'x()'])
      expect(() => infer('BooleanIndexAxis', { in: x, mask }, { dim })).toThrow();
    for (const count of [-1, 1041, 0.5])
      expect(() => infer('BooleanIndexAxis', { in: x, mask, count: integer(count) }, { declared_count: true })).toThrow();
    expect(() => infer('BooleanIndexAxis', { in: x, mask, count: x }, { declared_count: true })).toThrow();
    expect(() => infer('BooleanIndexAxis', { in: x, mask }, { declared_count: true })).toThrow();
  });
});
describe('Python integer selection arithmetic', () => {
  it.each([
    ['IntegerSubtract', 0, 5, -5], ['IntegerSubtract', 10, 5, 5],
    ['IntegerMinimum', -4, 6, -4], ['IntegerMinimum', 7, 6, 6],
    ['IntegerMaximum', 1, -4, 1], ['IntegerMaximum', 1, 6, 6],
  ])('%s(%s, %s) = %s', (type, a, b, result) => {
    expect(infer(String(type), { a: integer(Number(a)), b: integer(Number(b)) }).outputs.out).toEqual(integer(Number(result)));
  });
  it.each(['IntegerSubtract', 'IntegerMinimum', 'IntegerMaximum'])('preserves symbols and rejects Tensor inputs for %s', type => {
    expect(infer(type, { a: integer('epoch'), b: integer(5) }).outputs.out).toMatchObject({ kind: 'int', value: expect.stringContaining('epoch') });
    expect(() => infer(type, { a: x, b: integer(5) })).toThrow();
  });
  it('rejects integer overflow', () => {
    expect(() => infer('IntegerSubtract', { a: integer(Number.MAX_SAFE_INTEGER), b: integer(-1) })).toThrow(/安全范围/);
  });
});
describe('Selection, detach and zero factory metadata integration', () => {
  it('preserves source IDs through selection/detach, but not through zero initialization', () => {
    const result = buildModel({ name: 'selection', nodes: [
      { id: 'x', type: 'Input', params: { shape: '2,1040,48,3', axes: 'batch,query,point,xyz' } },
      { id: 'mask', type: 'Input', params: { shape: '1040', axes: 'query', dtype: 'bool' } },
      { id: 'count', type: 'IntegerParameter', params: { expression: '720' } },
      { id: 'select', type: 'BooleanIndexAxis', params: { declared_count: true } },
      { id: 'detach', type: 'Detach' },
      { id: 'dtype', type: 'GetDtype' },
      { id: 'device', type: 'GetDevice', params: { tensor_name: 'query_pos' } },
      { id: 'zero', type: 'NewZeros', params: { metadata_inputs: true, shape: '2,d0,48,1', axes: 'batch,query,point,time' } },
      { id: 'out', type: 'Output' },
    ], connections: [
      ['x.out','select.in'], ['mask.out','select.mask'], ['count.out','select.count'], ['select.out','detach.in'],
      ['x.out','dtype.in'], ['x.out','device.in'], ['dtype.out','zero.dtype'], ['device.out','zero.device'],
      ['count.out','zero.d0'], ['zero.out','out.in'],
    ] });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    result.project!.lineageEnabled = true;
    const checked = validateModel(result.project!);
    const node = (id: string) => checked.nodes.find(n => n.path.join('/') === id)!.outputs.out;
    expect(node('select')).toMatchObject({ shape: [2,720,48,3], lineage: { sources: [{ id: 'x' }, { id: 'mask' }] } });
    expect(node('detach')).toMatchObject({ autograd: { requiresGrad: false }, lineage: { sources: [{ id: 'x' }, { id: 'mask' }] } });
    expect(node('zero')).toMatchObject({ shape: [2,720,48,1], dtype: 'float32', autograd: { requiresGrad: false }, lineage: { sources: [], initializations: ['zeros'] } });
    expect(ports(result.project!.graph.nodes.find(n => n.id === 'zero')!, registry).inputs.map(p => p.id)).toEqual(['dtype','device','d0']);
    const updated = patchModel(result.project!, { operations: [{ op: 'set_params', id: 'count', params: { expression: '60' } }] });
    expect(updated.ok, JSON.stringify(updated.diagnostics)).toBe(true);
    expect(updated.outputs.out).toMatchObject({ shape: [2,60,48,1] });
    expect(validateModel(parseProject(serializeProject(updated.project!), registry)).ok).toBe(true);
  });
  it('requires actual dtype/device metadata instead of accepting arbitrary tensors', () => {
    const p = { metadata_inputs: true, shape: '2,720,48,1' };
    const values = { dtype: { kind: 'dtype', value: 'float16' } as ValueSpec,
      device: { kind: 'device', reference: 'query_pos.device' } as ValueSpec };
    expect(infer('NewZeros', values, p).outputs.out).toMatchObject({ shape: [2,720,48,1], dtype: 'float16' });
    expect(() => infer('NewZeros', { ...values, dtype: x }, p)).toThrow(/GetDtype/);
    expect(() => infer('NewZeros', { ...values, device: x }, p)).toThrow(/GetDevice/);
  });
  it('keeps legacy NewZeros projects valid without adding a missing mode flag', () => {
    const built = buildModel({ name: 'legacy zeros', nodes: [
      { id: 'x', type: 'Input', params: { shape: '2,3', axes: '' } },
      { id: 'zeros', type: 'NewZeros', params: { shape: '2,3' } },
      { id: 'out', type: 'Output' },
    ], connections: [['x.out','zeros.in'], ['zeros.out','out.in']] });
    const p = built.project!;
    delete p.graph.nodes.find(n => n.id === 'zeros')!.params.metadata_inputs;
    const before = structuredClone(p);
    expect(validateModel(p).ok).toBe(true);
    expect(p).toEqual(before);
  });
});
