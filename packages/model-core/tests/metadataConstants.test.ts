import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { deviceCode, deviceValue, literalDeviceName } from '../src/modules/device';

function infer(type: string, params: Params = {}, values: Record<string, ValueSpec> = {}) {
  const node = registry.create(type, params);
  return registry.get(type)!.inferValue!(values, node.params);
}

describe('Explicit metadata constants', () => {
  it.each(['cpu', 'cuda', 'cuda:0', 'cuda:12', 'mps', 'xpu:1', 'meta'])('provides device %s without a Tensor input', device => {
    const node = registry.create('DeviceConstant', { device });
    expect(ports(node, registry).inputs).toEqual([]);
    const value = infer('DeviceConstant', { device }).outputs.out;
    expect(value).toEqual({ kind: 'device', reference: `torch.device("${device}")` });
    expect(deviceValue(value)).toEqual(value);
    expect(literalDeviceName(deviceValue(value).reference)).toBe(device);
    expect(deviceCode(node)).toContain(`out = torch.device("${device}")`);
    expect(infer('DeviceVariable', {}, { in: value }).outputs.out).toEqual(value);
    expect(infer('Arange', {}, { device: value, end: { kind: 'int', value: 4 } }).outputs.out)
      .toMatchObject({ shape: [4], dtype: 'int64', autograd: { requiresGrad: false } });
  });
  it.each(['', 'gpu', 'CUDA:0', 'cuda:-1', 'cuda:0 ', 'cuda:01', 'cuda:1234567', 'cuda:0");evil()'])('rejects malformed device %s', device => {
    expect(() => infer('DeviceConstant', { device })).toThrow();
    expect(literalDeviceName(`torch.device("${device}")`)).toBeUndefined();
    expect(() => deviceValue({ kind: 'device', reference: `torch.device("${device}")` })).toThrow();
  });
  it.each(['float16', 'bfloat16', 'float32', 'float64', 'bool', 'int64', 'complex64'])('provides dtype %s without casting a Tensor', dtype => {
    const node = registry.create('DtypeConstant', { dtype });
    expect(ports(node, registry).inputs).toEqual([]);
    expect(infer('DtypeConstant', { dtype }).outputs.out).toEqual({ kind: 'dtype', value: dtype });
    expect(deviceCode(node)).toContain(`out = torch.${dtype}`);
  });
  it('rejects unknown dtype rather than silently using float32', () => {
    expect(() => infer('DtypeConstant', { dtype: 'detype' })).toThrow();
  });
  it('keeps new_tensor numerical lineage separate and preserves independent gradients for factory outputs', () => {
    const device = infer('DeviceConstant', { device: 'cuda:0' }).outputs.out;
    const dtype = infer('DtypeConstant').outputs.out;
    const result = infer('ArrayToTensor', { mode: 'new_tensor' }, {
      in: { kind: 'ndarray', dims: [2, 4, 4], axes: ['batch', 'row', 'col'], dtype: 'float64' }, device, dtype,
    });
    expect(result.outputs.out).toMatchObject({ shape: [2, 4, 4], dtype: 'float32', autograd: { requiresGrad: false } });
    const zeros = infer('NewZeros', { metadata_inputs: true, shape: '2,1,3' }, { device, dtype });
    expect(zeros.outputs.out).toMatchObject({ shape: [2, 1, 3], dtype: 'float32', autograd: { requiresGrad: false } });
  });
  it('supports one-consumer aliases, shared edits and serialization without inventing root sources', () => {
    const result = buildModel({ name: 'metadata', nodes: [
      { id: 'input', type: 'Input', params: { shape: '2,3', axes: 'batch,feature' } },
      { id: 'input_output', type: 'Output' },
      { id: 'device', type: 'DeviceConstant', params: { device: 'cuda:0' } },
      { id: 'dtype', type: 'DtypeConstant' },
      { id: 'ones', type: 'NewOnes', params: { metadata_inputs: true, shape: '2,3' } },
      { id: 'output', type: 'Output' },
    ], connections: [['input.out', 'input_output.in'], ['device.out', 'ones.device'], ['dtype.out', 'ones.dtype'], ['ones.out', 'output.in']] });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    const copied = patchModel(result.project!, { operations: [
      { op: 'duplicate_linked', source: 'device', id: 'device_copy' },
      { op: 'duplicate_linked', source: 'dtype', id: 'dtype_copy' },
      { op: 'disconnect', from: 'device.out', to: 'ones.device' },
      { op: 'disconnect', from: 'dtype.out', to: 'ones.dtype' },
      { op: 'connect', from: 'device_copy.out', to: 'ones.device' },
      { op: 'connect', from: 'dtype_copy.out', to: 'ones.dtype' },
      { op: 'set_params', id: 'dtype_copy', params: { dtype: 'float64' } },
      { op: 'set_params', id: 'device_copy', params: { device: 'cpu' } },
    ] });
    expect(copied.ok, JSON.stringify(copied.diagnostics)).toBe(true);
    expect(copied.nodes.find(n => n.path[0] === 'ones')!.outputs.out).toMatchObject({ dtype: 'float64' });
    for (const id of ['device', 'device_copy', 'dtype', 'dtype_copy']) {
      const spec = copied.nodes.find(n => n.path[0] === id)!.outputs.out;
      expect(spec.lineage?.sources ?? []).toEqual([]);
      expect(spec).not.toHaveProperty('shape');
      expect(spec).not.toHaveProperty('autograd');
    }
    const reopened = validateModel(parseProject(serializeProject(copied.project!), registry));
    expect(reopened.ok).toBe(true);
    expect(reopened.nodes).toEqual(copied.nodes);
  });
});
