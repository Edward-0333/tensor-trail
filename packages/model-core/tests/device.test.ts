import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferGraph } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import { isDeviceSpec, shapeText, type Params, type ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { deviceCode } from '../src/modules/device';
import { matchingPreparationCode } from '../src/modules/matchingPreparation';

const tensor = {
  shape: [1040, 7],
  axes: ['query', 'time'],
  dtype: 'float64',
  autograd: { requiresGrad: true, source: 'manual' as const },
};
const device = { kind: 'device' as const, reference: 'maps.device' };
const count = { kind: 'int' as const, value: 1040 };
function infer(type: string, values: Record<string, ValueSpec>, params: Params = {}) {
  const node = registry.create(type, params);
  return registry.get(type)!.inferValue!(values, node.params);
}

describe('Explicit device metadata', () => {
  it('reads a symbolic device expression, not tensor data, dtype, grad or actual CUDA placement', () => {
    const before = structuredClone(tensor);
    const result = infer('GetDevice', { in: tensor }, { tensor_name: 'maps' });
    expect(result.outputs.out).toEqual(device);
    expect(isDeviceSpec(result.outputs.out)).toBe(true);
    expect(shapeText(result.outputs.out)).toBe('Device · maps.device');
    expect(tensor).toEqual(before);
    expect(result.outputs.out).not.toHaveProperty('shape');
    expect(result.outputs.out).not.toHaveProperty('autograd');
    expect(result.outputs.out).not.toHaveProperty('dtype');
    expect(deviceCode(registry.create('GetDevice', { tensor_name: 'maps' }))).toContain(
      'device = maps.device',
    );
  });
  it.each(['', 'maps()', 'maps[0]', '__proto__', 'maps.constructor', 'a'.repeat(250)])(
    'rejects invalid display references %s',
    (tensor_name) => {
      expect(() => infer('GetDevice', { in: tensor }, { tensor_name })).toThrow();
    },
  );
  it('requires a Tensor for GetDevice and a Device for DeviceVariable', () => {
    expect(() => infer('GetDevice', { in: device })).toThrow();
    expect(() => infer('GetDevice', {})).toThrow();
    expect(() => infer('DeviceVariable', { in: tensor })).toThrow(/GetDevice/);
    expect(infer('DeviceVariable', { in: device }).outputs.out).toEqual(device);
  });
  it.each(['Arange', 'OnesScaled'])(
    '%s defaults to a distinct device port, not a reference Tensor input',
    (type) => {
      const node = registry.create(type);
      expect(ports(node, registry).inputs.map((p) => p.id)).toEqual(
        type === 'Arange' ? ['device', 'end'] : ['device', 'd0'],
      );
    const size: Record<string, ValueSpec> = type === 'Arange' ? { end: count } : { d0: count };
      const out = infer(type, { device, ...size }).outputs.out;
      expect(out).toMatchObject({
        shape: [1040],
        dtype: type === 'Arange' ? 'int64' : 'float32',
        autograd: { requiresGrad: false },
      });
      expect(out).not.toHaveProperty('reference');
      expect(() => infer(type, { device: tensor, ...size })).toThrow(/GetDevice/);
      expect(() => infer(type, { in: tensor, ...size })).toThrow();
      expect(() => infer(type, { device: { kind: 'int', value: 0 }, ...size })).toThrow();
      expect(matchingPreparationCode(node)).toContain('device=device');
      expect(matchingPreparationCode(node)).not.toContain('device=reference.device');
    },
  );
  it.each(['Arange', 'OnesScaled'])(
    'keeps old serialized %s nodes loadable without inventing a new required field',
    (type) => {
      const input = registry.create('Input', { shape: '1040,7', axes: '' }, 'maps');
      const size = registry.create('IntegerParameter', { expression: '1040' }, 'Q');
      const factory = registry.create(type, {}, 'factory');
      delete factory.params.device_source;
      const def = registry.get(type)!;
      expect(def.getFields!(factory.params).map((f) => f.key)).not.toContain('device_source');
      expect(ports(factory, registry).inputs[0].id).toBe('in');
      const graph = {
        nodes: [input, size, factory],
        edges: [
          { id: 'device', source: 'maps', sourcePort: 'out', target: 'factory', targetPort: 'in' },
          {
            id: 'size',
            source: 'Q',
            sourcePort: 'out',
            target: 'factory',
            targetPort: type === 'Arange' ? 'end' : 'd0',
          },
        ],
      };
      const project = { schemaVersion: 1 as const, name: 'legacy', graph, variants: [] };
      const serialized = serializeProject(project);
      const parsed = parseProject(serialized, registry);
      expect(serializeProject(parsed)).toBe(serialized);
      expect(inferGraph(parsed.graph, registry).nodes.factory.status).toBe('ok');
      expect(matchingPreparationCode(factory)).toContain('device=reference.device');
      expect(
        infer(type, { in: tensor, end: count, d0: count }, { device_source: 'tensor_legacy' })
          .outputs.out,
      ).toHaveProperty('shape');
    },
  );
  it('rejects a malformed device value or source mode rather than interpreting it as a tensor', () => {
    expect(() =>
      infer('Arange', { device: { kind: 'device', reference: 'cuda:0' }, end: count }),
    ).toThrow();
    expect(() => infer('Arange', { device, end: count }, { device_source: 'guess' })).toThrow();
  });
});

describe('Device API and linked copies', () => {
  it('shares a Device through aliases and a submodule boundary, and round trips without adding observed devices', () => {
    const built = buildModel({
      name: 'device',
      nodes: [
        { id: 'maps', type: 'Input', params: { shape: '1040,7', axes: 'query,time' } },
        { id: 'read', type: 'GetDevice', params: { tensor_name: 'maps' } },
        { id: 'device', type: 'DeviceVariable', label: 'maps.device' },
        { id: 'count', type: 'IntegerParameter', params: { expression: '1040' } },
        { id: 'arange', type: 'Arange' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['maps.out', 'read.in'],
        ['read.out', 'device.in'],
        ['device.out', 'arange.device'],
        ['count.out', 'arange.end'],
        ['arange.out', 'out.in'],
      ],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const patched = patchModel(
      built.project!,
      {
        operations: [
          {
            op: 'duplicate_linked',
            source: 'device',
            id: 'device_copy',
            position: { x: 10, y: 300 },
          },
          {
            op: 'add_node',
            node: {
              id: 'boundary',
              type: 'Submodule',
              subgraph: {
                nodes: [
                  { id: 'device', type: 'Inlet', params: { port: 'device' } },
                  { id: 'value', type: 'DeviceVariable' },
                  { id: 'return', type: 'Outlet', params: { port: 'device' } },
                ],
                connections: [
                  ['device.out', 'value.in'],
                  ['value.out', 'return.in'],
                ],
              },
            },
          },
          { op: 'connect', from: 'device_copy.out', to: 'boundary.device' },
        ],
      },
      { layout: 'preserve' },
    );
    expect(patched.ok, JSON.stringify(patched.diagnostics)).toBe(true);
    expect(patched.nodes.find((n) => n.path.join('/') === 'boundary')!.outputs.device).toEqual(
      device,
    );
    expect(
      ports(patched.project!.graph.nodes.find((n) => n.id === 'device_copy')!, registry).inputs,
    ).toEqual([]);
    const reopened = validateModel(parseProject(serializeProject(patched.project!), registry));
    expect(reopened.nodes).toEqual(patched.nodes);
    expect(serializeProject(patched.project!)).not.toContain('cuda:0');
    const invalid = patchModel(patched.project!, {
      operations: [
        { op: 'disconnect', from: 'device.out', to: 'arange.device' },
        { op: 'connect', from: 'maps.out', to: 'arange.device' },
      ],
    });
    expect(invalid.ok).toBe(false);
  });
});
