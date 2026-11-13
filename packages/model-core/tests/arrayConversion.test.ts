import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { annotateInformationSources } from '../src/core/informationSources';
import { annotateTensorOutputs } from '../src/core/tensorAppearance';
import { parseProject, serializeProject } from '../src/core/serialization';
import { isTensorSpec, shapeText, tensorValue, type Params, type ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { arrayConversionCode } from '../src/modules/arrayConversion';

const meta: ValueSpec = { kind: 'forward', text: 'list[dict], not captured' };
const device: ValueSpec = { kind: 'device', reference: 'query_points.device' };
function run(type: string, v: Record<string, ValueSpec>, params: Params = {}) {
  const n = registry.create(type, params);
  const result = registry.get(type)!.inferValue!(v, n.params);
  return annotateInformationSources(n, v, annotateTensorOutputs(n, v, result.outputs)).out;
}
function sourced(v: ValueSpec, index: number): ValueSpec {
  return { ...v, lineage: { sources: [{ id: `source${index}`, index, label: `input${index}` }],
    origins: [], initializations: [], unknown: false, evidence: 'root' } };
}
const array = () => run('BatchMetaArray', { img_metas: sourced(meta, 2) });
const tensor: ValueSpec = { shape: [2, 1040, 256], axes: ['B', 'Q', 'D'], dtype: 'float16',
  autograd: { requiresGrad: true, source: 'manual' } };

describe('NumPy / Tensor conversion drawing', () => {
  it('splits asarray from astype without guessing the original dtype', () => {
    const raw = run('BatchMetaArray', { img_metas: sourced(meta, 2) }, { operation: 'asarray' });
    const before = structuredClone(raw);
    expect(raw).toMatchObject({ kind: 'ndarray', dtype: 'unknown', dims: [2, 30, 4, 4] });
    expect(shapeText(raw)).toContain('dtype 未采集');
    const code = arrayConversionCode(registry.create('BatchMetaArray', { operation: 'asarray' }));
    expect(code?.split('\n')[0]).not.toContain('.astype(');
    const cast = run('NumpyAstype', { in: raw });
    expect(cast).toEqual(array());
    expect(cast).not.toHaveProperty('autograd');
    expect(raw).toEqual(before);
    expect(arrayConversionCode(registry.create('NumpyAstype'))).toContain('values.astype(np.float32)');
    expect(() => run('ArrayToTensor', { in: raw, device })).toThrow();
    expect(run('ArrayToTensor', { in: cast, device })).toMatchObject({ dtype: 'float32' });
  });
  it('keeps legacy merged nodes compatible when operation is absent', () => {
    const old = registry.get('BatchMetaArray')!.inferValue!({ img_metas: meta },
      { field: 'lidar2img', shape: '2,30,4,4', axes: '', dtype: 'float32' });
    expect(old.outputs.out).toMatchObject({ dtype: 'float32' });
    const explicit = run('BatchMetaArray', { img_metas: meta },
      { operation: 'asarray', source_dtype: 'float64', dtype: 'float32' });
    expect(explicit).toMatchObject({ dtype: 'float64' });
    expect(run('NumpyAstype', { in: explicit }, { dtype: 'float32' })).toMatchObject({ dtype: 'float32' });
    const legacy = buildModel({ name: 'legacy', nodes: [
      { id: 'meta', type: 'ForwardInput', params: { text: 'list[dict]' } },
      { id: 'array', type: 'BatchMetaArray' },
    ], connections: [['meta.out', 'array.img_metas']] }, { mode: 'draft' }).project!;
    delete legacy.graph.nodes[1].params.operation;
    delete legacy.graph.nodes[1].params.source_dtype;
    const report = validateModel(legacy, { mode: 'draft' });
    expect(report.ok, JSON.stringify(report.diagnostics)).toBe(true);
    expect(report.nodes.find(n => n.path[0] === 'array')?.outputs.out).toMatchObject({ dtype: 'float32' });
  });
  it('rejects Tensor astype inputs and unsupported target dtypes', () => {
    expect(() => run('NumpyAstype', { in: tensor })).toThrow('ndarray');
    expect(() => run('NumpyAstype', { in: array() }, { dtype: 'object' })).toThrow();
    expect(() => run('BatchMetaArray', { img_metas: meta }, { operation: 'unknown' })).toThrow();
  });
  it('distinguishes an ndarray from a Tensor and retains only metadata value sources', () => {
    const a = array();
    expect(a).toMatchObject({ kind: 'ndarray', dims: [2, 30, 4, 4], dtype: 'float32' });
    expect(isTensorSpec(a)).toBe(false);
    expect(() => tensorValue(a)).toThrow();
    expect(a).not.toHaveProperty('autograd');
    expect(a.lineage?.sources.map(s => s.index)).toEqual([2]);
    expect(shapeText(a)).toBe('np.ndarray [2, 30, 4, 4] · float32');
    expect(arrayConversionCode(registry.create('BatchMetaArray'))).toContain('astype(np.float32)');
  });
  it('supports declared ego matrix dimensions without making shape/device new data sources', () => {
    const a = run('BatchMetaArray', { img_metas: sourced(meta, 2),
      d0: sourced({ kind: 'int', value: 1 }, 3) }, { shape: 'd0,4,4', field: 'ego2lidar' });
    expect(a).toMatchObject({ kind: 'ndarray', dims: [1, 4, 4] });
    expect(a.lineage?.sources.map(s => s.index)).toEqual([2]);
  });
  it('reads dtype as metadata, not features or an autograd edge', () => {
    const dtype = run('GetDtype', { in: sourced(tensor, 3) });
    expect(dtype).toMatchObject({ kind: 'dtype', value: 'float16' });
    expect(dtype.lineage?.sources).toEqual([]);
    expect(dtype).not.toHaveProperty('autograd');
    expect(shapeText(dtype)).toBe('torch.float16');
  });
  it('torch.tensor copies the array dtype and starts with no gradient', () => {
    const a = array(), before = structuredClone(a);
    const out = run('ArrayToTensor', { in: a, device: sourced(device, 3) });
    expect(out).toMatchObject({ shape: [2, 30, 4, 4], dtype: 'float32',
      autograd: { requiresGrad: false, source: 'inferred' } });
    expect(out.lineage?.sources.map(s => s.index)).toEqual([2]);
    expect(a).toEqual(before);
  });
  it('new_tensor uses receiver dtype but not receiver features or requires_grad', () => {
    const out = run('ArrayToTensor', { in: array(), device: sourced(device, 3),
      dtype: run('GetDtype', { in: sourced(tensor, 3) }) }, { mode: 'new_tensor' });
    expect(out).toMatchObject({ shape: [2, 30, 4, 4], dtype: 'float16',
      autograd: { requiresGrad: false, source: 'inferred' } });
    expect(out.lineage?.sources.map(s => s.index)).toEqual([2]);
  });
  it.each(['in', 'device', 'dtype'])('rejects a full Tensor used as %s', key => {
    expect(() => run('ArrayToTensor', { in: array(), device,
      dtype: { kind: 'dtype', value: 'float32' }, [key]: tensor }, { mode: 'new_tensor' })).toThrow();
  });
  it('requires dtype in new_tensor mode, and forbids Tensor input to metadata extraction', () => {
    expect(() => run('ArrayToTensor', { in: array(), device }, { mode: 'new_tensor' })).toThrow();
    expect(() => run('BatchMetaArray', { img_metas: tensor })).toThrow();
    expect(() => run('GetDtype', { in: array() })).toThrow();
  });
  it.each<Params>([{ shape: '-1,4,4' }, { dtype: 'object' }, { field: '../invalid' },
    { axes: 'one,axis' }])('rejects invalid array contracts %j', params => {
    expect(() => run('BatchMetaArray', { img_metas: meta }, params)).toThrow();
  });
  it('survives API save/reopen and dtype changes without modifying unrelated positions', () => {
    const built = buildModel({ name: 'matrix conversion', nodes: [
      { id: 'metas', type: 'ForwardInput', params: { text: 'list[dict]' }, position: { x: 42, y: 25 } },
      { id: 'q', type: 'Input', params: { shape: '2,1040,256', axes: 'B,Q,D', dtype: 'float32' } },
      { id: 'array', type: 'BatchMetaArray' },
      { id: 'device', type: 'GetDevice' }, { id: 'dtype', type: 'GetDtype' },
      { id: 'convert', type: 'ArrayToTensor', params: { mode: 'new_tensor' } },
    ], connections: [['metas.out', 'array.img_metas'], ['q.out', 'device.in'], ['q.out', 'dtype.in'],
      ['array.out', 'convert.in'], ['device.out', 'convert.device'], ['dtype.out', 'convert.dtype']] },
    { mode: 'draft' });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const reopened = parseProject(serializeProject(built.project!), registry);
    expect(validateModel(reopened, { mode: 'draft' }).ok).toBe(true);
    const changed = patchModel(reopened, { operations: [
      { op: 'set_params', id: 'q', params: { dtype: 'float16' } },
    ] }, { mode: 'draft', layout: 'preserve' });
    expect(changed.ok).toBe(true);
    expect(changed.nodes.find(n => n.path[0] === 'convert')?.outputs.out).toMatchObject({ dtype: 'float16' });
    expect(changed.project!.graph.nodes[0].position).toEqual(reopened.graph.nodes[0].position);
  });
});
