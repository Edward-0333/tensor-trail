import { describe, expect, it } from 'vitest';
import { buildModel, patchModel, validateModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { decoderLayerOpCode } from '../src/modules/decoderLayerOps';
import { newZerosCode } from '../src/modules/newZeros';

const tensor = (shape: number[], dtype = 'float32'): TensorSpec => ({ shape, axes: shape.map(() => ''), dtype });
const attentionInputs = () => ({
  query_points: tensor([2, 1, 3]), query_feat: tensor([2, 1, 256]),
  key_points: tensor([2, 720, 48, 3]), key_feat: tensor([2, 720, 256]),
});
function attention(values: Record<string, ValueSpec> = {}, params: Params = {}) {
  const node = registry.create('OPUSCrossAttentionContract', params);
  return registry.get(node.type)!.inferValue!({ ...attentionInputs(), ...values }, node.params);
}

describe('forecast ego attention interface', () => {
  it('keeps the Query shape, allowing a different number of Keys and arbitrary refined-point count', () => {
    expect(attention().outputs.out).toMatchObject({ shape: [2, 1, 256], dtype: 'float32' });
    expect(attention({ key_points: tensor([2, 1040, 4, 3]), key_feat: tensor([2, 1040, 256]) }).outputs.out)
      .toMatchObject({ shape: [2, 1, 256] });
    expect(attention().outputs.out).not.toHaveProperty('autograd');
    expect(Object.keys(attention().outputs)).toEqual(['out']);
    expect(decoderLayerOpCode(registry.create('OPUSCrossAttentionContract'))).toContain('第二个注意力权重');
  });

  it.each<Record<string, ValueSpec>>([
    { query_points: tensor([2, 1, 1, 3]) }, { query_points: tensor([2, 3, 3]) },
    { query_points: tensor([2, 1, 4]) }, { key_points: tensor([2, 720, 3]) },
    { key_points: tensor([2, 720, 0, 3]) }, { key_feat: tensor([2, 719, 256]) },
    { key_feat: tensor([3, 720, 256]) }, { key_feat: tensor([2, 720, 128]) },
    { key_feat: tensor([2, 720, 256], 'float64') }, { query_points: tensor([2, 1, 3], 'int64') },
  ])('rejects incompatible attention input %j', inputs => {
    expect(() => attention(inputs)).toThrow();
  });

  it('rejects invalid attention configuration', () => {
    expect(() => attention({}, { num_heads: 7 })).toThrow(/整除/);
    expect(() => attention({}, { embed_dims: 0 })).toThrow(/正整数/);
    expect(() => attention({ key_points: tensor([2, 0, 48, 3]), key_feat: tensor([2, 0, 256]) })).toThrow(/不能为空/);
  });

  it.each(['float16', 'bfloat16', 'float32', 'float64', 'int64'])(
    'new_ones inherits dtype %s, not reference data, shape or grad state', dtype => {
      const node = registry.create('NewOnes', { shape: 'd0,1,3', metadata_inputs: true, axes: 'B,ego,xyz' });
      const out = registry.get(node.type)!.inferValue!({
        dtype: { kind: 'dtype', value: dtype }, device: { kind: 'device', reference: 'ego_feat.device' },
        d0: { kind: 'int', value: 2 },
      }, node.params).outputs.out;
      expect(out).toMatchObject({ shape: [2, 1, 3], dtype, autograd: { requiresGrad: false } });
      expect(out).not.toHaveProperty('parameterRef');
      expect(newZerosCode(node)).toContain('new_ones');
      expect(newZerosCode(node)).not.toContain('全零');
    },
  );

  it('tracks fused data sources through detach but not through metadata-only factory dependencies', () => {
    const built = buildModel({ name: 'forecast attention', nodes: [
      { id: 'ego', type: 'Input', params: { shape: '2,1,256', axes: '' } },
      { id: 'pos', type: 'Input', params: { shape: '2,720,48,3', axes: '' } },
      { id: 'feat', type: 'Input', params: { shape: '2,720,256', axes: '' } },
      { id: 'B', type: 'IntegerParameter', params: { expression: '2' } },
      { id: 'dtype', type: 'GetDtype' }, { id: 'device', type: 'GetDevice' },
      { id: 'ones', type: 'NewOnes', params: { shape: 'd0,1,3', metadata_inputs: true } },
      { id: 'center', type: 'Function', params: { expression: 'x * 0.5' } },
      { id: 'detach_pos', type: 'Detach' }, { id: 'detach_feat', type: 'Detach' },
      { id: 'attn', type: 'OPUSCrossAttentionContract' }, { id: 'out', type: 'Output' },
    ], connections: [
      ['ego.out', 'dtype.in'], ['ego.out', 'device.in'], ['dtype.out', 'ones.dtype'],
      ['device.out', 'ones.device'], ['B.out', 'ones.d0'], ['ones.out', 'center.in'],
      ['pos.out', 'detach_pos.in'], ['feat.out', 'detach_feat.in'],
      ['center.out', 'attn.query_points'], ['ego.out', 'attn.query_feat'],
      ['detach_pos.out', 'attn.key_points'], ['detach_feat.out', 'attn.key_feat'], ['attn.out', 'out.in'],
    ] });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const enabled = patchModel(built.project!, { operations: [{ op: 'set_lineage_enabled', enabled: true }] });
    expect(enabled.ok).toBe(true);
    const nodes = inferProject(enabled.project!, registry).nodes;
    expect(nodes.center.outputs.out).toMatchObject({ autograd: { requiresGrad: false },
      lineage: { sources: [], origins: ['constant'], initializations: ['ones'], unknown: false } });
    expect(nodes.detach_feat.outputs.out).toMatchObject({ autograd: { requiresGrad: false } });
    expect((nodes.attn.outputs.out as TensorSpec).lineage?.sources.map(s => s.id)).toEqual(['ego', 'pos', 'feat']);
    expect(nodes.attn.outputs.out).not.toHaveProperty('autograd');
    expect(validateModel(parseProject(serializeProject(enabled.project!), registry)).ok).toBe(true);
  });
});
