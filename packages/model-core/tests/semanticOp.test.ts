import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { ports } from '../src/core/graph';
import { parseProject, serializeProject } from '../src/core/serialization';
import type { Params, TensorSpec, ValueSpec } from '../src/core/types';
import { registry } from '../src/modules';
import { semanticOp } from '../src/modules/semanticOp';

const x: TensorSpec = { shape: [2, 3], axes: ['batch', 'channel'], dtype: 'float16' };
const dict: ValueSpec = { kind: 'dict', fields: { coords: x } };
function infer(params: Params = {}, values: Record<string, ValueSpec> = { x }) {
  const node = registry.create('SemanticOp', params);
  return semanticOp.inferValue!(values, node.params, {
    hyperparameters: { B: { type: 'int', value: 4 } },
  });
}

describe('SemanticOp', () => {
  it('inherits from the first declared Tensor after arbitrary Python values and retains only its tensor contract', () => {
    const result = infer(
      { inputs: 'meta, steps, x, later', summary: '读取特征', formula: 'features(x)' },
      {
        later: { ...x, dtype: 'int64' },
        meta: dict,
        steps: { kind: 'int', value: 3 },
        x: {
          ...x,
          parameterRef: { module: 'a', name: 'weight' },
          autograd: { requiresGrad: true, source: 'manual' },
        },
      },
    );
    expect(result).toEqual({ outputs: { out: x }, notes: ['读取特征', 'features(x)'] });
  });

  it('resolves expressions, generates axes and supports explicit scalar and empty dimensions', () => {
    expect(infer({ inputs: 'meta', shape: '(B + 2) // 2, 0' }, { meta: dict }).outputs.out).toEqual(
      { shape: [3, 0], axes: ['axis0', 'axis1'], dtype: 'float32' },
    );
    expect(infer({ shape: '[]', dtype: 'float64', requires_grad: 'on' }).outputs.out).toEqual({
      shape: [],
      axes: [],
      dtype: 'float64',
      autograd: { requiresGrad: true, source: 'manual' },
    });
  });

  it('uses per-output specifications independently of the shared fields, inheriting unspecified ports', () => {
    const specs = {
      index: { shape: 'B', axes: 'batch', dtype: 'int64', requires_grad: false },
      scores: { shape: 'B,3', dtype: 'float64', requires_grad: true },
    };
    const { outputs } = infer({
      outputs: 'index, scores, residual',
      dtype: 'bool',
      shape: '99',
      output_specs: JSON.stringify(specs),
    });
    expect(outputs.index).toEqual({
      shape: [4],
      axes: ['batch'],
      dtype: 'int64',
      autograd: { requiresGrad: false, source: 'manual' },
    });
    expect(outputs.scores).toEqual({
      shape: [4, 3],
      axes: ['axis0', 'axis1'],
      dtype: 'float64',
      autograd: { requiresGrad: true, source: 'manual' },
    });
    expect(outputs.residual).toEqual(x);
    expect(infer({ outputs: 'a, b', axes: 'b,c' }).outputs).toEqual({
      a: { ...x, axes: ['b', 'c'] },
      b: { ...x, axes: ['b', 'c'] },
    });
  });

  it('infers grad:off only with a dtype constraint or known off tensor inputs', () => {
    const off = { requiresGrad: false, source: 'inferred' };
    expect(infer({ dtype: 'int64' }).outputs.out).toMatchObject({ autograd: off });
    expect(
      infer(
        { inputs: 'i, x, d' },
        {
          i: { ...x, dtype: 'bool' },
          d: dict,
          x: { ...x, autograd: { requiresGrad: false, source: 'manual' } },
        },
      ).outputs.out,
    ).toMatchObject({ autograd: off });
    expect(infer({ inputs: 'x, meta' }, { x, meta: dict }).outputs.out).not.toHaveProperty(
      'autograd',
    );
  });

  it.each(['int64', 'bool'])('rejects enabled gradients for %s in both spec forms', (dtype) => {
    expect(() => infer({ dtype, requires_grad: 'on' })).toThrow(/浮点/);
    expect(() =>
      infer({ output_specs: JSON.stringify({ out: { shape: '2', dtype, requires_grad: true } }) }),
    ).toThrow(/浮点/);
  });
  const invalidSpecs: Params[] = [
    { summary: ' ' },
    { inputs: 'x,x' },
    { inputs: '0a' },
    { outputs: 'a-b' },
    { inputs: 'x,' },
    { outputs: '__proto__' },
    { inputs: 'a,b,c,d,e,f,g,h,i,j,k,l,m' },
    { outputs: 'a,b,c,d,e,f,g,h,i' },
    { inputs: 'x,missing' },
    { shape: 'B + unknown' },
    { shape: '-2' },
    { shape: '1,2', axes: 'one' },
    { output_specs: '[]' },
    { output_specs: '{' },
    { output_specs: '{"typo":{"shape":"1"}}' },
    { output_specs: '{"out":null}' },
    { output_specs: '{"out":{"dtype":"bool"}}' },
    { output_specs: '{"out":{"shape":"1","axes":null}}' },
    { output_specs: '{"out":{"shape":"1","dtype":"invalid"}}' },
    { output_specs: '{"out":{"shape":"1","requires_grad":"yes"}}' },
    { output_specs: '{"out":{"shape":"1","axis":"a"}}' },
  ];
  it.each(invalidSpecs)('rejects an invalid semantic contract %j', (params) => {
    expect(() => infer(params)).toThrow();
  });
  it('provides usable editing handles while a port declaration is incomplete', () => {
    expect(ports(registry.create('SemanticOp', { inputs: 'bad,', outputs: '' }), registry)).toEqual(
      { inputs: [{ id: 'x', label: 'x' }], outputs: [{ id: 'out', label: 'out' }] },
    );
    for (const specs of ['', '{"a":{"shape":"B"}}']) {
      expect(() =>
        infer({ inputs: 'meta', outputs: 'a,b', output_specs: specs }, { meta: dict }),
      ).toThrow(/没有 Tensor 输入/);
    }
  });
  it('merges lineages for every output and preserves heterogeneous autograd through the inference engine and save/load', () => {
    const model = buildModel({
      name: 'mixed',
      nodes: [
        { id: 'meta', type: 'ForwardInput', params: { schema: dict } },
        { id: 'xy', type: 'Input' },
        {
          id: 's',
          type: 'SemanticOp',
          params: {
            inputs: 'meta,xy',
            outputs: 'a,b',
            output_specs:
              '{"a":{"shape":"2","dtype":"bool"},"b":{"shape":"","requires_grad":"on"}}',
          },
        },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['xy.out', 's.xy'],
        ['meta.out', 's.meta'],
        ['s.b', 'out.in'],
      ],
    });
    expect(model.diagnostics).toEqual([]);
    const p = model.project!;
    p.lineageEnabled = true;
    const restored = parseProject(serializeProject(p), registry);
    expect(restored).toEqual(p);
    const { a, b } = inferProject(restored, registry).nodes.s.outputs;
    for (const value of [a, b])
      expect(value.lineage?.sources.map((s) => s.id)).toEqual(['meta', 'xy']);
    expect(a).toMatchObject({
      dtype: 'bool',
      autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(b).toMatchObject({
      dtype: 'float32',
      autograd: { requiresGrad: true, source: 'manual' },
    });
  });
  it('strictly builds the dual-mode recipe and validates the saved project', () => {
    const recipe = JSON.parse(
      readFileSync(
        new URL('../../../examples/semantic-dual-mode.recipe.json', import.meta.url),
        'utf8',
      ),
    );
    const model = buildModel(recipe);
    expect(model).toMatchObject({
      ok: true,
      diagnostics: [],
      outputs: { time: { shape: [2], dtype: 'int64' } },
    });
    expect(validateModel(model.project)).toMatchObject({ ok: true, diagnostics: [] });
  });
});
