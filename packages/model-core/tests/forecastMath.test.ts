import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import type {
  BooleanSpec, DictionarySpec, Dim, IntegerSpec, Params, Project, TensorSpec, ValueSpec,
} from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { forecastMath, forecastMathCode } from '../src/modules/forecastMath';
import { tensor } from '../src/modules/math';
import { primitives } from '../src/modules/primitives';
import { ModuleRegistry } from '../src/modules/registry';

const registry = new ModuleRegistry();
for (const definition of [...primitives, ...forecastMath]) registry.register(definition);

const t = (shape: Dim[], dtype = 'float32'): TensorSpec => tensor(shape, shape.map((_, i) => `axis${i}`), dtype);
const integer = (value: Dim): IntegerSpec => ({ kind: 'int', value });
const boolean = (value: boolean): BooleanSpec => ({ kind: 'bool', value });
const dict = (fields: DictionarySpec['fields'], keyType?: 'str' | 'int'): DictionarySpec => ({
  kind: 'dict', fields, ...(keyType === undefined ? {} : { keyType }),
});
const rule = (type: string) => registry.get(type)!;
function infer(type: string, inputs: Record<string, ValueSpec>, params: Params = {}) {
  const definition = rule(type), values = registry.create(type, params).params;
  return definition.inferValue
    ? definition.inferValue(inputs, values)
    : definition.infer!(inputs as Record<string, TensorSpec>, values);
}
const out = (type: string, inputs: Record<string, ValueSpec>, params: Params = {}) => infer(type, inputs, params).outputs.out;
const tensorOut = (type: string, inputs: Record<string, ValueSpec>, params: Params = {}) => tensorValue(out(type, inputs, params));

describe('forecastMath contract', () => {
  it.each([
    ['Unflatten', ['in'], { dim: -1, sizes: '-1,3' }],
    ['TensorCompareScalar', ['in'], { operator: 'ge', value: '0' }],
    ['LogicalAnd', ['a', 'b'], {}],
    ['MatrixInverse', ['in'], {}],
    ['TensorToDevice', ['in'], { target: 'cpu' }],
    ['DictionaryUpdate', ['base', 'updates'], {}],
    ['SequenceLength', ['in'], {}],
    ['SelectValue', ['condition', 'true_value', 'false_value'], {}],
    ['DynamicSlice', ['in', 'start', 'stop'], { dim: 1 }],
    ['BooleanParameter', [], { value: false }],
    ['BooleanAnd', ['a', 'b'], {}],
    ['BooleanNot', ['in'], {}],
    ['IntegerCompare', ['a', 'b'], { operator: 'lt' }],
  ] as [string, string[], Params][])('%s exports stable ports, defaults and readable code', (type, inputs, params) => {
    expect(rule(type).inputs.map((port) => port.id)).toEqual(inputs);
    expect(rule(type).outputs.map((port) => port.id)).toEqual(['out']);
    const node = registry.create(type);
    expect(node.params).toEqual(params);
    expect(forecastMathCode(node)).toContain('out =');
  });

  it('exports only the requested definitions and ignores unrelated code requests', () => {
    expect(forecastMath).toHaveLength(13);
    expect(new Set(forecastMath.map((definition) => definition.type)).size).toBe(13);
    expect(forecastMathCode(registry.create('Input'))).toBeUndefined();
    expect(rule('TensorCompareScalar').fields.find((field) => field.key === 'operator')?.options)
      .toEqual(['ge', 'le', 'lt', 'gt']);
    expect(rule('IntegerCompare').fields[0].options).toEqual(['lt', 'le', 'eq', 'ge', 'gt']);
    expect(rule('TensorToDevice').fields[0].options).toEqual(['cpu', 'cuda']);
  });
});

describe('Unflatten', () => {
  it.each([1, -2])('splits only dim=%s, preserving the other dimensions and labels', (dim) => {
    const input = { ...t(['B', 12, 'T'], 'float16'), axes: ['batch', 'packed', 'time'] };
    const before = structuredClone(input);
    const result = tensorOut('Unflatten', { in: input }, { dim, sizes: '-1,3' });
    expect(result).toEqual(tensor(['B', 4, 3, 'T'], ['batch', 'packed_0', 'packed_1', 'time'], 'float16'));
    expect(result.shape).not.toBe(input.shape);
    expect(result.axes).not.toBe(input.axes);
    expect(input).toEqual(before);
  });

  it('supports splitting the first/last axis and expanding a size into itself', () => {
    expect(tensorOut('Unflatten', { in: t([6, 5]) }, { dim: 0, sizes: '2, +3' }).shape).toEqual([2, 3, 5]);
    expect(tensorOut('Unflatten', { in: t([2, 12]) }).shape).toEqual([2, 4, 3]);
    const result = infer('Unflatten', { in: t(['B', 'C']) }, { sizes: '-1' });
    expect(result.outputs.out).toEqual(t(['B', 'C']));
    expect(result.warnings).toEqual([]);
  });

  it('preserves symbolic dimensions and states unresolved product constraints', () => {
    const result = infer('Unflatten', { in: t(['B', 'C']) });
    expect(tensorValue(result.outputs.out).shape).toEqual(['B', '(C)/(3)', 3]);
    expect(result.warnings?.join()).toContain('divide');
    const named = infer('Unflatten', { in: t([12]) }, { sizes: 'Q,3' });
    expect(tensorValue(named.outputs.out).shape).toEqual(['Q', 3]);
    expect(named.warnings?.join()).toContain('nonnegative');
    expect(named.warnings?.join()).toContain('12');
    const inferred = infer('Unflatten', { in: t([12]) }, { sizes: '-1,Q' });
    expect(tensorValue(inferred.outputs.out).shape).toEqual(['(12)/(Q)', 'Q']);
    expect(inferred.warnings?.join()).toContain('nonzero');
    expect(infer('Unflatten', { in: t(['Q']) }, { sizes: 'Q,1' }).warnings).toEqual([
      'Symbolic sizes must resolve to nonnegative integers at runtime.',
    ]);
  });

  it('handles zero axes without hiding axis mismatches or ambiguous -1', () => {
    expect(tensorOut('Unflatten', { in: t([2, 0, 5]) }, { dim: 1 }).shape).toEqual([2, 0, 3, 5]);
    expect(tensorOut('Unflatten', { in: t([0]) }, { sizes: '0,3' }).shape).toEqual([0, 3]);
    expect(tensorOut('Unflatten', { in: t([0]) }, { sizes: '0,Q' }).shape).toEqual([0, 'Q']);
    expect(() => infer('Unflatten', { in: t([0]) }, { sizes: '0,-1' })).toThrow('ambiguous');
    expect(() => infer('Unflatten', { in: t([0, 5]) }, { sizes: '2,3' })).toThrow('axis size');
    expect(() => infer('Unflatten', { in: t([5]) })).toThrow('divisible');
  });

  it.each(['', ' ', '2,,3', '-1,-1', '-2,3', '1.5,2', 'N*3', '[]', '1e2', 'NaN!', '9007199254740992'])
    ('rejects invalid sizes %j', (sizes) => {
      expect(() => infer('Unflatten', { in: t([12]) }, { sizes })).toThrow();
    });

  it('rejects invalid axes, malformed dimensions and unsafe products', () => {
    for (const dim of [1, -2, 0.5, Infinity, '0'])
      expect(() => infer('Unflatten', { in: t([12]) }, { dim })).toThrow();
    expect(() => infer('Unflatten', { in: t([]) })).toThrow();
    expect(() => infer('Unflatten', { in: t([-1]) })).toThrow('nonnegative');
    expect(() => infer('Unflatten', { in: t([12]) }, { sizes: '9007199254740991,2' })).toThrow();
    expect(() => infer('Unflatten', { in: t([12]) }, { sizes: 12 })).toThrow('text');
    expect(() => infer('Unflatten', {})).toThrow();
  });
});

describe('TensorCompareScalar / LogicalAnd', () => {
  it.each(['ge', 'le', 'lt', 'gt'])('compares using %s without claiming to know Tensor contents', (operator) => {
    const source = { ...t(['B', 0, 3]), autograd: { requiresGrad: true, source: 'manual' as const } };
    expect(out('TensorCompareScalar', { in: source }, { operator, value: ' -1.25e-2 ' })).toEqual({
      ...t(['B', 0, 3], 'bool'), autograd: { requiresGrad: false, source: 'inferred' },
    });
    expect(source.autograd.requiresGrad).toBe(true);
  });

  it.each(['float16', 'bfloat16', 'float32', 'float64', 'int64', 'int8', 'uint8', 'bool'])
    ('accepts scalar and dimensioned real %s Tensors', (dtype) => {
      expect(tensorOut('TensorCompareScalar', { in: t([], dtype) }).dtype).toBe('bool');
      expect(tensorOut('TensorCompareScalar', { in: t([2, 3], dtype) }).shape).toEqual([2, 3]);
    });

  it.each(['', ' ', 'NaN', 'Infinity', '1e999', '0x10', '1; exit()', 'False'])
    ('rejects non-decimal or non-finite comparison value %j', (value) => {
      expect(() => infer('TensorCompareScalar', { in: t([2]) }, { value })).toThrow();
    });

  it('validates operator/dtype/text and emits valid Python numeric literals', () => {
    expect(() => infer('TensorCompareScalar', { in: t([2]) }, { value: 1 })).toThrow('text');
    for (const operator of ['eq', 'ne', '__proto__', 'toString'])
      expect(() => infer('TensorCompareScalar', { in: t([2]) }, { operator })).toThrow('operator');
    expect(() => infer('TensorCompareScalar', { in: t([2], 'complex64') })).toThrow();
    expect(forecastMathCode(registry.create('TensorCompareScalar', { value: '0002' }))).toContain('x >= 2');
    expect(forecastMathCode(registry.create('TensorCompareScalar', { value: '9007199254740993' })))
      .toContain('x >= 9007199254740993');
  });

  it('broadcasts bool tensors, preserving empty and scalar shapes', () => {
    const a = { ...t([2, 1, 3], 'bool'), axes: ['batch', 'singleton', 'xyz'] };
    const b = { ...t([4, 1], 'bool'), axes: ['query', 'singleton'] };
    const result = tensorOut('LogicalAnd', { a, b });
    expect(result.shape).toEqual([2, 4, 3]);
    expect(result.axes).toEqual(['batch', 'query', 'xyz']);
    expect(result.autograd).toEqual({ requiresGrad: false, source: 'inferred' });
    expect(tensorOut('LogicalAnd', { a: t([], 'bool'), b: t([], 'bool') }).shape).toEqual([]);
    expect(tensorOut('LogicalAnd', { a: t([0, 3], 'bool'), b: t([1, 3], 'bool') }).shape).toEqual([0, 3]);
    expect(infer('LogicalAnd', { a: t(['B'], 'bool'), b: t(['C'], 'bool') }).warnings).toHaveLength(1);
  });

  it('rejects non-bool tensors, missing operands and incompatible broadcasts', () => {
    expect(() => infer('LogicalAnd', { a: t([2]), b: t([2], 'bool') })).toThrow('bool');
    expect(() => infer('LogicalAnd', { a: t([2], 'bool'), b: t([3], 'bool') })).toThrow();
    expect(() => infer('LogicalAnd', { a: t([2], 'bool') })).toThrow();
  });
});

describe('MatrixInverse / TensorToDevice', () => {
  it.each(['float32', 'float64'])('preserves batched square shapes and %s dtype', (dtype) => {
    const source = t(['B', 3, 4, 4], dtype);
    const result = infer('MatrixInverse', { in: source });
    expect(result.outputs.out).toEqual(source);
    expect(tensorValue(result.outputs.out).shape).not.toBe(source.shape);
    expect(result.notes?.join()).toContain('invertibility is not checked');
    expect(tensorOut('MatrixInverse', { in: t([0, 0], dtype) }).shape).toEqual([0, 0]);
  });

  it('checks square axes without pretending symbolic dimensions are established', () => {
    expect(infer('MatrixInverse', { in: t(['N', 'N']) }).warnings).toEqual([]);
    expect(infer('MatrixInverse', { in: t(['N', 4]) }).warnings).toHaveLength(1);
    expect(() => infer('MatrixInverse', { in: t([2, 3]) })).toThrow('square');
    expect(() => infer('MatrixInverse', { in: t([2]) })).toThrow('two axes');
    expect(() => infer('MatrixInverse', { in: t([]) })).toThrow('two axes');
    expect(() => infer('MatrixInverse', { in: t([-1, -1]) })).toThrow('nonnegative');
    for (const dtype of ['bool', 'int64', 'float16', 'bfloat16', 'complex64'])
      expect(() => infer('MatrixInverse', { in: t([2, 2], dtype) })).toThrow('float32 or float64');
  });

  it.each(['cpu', 'cuda'])('records %s transfer but preserves values and gradient metadata', (target) => {
    for (const requiresGrad of [true, false, undefined]) {
      const source: TensorSpec = {
        ...t(['B', 3], 'float16'),
        ...(requiresGrad === undefined ? {} : { autograd: { requiresGrad, source: 'manual' as const } }),
      };
      const result = infer('TensorToDevice', { in: source }, { target });
      const moved = tensorValue(result.outputs.out);
      expect(moved).toEqual(source);
      expect(moved).not.toBe(source);
      expect(moved.shape).not.toBe(source.shape);
      expect(moved).not.toHaveProperty('device');
      expect(result.notes?.join()).toContain(target);
      expect(result.notes?.join()).toContain('no transfer is executed');
    }
    expect(out('TensorToDevice', { in: t([], 'int64') }, { target })).toEqual(t([], 'int64'));
  });

  it('does not silently accept unknown target devices', () => {
    for (const target of ['cuda:0', 'meta', 'CPU', 0])
      expect(() => infer('TensorToDevice', { in: t([2]) }, { target })).toThrow('cpu or cuda');
  });
});

describe('DictionaryUpdate / SequenceLength', () => {
  it('replaces nested fields shallowly, preserving full field specs and input immutability', () => {
    const feature: TensorSpec = {
      ...t(['B', 3]), autograd: { requiresGrad: true, source: 'runtime' },
      lineage: { sources: [], origins: ['parameter'], initializations: ['checkpoint'], unknown: false, evidence: 'inferred' },
      parameterRef: { module: 'decoder', name: 'weight' },
    };
    const base = dict({ kept: feature, nested: dict({ old: t([1]), shared: t([2]) }), replaced: t([9]) });
    const updates = dict({ nested: dict({ new: t([3]) }), replaced: { kind: 'none' }, history: { kind: 'list', items: [feature] } });
    const before = structuredClone({ base, updates });
    const result = out('DictionaryUpdate', { base, updates }) as DictionarySpec;
    expect(result).toEqual(dict({ ...base.fields, ...updates.fields }));
    expect(result.fields.nested).toEqual(dict({ new: t([3]) }));
    expect(result.fields.kept).toEqual(feature);
    expect(result.fields.kept).not.toBe(feature);
    tensorValue(result.fields.kept).shape[0] = 99;
    expect({ base, updates }).toEqual(before);
    expect(forecastMathCode(registry.create('DictionaryUpdate'))).toContain('base.update(updates)\nout = base');
  });

  it('preserves integer keys and refuses unrepresentable mixed key types', () => {
    expect(out('DictionaryUpdate', { base: dict({ '0': t([2]) }, 'int'), updates: dict({ '0': t([3]), '1': t([4]) }, 'int') }))
      .toEqual(dict({ '0': t([3]), '1': t([4]) }, 'int'));
    expect(out('DictionaryUpdate', { base: dict({}, 'int'), updates: dict({ x: t([2]) }) }))
      .toEqual(dict({ x: t([2]) }));
    expect(out('DictionaryUpdate', { base: dict({ '0': t([2]) }, 'int'), updates: dict({}) }))
      .toEqual(dict({ '0': t([2]) }, 'int'));
    expect(out('DictionaryUpdate', { base: dict({}), updates: dict({ '0': t([2]) }, 'int') }))
      .toEqual(dict({ '0': t([2]) }, 'int'));
    expect(() => infer('DictionaryUpdate', { base: dict({ '0': t([2]) }, 'int'), updates: dict({ '0': t([3]) }) }))
      .toThrow('Mixed str/int');
    expect(() => infer('DictionaryUpdate', { base: t([2]), updates: dict({}) })).toThrow('base');
    expect(() => infer('DictionaryUpdate', { base: dict({}) })).toThrow('updates');
    expect(() => infer('DictionaryUpdate', { base: dict({ '01': t([2]) }, 'int'), updates: dict({}) })).toThrow();
    expect(() => infer('DictionaryUpdate', { base: dict({}), updates: dict({ ['__proto__']: t([2]) }) })).toThrow();
  });

  it.each(['list', 'tuple'] as const)('counts %s entries without adding their Tensor element counts', (kind) => {
    expect(out('SequenceLength', { in: { kind, items: [t([20, 30]), t([100]), t([])] } })).toEqual(integer(3));
    expect(out('SequenceLength', { in: { kind, items: [] } })).toEqual(integer(0));
  });

  it('also accepts list[int], but not an opaque container or a Tensor', () => {
    expect(out('SequenceLength', { in: { kind: 'int-list', values: [3, 8] } })).toEqual(integer(2));
    expect(out('SequenceLength', { in: { kind: 'int-list', values: [] } })).toEqual(integer(0));
    for (const value of [t([6]), dict({}), { kind: 'list', text: 'unknown' }] as ValueSpec[])
      expect(() => infer('SequenceLength', { in: value })).toThrow('known items');
    expect(() => infer('SequenceLength', {})).toThrow('known items');
  });
});

describe('SelectValue / Python Boolean operators', () => {
  it.each([true, false])('selects only the correct complete spec for condition=%s', (condition) => {
    const trueValue = { ...t([2, 3]), autograd: { requiresGrad: true, source: 'runtime' as const } };
    const falseValue = dict({ history: { kind: 'tuple', items: [t([9], 'int64')] } });
    const chosen = condition ? trueValue : falseValue;
    const result = out('SelectValue', { condition: boolean(condition), true_value: trueValue, false_value: falseValue });
    expect(result).toEqual(chosen);
    expect(result).not.toBe(chosen);
    expect(out('SelectValue', { condition: boolean(condition), true_value: integer(7), false_value: { kind: 'none' } }))
      .toEqual(condition ? integer(7) : { kind: 'none' });
  });

  it.each([
    t([], 'bool'), t([1], 'bool'), boolean(undefined as unknown as boolean),
    { kind: 'control', expression: 'self.training', branch: true }, integer(1),
  ] as ValueSpec[])('rejects unknown or non-Python conditions %# even with identical alternatives', (condition) => {
    expect(() => infer('SelectValue', { condition, true_value: t([2]), false_value: t([2]) }))
      .toThrow('known Python BooleanSpec');
  });

  it('requires both graph alternatives rather than claiming lazy branch execution', () => {
    expect(() => infer('SelectValue', { condition: boolean(true), true_value: t([2]) })).toThrow('both');
    expect(() => infer('SelectValue', { true_value: t([2]), false_value: t([2]) })).toThrow('condition');
  });

  it.each([true, false])('declares and negates Python bool %s', (value) => {
    expect(out('BooleanParameter', {}, { value })).toEqual(boolean(value));
    expect(out('BooleanNot', { in: boolean(value) })).toEqual(boolean(!value));
  });

  it.each([[true, true], [true, false], [false, true], [false, false]])
    ('computes %s and %s as a Python bool', (a, b) => {
      expect(out('BooleanAnd', { a: boolean(a), b: boolean(b) })).toEqual(boolean(a && b));
    });

  it('rejects scalar tensors, ControlSpec, missing inputs and non-Boolean parameters', () => {
    for (const value of ['True', 1, 0])
      expect(() => infer('BooleanParameter', {}, { value })).toThrow('Boolean');
    expect(() => infer('BooleanAnd', { a: boolean(false), b: t([], 'bool') })).toThrow('b');
    expect(() => infer('BooleanAnd', { a: boolean(false) })).toThrow('b');
    expect(() => infer('BooleanNot', { in: { kind: 'control', expression: 'flag', branch: false } })).toThrow('BooleanSpec');
    expect(() => infer('BooleanNot', { in: t([], 'bool') })).toThrow('BooleanSpec');
  });
});

describe('IntegerCompare', () => {
  it.each(['lt', 'le', 'eq', 'ge', 'gt'])('compares concrete integers using %s', (operator) => {
    for (const [a, b] of [[-2, 1], [0, 0], [6, 3], [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER - 1]]) {
      const expected = { lt: a < b, le: a <= b, eq: a === b, ge: a >= b, gt: a > b }[operator];
      expect(out('IntegerCompare', { a: integer(a), b: integer(b) }, { operator })).toEqual(boolean(expected!));
    }
  });

  it('refuses symbolic inputs even if the symbols match', () => {
    expect(() => infer('IntegerCompare', { a: integer('N'), b: integer(6) })).toThrow('concrete integers');
    expect(() => infer('IntegerCompare', { a: integer('N'), b: integer('N') }, { operator: 'eq' })).toThrow('symbolic');
    for (const a of [integer(1.2), integer(NaN), integer(Number.MAX_SAFE_INTEGER + 1), t([], 'int64'), boolean(true)])
      expect(() => infer('IntegerCompare', { a, b: integer(6) })).toThrow();
    expect(() => infer('IntegerCompare', { a: integer(1), b: integer(6) }, { operator: 'ne' })).toThrow('operator');
    expect(() => infer('IntegerCompare', { a: integer(1) })).toThrow();
  });
});

describe('DynamicSlice', () => {
  it.each([
    [0, 1, 1], [3, 4, 1], [9, 10, 1], [10, 11, 0], [20, 21, 0],
    [-1, 10, 1], [-3, -1, 2], [-20, 20, 10], [4, 2, 0], [-1, 0, 0],
    [0, -1, 9], [0, 0, 0], [-20, -11, 0], [20, -1, 0],
  ])('slices start=%s stop=%s into length %s with Python clipping', (start, stop, length) => {
    const source = { ...t(['B', 10, 3], 'float64'), axes: ['batch', 'time', 'xyz'] };
    const before = structuredClone(source);
    const result = tensorOut('DynamicSlice', { in: source, start: integer(start), stop: integer(stop) });
    expect(result).toEqual(tensor(['B', length, 3], source.axes, source.dtype));
    expect(result.axes).not.toBe(source.axes);
    expect(source).toEqual(before);
  });

  it('matches array slicing across concrete lengths, negative bounds and extreme safe integers', () => {
    for (const length of [0, 1, 3, 10]) {
      const values = Array.from({ length });
      for (const start of [-Number.MAX_SAFE_INTEGER, -12, -3, -1, 0, 1, 4, 12, Number.MAX_SAFE_INTEGER]) {
        for (const stop of [-Number.MAX_SAFE_INTEGER, -12, -3, -1, 0, 1, 4, 12, Number.MAX_SAFE_INTEGER]) {
          const result = tensorOut('DynamicSlice', { in: t([length]), start: integer(start), stop: integer(stop) }, { dim: -1 });
          expect(result.shape).toEqual([values.slice(start, stop).length]);
        }
      }
    }
  });

  it('does not pretend symbolic interval:interval+1 always contains one item', () => {
    const result = infer('DynamicSlice', { in: t(['B', 10, 3]), start: integer('interval'), stop: integer('(interval + 1)') });
    expect(tensorValue(result.outputs.out).shape).toEqual(['B', 'slice(10,interval:(interval + 1):1)', 3]);
    expect(result.warnings?.join()).toContain('unresolved');
    const symbolicAxis = infer('DynamicSlice', { in: t(['B', 'T', 3]), start: integer(2), stop: integer(3) });
    expect(tensorValue(symbolicAxis.outputs.out).shape[1]).toBe('slice(T,2:3:1)');
    expect(symbolicAxis.warnings).toHaveLength(1);
  });

  it('can establish empty slices despite other symbolic dimensions', () => {
    expect(tensorOut('DynamicSlice', { in: t([0]), start: integer('i'), stop: integer('j') }, { dim: 0 }).shape).toEqual([0]);
    expect(tensorOut('DynamicSlice', { in: t(['T']), start: integer('i'), stop: integer('i') }, { dim: 0 }).shape).toEqual([0]);
    expect(tensorOut('DynamicSlice', { in: t(['T']), start: integer(4), stop: integer(2) }, { dim: 0 }).shape).toEqual([0]);
    expect(tensorOut('DynamicSlice', { in: t(['T']), start: integer(-2), stop: integer(-4) }, { dim: 0 }).shape).toEqual([0]);
    expect(typeof tensorOut('DynamicSlice', { in: t(['T']), start: integer(4), stop: integer(-2) }, { dim: 0 }).shape[0]).toBe('string');
  });

  it('validates axes and requires integer metadata rather than scalar Tensor bounds', () => {
    const inputs = { in: t([2, 10, 3]), start: integer(2), stop: integer(3) };
    expect(tensorOut('DynamicSlice', inputs, { dim: -2 }).shape).toEqual([2, 1, 3]);
    for (const dim of [-4, 3, 0.5, '1']) expect(() => infer('DynamicSlice', inputs, { dim })).toThrow();
    expect(() => infer('DynamicSlice', { ...inputs, start: t([], 'int64') })).toThrow();
    expect(() => infer('DynamicSlice', { ...inputs, stop: integer(1.5) })).toThrow();
    expect(() => infer('DynamicSlice', { in: t([2]), start: integer(0) }, { dim: 0 })).toThrow();
    expect(() => infer('DynamicSlice', { ...inputs, in: t([]) }, { dim: 0 })).toThrow();
  });
});

describe('isolated registry graph integration', () => {
  it('recomputes training/pretrain selection, sequence length and len < 6 without global registration', () => {
    const local = new ModuleRegistry();
    for (const definition of [...primitives, ...forecastMath]) local.register(definition);
    const sources: Record<string, ValueSpec> = {
      ShortHistory: { kind: 'list', items: [t([2, 3])] },
      LongHistory: { kind: 'tuple', items: Array.from({ length: 6 }, () => t([2, 3])) },
      Six: integer(6),
    };
    for (const [type, value] of Object.entries(sources)) local.register({
      type, label: type, category: 'test', description: 'test fixture', fields: [], inputs: [],
      outputs: [{ id: 'out', label: 'out' }], inferValue: () => ({ outputs: { out: value } }),
    });
    const nodes = [
      local.create('BooleanParameter', { value: true }, 'training'),
      local.create('BooleanParameter', { value: false }, 'pretrain'),
      local.create('BooleanNot', {}, 'not_pretrain'),
      local.create('BooleanAnd', {}, 'condition'),
      local.create('ShortHistory', {}, 'short'), local.create('LongHistory', {}, 'long'),
      local.create('SelectValue', {}, 'select'), local.create('SequenceLength', {}, 'length'),
      local.create('Six', {}, 'six'), local.create('IntegerCompare', {}, 'compare'),
      local.create('Output', {}, 'result'),
    ];
    const links = [
      ['pretrain', 'not_pretrain', 'in'], ['training', 'condition', 'a'], ['not_pretrain', 'condition', 'b'],
      ['condition', 'select', 'condition'], ['short', 'select', 'true_value'], ['long', 'select', 'false_value'],
      ['select', 'length', 'in'], ['length', 'compare', 'a'], ['six', 'compare', 'b'], ['compare', 'result', 'in'],
    ];
    const project: Project = {
      schemaVersion: 1, name: 'known configuration selection', variants: [],
      graph: { nodes, edges: links.map(([source, target, targetPort], i) => ({ id: `e${i}`, source, sourcePort: 'out', target, targetPort })) },
    };
    const before = structuredClone(project);
    const training = inferProject(project, local);
    expect(training.issues).toEqual([]);
    expect(Object.values(training.nodes).map((node) => node.status)).not.toContain('error');
    expect(training.nodes.length.outputs.out).toEqual(integer(1));
    expect(training.nodes.result.outputs.out).toEqual(boolean(true));
    expect(project).toEqual(before);
    nodes.find((node) => node.id === 'pretrain')!.params.value = true;
    const pretrain = inferProject(project, local);
    expect(pretrain.nodes.length.outputs.out).toEqual(integer(6));
    expect(pretrain.nodes.result.outputs.out).toEqual(boolean(false));
    nodes.find((node) => node.id === 'training')!.params.value = false;
    nodes.find((node) => node.id === 'pretrain')!.params.value = false;
    expect(inferProject(project, local).nodes.result.outputs.out).toEqual(boolean(false));
  });
});
