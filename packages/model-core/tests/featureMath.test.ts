import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import {
  tensorValue,
  type Dim,
  type IntegerSpec,
  type Project,
  type TensorSpec,
  type ValueSpec,
} from '../src/core/types';
import { registry } from '../src/modules/index';
import { featureMathModules, viewLabel, viewPorts } from '../src/modules/featureMath';

const integer = (value: Dim): IntegerSpec => ({ kind: 'int', value });
const tensor = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, index) => `axis${index}`),
  dtype,
});
const rule = (type: string) => featureMathModules.find((module) => module.type === type)!;
const multiply = rule('IntegerMultiply').inferValue!;
const view = rule('View').inferValue!;

function project(time = 20): Project {
  return {
    schemaVersion: 1,
    name: 'agent_feature.view(bs * A, T, -1)',
    variants: [],
    graph: {
      nodes: [
        registry.create(
          'Input',
          { shape: `2,49,${time},9`, axes: 'batch,agent,time,feature' },
          'feat',
        ),
        registry.create('TensorSize', { axes: '0,1,2', names: 'bs,A,T' }, 'size'),
        registry.create('IntegerMultiply', {}, 'product'),
        registry.create('View', {}, 'view'),
        registry.create('Output', {}, 'out'),
      ],
      edges: [
        { id: 'size', source: 'feat', sourcePort: 'out', target: 'size', targetPort: 'in' },
        { id: 'tensor', source: 'feat', sourcePort: 'out', target: 'view', targetPort: 'in' },
        { id: 'bs', source: 'size', sourcePort: 'dim0', target: 'product', targetPort: 'a' },
        { id: 'agents', source: 'size', sourcePort: 'dim1', target: 'product', targetPort: 'b' },
        { id: 'product', source: 'product', sourcePort: 'out', target: 'view', targetPort: 'd0' },
        { id: 'time', source: 'size', sourcePort: 'dim2', target: 'view', targetPort: 'd1' },
        { id: 'result', source: 'view', sourcePort: 'out', target: 'out', targetPort: 'in' },
      ],
    },
  };
}

describe('Cos / Sin', () => {
  it.each(['Cos', 'Sin'])('%s 保留浮点类型，整数和 bool 提升到 float32', (type) => {
    for (const dtype of [
      'float16',
      'bfloat16',
      'float32',
      'float64',
      'bool',
      'uint8',
      'int8',
      'int16',
      'int32',
      'int64',
    ]) {
      const input = tensor(['B', 0, 'T'], dtype),
        before = structuredClone(input);
      const out = rule(type).infer!({ in: input }, {}).outputs.out;
      expect(out).toEqual({ ...input, dtype: dtype.includes('float') ? dtype : 'float32' });
      expect(out.shape).not.toBe(input.shape);
      expect(out.axes).not.toBe(input.axes);
      expect(input).toEqual(before);
    }
    expect(rule(type).infer!({ in: tensor([], 'int64') }, {}).outputs.out).toEqual(
      tensor([], 'float32'),
    );
    expect(() => rule(type).infer!({ in: tensor([2], 'complex64') }, {})).toThrow('dtype');
  });
});

describe('IntegerFloorDivide', () => {
  const divide = rule('IntegerFloorDivide').inferValue!;
  it.each([
    [256, 4, 64], [7, 3, 2], [-7, 3, -3], [7, -3, -3], [-7, -3, 2],
    [0, -3, 0], [-6, 3, -2], [Number.MAX_SAFE_INTEGER, 3, 3002399751580330],
  ])('computes %s // %s = %s using floor division', (a, b, expected) => {
    expect(divide({ a: integer(a), b: integer(b) }, {}).outputs.out).toEqual(integer(expected));
  });
  it('keeps symbolic dimensions without inventing a concrete count', () => {
    const result = divide({ a: integer('GC'), b: integer('G') }, {});
    expect(result.outputs.out).toEqual(integer('(GC) // (G)'));
    expect(result.warnings).toHaveLength(1);
  });
  it('rejects a zero divisor and non-integer inputs', () => {
    expect(() => divide({ a: integer(256), b: integer(0) }, {})).toThrow('不能为 0');
    expect(() => divide({ a: tensor([]), b: integer(4) }, {})).toThrow('局部整数');
    expect(() => divide({ a: integer(1.5), b: integer(4) }, {})).toThrow('安全整数');
    expect(() => divide({ a: integer(256) }, {})).toThrow('b');
  });
});

describe('局部整数乘法', () => {
  it('计算 bs × A 并保留负整数、符号与零乘积', () => {
    expect(multiply({ a: integer(2), b: integer(49) }, {}).outputs.out).toEqual(integer(98));
    expect(multiply({ a: integer(-2), b: integer(3) }, {}).outputs.out).toEqual(integer(-6));
    expect(multiply({ a: integer('B'), b: integer('A') }, {}).outputs.out).toEqual(
      integer('(B)×(A)'),
    );
    expect(multiply({ a: integer(1), b: integer('B') }, {}).outputs.out).toEqual(integer('B'));
    expect(multiply({ a: integer('B'), b: integer(0) }, {}).outputs.out).toEqual(integer(0));
  });

  it.each([
    tensor([]),
    { kind: 'dict', fields: {} },
    integer(1.5),
    integer(NaN),
    integer(Number.MAX_SAFE_INTEGER + 1),
  ])('拒绝非整数输入 %#', (value) => {
    expect(() => multiply({ a: value as ValueSpec, b: integer(2) }, {})).toThrow();
  });

  it('拒绝缺少输入与具体乘积溢出', () => {
    expect(() => multiply({ a: integer(2) }, {})).toThrow('b');
    expect(() => multiply({ a: integer(Number.MAX_SAFE_INTEGER), b: integer(2) }, {})).toThrow(
      '安全整数',
    );
  });
});

describe('动态 View', () => {
  it('读取独立整数端口；重复占位符复用同一端口，按首次出现顺序排列', () => {
    expect(viewPorts({ shape: 'd1, d0, d1' }).inputs.map((port) => port.id)).toEqual([
      'in',
      'd1',
      'd0',
    ]);
    expect(viewPorts({ shape: 'd0, ' }).inputs.map((port) => port.id)).toEqual(['in', 'd0']);
    expect(viewLabel({ shape: 'd0,d1,-1' })).toBe('view(d0, d1, -1)');
    expect(view({ in: tensor([4]), d0: integer(2) }, { shape: 'd0, d0' }).outputs.out).toEqual(
      tensor([2, 2]),
    );
  });

  it('实时使用 bs × A 与新的 T，保存重开后仍随上游变化', () => {
    const p = project(),
      before = structuredClone(p);
    const result = inferProject(p, registry);
    expect(result.issues).toEqual([]);
    expect(result.nodes.product.outputs.out).toEqual(integer(98));
    expect(result.nodes.view.status, result.nodes.view.messages.join()).toBe('ok');
    expect(tensorValue(result.nodes.out.outputs.out).shape).toEqual([98, 20, 9]);
    expect(result.nodes.view.notes.join()).toContain('stride');
    expect(p).toEqual(before);
    const reopened = parseProject(serializeProject(p), registry);
    expect(reopened).toEqual(p);
    reopened.graph.nodes.find((node) => node.id === 'feat')!.params.shape = '2,49,7,9';
    expect(tensorValue(inferProject(reopened, registry).nodes.out.outputs.out).shape).toEqual([
      98, 7, 9,
    ]);
  });

  it('支持安全常量和动态 -1，保留 dtype；标量输入可以变为长度 1', () => {
    expect(view({ in: tensor([2, 3], 'float16') }, { shape: '+3, 2' }).outputs.out).toEqual(
      tensor([3, 2], 'float16'),
    );
    expect(view({ in: tensor([2, 3]), d0: integer(-1) }, { shape: 'd0, 3' }).outputs.out).toEqual(
      tensor([2, 3]),
    );
    expect(view({ in: tensor([], 'bool') }, { shape: '1' }).outputs.out).toEqual(
      tensor([1], 'bool'),
    );
  });

  it('区分明确的零尺寸与零尺寸加 -1 的歧义', () => {
    expect(view({ in: tensor([0, 3]) }, { shape: '0, 3' }).outputs.out).toEqual(tensor([0, 3]));
    expect(view({ in: tensor([0, 3]) }, { shape: '-1, 3' }).outputs.out).toEqual(tensor([0, 3]));
    expect(() => view({ in: tensor([0, 3]) }, { shape: '0, -1' })).toThrow('歧义');
    expect(() => view({ in: tensor([0, 3]), d0: integer(0) }, { shape: 'd0, -1' })).toThrow('歧义');
    expect(() => view({ in: tensor([2, 3]), d0: integer('B') }, { shape: '0, d0' })).toThrow(
      '元素总数',
    );
  });

  it('符号乘积与推导维度保留表达式并提示整除约束', () => {
    const size = multiply({ a: integer('B'), b: integer('A') }, {}).outputs.out;
    const result = view(
      { in: tensor(['B', 'A', 'T', 9]), d0: size, d1: integer('T') },
      { shape: 'd0, d1, -1' },
    );
    const out = tensorValue(result.outputs.out);
    expect(out.shape.slice(0, 2)).toEqual(['(B)×(A)', 'T']);
    expect(String(out.shape[2])).toContain('/');
    expect(result.warnings?.join()).toContain('整除');
    expect(result.warnings?.join()).toContain('非负整数');
    expect(out.dtype).toBe('float32');
  });

  it('零元素总数不能掩盖符号目标尺寸可能为负的约束', () => {
    const negative = multiply({ a: integer(-1), b: integer('B') }, {}).outputs.out;
    const result = view({ in: tensor([0]), d0: negative }, { shape: '0, d0' });
    expect(tensorValue(result.outputs.out).shape).toEqual([0, '-1×(B)']);
    expect(result.warnings?.join()).toContain('非负整数');
  });

  it.each([
    '',
    '[]',
    'd8, 3',
    'bs, A, T',
    'd0*2, -1',
    'd0,,2',
    '-2, 3',
    '1.5, 4',
    '-1,-1',
    '9007199254740992',
    '1,1,1,1,1,1,1,1,1',
  ])('拒绝未知表达式或不支持的目标形状 %s', (shape) => {
    expect(() => view({ in: tensor([2, 3]), d0: integer(2) }, { shape })).toThrow();
  });

  it('拒绝缺少端口、Tensor 尺寸参数及多个动态 -1', () => {
    expect(() => view({ in: tensor([2, 3]) }, { shape: 'd0, 3' })).toThrow('d0');
    expect(() => view({ in: tensor([2, 3]), d0: tensor([], 'int64') }, { shape: 'd0, 3' })).toThrow(
      '局部整数',
    );
    expect(() => view({ in: tensor([2, 3]), d0: integer(-2) }, { shape: 'd0, 3' })).toThrow('非负');
    expect(() =>
      view({ in: tensor([2, 3]), d0: integer(-1), d1: integer(-1) }, { shape: 'd0, d1' }),
    ).toThrow('一个 -1');
    const p = project();
    p.graph.edges = p.graph.edges.filter((edge) => edge.id !== 'time');
    const result = inferProject(p, registry).nodes.view;
    expect(result.status).toBe('blocked');
    expect(result.outputs).toEqual({});
  });

  it('拒绝元素总数不匹配、不整除以及目标尺寸乘积溢出', () => {
    expect(() => view({ in: tensor([2, 3]) }, { shape: '2, 2' })).toThrow('元素总数');
    expect(() => view({ in: tensor([2, 3]) }, { shape: '4, -1' })).toThrow('整除');
    expect(() => view({ in: tensor([2, 3]) }, { shape: '9007199254740991, 2' })).toThrow(
      '安全整数',
    );
  });
});
