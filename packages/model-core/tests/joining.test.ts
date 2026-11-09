import { describe, expect, it } from 'vitest';
import type { Dim, Params, Project, TensorSpec } from '../src/core/types';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { ports } from '../src/core/graph';
import { joiningModules, joiningPorts } from '../src/modules/joining';
import { primitives } from '../src/modules/primitives';
import { ModuleRegistry } from '../src/modules/registry';

const registry = new ModuleRegistry();
[...primitives, ...joiningModules].forEach((definition) => registry.register(definition));
const spec = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  dtype,
  axes: shape.map((_, index) => `axis${index}`),
});
const infer = (type: string, values: TensorSpec[], params: Params = { dim: -1 }) =>
  registry.get(type)!.infer!(
    Object.fromEntries(values.map((value, index) => [String.fromCharCode(97 + index), value])),
    params,
  );

describe('多路 Cat 与 Stack', () => {
  it('保留 Concat 类型与 a/b 端口，新模块默认两路输入', () => {
    expect(primitives.some((module) => module.type === 'Concat')).toBe(false);
    for (const type of ['Concat', 'Stack']) {
      const node = registry.create(type);
      expect(node.params).toMatchObject({ dim: -1, input_count: 2 });
      expect(ports(node, registry).inputs.map((port) => port.id)).toEqual(['a', 'b']);
      expect(ports(node, registry).outputs.map((port) => port.id)).toEqual(['out']);
    }
    expect(joiningPorts({ input_count: 5 }).inputs.map((port) => port.id)).toEqual([
      'a',
      'b',
      'c',
      'd',
      'e',
    ]);
    expect(joiningPorts({ input_count: 16 }).inputs.at(-1)!.id).toBe('p');
  });

  it('PLUTO 五路特征 2+2+2+2+1 沿末轴拼接成9个通道', () => {
    const features = [2, 2, 2, 2, 1].map((size) => spec([2, 49, 20, size]));
    const result = infer('Concat', features, { dim: -1, input_count: 5 });
    expect(result.outputs.out).toMatchObject({ shape: [2, 49, 20, 9], dtype: 'float32' });
    expect(result.warnings).toEqual([]);
  });

  it('cos/sin 两路同形张量 Stack 到新末轴，得到方向向量', () => {
    const heading = spec([2, 49, 20]);
    const result = infer('Stack', [heading, heading], { dim: -1, input_count: 2 });
    expect(result.outputs.out).toEqual({
      shape: [2, 49, 20, 2],
      axes: [...heading.axes, 'stack'],
      dtype: 'float32',
    });
  });

  it.each([0, 1, 2, 3])('Stack dim=%i 在对应位置插入新轴', (axis) => {
    const source = spec([2, 3, 4]);
    const expected = [...source.shape];
    expected.splice(axis, 0, 2);
    const positive = infer('Stack', [source, source], { dim: axis }).outputs.out;
    const negative = infer('Stack', [source, source], { dim: axis - 4 }).outputs.out;
    expect(positive.shape).toEqual(expected);
    expect(negative).toEqual(positive);
  });

  it('Cat 的正负轴等价，只累加已有操作轴', () => {
    const a = spec([2, 3]),
      b = spec([4, 3]);
    expect(infer('Concat', [a, b], { dim: 0 }).outputs.out.shape).toEqual([6, 3]);
    expect(infer('Concat', [a, b], { dim: -2 }).outputs).toEqual(
      infer('Concat', [a, b], { dim: 0 }).outputs,
    );
  });

  it('高阶 Stack 的负轴范围按输入阶数判断，不固定限制为 -8', () => {
    const source = spec(Array(8).fill(2));
    const input = registry.create(
      'Input',
      { shape: source.shape.join(', '), axes: source.axes.join(', ') },
      'input',
    );
    const stack = registry.create('Stack', { dim: -9 }, 'stack');
    const result = inferProject(
      {
        schemaVersion: 1,
        name: '高阶 Stack',
        variants: [],
        graph: {
          nodes: [input, stack],
          edges: ['a', 'b'].map((port) => ({
            id: port,
            source: 'input',
            sourcePort: 'out',
            target: 'stack',
            targetPort: port,
          })),
        },
      },
      registry,
    );
    expect(result.nodes.stack.status).toBe('ok');
    expect(result.nodes.stack.outputs.out).toMatchObject({ shape: Array(9).fill(2) });
  });

  it('输入数量改为1时 Cat 为独立拷贝，Stack 增加单例轴', () => {
    const source = spec([2, 3]);
    const cat = infer('Concat', [source], { dim: -1, input_count: 1 }).outputs.out;
    const stack = infer('Stack', [source], { dim: -1, input_count: 1 }).outputs.out;
    expect(cat).toEqual(source);
    expect(cat.shape).not.toBe(source.shape);
    expect(stack.shape).toEqual([2, 3, 1]);
  });

  it.each([0, 17, -2, 2.5, '3', Number.NaN, Number.POSITIVE_INFINITY])(
    '输入数量 %s 渲染受界限保护，但推导严格报错',
    (count) => {
      const params = { dim: -1, input_count: count };
      expect(joiningPorts(params).inputs.length).toBeGreaterThanOrEqual(1);
      expect(joiningPorts(params).inputs.length).toBeLessThanOrEqual(16);
      expect(() => infer('Concat', [spec([2]), spec([2])], params)).toThrow('1–16');
      expect(() => infer('Stack', [spec([2]), spec([2])], params)).toThrow('1–16');
    },
  );

  it('新增输入端口后不能沿用旧两路数据悄悄推导', () => {
    expect(() => infer('Concat', [spec([2]), spec([2])], { dim: -1, input_count: 3 })).toThrow(
      '输入 c',
    );
    expect(() => infer('Stack', [spec([2]), spec([2])], { dim: -1, input_count: 3 })).toThrow(
      '输入 c',
    );
  });

  it.each([-4, 3, 0.5, '0', Number.NaN])('对二维输入拒绝无效 dim=%s', (dim) => {
    expect(() => infer('Concat', [spec([2, 3]), spec([2, 3])], { dim })).toThrow();
    expect(() => infer('Stack', [spec([2, 3]), spec([2, 3])], { dim })).toThrow();
  });

  it('不使用广播：Cat 非拼接轴和 Stack 每个轴都必须相同', () => {
    expect(() => infer('Concat', [spec([2, 1, 3]), spec([2, 4, 3])])).toThrow('不进行广播');
    expect(() => infer('Stack', [spec([2, 1, 3]), spec([2, 4, 3])])).toThrow('不进行广播');
    expect(() => infer('Concat', [spec([2, 3]), spec([3])])).toThrow('相同维数');
    expect(() => infer('Stack', [spec([2, 3]), spec([3])])).toThrow('形状相同');
  });

  it('Stack 可堆叠零维Tensor，Cat 明确拒绝零维Tensor', () => {
    expect(infer('Stack', [spec([]), spec([])]).outputs.out).toEqual({
      shape: [2],
      axes: ['stack'],
      dtype: 'float32',
    });
    expect(() => infer('Concat', [spec([]), spec([])])).toThrow('零维');
    expect(() => infer('Concat', [spec([0]), spec([])])).toThrow('零维');
  });

  it('Cat 可忽略1维[0]空Tensor，但空Tensor的dtype参与提升', () => {
    const result = infer('Concat', [spec([0], 'float64'), spec([2, 3]), spec([0], 'int64')], {
      dim: -1,
      input_count: 3,
    });
    expect(result.outputs.out).toMatchObject({ shape: [2, 3], dtype: 'float64' });
    expect(infer('Concat', [spec([2, 0]), spec([0])], { dim: 1 }).outputs.out.shape).toEqual([
      2, 0,
    ]);
    expect(infer('Concat', [spec([0]), spec([0])], { dim: 3 }).outputs.out.shape).toEqual([0]);
  });

  it('多维零长度输入遵循正常轴规则，Stack 不跳过1维空Tensor', () => {
    expect(infer('Concat', [spec([0, 3]), spec([2, 3])], { dim: 0 }).outputs.out.shape).toEqual([
      2, 3,
    ]);
    expect(() => infer('Concat', [spec([0, 3]), spec([2, 3])], { dim: 1 })).toThrow('不进行广播');
    expect(infer('Stack', [spec([2, 0, 3]), spec([2, 0, 3])]).outputs.out.shape).toEqual([
      2, 0, 3, 2,
    ]);
    expect(() => infer('Stack', [spec([0]), spec([2, 3])])).toThrow('形状相同');
  });

  it.each([
    ['float16', 'bfloat16', 'float32'],
    ['uint8', 'int8', 'int16'],
    ['int64', 'float16', 'float16'],
    ['bool', 'int32', 'int32'],
    ['float32', 'float64', 'float64'],
  ])('Cat/Stack 支持混合dtype：%s 与 %s → %s', (a, b, expected) => {
    for (const type of ['Concat', 'Stack'])
      expect(infer(type, [spec([2], a), spec([2], b)]).outputs.out.dtype).toBe(expected);
  });

  it('拒绝未知dtype，不将其误当作整数或默认浮点', () => {
    for (const type of ['Concat', 'Stack'])
      expect(() => infer(type, [spec([2], 'complex64'), spec([2])])).toThrow('暂不支持 dtype');
  });

  it('符号拼接轴保留求和表达式，非操作轴差异保留待验证提示', () => {
    expect(infer('Concat', [spec(['B', 'T']), spec(['B', 'S'])]).outputs.out.shape).toEqual([
      'B',
      '(T+S)',
    ]);
    expect(infer('Concat', [spec(['B', 2]), spec(['N', 3])]).warnings?.join(' ')).toContain(
      '尚待验证',
    );
    expect(infer('Stack', [spec(['B', 'T']), spec(['B', 'S'])]).warnings?.join(' ')).toContain(
      '尚待验证',
    );
    expect(() => infer('Concat', [spec([Number.MAX_SAFE_INTEGER]), spec([1])])).toThrow('安全整数');
  });

  it.each(['Concat', 'Stack'])('%s 首路为符号尺寸时仍拒绝后续已知尺寸冲突', (type) => {
    for (const sizes of [
      ['B', 2, 3],
      [2, 'B', 3],
      ['B', 'N', 2, 3],
      ['B', 0, 2],
    ] as Dim[][]) {
      const values = sizes.map((size) => spec([size, 1]));
      expect(() => infer(type, values, { dim: 1, input_count: values.length })).toThrow('不匹配');
    }
  });

  it.each(['Concat', 'Stack'])('%s 多路具体尺寸一致时仅对未决符号给出提示', (type) => {
    const result = infer(type, [spec(['B', 1]), spec([2, 1]), spec([2, 1])], {
      dim: 1,
      input_count: 3,
    });
    expect(result.outputs.out.shape).toEqual(type === 'Concat' ? ['B', 3] : ['B', 3, 1]);
    expect(result.warnings?.join(' ')).toContain('尚待验证');
  });

  it('Cat 操作轴上的不同具体长度仍可与符号长度累加', () => {
    const result = infer('Concat', [spec([2, 'T']), spec([2, 3]), spec([2, 4])], {
      dim: 1,
      input_count: 3,
    });
    expect(result.outputs.out.shape).toEqual([2, '(T+3+4)']);
    expect(result.warnings).toEqual([]);
  });

  it('旧params只有dim的Concat序列化与推导正常，不强制新增字段', () => {
    const a = registry.create('Input', { shape: '2, 3', axes: 'batch, feature' }, 'a');
    const b = registry.create('Input', { shape: '2, 4', axes: 'batch, feature' }, 'b');
    const concat = registry.create('Concat', {}, 'cat');
    concat.params = { dim: -1 };
    const old: Project = {
      schemaVersion: 1,
      name: '旧 Cat',
      variants: [],
      graph: {
        nodes: [a, b, concat],
        edges: [
          { id: 'a_cat', source: 'a', sourcePort: 'out', target: 'cat', targetPort: 'a' },
          { id: 'b_cat', source: 'b', sourcePort: 'out', target: 'cat', targetPort: 'b' },
        ],
      },
    };
    const serialized = serializeProject(old);
    const restored = parseProject(serialized, registry);
    expect(serializeProject(restored)).toBe(serialized);
    expect(registry.get('Concat')!.getFields!(concat.params).map((field) => field.key)).toEqual([
      'dim',
    ]);
    const result = inferProject(restored, registry);
    expect(result.issues).toEqual([]);
    expect(result.nodes.cat.status).toBe('ok');
    expect(result.nodes.cat.outputs.out).toMatchObject({ shape: [2, 7], dtype: 'float32' });
  });
});
