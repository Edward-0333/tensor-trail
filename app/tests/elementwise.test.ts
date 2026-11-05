import { describe, expect, it } from 'vitest';
import type { Dim, TensorSpec } from '../src/core/types';
import { elementwiseModules } from '../src/modules/elementwise';
import { indexingModules } from '../src/modules/indexing';
import { tensorIndexDefaults } from '../src/core/indexingTypes';

const spec = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  dtype,
  axes: shape.map((_, index) => `axis${index}`),
});
const infer = (type: string, inputs: Record<string, TensorSpec>) =>
  elementwiseModules.find((module) => module.type === type)!.infer!(inputs, {});
const out = (type: string, inputs: Record<string, TensorSpec>) => infer(type, inputs).outputs.out;

describe('逐元素运算与掩码维度对齐', () => {
  it('相邻有效掩码逐元素与，输出保留 bool 及时间轴', () => {
    const mask = spec([2, 49, 20], 'bool');
    expect(out('BitwiseAnd', { a: mask, b: mask })).toEqual(mask);
  });

  it.each([0, 1, 2, 21])('T=%i 时相邻切片与运算正确保留空时间轴', (time) => {
    const mask = spec([2, 49, time], 'bool');
    const slice = indexingModules.find((module) => module.type === 'TensorIndex')!.infer!;
    const common = { ...tensorIndexDefaults, mode: 'slice', dim: '2' };
    const current = slice({ in: mask }, { ...common, start: '1', stop: '' }).outputs.out;
    const previous = slice({ in: mask }, { ...common, start: '', stop: '-1' }).outputs.out;
    expect(out('BitwiseAnd', { a: current, b: previous }).shape).toEqual([
      2,
      49,
      Math.max(0, time - 1),
    ]);
  });

  it('bool 掩码可广播；整数 & 按 dtype 提升，浮点 & 拒绝', () => {
    expect(
      out('BitwiseAnd', { a: spec([2, 1, 20], 'bool'), b: spec([1, 49, 20], 'bool') }),
    ).toMatchObject({ shape: [2, 49, 20], dtype: 'bool' });
    expect(out('BitwiseAnd', { a: spec([2], 'uint8'), b: spec([2], 'int8') }).dtype).toBe('int16');
    expect(out('BitwiseAnd', { a: spec([2], 'bool'), b: spec([2], 'int64') }).dtype).toBe('int64');
    expect(() => out('BitwiseAnd', { a: spec([2]), b: spec([2], 'bool') })).toThrow('浮点');
  });

  it('heading 差分经有效掩码选择，保持浮点轨迹形状', () => {
    const heading = spec([2, 49, 20]);
    const delta = out('Subtract', { a: heading, b: heading });
    const zero = out('ZerosLike', { in: delta });
    expect(out('Where', { condition: spec([2, 49, 20], 'bool'), x: delta, y: zero })).toEqual(
      heading,
    );
  });

  it('向量特征的掩码仅补末尾单例轴，再由 Where 广播', () => {
    const mask = spec([2, 49, 20], 'bool');
    const feature = spec([2, 49, 20, 2]);
    const expanded = out('AlignTrailingDims', { in: mask, like: feature });
    expect(expanded).toEqual({
      shape: [2, 49, 20, 1],
      axes: [...mask.axes, 'singleton'],
      dtype: 'bool',
    });
    expect(
      out('Where', { condition: expanded, x: feature, y: out('ZerosLike', { in: feature }) }),
    ).toEqual(feature);
  });

  it('补齐可一次添加多个轴，同阶不改变输入，且不提前承诺广播成功', () => {
    const mask = spec([2], 'bool');
    expect(out('AlignTrailingDims', { in: mask, like: spec([2, 3, 4]) })).toEqual({
      shape: [2, 1, 1],
      axes: ['axis0', 'singleton', 'singleton'],
      dtype: 'bool',
    });
    expect(out('AlignTrailingDims', { in: spec([3], 'bool'), like: spec([5]) }).shape).toEqual([3]);
    expect(() => out('AlignTrailingDims', { in: spec([2, 3], 'bool'), like: spec([2]) })).toThrow(
      '维数高于',
    );
  });

  it.each(['BitwiseAnd', 'Subtract'])('%s 拒绝不能广播的对应尺寸', (type) => {
    const dtype = type === 'BitwiseAnd' ? 'bool' : 'float32';
    expect(() => out(type, { a: spec([2, 49, 20], dtype), b: spec([2, 48, 20], dtype) })).toThrow(
      '不匹配',
    );
  });

  it('Where 同时检查条件与两路候选形状，非 bool 条件拒绝', () => {
    expect(() =>
      out('Where', { condition: spec([2, 3], 'bool'), x: spec([2, 4]), y: spec([2, 4]) }),
    ).toThrow('不匹配');
    expect(() =>
      out('Where', { condition: spec([2, 3], 'bool'), x: spec([2, 3]), y: spec([2, 4]) }),
    ).toThrow('不匹配');
    for (const dtype of ['float32', 'int64', 'uint8'])
      expect(() =>
        out('Where', { condition: spec([2], dtype), x: spec([2]), y: spec([2]) }),
      ).toThrow('condition 必须是 bool');
  });

  it('零长度轴与单例轴可广播为零长度，但不能与其它正尺寸广播', () => {
    expect(out('Subtract', { a: spec([2, 0]), b: spec([2, 1]) }).shape).toEqual([2, 0]);
    expect(
      out('Where', { condition: spec([2, 0], 'bool'), x: spec([]), y: spec([1]) }).shape,
    ).toEqual([2, 0]);
    expect(() => out('Subtract', { a: spec([2, 0]), b: spec([2, 2]) })).toThrow('不匹配');
  });

  it.each([
    ['uint8', 'int8', 'int16'],
    ['int8', 'int16', 'int16'],
    ['int32', 'int64', 'int64'],
    ['int64', 'float16', 'float16'],
    ['float16', 'bfloat16', 'float32'],
    ['bfloat16', 'float32', 'float32'],
    ['float32', 'float64', 'float64'],
  ])('数值 Tensor 类型提升 %s 与 %s → %s', (a, b, expected) => {
    const inputs = { a: spec([2], a), b: spec([2], b) };
    expect(out('Subtract', inputs).dtype).toBe(expected);
    expect(out('Where', { condition: spec([2], 'bool'), x: inputs.a, y: inputs.b }).dtype).toBe(
      expected,
    );
  });

  it.each([
    ['float64', 'float32', 'float32'],
    ['int64', 'int8', 'int8'],
    ['int8', 'uint8', 'uint8'],
    ['float16', 'bfloat16', 'bfloat16'],
    ['float64', 'int64', 'float64'],
  ])('零维 %s 与非零维 %s 遵循 Tensor 标量提升规则 → %s', (scalar, dimensioned, expected) => {
    const a = spec([], scalar),
      b = spec([2], dimensioned);
    expect(out('Subtract', { a, b }).dtype).toBe(expected);
    expect(out('Subtract', { a: b, b: a }).dtype).toBe(expected);
    expect(out('Where', { condition: spec([2], 'bool'), x: a, y: b }).dtype).toBe(expected);
  });

  it('Where 两路均为零维时先提升数值类型，条件维数不影响 dtype', () => {
    const value = out('Where', {
      condition: spec([3, 4], 'bool'),
      x: spec([], 'float64'),
      y: spec([], 'float32'),
    });
    expect(value).toMatchObject({ shape: [3, 4], dtype: 'float64' });
    expect(
      out('Where', { condition: spec([], 'bool'), x: spec([], 'bool'), y: spec([], 'int8') }).dtype,
    ).toBe('int8');
  });

  it('减法任一路为 bool 均拒绝，即使另一分支可以提升成数值', () => {
    for (const dtype of ['bool', 'int64', 'float32']) {
      expect(() => out('Subtract', { a: spec([2], 'bool'), b: spec([2], dtype) })).toThrow(
        '不支持 bool',
      );
      expect(() => out('Subtract', { a: spec([2], dtype), b: spec([2], 'bool') })).toThrow(
        '不支持 bool',
      );
    }
  });

  it.each([
    'bool',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'bfloat16',
    'float32',
    'float64',
  ])('ZerosLike 保持 %s 的形状/类型并且不修改输入', (dtype) => {
    for (const shape of [[], [2, 0, 4], ['B', 'T']] as Dim[][]) {
      const source = spec(shape, dtype),
        before = structuredClone(source);
      const result = out('ZerosLike', { in: source });
      expect(result).toEqual(source);
      expect(result.shape).not.toBe(source.shape);
      expect(result.axes).not.toBe(source.axes);
      expect(source).toEqual(before);
    }
  });

  it('符号维不一致保留待验证提示，不假定具体尺寸一致', () => {
    const result = infer('Subtract', { a: spec(['B', 'T']), b: spec(['B', 'S']) });
    expect(result.warnings?.join(' ')).toContain('符号维度尚待验证');
  });

  it.each(['complex64', 'qint8', 'float128'])('拒绝未支持的 dtype=%s', (dtype) => {
    expect(() => out('ZerosLike', { in: spec([2], dtype) })).toThrow('暂不支持 dtype');
    expect(() => out('Subtract', { a: spec([2], dtype), b: spec([2]) })).toThrow('暂不支持 dtype');
  });
});
