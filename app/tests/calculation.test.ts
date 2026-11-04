import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import {
  calculate,
  parseCalculations,
  sliceLabel,
  type Calculation,
} from '../src/modules/calculation';
import { buildModel, describeModules } from '../src/agent';
import type { TensorSpec } from '../src/core/types';

const spec = (shape: (number | string)[], dtype = 'float32'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const slice = (values: Partial<Extract<Calculation, { op: 'slice' }>> = {}): Calculation => ({
  op: 'slice',
  dim: -1,
  start: '',
  stop: '2',
  step: 1,
  ...values,
});

describe('切片与计算操作', () => {
  it('PLUTO 历史掩码切片跟随 history_steps，保留 bool 类型与表达式显示', () => {
    const rule = registry.get('Slice')!.infer!;
    const params = { dim: 2, start: '', stop: 'self.history_steps', step: 1 };
    for (const steps of [21, 10, 0, 120]) {
      const result = rule({ in: spec([2, 49, 101], 'bool') }, params, {
        hyperparameters: { history_steps: { type: 'int', value: steps } },
      });
      expect(result.outputs.out).toEqual(spec([2, 49, Math.min(101, steps)], 'bool'));
    }
    expect(sliceLabel(params)).toBe('[轴2: :self.history_steps]');
    expect(
      rule(
        { in: spec([2, 49, 101], 'bool') },
        { ...params, start: 'history_steps - 3', stop: 'history_steps' },
        { hyperparameters: { history_steps: { type: 'int', value: 21 } } },
      ).outputs.out.shape,
    ).toEqual([2, 49, 3]);
  });
  it.each(['missing', '1.5', 'history_steps / 2', 'history_steps()', '1;alert(1)'])(
    '拒绝切片的无效或非整数边界 %s',
    (stop) => {
      expect(() =>
        registry.get('Slice')!.infer!(
          { in: spec([2, 49, 101]) },
          { dim: 2, start: '', stop, step: 1 },
          { hyperparameters: { history_steps: { type: 'int', value: 21 } } },
        ),
      ).toThrow();
    },
  );
  it('默认 [..., :2] 只裁剪最后一维，并保留轴名称', () => {
    const input = spec([8, 100, 64]);
    expect(calculate(input, slice())).toEqual({ ...input, shape: [8, 100, 2] });
    expect(input.shape).toEqual([8, 100, 64]);
  });
  it.each([
    [{ start: '-4', stop: '-1', step: 2 }, 2],
    [{ start: '-100', stop: '100' }, 10],
    [{ start: '7', stop: '2' }, 0],
    [{ start: '1', stop: '', step: 3 }, 3],
    [{ start: '', stop: '2' }, 2],
  ])('按照边界及步长推导切片长度 %j', (options, length) => {
    expect(calculate(spec([2, 10]), slice(options)).shape).toEqual([2, length]);
  });
  it('支持指定中间轴和空的切片结果', () => {
    expect(calculate(spec([2, 10, 5]), slice({ dim: 1, stop: '3' })).shape).toEqual([2, 3, 5]);
    expect(calculate(spec([2, 0]), slice()).shape).toEqual([2, 0]);
  });
  it('any 全轴归约输出 bool 标量，keepdim 保留单例轴', () => {
    expect(calculate(spec([2, 3]), { op: 'any', dim: null, keepdim: false })).toEqual(
      spec([], 'bool'),
    );
    expect(calculate(spec([2, 3]), { op: 'any', dim: null, keepdim: true }).shape).toEqual([1, 1]);
  });
  it('any 指定轴同时删除相应的轴名称', () => {
    expect(calculate(spec([2, 3, 4]), { op: 'any', dim: 1, keepdim: false })).toEqual({
      shape: [2, 4],
      axes: ['axis0', 'axis2'],
      dtype: 'bool',
    });
    expect(calculate(spec([2, 3, 4]), { op: 'any', dim: -1, keepdim: true }).shape).toEqual([
      2, 3, 1,
    ]);
  });
  it('取反接受 bool/int64，拒绝浮点输入', () => {
    for (const dtype of ['bool', 'int64'])
      expect(calculate(spec([2, 3], dtype), { op: 'invert' })).toEqual(spec([2, 3], dtype));
    expect(() => calculate(spec([2, 3]), { op: 'invert' })).toThrow('浮点');
  });
  it.each([
    [{ op: 'eval', code: 'run()' }],
    [{ op: 'slice', dim: -1, start: '', stop: '2', step: 0 }],
    [{ op: 'slice', dim: -1, start: 'x', stop: '2', step: 1 }],
    [{ op: 'any', dim: null, keepdim: 'yes' }],
    [{ op: 'invert', extra: true }],
    [],
    Array.from({ length: 17 }, () => ({ op: 'invert' })),
  ])('拒绝非法或超量运算 %j', (...ops) => {
    // Vitest spreads each array row into arguments.
    expect(() => parseCalculations(JSON.stringify(ops))).toThrow();
  });
  it('组合计算按顺序返回形状与类型，并定位失败步骤', () => {
    const node = registry.create('Compute', {
      operations: JSON.stringify([{ op: 'invert' }, { op: 'any', dim: -1, keepdim: false }]),
    });
    const rule = registry.get('Compute')!.infer!;
    expect(rule({ in: spec([2, 3], 'bool') }, node.params).outputs.out).toEqual({
      shape: [2],
      axes: ['axis0'],
      dtype: 'bool',
    });
    expect(() => rule({ in: spec([2, 3]) }, node.params)).toThrow('第 1 步');
  });
  it('符号维度产生待验证警告，轴越界会报错', () => {
    const node = registry.create('Slice');
    expect(
      registry.get('Slice')!.infer!({ in: spec(['B', 'T']) }, node.params).warnings?.length,
    ).toBeGreaterThan(0);
    expect(() => calculate(spec([2, 3]), slice({ dim: 4 }))).toThrow('范围');
  });
  it('新模块及 bool 输入可通过 AI 配方创建和校验', () => {
    const result = buildModel({
      name: '计算',
      nodes: [
        {
          id: 'x',
          type: 'Input',
          params: { dtype: 'bool', shape: '2, 3', axes: 'batch, feature' },
        },
        {
          id: 'c',
          type: 'Compute',
          params: {
            operations: JSON.stringify([{ op: 'invert' }, { op: 'any', dim: -1, keepdim: false }]),
          },
        },
        { id: 'y', type: 'Output' },
      ],
      connections: [
        ['x.out', 'c.in'],
        ['c.out', 'y.in'],
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.outputs.y).toEqual({ shape: [2], axes: ['batch'], dtype: 'bool' });
    expect(describeModules().modules.some((m) => m.type === 'Slice')).toBe(true);
  });
});
