import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import type { Dim, Params, TensorSpec, ValueSpec } from '../src/core/types';
import { tensorValue } from '../src/core/types';
import { ports } from '../src/core/graph';

const t = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
  shape,
  dtype,
  axes: shape.map((_, i) => `axis${i}`),
});
const size = (value: Dim): ValueSpec => ({ kind: 'int', value });
const zeros = (params: Params = {}, values: Record<string, ValueSpec> = {}) => {
  const n = registry.create('Zeros', params);
  return registry.get(n.type)!.inferValue!(
    { d0: size(98), d1: size(128), device_like: t([2, 49, 21, 2]), ...values },
    n.params,
  );
};
const write = (values: Record<string, TensorSpec> = {}, params: Params = {}) => {
  const n = registry.create('BooleanAssign', params);
  return registry.get(n.type)!.infer!(
    { in: t([98, 128]), mask: t([98], 'bool'), values: t(['N_valid', 128]), ...values },
    n.params,
  );
};

describe('零张量创建', () => {
  it('仅使用设备参考，不复制其 dtype、尺寸或轴', () => {
    expect(zeros({}, { device_like: t([9, 8, 7], 'float64') }).outputs.out).toEqual(t([98, 128]));
    expect(zeros({ dtype: 'float64' }).outputs.out).toEqual(t([98, 128], 'float64'));
  });
  it('支持常量、符号和零尺寸，尺寸接口动态去重', () => {
    expect(zeros({ shape: '0, d0, d0, 7' }, { d0: size('A') }).outputs.out).toEqual(
      t([0, 'A', 'A', 7]),
    );
    expect(
      ports(registry.create('Zeros', { shape: '0, d3, d3' }), registry).inputs.map((p) => p.id),
    ).toEqual(['d3', 'device_like']);
  });
  it.each(['-1,128', '', 'd9,128', 'd0,,d1', '3.2,128', '9007199254740992'])(
    '拒绝错误目标形状 %s',
    (shape) => {
      expect(() => zeros({ shape })).toThrow();
    },
  );
  it('拒绝非法整数、Tensor 尺寸和缺失设备参考', () => {
    for (const d0 of [size(-1), size(1.2), t([1])]) expect(() => zeros({}, { d0 })).toThrow();
    expect(() => zeros({}, { device_like: size(1) })).toThrow();
  });
});

describe('布尔索引逐行赋值', () => {
  it('保持目标完整形状、轴和 dtype，而不是输出有效行形状', () => {
    const target = { ...t([98, 128]), axes: ['all_agents', 'features'] },
      before = structuredClone(target);
    const r = write({ in: target });
    expect(r.outputs.out).toEqual(target);
    expect(r.outputs.out).not.toBe(target);
    expect(target).toEqual(before);
    expect(r.warnings).toEqual([]);
    expect(r.notes!.join()).toContain('运行时验证');
  });
  it.each<Record<string, TensorSpec>>([
    { mask: t([98], 'int64') },
    { mask: t([98, 1], 'bool') },
    { mask: t([97], 'bool') },
    { values: t(['N_valid', 127]) },
    { values: t(['N_valid', 128], 'float16') },
    { values: t([99, 128]) },
    { values: t([128]) },
  ])('拒绝形状或类型不匹配 %j', (values) => expect(() => write(values)).toThrow());
  it('具体行数不会被当成已核实的 mask 计数', () => {
    expect(write({ values: t([3, 128]) }).warnings!.join()).toContain('count_nonzero');
  });
  it('支持空 mask 和空 values', () => {
    expect(
      write({ in: t([0, 128]), mask: t([0], 'bool'), values: t([0, 128]) }).outputs.out,
    ).toEqual(t([0, 128]));
    expect(() => write({ in: t([0, 128]), mask: t([0], 'bool'), values: t([1, 128]) })).toThrow();
  });
  it('恢复 bs、A、dim 后为 [2,49,128]，错误尺寸会被 View 拒绝', () => {
    const z = tensorValue(zeros().outputs.out);
    const assigned = write({ in: z }).outputs.out;
    const n = registry.create('View', { shape: 'd0, d1, d2' });
    const input = { in: assigned, d0: size(2), d1: size(49), d2: size(128) };
    expect(registry.get('View')!.inferValue!(input, n.params).outputs.out).toEqual(t([2, 49, 128]));
    expect(() => registry.get('View')!.inferValue!({ ...input, d1: size(50) }, n.params)).toThrow();
  });
});
