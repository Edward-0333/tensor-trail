import { describe, it, expect } from 'vitest';
import { registry } from '../src/modules/index';
import { anyLabel } from '../src/modules/logical';
import { calculate } from '../src/modules/calculation';
import { DICTIONARY_DTYPES } from '../src/core/dictionary';
import { isCalculation } from '../src/modules/libraryCategories';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';
import { buildModel, describeModules } from '../src/agent/index';
import { type TensorSpec, type Project } from '../src/core/types';

const spec = (shape: (number | string)[], dtype = 'bool'): TensorSpec => ({
  shape,
  axes: shape.map((_, i) => `axis${i}`),
  dtype,
});
const any = registry.get('Any')!.infer!;
const invert = registry.get('Invert')!.infer!;

describe('独立 Any / Invert', () => {
  it.each([
    [2, 49, 21],
    [2, 58, 20],
  ])('PLUTO 掩码 %j 先 any(-1) 再取反，支持合并和保存', (...shape) => {
    const result = buildModel({
      name: 'padding mask',
      nodes: [
        {
          id: 'mask',
          type: 'Input',
          params: { shape: shape.join(','), axes: 'batch,entity,sample', dtype: 'bool' },
        },
        { id: 'any', type: 'Any' },
        { id: 'invert', type: 'Invert' },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['mask.out', 'any.in'],
        ['any.out', 'invert.in'],
        ['invert.out', 'out.in'],
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(result.outputs.out).toEqual({
      shape: shape.slice(0, -1),
      axes: ['batch', 'entity'],
      dtype: 'bool',
    });
    const project = result.project as Project;
    const group = mergeCalculations(project.graph, 'any', 'invert', registry);
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
    expect(inferProject(project, registry).nodes.out.outputs.out).toEqual(result.outputs.out);
    splitCalculationGroup(project.graph, group);
    expect(inferProject(project, registry).nodes.out.outputs.out).toEqual(result.outputs.out);
  });
  it('归约可保留轴、选全部轴，显示摘要与形状同步', () => {
    const input = spec([2, 49, 21]),
      before = structuredClone(input);
    expect(any({ in: input }, { dim: '-1', keepdim: true }).outputs.out).toEqual(spec([2, 49, 1]));
    expect(any({ in: input }, { dim: '', keepdim: false }).outputs.out).toEqual(spec([]));
    expect(any({ in: input }, { dim: ' ', keepdim: true }).outputs.out).toEqual(spec([1, 1, 1]));
    expect(anyLabel({ dim: '-1', keepdim: false })).toBe('any(-1)');
    expect(anyLabel({ dim: '', keepdim: false })).toBe('any()');
    expect(anyLabel({ dim: '', keepdim: true })).toBe('any(keepdim=True)');
    expect(input).toEqual(before);
  });
  it.each(DICTIONARY_DTYPES)('Any 的 %s 类型与 PyTorch 一致', (dtype) => {
    const expected = dtype === 'uint8' ? 'uint8' : 'bool';
    for (const dim of ['-1', ''])
      expect(any({ in: spec([2, 3], dtype) }, { dim, keepdim: false }).outputs.out.dtype).toBe(
        expected,
      );
    expect(calculate(spec([2, 3], dtype), { op: 'any', dim: -1, keepdim: false }).dtype).toBe(
      expected,
    );
  });
  it('标量可按 0、-1 或全部轴归约，keepdim 也仍为标量', () => {
    for (const dtype of ['bool', 'uint8'])
      for (const dim of ['0', '-1', ''])
        for (const keepdim of [false, true])
          expect(any({ in: spec([], dtype) }, { dim, keepdim }).outputs.out).toEqual(
            spec([], dtype),
          );
    for (const dim of ['1', '-2'])
      expect(() => any({ in: spec([]) }, { dim, keepdim: false })).toThrow('标量');
  });
  it('符号和零长度轴可归约，保留未归约轴的含义', () => {
    const source = spec(['B', 0, 'N']);
    expect(any({ in: source }, { dim: '1', keepdim: false }).outputs.out).toEqual({
      shape: ['B', 'N'],
      axes: ['axis0', 'axis2'],
      dtype: 'bool',
    });
    expect(any({ in: source }, { dim: '1', keepdim: true }).outputs.out).toEqual({
      ...source,
      shape: ['B', 1, 'N'],
    });
  });
  it('取反支持布尔及全部输入整型，拒绝浮点；不修改输入', () => {
    for (const dtype of ['bool', 'int64', 'int32', 'int16', 'int8', 'uint8']) {
      const input = spec([2, 3], dtype);
      const before = structuredClone(input);
      const out = invert({ in: input }, {}).outputs.out;
      expect(out).toEqual(input);
      expect(out).not.toBe(input);
      expect(input).toEqual(before);
    }
    for (const dtype of ['float16', 'float32', 'float64', 'bfloat16'])
      expect(() => invert({ in: spec([2, 3], dtype) }, {})).toThrow('浮点');
  });
  it.each(['1.5', '0,1', 'missing', '9007199254740992', '2', '-3'])(
    '非法或越界轴 %s 明确报错',
    (dim) => {
      expect(() => any({ in: spec([2, 3]) }, { dim, keepdim: false })).toThrow();
    },
  );
  it('现有 Compute 和独立节点使用相同的归约/取反规则', () => {
    const input = spec([2, 58, 20]);
    const separate = invert(
      { in: any({ in: input }, { dim: '-1', keepdim: false }).outputs.out },
      {},
    ).outputs.out;
    const combined = registry.get('Compute')!.infer!(
      { in: input },
      { operations: JSON.stringify([{ op: 'any', dim: -1, keepdim: false }, { op: 'invert' }]) },
    ).outputs.out;
    expect(combined).toEqual(separate);
    for (const type of ['Any', 'Invert']) {
      expect(isCalculation(registry.get(type))).toBe(true);
      expect(describeModules().modules.filter((m) => m.type === type)).toHaveLength(1);
    }
  });
});
