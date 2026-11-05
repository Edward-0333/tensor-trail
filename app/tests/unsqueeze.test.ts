import { describe, expect, it } from 'vitest';
import { registry } from '../src/modules';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { mergeCalculations, splitCalculationGroup } from '../src/core/calculationGroups';
import { tensorValue, type Project } from '../src/core/types';
import { buildModel } from '../src/agent';

const rule = registry.get('Unsqueeze')!.infer!;
const source = { shape: [2, 107], axes: ['batch', 'entity'], dtype: 'float32' };

describe('Unsqueeze', () => {
  it.each([
    [0, [1, 2, 107], ['singleton', 'batch', 'entity']],
    [1, [2, 1, 107], ['batch', 'singleton', 'entity']],
    [2, [2, 107, 1], ['batch', 'entity', 'singleton']],
    [-1, [2, 107, 1], ['batch', 'entity', 'singleton']],
    [-2, [2, 1, 107], ['batch', 'singleton', 'entity']],
    [-3, [1, 2, 107], ['singleton', 'batch', 'entity']],
  ])('dim=%s 的正负索引语义和轴含义正确', (dim, shape, axes) => {
    const before = structuredClone(source);
    const out = rule({ in: source }, { dim }).outputs.out;
    expect(out).toEqual({ shape, axes, dtype: 'float32' });
    expect(source).toEqual(before);
    expect(out.shape).not.toBe(source.shape);
    expect(out.axes).not.toBe(source.axes);
  });
  it.each([0, -1])('标量可用 dim=%s 转成长度为 1 的张量', (dim) => {
    expect(rule({ in: { shape: [], axes: [], dtype: 'bool' } }, { dim }).outputs.out).toEqual({
      shape: [1],
      axes: ['singleton'],
      dtype: 'bool',
    });
  });
  it.each(['bool', 'int64', 'float16', 'float32', 'float64'])(
    '保留 %s、符号尺寸与零长度轴',
    (dtype) => {
      const out = rule(
        { in: { shape: ['B', 0, 'N'], axes: ['batch', 'empty', 'entity'], dtype } },
        { dim: -1 },
      ).outputs.out;
      expect(out).toEqual({
        shape: ['B', 0, 'N', 1],
        axes: ['batch', 'empty', 'entity', 'singleton'],
        dtype,
      });
    },
  );
  it.each([-4, 3, 0.5, NaN, Infinity, '-1', true])('拒绝非法 dim=%s', (dim) => {
    expect(() => rule({ in: source }, { dim })).toThrow('dim');
  });
  it('对画布支持的最大维数明确提示，不产生不能保存的规格', () => {
    const seven = { shape: Array(7).fill(2), axes: Array(7).fill('axis'), dtype: 'float32' };
    expect(rule({ in: seven }, { dim: -8 }).outputs.out.shape).toEqual([1, ...seven.shape]);
    const eight = { shape: Array(8).fill(2), axes: Array(8).fill('axis'), dtype: 'float32' };
    expect(() => rule({ in: eight }, { dim: -1 })).toThrow('最多支持 8 维');
  });
  it('PLUTO 的 angle 扩维后可与 position 拼接，保存与计算组合保持结果', () => {
    const built = buildModel({
      name: 'PLUTO pos',
      nodes: [
        { id: 'angle', type: 'Input', params: { shape: '2,107', axes: 'batch,entity' } },
        {
          id: 'normalize',
          type: 'Function',
          params: { argument: 'x', expression: '(x + pi) % (2 * pi) - pi' },
        },
        { id: 'expand', type: 'Unsqueeze' },
        { id: 'position', type: 'Input', params: { shape: '2,107,2', axes: 'batch,entity,xy' } },
        { id: 'cat', type: 'Concat', params: { dim: -1 } },
        { id: 'pos', type: 'Output' },
      ],
      connections: [
        ['angle.out', 'normalize.in'],
        ['normalize.out', 'expand.in'],
        ['expand.out', 'cat.b'],
        ['position.out', 'cat.a'],
        ['cat.out', 'pos.in'],
      ],
    });
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    expect(tensorValue(built.outputs.pos).shape).toEqual([2, 107, 3]);
    const project = built.project as Project;
    const group = mergeCalculations(project.graph, 'normalize', 'expand', registry);
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
    expect(inferProject(project, registry).nodes.pos.outputs.out).toEqual(built.outputs.pos);
    splitCalculationGroup(project.graph, group);
    expect(inferProject(project, registry).nodes.pos.outputs.out).toEqual(built.outputs.pos);
  });
});
