import { describe, expect, it } from 'vitest';
import { evaluateFunctionFormula, parseFunctionFormula } from '@tensor-trail/model-core/core/functionFormula';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { tensorValue, type Project } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules/index';
import { isCalculation } from '@tensor-trail/model-core/modules/libraryCategories';
import { useEditor } from '../src/editor/store';
import { buildModel, describeModules, validateModel } from '@tensor-trail/model-core/agent/index';

const expression = '(angle + math.pi) % (2 * math.pi) - math.pi';
function project(): Project {
  return {
    schemaVersion: 1,
    name: '角度归一化',
    variants: [],
    graph: {
      nodes: [
        registry.create(
          'Input',
          { shape: '2, 49', axes: 'batch, entity', dtype: 'float32' },
          'agent',
        ),
        registry.create(
          'Input',
          { shape: '2, 58', axes: 'batch, entity', dtype: 'float32' },
          'polygon',
        ),
        registry.create('Concat', { dim: 1 }, 'cat'),
        {
          ...registry.create('Function', { argument: 'angle', expression }, 'normalize'),
          label: '角度归一化',
        },
        registry.create('Output', {}, 'angle'),
      ],
      edges: [
        { id: 'a', source: 'agent', sourcePort: 'out', target: 'cat', targetPort: 'a' },
        { id: 'b', source: 'polygon', sourcePort: 'out', target: 'cat', targetPort: 'b' },
        { id: 'c', source: 'cat', sourcePort: 'out', target: 'normalize', targetPort: 'in' },
        { id: 'd', source: 'normalize', sourcePort: 'out', target: 'angle', targetPort: 'in' },
      ],
    },
  };
}

describe('函数公式的数学语义', () => {
  it.each([
    [-7 * Math.PI, -Math.PI],
    [-4 * Math.PI, 0],
    [-Math.PI - 0.25, Math.PI - 0.25],
    [-Math.PI, -Math.PI],
    [-1, -1],
    [0, 0],
    [1, 1],
    [Math.PI, -Math.PI],
    [3 * Math.PI, -Math.PI],
    [7 * Math.PI, -Math.PI],
  ])('角度 %s 归一化为 %s，正确处理负数和半开区间端点', (input, expected) => {
    const result = evaluateFunctionFormula(parseFunctionFormula(expression, 'angle'), input);
    expect(result).toBeCloseTo(expected, 12);
    expect(result).toBeGreaterThanOrEqual(-Math.PI);
    expect(result).toBeLessThan(Math.PI);
  });

  it('保持算术优先级和括号，支持科学记数法以及 pi 别名', () => {
    expect(evaluateFunctionFormula(parseFunctionFormula('x + 2 * 3 / 2'), 4)).toBe(7);
    expect(evaluateFunctionFormula(parseFunctionFormula('-(x + 2) * 1e-1'), 3)).toBe(-0.5);
    expect(evaluateFunctionFormula(parseFunctionFormula('x / pi'), Math.PI)).toBe(1);
    expect(evaluateFunctionFormula(parseFunctionFormula('x % -3'), 5)).toBe(-1);
    expect(evaluateFunctionFormula(parseFunctionFormula('x % 3'), -5)).toBe(1);
  });

  it.each([
    '',
    ' ',
    'x +',
    '(x + 2',
    'x ** 2',
    'x // 2',
    'x = x + 1',
    'x[0]',
    'x.shape',
    'x.sum()',
    'torch.sin(x)',
    'eval(x)',
    'y + 1',
    '1; x',
    '1e309 + x',
    'x / (2 - 2)',
    'x % 0',
    '2 * math.pi',
  ])('拒绝不支持或无效的公式 %s', (formula) =>
    expect(() => parseFunctionFormula(formula)).toThrow(),
  );

  it('限制过长/过深表达式，拒绝保留变量名', () => {
    expect(() => parseFunctionFormula('x+'.repeat(501))).toThrow();
    expect(() => parseFunctionFormula('('.repeat(25) + 'x' + ')'.repeat(25))).toThrow();
    for (const argument of ['pi', 'math', '__proto__', 'constructor', 'x.y', '', 'a'.repeat(65)])
      expect(() => parseFunctionFormula('x', argument)).toThrow();
  });
});

describe('函数模块与数据流', () => {
  it('PLUTO 拼接后的 angle 保持 [2,107]、float32 和轴含义', () => {
    const p = project(),
      before = structuredClone(p);
    const result = inferProject(p, registry);
    expect(result.nodes.normalize.inputs.in).toEqual({
      shape: [2, 107],
      axes: ['batch', 'entity'],
      dtype: 'float32',
    });
    expect(result.nodes.angle.outputs.out).toEqual(result.nodes.normalize.inputs.in);
    expect(result.nodes.normalize.outputs.out).not.toBe(result.nodes.normalize.inputs.in);
    expect(result.nodes.angle.status).toBe('ok');
    expect(p).toEqual(before);
    expect(validateModel(p).ok).toBe(true);
  });
  it('浮点标量和符号形状可以传播，整数或布尔输入不会被假定为浮点输出', () => {
    const node = registry.create('Function');
    const rule = registry.get('Function')!.infer!;
    for (const dtype of ['float16', 'float32', 'float64']) {
      const input = { shape: [], axes: [], dtype };
      expect(rule({ in: input }, node.params).outputs.out).toEqual(input);
    }
    const symbolic = { shape: ['B', 'P'], axes: ['batch', 'polygon'], dtype: 'float32' };
    expect(rule({ in: symbolic }, node.params).outputs.out).toEqual(symbolic);
    for (const dtype of ['bool', 'int64'])
      expect(() => rule({ in: { ...symbolic, dtype } }, node.params)).toThrow('浮点');
  });
  it('公式修改能保存和撤销；无效公式使下游等待修正', () => {
    useEditor.getState().load(project());
    useEditor.setState({ past: [], future: [], path: [] });
    const before = useEditor.getState().project;
    useEditor.getState().setParam('normalize', 'expression', 'angle * 2 + 1');
    const changed = useEditor.getState().project;
    expect(parseProject(serializeProject(changed), registry)).toEqual(changed);
    expect(inferProject(changed, registry).nodes.angle.status).toBe('ok');
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(changed);
    useEditor.getState().setParam('normalize', 'expression', 'angle[0]');
    const result = inferProject(useEditor.getState().project, registry);
    expect(result.nodes.normalize.status).toBe('error');
    expect(result.nodes.angle.status).toBe('blocked');
  });
  it('库和配方暴露一个具备公式参数、输入输出端口的函数模块', () => {
    const def = registry.get('Function')!;
    expect(isCalculation(def)).toBe(true);
    expect(
      describeModules()
        .modules.find((m) => m.type === 'Function')
        ?.fields.map((f) => f.key),
    ).toEqual(['argument', 'expression']);
    const r = buildModel({
      name: '函数配方',
      nodes: [
        { id: 'x', type: 'Input', params: { shape: '2, 107', axes: 'batch, entity' } },
        { id: 'f', type: 'Function', params: { argument: 'angle', expression } },
        { id: 'y', type: 'Output' },
      ],
      connections: [
        ['x.out', 'f.in'],
        ['f.out', 'y.in'],
      ],
    });
    expect(r.ok, JSON.stringify(r.diagnostics)).toBe(true);
    expect(tensorValue(r.outputs.y).shape).toEqual([2, 107]);
  });
});
