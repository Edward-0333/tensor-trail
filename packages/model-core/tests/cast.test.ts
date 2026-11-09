import { describe, expect, it } from 'vitest';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Project, type TensorSpec } from '../src/core/types';
import { castLabel, castModule } from '../src/modules/cast';
import { containerInputs } from '../src/modules/containers';
import { indexingModules } from '../src/modules/indexing';
import { plutoInputSchema } from '../src/modules/plutoInputExample';
import { primitives } from '../src/modules/primitives';
import { ModuleRegistry } from '../src/modules/registry';

const registry = new ModuleRegistry();
[...primitives, ...containerInputs, ...indexingModules, castModule].forEach((definition) =>
  registry.register(definition),
);
const rule = castModule.infer!;
const category: TensorSpec = {
  shape: ['B', 'A'],
  axes: ['batch', 'agent'],
  dtype: 'int8',
};

function project(path = 'agent.category'): Project {
  return {
    schemaVersion: 1,
    name: 'PLUTO category 类型转换',
    variants: [],
    graph: {
      nodes: [
        registry.create('ForwardInput', { schema: JSON.stringify(plutoInputSchema()) }, 'data'),
        registry.create('DictionaryGet', { path }, 'field'),
        registry.create('Cast', {}, 'cast'),
        registry.create('Output', {}, 'category'),
      ],
      edges: [
        {
          id: 'read',
          source: 'data',
          sourcePort: 'out',
          target: 'field',
          targetPort: 'in',
        },
        {
          id: 'convert',
          source: 'field',
          sourcePort: 'out',
          target: 'cast',
          targetPort: 'in',
        },
        {
          id: 'return',
          source: 'cast',
          sourcePort: 'out',
          target: 'category',
          targetPort: 'in',
        },
      ],
    },
  };
}

describe('Cast 类型转换', () => {
  it('long() 将 int8 转成 int64，保留形状与轴含义且不修改输入', () => {
    const input = structuredClone(category),
      before = structuredClone(input);
    const node = registry.create('Cast');
    const out = rule({ in: input }, node.params).outputs.out;
    expect(castLabel(node.params)).toBe('long()');
    expect(out).toEqual({ ...category, dtype: 'int64' });
    expect(input).toEqual(before);
    expect(out.shape).not.toBe(input.shape);
    expect(out.axes).not.toBe(input.axes);
  });

  it('float() 将 bool 掩码转为 float32，dtype 可继续切换', () => {
    const input = { ...category, dtype: 'bool' };
    expect(castLabel({ dtype: 'float32' })).toBe('float()');
    expect(rule({ in: input }, { dtype: 'float32' }).outputs.out).toEqual({
      ...input,
      dtype: 'float32',
    });
    expect(castLabel({ dtype: 'float16' })).toBe('to(float16)');
    expect(rule({ in: input }, { dtype: 'float16' }).outputs.out.dtype).toBe('float16');
  });

  it.each(['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool'])(
    '目标 %s 支持标量、符号尺寸与零长度轴',
    (dtype) => {
      for (const input of [
        { shape: [], axes: [], dtype: 'bool' },
        {
          shape: ['B', 0, 'N'],
          axes: ['batch', 'empty', 'entity'],
          dtype: 'float32',
        },
      ]) {
        expect(rule({ in: input }, { dtype }).outputs.out).toEqual({
          ...input,
          dtype,
        });
      }
    },
  );

  it.each(['int128', 'float', 'torch.int64', '', true, 64])(
    '直接推导也拒绝非法 dtype=%s',
    (dtype) => {
      expect(() => rule({ in: category }, { dtype })).toThrow('dtype');
      expect(() => castLabel({ dtype })).toThrow('dtype');
    },
  );

  it('缺少目标类型或直接传入字典时给出明确错误', () => {
    expect(() => rule({ in: category }, {})).toThrow('dtype');
    const dictionary = { kind: 'dict', fields: {} } as unknown as TensorSpec;
    expect(() => rule({ in: dictionary }, { dtype: 'int64' })).toThrow('需要 Tensor');
  });

  it('PLUTO category 字段经过 long()，参数更改及保存重开后保持形状和连线', () => {
    const p = project(),
      before = structuredClone(p);
    const result = inferProject(p, registry);
    expect(result.issues).toEqual([]);
    expect(Object.values(result.nodes).every((node) => node.status === 'ok')).toBe(true);
    const source = tensorValue(result.nodes.field.outputs.out);
    expect(source).toMatchObject({ shape: ['B', 'A'], dtype: 'int8' });
    expect(result.nodes.category.outputs.out).toEqual({
      ...source,
      dtype: 'int64',
    });
    expect(p).toEqual(before);
    expect(parseProject(serializeProject(p), registry)).toEqual(p);

    p.graph.nodes.find((node) => node.id === 'cast')!.params.dtype = 'float32';
    const reopened = parseProject(serializeProject(p), registry);
    expect(reopened.graph.edges).toEqual(before.graph.edges);
    expect(inferProject(reopened, registry).nodes.category.outputs.out).toEqual({
      ...source,
      dtype: 'float32',
    });
  });

  it('引擎拒绝字典输入和无效目标类型，不留下可传递的错误输出', () => {
    const dictionary = inferProject(project('agent'), registry);
    expect(dictionary.nodes.cast.status).toBe('error');
    expect(dictionary.nodes.cast.messages.join()).toContain('需要 Tensor');
    expect(dictionary.nodes.cast.outputs).toEqual({});
    expect(dictionary.nodes.category.status).toBe('blocked');

    const invalid = project();
    invalid.graph.nodes.find((node) => node.id === 'cast')!.params.dtype = 'float';
    const result = inferProject(invalid, registry);
    expect(result.nodes.cast.status).toBe('error');
    expect(result.nodes.cast.outputs).toEqual({});
    expect(result.nodes.category.status).toBe('blocked');
  });
});
