import type { ModelNode, ModuleDefinition, ModuleSpec, TensorSpec } from '../core/types';
import { clone, isModuleSpec, isTensorSpec, tensorValue } from '../core/types';
import { checkDictionaryKey, parseDictionarySchema } from '../core/dictionary';
import { requireThat } from './math';
import { input, output, text } from './primitives';

export function parameterName(value: unknown): string {
  requireThat(
    typeof value === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(value),
    '参数名需要单个 Python 属性名称，例如 weight 或 bias。',
  );
  checkDictionaryKey(value);
  return value;
}

export function moduleReferenceName(value: unknown): string {
  requireThat(
    typeof value === 'string' && value.length <= 256,
    '模块引用需要不超过 256 字符的属性路径。',
  );
  value.split('.').forEach(parameterName);
  return value;
}

export function moduleValue(value: unknown): ModuleSpec {
  requireThat(
    value && typeof value === 'object' && isModuleSpec(value as ModuleSpec),
    '需要 nn.Module 模块引用，不是 Tensor 或 Parameter。',
  );
  return value as ModuleSpec;
}

export function moduleReferenceCode(node: ModelNode): string | undefined {
  if (node.type === 'ModuleReference')
    return `# 声明已有模块引用，不构造或执行模块。\nmodule = ${node.params.reference}\n# 类型：${node.params.module_type}\n# 参数表是静态规格，不含真实权重数值。`;
  if (node.type === 'GetParameter')
    return `out = module.${node.params.name}\n# 读取已有 nn.Parameter 的引用；不 clone、不 detach、不重新初始化。\n# 不调用 module.forward，也不创建新的可训练参数或 autograd 运算。`;
  if (node.type === 'ModuleParameterState')
    return `# 初始化语句完成后，继续使用同一个模块对象。\n# initialized_parameter 必须仍是 module.${node.params.name} 的完整参数。\nout = module\n# 这是原地修改的顺序/状态标记，不执行属性赋值，也不重新注册参数。`;
}

export const moduleReferenceModules: ModuleDefinition[] = [
  {
    type: 'ModuleReference',
    label: '模块引用',
    category: '输入输出',
    description:
      '声明已有 nn.Module 对象及其直接参数表；不是 Tensor，不执行模型构造或 forward。可供 GetParameter 读取 weight/bias。',
    fields: [
      { ...text('reference', '模块引用路径', 'self.linear'), maxLength: 256 },
      { ...text('module_type', '模块类型', 'nn.Linear'), maxLength: 128 },
      {
        ...text(
          'parameters',
          '参数表（字典结构 JSON）',
          '{"kind":"dict","fields":{"weight":{"shape":[8,8],"dtype":"float32"},"bias":{"shape":[8],"dtype":"float32"}}}',
        ),
        maxLength: 16000,
      },
    ],
    inputs: [],
    outputs: output,
    inferValue: (_, p) => {
      const reference = moduleReferenceName(p.reference);
      requireThat(
        typeof p.module_type === 'string' &&
          p.module_type.trim().length > 0 &&
          p.module_type.length <= 128,
        '请填写模块类型。',
      );
      requireThat(
        typeof p.parameters === 'string' && p.parameters.length <= 16000,
        '参数表需要不超过 16000 字符的 JSON 文本。',
      );
      const schema = parseDictionarySchema(p.parameters);
      requireThat(
        schema.keyType !== 'int' && Object.keys(schema.fields).length <= 64,
        '参数表需要至多 64 个直接属性。',
      );
      const parameters = Object.fromEntries(
        Object.entries(schema.fields).map(([name, value]) => {
          parameterName(name);
          requireThat(
            isTensorSpec(value),
            `参数 ${name} 必须是 Tensor 规格；不接受 None、子字典或 buffer。`,
          );
          return [name, { ...clone(value), parameterRef: { module: reference, name } }];
        }),
      );
      return {
        outputs: { out: { kind: 'module', reference, moduleType: p.module_type, parameters } },
        notes: [
          '模块对象本身没有 shape、dtype 或 requires_grad；这些属性属于它的参数。',
          '这里只声明直接注册的参数规格；不连接实际 Python 对象，不推测未声明的参数。',
        ],
      };
    },
  },
  {
    type: 'GetParameter',
    label: '读取参数',
    category: '输入输出',
    description:
      '从模块引用读取已存在的 nn.Parameter，如 module.weight 或 module.bias；保留参数身份与梯度标注，不执行前向计算、复制或重新初始化。',
    fields: [text('name', '参数名称', 'weight')],
    inputs: input,
    outputs: output,
    inferValue: ({ in: source }, p) => {
      const module = moduleValue(source),
        name = parameterName(p.name);
      requireThat(
        Object.hasOwn(module.parameters, name),
        `模块 ${module.reference} 没有已声明的参数 ${name}。`,
      );
      const parameter = tensorValue(module.parameters[name]);
      return {
        outputs: { out: { ...clone(parameter), parameterRef: { module: module.reference, name } } },
        notes: [
          `读取 ${module.reference}.${name}；是同一个 nn.Parameter 的引用，不是新张量运算。`,
          '画布只复制规格描述以避免编辑副作用；不代表 Python 中 clone 参数或分配真实张量。',
        ],
      };
    },
  },
  {
    type: 'ModuleParameterState',
    label: '初始化后的模块',
    category: '输入输出',
    description:
      '将参数原地初始化的完成状态与所属模块关联；输出同一个模块引用，不执行参数替换或重新注册。',
    fields: [text('name', '已初始化的参数', 'weight')],
    inputs: [
      { id: 'module', label: '模块' },
      { id: 'parameter', label: '初始化后参数' },
    ],
    outputs: output,
    inferValue: ({ module: source, parameter: updated }, p) => {
      const module = moduleValue(source),
        name = parameterName(p.name),
        parameter = tensorValue(updated);
      requireThat(Object.hasOwn(module.parameters, name), `模块中不存在参数 ${name}。`);
      const previous: TensorSpec = module.parameters[name];
      requireThat(
        parameter.parameterRef?.module === module.reference && parameter.parameterRef.name === name,
        '初始化结果必须保留同一模块、同一参数的身份；不能使用派生 Tensor 或其他模块的参数。',
      );
      requireThat(
        JSON.stringify(previous.shape) === JSON.stringify(parameter.shape) &&
          previous.dtype === parameter.dtype &&
          JSON.stringify(previous.autograd) === JSON.stringify(parameter.autograd),
        '原地初始化不能改变参数形状、dtype 或 requires_grad 标注。',
      );
      const out = clone(module);
      out.parameters[name] = clone(parameter);
      return {
        outputs: { out },
        notes: [
          '表示参数原地修改已完成，之后读取模块可取得该参数；不创建新模块或新参数。',
          '只维护静态依赖和元数据；不模拟随机数值或 Python 对象的实际内存。',
        ],
      };
    },
  },
];
