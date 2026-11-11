import type { ModelNode, ModuleDefinition } from '../core/types';
import { attributeBinding, attributeOwnerLabel } from '../core/attributes';
import { tensorValue } from '../core/types';
import { input, output, text } from './primitives';

const fields = [
  {
    ...text('owner', '所属模块路径（节点 ID 的 JSON 数组）', '[]'),
    help: '从根图定位模块；例如 ["pts_bbox_head"]，[] 表示根模型。读写节点使用相同路径和属性名；不是 forward 形参。',
  },
  text('name', '属性名称', 'points_scale'),
  { ...text('version', '赋值版本（可选）', ''), help: '同一属性多次赋值时显式区分写入点，例如 initial / pretrain。GET 必须选择相同版本；不是新的 Python 属性，也不自动判断哪个条件生效。' },
];

export function attributeCode(node: ModelNode, scope: string[] = []): string | undefined {
  if (!['SetAttribute', 'GetAttribute'].includes(node.type)) return;
  let binding;
  try {
    binding = attributeBinding(node.params);
  } catch {
    return '# 属性路径或属性名待修正；请查看节点错误提示。';
  }
  const target = `${JSON.stringify(scope) === JSON.stringify(binding.owner) ? 'self' : attributeOwnerLabel(binding.owner)}.${binding.name}`;
  return node.type === 'SetAttribute'
    ? `# 属性绑定示意；路径按画布节点 ID 定位，不在浏览器执行 Python。\n${target} = value\n${binding.version ? `# 赋值版本：${binding.version}，仅标识此写入点，不是另一个属性。\n` : ''}# 写入/覆盖普通 Tensor 属性；不 clone、不 detach、不新增 Parameter。\n# 不向该模块的 forward() 添加参数，也不生成独立 Tensor 输出。`
    : `# 从同一模块、同一属性${binding.version ? `、赋值版本 ${binding.version}` : ''}的唯一写入点读取 Tensor 引用。\nvalue = ${target}\n# 在所属模块的方法内，等价于 self.${binding.name}。\n# 保留上游形状、dtype 和梯度标注；不复制真实张量。`;
}

export const attributeModules: ModuleDefinition[] = [
  {
    type: 'SetAttribute',
    label: '写属性 SET',
    category: '输入输出',
    description:
      '将输入 Tensor 写入指定模块属性。首次创建，之后覆盖；只描述属性绑定，不新增可训练参数或 forward 输入。每次图推导重新建立状态。',
    fields,
    inputs: input,
    outputs: [],
    inferValue: ({ in: value }, params) => {
      attributeBinding(params);
      const tensor = tensorValue(value);
      if (tensor.parameterRef)
        throw new Error(
          '写属性节点仅支持普通 Tensor；注册或替换 nn.Parameter 不属于普通属性赋值。',
        );
      return {
        outputs: {},
        notes: [
          '输入 Tensor 的引用绑定到模块属性，不 clone、不 detach；真实张量未在画布中执行。',
          'SET 没有 Tensor 输出端口；对应 GET 通过属性依赖读取该值，不占用 forward 输入端口。',
        ],
      };
    },
  },
  {
    type: 'GetAttribute',
    label: '读属性 GET',
    category: '输入输出',
    description:
      '读取对应 SetAttribute 的 Tensor 引用。通过所属模块路径和属性名跨层绑定；无需形参端口，形状与梯度标注跟随真实上游。',
    fields,
    inputs: [],
    outputs: output,
    inferValue: () => {
      throw new Error('属性读取需要在完整图中解析对应 SetAttribute，不能伪造独立输入。');
    },
  },
];
