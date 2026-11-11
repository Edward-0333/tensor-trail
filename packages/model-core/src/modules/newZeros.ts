import type { ModelNode, ModuleDefinition, Params } from '../core/types';
import { tensorValue } from '../core/types';
import { validateAutograd } from '../core/tensorAppearance';
import { tensor } from './math';
import { factoryShape } from './factoryShape';
import { bool, output, text } from './primitives';
import { viewPorts } from './featureMath';

const defaultShape = 'd0, d1, d2';
const getPorts = (params: Params) => ({
  inputs: viewPorts(params).inputs.map((port) =>
    port.id === 'in' ? { id: 'in', label: '参考 Tensor（仅继承 dtype/device）' } : port,
  ),
  outputs: output,
});

export function newZerosCode(node: ModelNode): string | undefined {
  if (node.type !== 'NewZeros') return;
  return `out = reference.new_zeros((${node.params.shape},), requires_grad=${node.params.requires_grad === true ? 'True' : 'False'})\n# d0、d1 等从整数尺寸接口传入；本图对应 B、Q、self.embed_dims。\n# 仅继承 reference.dtype 和 reference.device，不继承形状、数值或 requires_grad。\n# 默认创建独立的全零 Tensor，不是 nn.Parameter，也不向 reference 回传梯度。\n# 画布仅推导规格，不分配内存；实际设备未采集，不据此声称已验证 CUDA。`;
}

export const newZeros: ModuleDefinition = {
  type: 'NewZeros',
  label: '同设备/类型零张量 NewZeros',
  category: '维度操作',
  description:
    '对应 Tensor.new_zeros(size)。按显式整数尺寸创建全零 Tensor，仅继承参考 Tensor 的 dtype/device，默认 requires_grad=False；不是 zeros_like，也不是坐标到特征的变换。',
  fields: [
    {
      ...text('shape', '目标形状', defaultShape),
      maxLength: 256,
      help: '非负整数或 d0–d7 尺寸接口；例如 d0,d1,d2。不支持 -1，不从参考 Tensor 推导形状。',
    },
    {
      ...text('axes', '输出轴含义', ''),
      maxLength: 256,
      help: '可选，与输出维数一致的逗号分隔标签；例如 batch,query,feature。',
    },
    bool('requires_grad', '新 Tensor 的 requires_grad', false),
  ],
  ...getPorts({ shape: defaultShape }),
  getPorts,
  inferValue: (values, params) => {
    const reference = tensorValue(values.in);
    const { shape, axes } = factoryShape(values, params, 'new_zeros');
    const autograd = { requiresGrad: params.requires_grad, source: 'inferred' };
    validateAutograd(autograd, reference.dtype);
    return {
      outputs: { out: { ...tensor(shape, axes, reference.dtype), autograd } },
      notes: [
        '创建独立全零 Tensor；形状由显式尺寸决定，dtype/device 继承参考 Tensor。',
        '参考输入仅表示元数据依赖，不使用其数值；new_zeros 不把梯度传回参考 Tensor。',
        'requires_grad 来自工厂参数，默认 False，不继承参考 Tensor 的梯度状态；不创建 nn.Parameter。',
        '设备和全零数值仅按算子语义描述；静态画布不实际分配 Tensor 或采集设备。',
      ],
    };
  },
};
