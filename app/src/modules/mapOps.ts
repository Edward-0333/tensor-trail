import type { Dim, ModelNode, ModuleDefinition, Params, ValueSpec, Inference } from '../core/types';
import { isIntegerSpec, tensorValue } from '../core/types';
import { axisIndex, product, requireThat, tensor } from './math';
import { input, output, text, num, bool } from './primitives';
import { featureMathModules, viewPorts } from './featureMath';

const view = featureMathModules.find((d) => d.type === 'View')!;
function reshape(values: Record<string, ValueSpec>, p: Params): Inference {
  try {
    const r = view.inferValue!(values, p);
    return {
      ...r,
      warnings: r.warnings?.map((w) => w.replaceAll('View', 'Reshape')),
      outputs: { out: tensorValue(r.outputs.out) },
      notes: [`reshape(${p.shape})；按需创建视图或复制，保持元素总数。`],
    };
  } catch (e) {
    throw new Error((e as Error).message.replaceAll('View', 'Reshape'));
  }
}
const repeatsPorts = (p: Params) => viewPorts({ shape: p.repeats });
export function mapOperationCode(n: ModelNode): string | undefined {
  if (n.type === 'Reshape')
    return `# d0、d1 等来自局部尺寸输入\nx.reshape(${n.params.shape})  # 不能用 view 的内存连续性限制代替 reshape`;
  if (n.type === 'Repeat')
    return `# d0、d1 等来自局部整数输入\nx.repeat(${n.params.repeats})  # 每个数表示对应维度重复次数，不是目标尺寸`;
  if (n.type === 'ReduceMax')
    return `x.max(dim=${n.params.dim}, keepdim=${n.params.keepdim ? 'True' : 'False'})[0]\n# 本节点只输出 values，不输出 argmax；填零位置同样参与最大值计算。`;
  if (n.type === 'ReduceSum')
    return `x.sum(dim=${n.params.dim}, keepdim=${n.params.keepdim ? 'True' : 'False'})`;
}
export const mapOps: ModuleDefinition[] = [
  {
    type: 'Reshape',
    label: 'Reshape',
    category: '维度操作',
    description: '改变形状，动态 d0–d7 接口接收局部尺寸；允许一个 -1，按需复制内存。',
    fields: [{ ...text('shape', '目标形状', '8, -1, 64'), maxLength: 256 }],
    inputs: input,
    outputs: output,
    getPorts: viewPorts,
    inferValue: reshape,
    infer: (v, p) => reshape(v, p),
  },
  {
    type: 'Repeat',
    label: '重复 Repeat',
    category: '维度操作',
    description: '按各轴次数重复 Tensor，允许新增前导轴。支持常量和 d0–d7 动态整数接口。',
    fields: [{ ...text('repeats', '重复次数', '1, d0, 1'), maxLength: 256 }],
    ...repeatsPorts({ repeats: '1, d0, 1' }),
    getPorts: repeatsPorts,
    inferValue: (values, p) => {
      const x = tensorValue(values.in);
      requireThat(typeof p.repeats === 'string' && p.repeats.length <= 256, '重复次数需要短文本。');
      const tokens = p.repeats.split(',').map((s) => s.trim());
      requireThat(
        tokens.length >= x.shape.length && tokens.length >= 1 && tokens.length <= 8,
        '重复次数的维数不能少于输入，最多 8 维。',
      );
      const factors: Dim[] = tokens.map((token) => {
        if (/^d[0-7]$/.test(token)) {
          const v = values[token];
          requireThat(v && isIntegerSpec(v), `${token} 需要局部整数。`);
          requireThat(
            typeof v.value === 'number'
              ? Number.isSafeInteger(v.value) && v.value >= 0
              : typeof v.value === 'string' && v.value.trim() !== '',
            '重复次数需要非负整数或符号。',
          );
          return v.value;
        }
        requireThat(
          /^\+?\d+$/.test(token) && Number.isSafeInteger(Number(token)),
          '重复次数需要非负整数或 d0–d7。',
        );
        return Number(token);
      });
      const padded = [...Array(factors.length - x.shape.length).fill(1), ...x.shape];
      const shape = factors.map((f, i) =>
        f === 0 || padded[i] === 0
          ? 0
          : f === 1
            ? padded[i]
            : padded[i] === 1
              ? f
              : product([padded[i], f]),
      );
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, i) => x.axes[i - (factors.length - x.shape.length)] ?? `axis${i}`),
            x.dtype,
          ),
        },
        notes: ['repeat 复制数据；对重复结果的梯度累加到来源 Tensor。'],
      };
    },
  },
  ...['ReduceMax', 'ReduceSum'].map(
    (type): ModuleDefinition => ({
      type,
      label: type === 'ReduceMax' ? '最大值 Max' : '求和 Sum',
      category: '逻辑与归约',
      description:
        type === 'ReduceMax'
          ? '沿指定轴取最大值，只输出 values。'
          : '沿指定轴求和，整数和 bool 默认累加为 int64。',
      fields: [num('dim', '归约轴', 1, -8), bool('keepdim', '保留归约轴', false)],
      inputs: input,
      outputs: output,
      infer: ({ in: x }, p) => {
        const axis = axisIndex(Number(p.dim), x.shape.length);
        requireThat(
          [
            'bool',
            'int8',
            'uint8',
            'int16',
            'int32',
            'int64',
            'float16',
            'bfloat16',
            'float32',
            'float64',
          ].includes(x.dtype),
          '不支持该归约数据类型。',
        );
        if (type === 'ReduceMax') requireThat(x.shape[axis] !== 0, '不能对长度为零的轴求最大值。');
        const shape = x.shape.flatMap((d, i) => (i !== axis ? [d] : p.keepdim ? [1] : [])),
          axes = x.axes.filter((_, i) => i !== axis || p.keepdim);
        const dtype = type === 'ReduceSum' && /^(bool|u?int)/.test(x.dtype) ? 'int64' : x.dtype;
        return {
          outputs: { out: tensor(shape, axes, dtype) },
          notes: [
            type === 'ReduceMax'
              ? '输出 max(...)[0]；不会自行忽略填零点。'
              : '空轴求和为零；输出形状与 keepdim 一致。',
          ],
        };
      },
    }),
  ),
];
