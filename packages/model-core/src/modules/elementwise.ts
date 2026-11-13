import type { ModelNode, ModuleDefinition, TensorSpec } from '../core/types';
import { tensorValue } from '../core/types';
import { broadcast, requireThat, tensor } from './math';
import { input, output, select } from './primitives';

const integers = ['uint8', 'int8', 'int16', 'int32', 'int64'];
const floats = ['float16', 'bfloat16', 'float32', 'float64'];
const supported = new Set(['bool', ...integers, ...floats]);

function checked(value: TensorSpec | undefined): TensorSpec {
  const source = tensorValue(value);
  requireThat(supported.has(source.dtype), `暂不支持 dtype=${source.dtype}。`);
  return source;
}

const category = (dtype: string) => (dtype === 'bool' ? 0 : integers.includes(dtype) ? 1 : 2);

/** torch.result_type for supported Tensor operands, including zero-dimensional tensors. */
export function resultDtype(a: TensorSpec, b: TensorSpec): string {
  if (a.dtype === b.dtype) return a.dtype;
  // Within one category, a dimensioned Tensor takes precedence over a scalar Tensor.
  if ((a.shape.length === 0) !== (b.shape.length === 0) && category(a.dtype) === category(b.dtype))
    return a.shape.length ? a.dtype : b.dtype;
  if (a.dtype === 'bool') return b.dtype;
  if (b.dtype === 'bool') return a.dtype;
  const af = floats.includes(a.dtype),
    bf = floats.includes(b.dtype);
  if (af || bf) {
    if (!af) return b.dtype;
    if (!bf) return a.dtype;
    if ([a.dtype, b.dtype].includes('float64')) return 'float64';
    // float16 + bfloat16 promotes to float32, as does either with float32.
    return 'float32';
  }
  if ([a.dtype, b.dtype].includes('uint8') && [a.dtype, b.dtype].includes('int8')) return 'int16';
  return integers[Math.max(integers.indexOf(a.dtype), integers.indexOf(b.dtype))];
}

function binaryShape(a: TensorSpec, b: TensorSpec, warnings: string[]): TensorSpec {
  const dtype = resultDtype(a, b);
  return broadcast({ ...a, dtype }, { ...b, dtype }, warnings);
}

const binaryInputs = [
  { id: 'a', label: 'a' },
  { id: 'b', label: 'b' },
];

export function divideCode(node: ModelNode): string | undefined {
  if (node.type !== 'Divide') return;
  return 'out = a / b\n# 逐元素广播真除法，不是整除或 Softmax。\n# 不原地修改输入，不自动添加 epsilon/clamp；分母为零时保留 PyTorch 的 inf/nan 语义。\n# 这里只推导形状与 dtype，不声称已执行数值运算或验证分母非零。';
}

export const elementwiseModules: ModuleDefinition[] = [
  {
    type: 'Divide', label: '逐元素真除法 /', category: '数学运算',
    description: '对应 a / b，支持 Tensor 广播和实数类型提升，不是矩阵运算或整除。不会自动添加 epsilon/clamp，也不会原地修改输入。',
    fields: [select('default_dtype', '整数真除法的默认浮点类型', 'float32', ['float32','float64'])],
    inputs: binaryInputs, outputs: output,
    infer: ({ a, b }, params) => {
      const left = checked(a), right = checked(b), warnings: string[] = [];
      requireThat(['float32','float64'].includes(String(params.default_dtype)), '默认浮点 dtype 无效。');
      const promoted = resultDtype(left, right);
      const dtype = category(promoted) < 2 ? String(params.default_dtype) : promoted;
      return { outputs: { out: broadcast({ ...left, dtype }, { ...right, dtype }, warnings) }, warnings,
        notes: ['逐元素真除法，从最后一维对齐并广播；不修改输入或注册 buffer。',
          '画布不执行数值计算，不能证明分母非零；不添加源码没有的 epsilon/clamp。',
          '整数/bool 真除法使用本节点记录的 torch 默认浮点类型；浮点输入按类型提升规则输出。'] };
    },
  },
  {
    type: 'BitwiseAnd',
    label: '按位与 &',
    category: '逻辑与归约',
    description:
      '对应 a & b。bool 输入逐元素判断两者同时为真；整数输入执行按位与，支持广播和整数类型提升。',
    fields: [],
    inputs: binaryInputs,
    outputs: output,
    infer: ({ a, b }) => {
      const left = checked(a),
        right = checked(b),
        warnings: string[] = [];
      requireThat(
        [left, right].every((source) => source.dtype === 'bool' || integers.includes(source.dtype)),
        '& 需要 bool 或整数 Tensor，浮点输入不支持按位与。',
      );
      return {
        outputs: { out: binaryShape(left, right, warnings) },
        warnings,
        notes: ['bool & bool 为逐元素逻辑与；混合整数输入按 PyTorch 规则提升 dtype。'],
      };
    },
  },
  {
    type: 'Subtract',
    label: '逐元素减法',
    category: '数学运算',
    description: '对应 a - b，支持 Tensor 广播及数值类型提升；bool Tensor 不支持减法。',
    fields: [],
    inputs: binaryInputs,
    outputs: output,
    infer: ({ a, b }) => {
      const left = checked(a),
        right = checked(b),
        warnings: string[] = [];
      requireThat(
        left.dtype !== 'bool' && right.dtype !== 'bool',
        '减法不支持 bool Tensor，请使用逻辑运算或先转换类型。',
      );
      return { outputs: { out: binaryShape(left, right, warnings) }, warnings };
    },
  },
  {
    type: 'Where',
    label: '条件选择 Where',
    category: '分支运算',
    description:
      '对应 torch.where(condition, x, y)。bool 条件为真时取 x，否则取 y；三路形状需要可以广播。',
    fields: [],
    inputs: [
      { id: 'condition', label: 'condition' },
      { id: 'x', label: 'x' },
      { id: 'y', label: 'y' },
    ],
    outputs: output,
    infer: ({ condition, x, y }) => {
      const mask = checked(condition),
        left = checked(x),
        right = checked(y),
        warnings: string[] = [];
      requireThat(mask.dtype === 'bool', 'Where 的 condition 必须是 bool Tensor。');
      const values = binaryShape(left, right, warnings);
      // Condition participates only in shape broadcasting, never in value dtype promotion.
      const result = broadcast(values, { ...mask, dtype: values.dtype }, warnings);
      return { outputs: { out: result }, warnings };
    },
  },
  {
    type: 'ZerosLike',
    label: '同形零张量',
    category: '数学运算',
    description:
      '对应 torch.zeros_like(in)，生成同形状、同数据类型的零张量；当前画布推导形状和类型。',
    fields: [],
    inputs: input,
    outputs: output,
    infer: ({ in: value }) => {
      const source = checked(value);
      return { outputs: { out: { ...tensor([...source.shape], [...source.axes], source.dtype),
        autograd: { requiresGrad: false, source: 'inferred' } } } };
    },
  },
  {
    type: 'AlignTrailingDims',
    label: '补齐末尾维度',
    category: '维度操作',
    description:
      '对应 while in.ndim < like.ndim: in = in.unsqueeze(-1)。仅在末尾补大小为 1 的轴，不扩展元素，也不保证后续可广播。',
    fields: [],
    inputs: [
      { id: 'in', label: '输入' },
      { id: 'like', label: '参照' },
    ],
    outputs: output,
    infer: ({ in: value, like }) => {
      const source = checked(value),
        reference = checked(like);
      const extra = reference.shape.length - source.shape.length;
      requireThat(extra >= 0, '输入维数高于参照 Tensor，补齐末尾维度不会删除已有维度。');
      return {
        outputs: {
          out: tensor(
            [...source.shape, ...Array<number>(extra).fill(1)],
            [...source.axes, ...Array<string>(extra).fill('singleton')],
            source.dtype,
          ),
        },
        notes: [
          `末尾补充 ${extra} 个单例轴；dtype 保持 ${source.dtype}，可广播性由后续逐元素操作校验。`,
        ],
      };
    },
  },
];
