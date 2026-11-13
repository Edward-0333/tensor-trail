import type { Dim, ModuleDefinition, Params, Port, ValueSpec } from '../core/types';
import { isIntegerSpec, tensorValue } from '../core/types';
import { product, requireThat, same, tensor } from './math';
import { input, output, text } from './primitives';

const floatDtypes = ['float16', 'bfloat16', 'float32', 'float64'];
const integerDtypes = ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64'];
const defaultViewShape = 'd0, d1, -1';
const placeholder = /^d[0-7]$/;

function integerValue(value: ValueSpec | undefined, port: string): Dim {
  requireThat(value && isIntegerSpec(value), `${port} 接口需要局部整数，不能使用 Tensor。`);
  requireThat(
    typeof value.value === 'number'
      ? Number.isSafeInteger(value.value)
      : typeof value.value === 'string' && value.value.trim().length > 0,
    `${port} 接口需要安全整数或符号尺寸。`,
  );
  return value.value;
}

function dimensionProduct(dimensions: Dim[]): Dim {
  // A known zero fixes the product even when other dimensions remain symbolic.
  if (dimensions.includes(0)) return 0;
  return product(dimensions);
}

function viewPattern(params: Params): (number | string)[] {
  requireThat(
    typeof params.shape === 'string' && params.shape.length <= 256,
    'View 的 shape 需要短文本，例如 d0, d1, -1。',
  );
  const tokens = params.shape.split(',').map((token) => token.trim());
  requireThat(
    tokens.length > 0 && tokens.length <= 8 && tokens.every(Boolean),
    'View 需要 1–8 个目标维度；暂不支持空形状的标量输出。',
  );
  const dimensions = tokens.map((token) => {
    if (placeholder.test(token)) return token;
    requireThat(
      /^(?:-1|\+?\d+)$/.test(token),
      'View 仅支持 d0–d7、非负整数和 -1；局部尺寸请通过接口连接。',
    );
    const dimension = Number(token);
    requireThat(Number.isSafeInteger(dimension), 'View 的目标尺寸超出安全整数范围。');
    return dimension;
  });
  requireThat(
    dimensions.filter((dimension) => dimension === -1).length <= 1,
    'View 最多允许一个 -1。',
  );
  return dimensions;
}

export function viewPorts(params: Params): { inputs: Port[]; outputs: Port[] } {
  // Retain recognized ports while a user is editing an incomplete expression.
  const names =
    typeof params.shape === 'string'
      ? params.shape
          .split(',')
          .map((token) => token.trim())
          .filter((token) => placeholder.test(token))
      : [];
  return {
    inputs: [...input, ...Array.from(new Set(names), (id) => ({ id, label: id }))],
    outputs: output,
  };
}

export function viewLabel(params: Params): string {
  return `view(${viewPattern(params).join(', ')})`;
}

export const featureMathModules: ModuleDefinition[] = [
  {
    type: 'NanToNum',
    label: 'torch.nan_to_num',
    category: '数学运算',
    description: '按 PyTorch 默认值逐元素替换 NaN 和正负无穷，保留形状和 dtype；不是 detach 或归一化。',
    fields: [],
    inputs: input,
    outputs: output,
    infer: ({ in: source }) => {
      const x = tensorValue(source);
      requireThat(
        [...floatDtypes, ...integerDtypes, 'complex64', 'complex128'].includes(x.dtype),
        `NanToNum 暂不支持 dtype=${x.dtype}。`,
      );
      return {
        outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) },
        notes: [
          'torch.nan_to_num(x)：NaN → 0；+inf / -inf → 对应浮点类型的最大 / 最小有限值。',
          '有限值保持不变；复数的实部和虚部分别处理，整数和 bool 无需替换。',
          '这里只推导结构，不执行数值替换，也不假定输入实际含有 NaN / inf；该操作不会主动切断梯度。',
        ],
      };
    },
  },
  {
    type: 'IntegerFloorDivide',
    label: '整数整除 a // b',
    category: '数学运算',
    description: '两个局部整数相除后向下取整，对应 Python a // b，例如 GC // G。',
    fields: [],
    inputs: [{ id: 'a', label: '被除数 a' }, { id: 'b', label: '除数 b' }],
    outputs: output,
    inferValue: ({ a, b }) => {
      const left = integerValue(a, 'a'), right = integerValue(b, 'b');
      requireThat(right !== 0, 'a // b 的除数不能为 0。');
      const numeric = typeof left === 'number' && typeof right === 'number';
      let value: Dim = `(${left}) // (${right})`;
      if (numeric) {
        // Avoid rounding a large integer quotient through floating-point division.
        const dividend = BigInt(left), divisor = BigInt(right);
        const correction = (left < 0) !== (right < 0) && dividend % divisor !== 0n ? 1n : 0n;
        value = Number(dividend / divisor - correction);
      }
      return {
        outputs: { out: { kind: 'int', value } },
        warnings: numeric ? [] : ['符号整除需验证除数非零，结果向下取整。'],
        notes: [`${left} // ${right} = ${value}；局部整数计算，不参与梯度。`],
      };
    },
  },
  {
    type: 'IntegerTruncDivide',
    label: 'int(a / b)',
    category: '数学运算',
    description: '两个局部整数相除后向零截断，对应 int(a / b)，不是 Tensor 运算或 floor 除法。',
    fields: [],
    inputs: [{ id: 'a', label: 'a' }, { id: 'b', label: 'b' }],
    outputs: output,
    inferValue: ({ a, b }) => {
      const left = integerValue(a, 'a'), right = integerValue(b, 'b');
      requireThat(right !== 0, 'int(a / b) 的除数不能为 0。');
      const numeric = typeof left === 'number' && typeof right === 'number';
      const value = numeric ? Math.trunc(left / right) || 0 : `int((${left})/(${right}))`;
      return {
        outputs: { out: { kind: 'int', value } },
        warnings: numeric ? [] : ['符号除法需验证除数非零，结果按 int 向零截断。'],
        notes: [`int(${left} / ${right}) = ${value}；读取尺寸并计算整数，不参与梯度。`],
      };
    },
  },
  ...['Cos', 'Sin'].map(
    (type): ModuleDefinition => ({
      type,
      label: type === 'Cos' ? '余弦 cos' : '正弦 sin',
      category: '数学运算',
      description: `对 Tensor 逐元素计算${type === 'Cos' ? '余弦' : '正弦'}，输入角度使用弧度。保留浮点类型；整数和 bool 输入输出 float32。`,
      fields: [],
      inputs: input,
      outputs: output,
      infer: ({ in: source }) => {
        const x = tensorValue(source);
        requireThat(
          floatDtypes.includes(x.dtype) || integerDtypes.includes(x.dtype),
          `${type} 不支持 dtype=${x.dtype}。`,
        );
        return {
          outputs: {
            out: tensor(
              [...x.shape],
              [...x.axes],
              floatDtypes.includes(x.dtype) ? x.dtype : 'float32',
            ),
          },
          notes: [`${type.toLowerCase()}(x)；角度使用弧度，保留形状和轴含义。`],
        };
      },
    }),
  ),
  {
    type: 'IntegerMultiply',
    label: '整数乘法',
    category: '数学运算',
    description: '计算两个局部整数的乘积，例如 bs × A；符号尺寸保留乘积表达式。',
    fields: [],
    inputs: [
      { id: 'a', label: 'a' },
      { id: 'b', label: 'b' },
    ],
    outputs: output,
    inferValue: ({ a, b }) => {
      const left = integerValue(a, 'a'),
        right = integerValue(b, 'b');
      const value =
        left === 0 || right === 0
          ? 0
          : left === 1
            ? right
            : right === 1
              ? left
              : dimensionProduct([left, right]);
      return {
        outputs: { out: { kind: 'int', value } },
        notes: [`${left} × ${right} = ${value}；输出为局部整数。`],
      };
    },
  },
  {
    type: 'View',
    label: 'View',
    category: '维度操作',
    description:
      '用局部整数接口指定目标形状，例如 view(bs × A, T, -1)。检查元素总数并推导一个 -1 维度。',
    fields: [
      {
        ...text('shape', '目标形状', defaultViewShape),
        maxLength: 256,
        help: 'd0–d7 从对应接口接收局部整数；可重复引用，也可填写非负整数或一个 -1。例如 d0, d1, -1。',
      },
    ],
    inputs: viewPorts({ shape: defaultViewShape }).inputs,
    outputs: output,
    getPorts: viewPorts,
    inferValue: (values, params) => {
      const source = tensorValue(values.in);
      const shape = viewPattern(params).map((dimension) =>
        typeof dimension === 'number' ? dimension : integerValue(values[dimension], dimension),
      );
      requireThat(
        shape.every((dimension) => typeof dimension !== 'number' || dimension >= -1),
        'View 的目标尺寸必须非负，或使用一个 -1 自动推导。',
      );
      requireThat(
        shape.filter((dimension) => dimension === -1).length <= 1,
        'View 最多允许一个 -1。',
      );
      requireThat(
        source.shape.every((dimension) =>
          typeof dimension === 'number'
            ? Number.isSafeInteger(dimension) && dimension >= 0
            : typeof dimension === 'string' && dimension.trim().length > 0,
        ),
        'View 的输入形状需要非负安全整数或符号尺寸。',
      );
      const warnings: string[] = [];
      if (shape.some((dimension) => typeof dimension === 'string'))
        warnings.push('View 的符号目标尺寸需在实际尺寸下验证为非负整数。');
      const total = dimensionProduct(source.shape);
      const index = shape.indexOf(-1);
      const known = dimensionProduct(shape.filter((dimension) => dimension !== -1));
      if (index >= 0) {
        requireThat(known !== 0, 'View 的零尺寸与 -1 同时出现时无法唯一推导，目标形状有歧义。');
        if (typeof total === 'number' && typeof known === 'number') {
          requireThat(total % known === 0, 'View 的元素总数无法整除指定的目标尺寸。');
          shape[index] = total / known;
        } else {
          shape[index] = `(${total})/(${known})`;
          warnings.push('符号尺寸下需验证已指定维度的乘积非零，且元素总数能够整除该乘积。');
        }
      } else same(total, known, 'View 前后元素总数必须相等', warnings);
      return {
        outputs: {
          out: tensor(
            shape,
            shape.map((_, axis) => `axis${axis}`),
            source.dtype,
          ),
        },
        warnings,
        notes: [
          viewLabel(params),
          '仅检查形状与元素总数；未记录 stride，不能保证原 Tensor 的内存布局允许 view。',
        ],
      };
    },
  },
];
