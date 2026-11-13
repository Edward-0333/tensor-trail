import type { BooleanSpec, Dim, ModelNode, ModuleDefinition, Params, ValueSpec } from '../core/types';
import {
  clone,
  isDictionarySpec,
  isIntegerListSpec,
  isTensorSequenceSpec,
  isTensorSpec,
  tensorValue,
} from '../core/types';
import { checkDictionaryKey } from '../core/dictionary';
import { calculate } from './calculation';
import { loopInteger } from './loopOps';
import { axisIndex, broadcast, product, requireThat, same, tensor } from './math';
import { bool, input, num, output, select, text } from './primitives';
import { requireRealTensor } from './sorting';

const named = (...names: string[]) => names.map((id) => ({ id, label: id }));
const noGrad = () => ({ requiresGrad: false, source: 'inferred' as const });
const comparisons = { lt: '<', le: '<=', eq: '==', ge: '>=', gt: '>' } as const;

function comparison(value: unknown, equality = false): keyof typeof comparisons {
  requireThat(
    typeof value === 'string' && Object.hasOwn(comparisons, value) && (equality || value !== 'eq'),
    `operator must be ${equality ? 'lt/le/eq/ge/gt' : 'ge/le/lt/gt'}.`,
  );
  return value as keyof typeof comparisons;
}

function dimension(params: Params, rank: number): number {
  requireThat(typeof params.dim === 'number' && Number.isSafeInteger(params.dim), 'dim must be an integer.');
  return axisIndex(params.dim, rank);
}

function validSize(value: Dim) {
  requireThat(
    typeof value === 'number'
      ? Number.isSafeInteger(value) && value >= 0
      : typeof value === 'string' && value.trim().length > 0,
    'Tensor dimensions must be nonnegative safe integers or nonempty symbols.',
  );
}

function unflattenSizes(value: unknown): Dim[] {
  requireThat(typeof value === 'string' && value.length <= 256, 'sizes must be short comma-separated text.');
  const sizes = value.split(',').map((entry): Dim => {
    const token = entry.trim();
    if (/^(?:-1|\+?\d+)$/.test(token)) {
      const size = Number(token);
      requireThat(Number.isSafeInteger(size), 'sizes exceed the safe integer range.');
      return size;
    }
    requireThat(/^[A-Za-z][A-Za-z0-9_]*$/.test(token), 'sizes require nonnegative integers, symbols or one -1.');
    return token;
  });
  requireThat(sizes.filter((size) => size === -1).length <= 1, 'sizes allow only one -1.');
  return sizes;
}

function sizeProduct(sizes: Dim[]): Dim {
  if (sizes.includes(0)) return 0;
  const factors = sizes.filter((size) => size !== 1);
  return factors.length === 1 ? factors[0] : product(factors);
}

function scalarNumber(value: unknown): string {
  requireThat(typeof value === 'string', 'value must be numeric text.');
  const literal = value.trim();
  requireThat(
    /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(literal) && Number.isFinite(Number(literal)),
    'value must be a finite decimal number.',
  );
  // Normalize integer leading zeros for valid Python while retaining integer precision.
  if (/^[+-]?\d+$/.test(literal)) return BigInt(literal).toString();
  return literal;
}

function booleanValue(value: ValueSpec | undefined, port: string): boolean {
  requireThat(
    value && !isTensorSpec(value) && value.kind === 'bool' && typeof value.value === 'boolean',
    `${port} requires a known Python BooleanSpec; Tensor contents and ControlSpec do not establish a Boolean value.`,
  );
  return value.value;
}

const pythonBool = (value: boolean): BooleanSpec => ({ kind: 'bool', value });

export function forecastMathCode(node: ModelNode): string | undefined {
  const p = node.params;
  switch (node.type) {
    case 'Unflatten':
      return `out = x.unflatten(${p.dim}, (${unflattenSizes(p.sizes).join(', ')},))\n# Split only dim; the product of sizes must equal x.shape[dim].`;
    case 'TensorCompareScalar':
      return `out = x ${comparisons[comparison(p.operator)]} ${scalarNumber(p.value)}\n# Elementwise bool Tensor, requires_grad=False; values are not evaluated here.`;
    case 'LogicalAnd':
      return 'out = torch.logical_and(a, b)\n# Broadcast bool Tensors; not Python short-circuit control flow.';
    case 'MatrixInverse':
      return 'out = torch.linalg.inv(x)\n# Static shape/dtype inference only; invertibility is not checked.';
    case 'TensorToDevice':
      return `out = x.to(device=${JSON.stringify(p.target)})\n# Device transfer only, not detach or a dtype cast; no actual transfer is performed here.`;
    case 'DictionaryUpdate':
      return 'base.update(updates)\nout = base\n# out describes the updated dictionary, not update()\'s None return. Nested fields are replaced, not merged.';
    case 'SequenceLength':
      return 'out = len(x)\n# Python integer: number of list/tuple items, not Tensor elements.';
    case 'SelectValue':
      return 'out = true_value if condition else false_value\n# Requires a known Python bool; selects the entire value, not torch.where.\n# This node does not implement lazy execution of upstream graph nodes.';
    case 'DynamicSlice':
      return `selection = [slice(None)] * x.ndim\nselection[${p.dim}] = slice(start, stop, 1)\nout = x[tuple(selection)]\n# Python integer bounds; negative indices and clipping apply. Symbolic bounds do not imply length 1.`;
    case 'BooleanParameter':
      return `out = ${p.value === true ? 'True' : 'False'}\n# Declared Python bool, not an nn.Parameter or an observed model flag.`;
    case 'BooleanAnd':
      return 'out = a and b\n# Both inputs must be known Python bools; this graph does not short-circuit upstream nodes.';
    case 'BooleanNot':
      return 'out = not x\n# Python bool, not elementwise Tensor inversion.';
    case 'IntegerCompare':
      return `out = a ${comparisons[comparison(p.operator, true)]} b\n# Known Python integers only; symbolic inputs cannot establish a Boolean result.`;
  }
}

export const forecastMath: ModuleDefinition[] = [
  {
    type: 'Unflatten', label: 'Unflatten', category: '维度操作',
    description: 'Split one axis into sizes, preserving every other dimension and axis label.',
    fields: [num('dim', 'dim', -1, -Number.MAX_SAFE_INTEGER), text('sizes', 'sizes', '-1,3')],
    inputs: input, outputs: output,
    infer: ({ in: value }, p) => {
      const x = tensorValue(value), dim = dimension(p, x.shape.length);
      x.shape.forEach(validSize);
      const sizes = unflattenSizes(p.sizes), warnings: string[] = [];
      const total = x.shape[dim], index = sizes.indexOf(-1);
      const known = sizeProduct(sizes.filter((size) => size !== -1));
      if (sizes.some((size) => typeof size === 'string'))
        warnings.push('Symbolic sizes must resolve to nonnegative integers at runtime.');
      if (index >= 0) {
        requireThat(known !== 0, 'Unflatten cannot infer -1 with a zero product: the size is ambiguous.');
        if (typeof total === 'number' && typeof known === 'number') {
          requireThat(total % known === 0, 'Unflatten axis size must be divisible by the specified sizes.');
          sizes[index] = total / known;
        } else {
          sizes[index] = known === 1 ? total : `(${total})/(${known})`;
          if (known !== 1)
            warnings.push('Verify that the specified sizes have a nonzero product and divide the original axis size exactly.');
        }
      } else same(total, known, 'Unflatten sizes must multiply to the original axis size', warnings);
      const shape = [...x.shape], axes = [...x.axes];
      shape.splice(dim, 1, ...sizes);
      axes.splice(dim, 1, ...sizes.map((_, i) => sizes.length === 1 ? x.axes[dim] : `${x.axes[dim] ?? `axis${dim}`}_${i}`));
      return { outputs: { out: tensor(shape, axes, x.dtype) }, warnings,
        notes: ['Only the selected axis is split; -1 is inferred from that axis, not the total Tensor element count.'] };
    },
  },
  {
    type: 'TensorCompareScalar', label: 'Tensor Compare Scalar', category: '逻辑与归约',
    description: 'Elementwise real Tensor comparison against a numeric scalar, returning a bool Tensor.',
    fields: [select('operator', 'operator', 'ge', ['ge', 'le', 'lt', 'gt']), text('value', 'value', '0')],
    inputs: input, outputs: output,
    infer: ({ in: value }, p) => {
      const x = tensorValue(value);
      if (x.dtype !== 'bool') requireRealTensor(x);
      comparison(p.operator);
      scalarNumber(p.value);
      return { outputs: { out: { ...tensor([...x.shape], [...x.axes], 'bool'), autograd: noGrad() } },
        notes: ['Only shape and dtype are inferred; the comparison result is not a known Python bool.'] };
    },
  },
  {
    type: 'LogicalAnd', label: 'Logical And', category: '逻辑与归约',
    description: 'Broadcast two bool Tensors and compute their elementwise conjunction.',
    fields: [], inputs: named('a', 'b'), outputs: output,
    infer: ({ a, b }) => {
      const left = tensorValue(a), right = tensorValue(b), warnings: string[] = [];
      requireThat(left.dtype === 'bool' && right.dtype === 'bool', 'LogicalAnd requires two bool Tensors.');
      return { outputs: { out: { ...broadcast(left, right, warnings), autograd: noGrad() } }, warnings };
    },
  },
  {
    type: 'MatrixInverse', label: 'Matrix Inverse', category: '数学运算',
    description: 'Invert square matrices on the last two axes; retain batch dimensions and float32/float64 dtype.',
    fields: [], inputs: input, outputs: output,
    infer: ({ in: value }) => {
      const x = tensorValue(value), warnings: string[] = [];
      requireThat(['float32', 'float64'].includes(x.dtype), 'MatrixInverse requires float32 or float64; no implicit cast is performed.');
      requireThat(x.shape.length >= 2, 'MatrixInverse requires at least two axes.');
      x.shape.forEach(validSize);
      same(x.shape.at(-2)!, x.shape.at(-1)!, 'MatrixInverse requires square last two axes', warnings);
      return { outputs: { out: tensor([...x.shape], [...x.axes], x.dtype) }, warnings,
        notes: ['Static inference only: invertibility is not checked and no inverse is computed.'] };
    },
  },
  {
    type: 'TensorToDevice', label: 'Tensor To Device', category: '维度操作',
    description: 'Record an explicit CPU/CUDA transfer without changing values, shape, dtype or gradient state.',
    fields: [select('target', 'target', 'cpu', ['cpu', 'cuda'])], inputs: input, outputs: output,
    infer: ({ in: value }, p) => {
      const x = tensorValue(value);
      requireThat(p.target === 'cpu' || p.target === 'cuda', 'target must be cpu or cuda.');
      return { outputs: { out: clone(x) }, notes: [
        `Requested device transfer to ${p.target}; no transfer is executed or device availability observed.`,
        'TensorSpec has no device-location field. The target is recorded in node parameters, not fabricated as observed metadata.',
        'No dtype conversion or detach; input values and gradient state are preserved.',
      ] };
    },
  },
  {
    type: 'DictionaryUpdate', label: 'Dictionary Update', category: '输入输出',
    description: 'Shallow dictionary update: replace matching fields while preserving complete field specifications.',
    fields: [], inputs: named('base', 'updates'), outputs: output,
    inferValue: ({ base, updates }) => {
      requireThat(base && isDictionarySpec(base), 'base requires a DictionarySpec with known fields.');
      requireThat(updates && isDictionarySpec(updates), 'updates requires a DictionarySpec with known fields.');
      for (const dict of [base, updates]) {
        requireThat(dict.keyType === undefined || dict.keyType === 'str' || dict.keyType === 'int', 'Invalid dictionary keyType.');
        Object.keys(dict.fields).forEach((key) => checkDictionaryKey(key, dict.keyType));
      }
      const hasBase = Object.keys(base.fields).length > 0, hasUpdates = Object.keys(updates.fields).length > 0;
      requireThat(!hasBase || !hasUpdates || (base.keyType ?? 'str') === (updates.keyType ?? 'str'),
        'Mixed str/int dictionary keys cannot be represented by DictionarySpec.');
      const keyType = hasBase || !hasUpdates ? base.keyType : updates.keyType;
      return { outputs: { out: {
        kind: 'dict', ...(keyType === undefined ? {} : { keyType }),
        fields: clone({ ...base.fields, ...updates.fields }),
      } }, notes: [
        'Shallow update: nested values are replaced as whole fields, not recursively merged.',
        'out is the updated dictionary state, not the None returned by dict.update(). Only specs are cloned to isolate editing.',
      ] };
    },
  },
  {
    type: 'SequenceLength', label: 'Sequence Length', category: '形状查询',
    description: 'Read the number of known list/tuple items, producing a Python integer.',
    fields: [], inputs: input, outputs: output,
    inferValue: ({ in: value }) => {
      requireThat(value && (isTensorSequenceSpec(value) || isIntegerListSpec(value)),
        'SequenceLength requires a list/tuple with known items; opaque containers do not establish a length.');
      const length = isTensorSequenceSpec(value) ? value.items.length : value.values.length;
      return { outputs: { out: { kind: 'int', value: length } },
        notes: ['Counts sequence entries, not Tensor elements; an empty sequence has length zero.'] };
    },
  },
  {
    type: 'SelectValue', label: 'Select Value', category: '分支运算',
    description: 'Select the full specification of one value using a known Python bool; unknown conditions are errors.',
    fields: [], inputs: named('condition', 'true_value', 'false_value'), outputs: output,
    inferValue: ({ condition, true_value, false_value }) => {
      const chooseTrue = booleanValue(condition, 'condition');
      requireThat(true_value && false_value, 'SelectValue requires both true_value and false_value inputs.');
      return { outputs: { out: clone(chooseTrue ? true_value : false_value) },
        notes: ['Selects only the chosen value spec; no branch merging, broadcasting or assumed Tensor contents.',
          'Both upstream graph inputs must be available; this node does not implement lazy branch execution.'] };
    },
  },
  {
    type: 'DynamicSlice', label: 'Dynamic Slice', category: '索引与切片',
    description: 'Slice one axis with Python integer start/stop inputs and a fixed step of one.',
    fields: [num('dim', 'dim', 1, -Number.MAX_SAFE_INTEGER)], inputs: named('in', 'start', 'stop'), outputs: output,
    inferValue: ({ in: value, start, stop }, p) => {
      const x = tensorValue(value), dim = dimension(p, x.shape.length);
      x.shape.forEach(validSize);
      const first = loopInteger(start), last = loopInteger(stop), size = x.shape[dim];
      let out;
      if (typeof first === 'number' && typeof last === 'number') {
        out = calculate(x, { op: 'slice', dim, start: String(first), stop: String(last), step: 1 });
      } else {
        const shape = [...x.shape];
        shape[dim] = `slice(${size},${first}:${last}:1)`;
        out = tensor(shape, [...x.axes], x.dtype);
      }
      if (size === 0 || first === last || (typeof first === 'number' && typeof last === 'number' && first >= last && (first < 0) === (last < 0)))
        out.shape[dim] = 0;
      return { outputs: { out },
        warnings: typeof out.shape[dim] === 'string'
          ? ['Symbolic slice length is unresolved: normalize negative bounds, clip to the axis, then compute max(stop - start, 0).'] : [],
        notes: ['The selected axis is retained, including length-zero slices. All other dimensions, labels and dtype are unchanged.',
          'Bounds are recomputed from current input specs; interval:interval+1 is not assumed to have length one.'] };
    },
  },
  {
    type: 'BooleanParameter', label: 'Boolean Parameter', category: '输入输出',
    description: 'Declare a Python bool configuration value, not a Tensor or an nn.Parameter.',
    fields: [bool('value', 'value', false)], inputs: [], outputs: output,
    inferValue: (_, p) => {
      requireThat(typeof p.value === 'boolean', 'value must be a Boolean.');
      return { outputs: { out: pythonBool(p.value) }, notes: ['Declared configuration, not an observed runtime model flag.'] };
    },
  },
  {
    type: 'BooleanAnd', label: 'Python Boolean And', category: '逻辑与归约',
    description: 'Conjunction of two known Python bools; no Tensor broadcasting or upstream short-circuiting.',
    fields: [], inputs: named('a', 'b'), outputs: output,
    inferValue: ({ a, b }) => {
      const left = booleanValue(a, 'a'), right = booleanValue(b, 'b');
      return { outputs: { out: pythonBool(left && right) } };
    },
  },
  {
    type: 'BooleanNot', label: 'Python Boolean Not', category: '逻辑与归约',
    description: 'Negate a known Python bool, not a Tensor.',
    fields: [], inputs: input, outputs: output,
    inferValue: ({ in: value }) => ({ outputs: { out: pythonBool(!booleanValue(value, 'in')) } }),
  },
  {
    type: 'IntegerCompare', label: 'Integer Compare', category: '逻辑与归约',
    description: 'Compare concrete Python integers; symbolic integers cannot establish a bool and are rejected.',
    fields: [select('operator', 'operator', 'lt', ['lt', 'le', 'eq', 'ge', 'gt'])], inputs: named('a', 'b'), outputs: output,
    inferValue: ({ a, b }, p) => {
      const left = loopInteger(a), right = loopInteger(b), op = comparison(p.operator, true);
      requireThat(typeof left === 'number' && typeof right === 'number',
        'IntegerCompare needs concrete integers; symbolic values cannot establish a known Boolean result.');
      const value = op === 'lt' ? left < right : op === 'le' ? left <= right
        : op === 'eq' ? left === right : op === 'ge' ? left >= right : left > right;
      return { outputs: { out: pythonBool(value) } };
    },
  },
];
