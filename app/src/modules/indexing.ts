import type { Field, ModuleDefinition, Params, ValueSpec } from '../core/types';
import { isDictionarySpec, isIntegerSpec, tensorValue, clone, shapeText } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { checkDictionaryKey } from '../core/dictionary';
import { text, bool, input, output } from './primitives';
import { requireThat, axisIndex, tensor } from './math';
import { tensorIndexDefaults } from '../core/indexingTypes';
import { calculate } from './calculation';

const selectionFields: Field[] = [
  {
    key: 'mode',
    label: '取值方式',
    kind: 'select',
    default: 'index',
    options: ['index', 'slice'],
    help: 'index：取单个元素并移除轴；slice：取范围并保留轴。',
  },
  { ...text('dim', '操作轴', String(tensorIndexDefaults.dim)), maxLength: 256 },
  {
    ...text('index', '索引值', String(tensorIndexDefaults.index)),
    maxLength: 256,
    help: '整数索引模式使用；支持超参数表达式。',
  },
  {
    ...text('start', '开始（留空从头）', String(tensorIndexDefaults.start)),
    maxLength: 256,
    help: '切片模式使用；包含开始位置。',
  },
  {
    ...text('stop', '结束（不含，留空到末尾）', String(tensorIndexDefaults.stop)),
    maxLength: 256,
    help: '切片模式使用；不包含结束位置。',
  },
  {
    ...bool('stopFromInput', '结束位置来自连线', false),
    help: '开启后从 stop 接口接收局部整数，例如 T；关闭前请先断开该连线。',
  },
  {
    ...text('stopLabel', '结束端口名称', 'stop'),
    help: '画布中显示的参数名称，例如 T。数值由 stop 连线决定。',
  },
  {
    key: 'step',
    label: '步长',
    kind: 'number',
    default: tensorIndexDefaults.step,
    min: 1,
    help: '切片模式使用；必须为正整数。',
  },
];

/** Python notation remains compact and preserves the user's parameter expressions. */
export function tensorIndexLabel(params: Params): string {
  const dim = String(params.dim ?? '').trim();
  const index = String(params.index ?? '').trim();
  const slice = params.mode === 'slice';
  if (!dim || (!slice && !index)) return '索引参数待填写';
  const stop = params.stopFromInput === true ? params.stopLabel || 'stop' : (params.stop ?? '');
  const selection = slice
    ? `${params.start ?? ''}:${stop}${params.step === 1 ? '' : `:${params.step}`}`
    : index;
  if (!/^[+-]?\d+$/.test(dim) || Math.abs(Number(dim)) > 8) return `[轴 ${dim}: ${selection}]`;
  const axis = Number(dim);
  const entries =
    axis < 0
      ? ['...', selection, ...Array(-axis - 1).fill(':')]
      : [...Array(axis).fill(':'), selection, '...'];
  return `[${entries.join(', ')}]`;
}

export function parseFieldPath(value: unknown): string[] {
  requireThat(typeof value === 'string' && value.length <= 1999, '请填写字段路径。');
  const trimmed = value.trim();
  let path: unknown;
  if (trimmed.startsWith('[')) {
    try {
      path = JSON.parse(trimmed);
    } catch {
      throw new Error('字段路径数组格式无效，例如 ["agent", "position"]。');
    }
  } else path = trimmed.split('.').map((key) => key.trim());
  requireThat(
    Array.isArray(path) &&
      path.length > 0 &&
      path.length <= 8 &&
      path.every((key) => typeof key === 'string'),
    '字段路径需要 1–8 个键。',
  );
  path.forEach(checkDictionaryKey);
  return path as string[];
}

const inferTensorSelection: NonNullable<ModuleDefinition['infer']> = (
  { in: x },
  params,
  context,
) => {
  const mode = params.mode ?? 'index';
  requireThat(mode === 'index' || mode === 'slice', '取值方式需要是整数索引或切片。');
  const dim = resolveIntegerExpression(params.dim, context?.hyperparameters);
  if (mode === 'slice') {
    const resolveBound = (value: unknown) =>
      value === '' ? '' : String(resolveIntegerExpression(value, context?.hyperparameters));
    requireThat(
      Number.isSafeInteger(params.step) && Number(params.step) > 0,
      '切片步长必须为正整数。',
    );
    const out = calculate(x, {
      op: 'slice',
      dim,
      start: resolveBound(params.start),
      stop: resolveBound(params.stop),
      step: Number(params.step),
    });
    return {
      outputs: { out },
      notes: [tensorIndexLabel(params), '切片保留操作轴，结束位置不包含在结果中。'],
      warnings: out.shape.some((d) => typeof d === 'string' && d.startsWith('slice('))
        ? ['符号维度的切片长度需要具体尺寸验证。']
        : [],
    };
  }
  const index = resolveIntegerExpression(params.index, context?.hyperparameters);
  const axis = axisIndex(dim, x.shape.length);
  const size = x.shape[axis];
  const warnings: string[] = [];
  if (typeof size === 'number') {
    requireThat(
      index >= -size && index < size,
      `索引 ${index} 超出轴 ${axis} 的范围（尺寸 ${size}）。`,
    );
  } else warnings.push(`轴 ${axis} 的尺寸为 ${size}，索引 ${index} 的边界需用实际尺寸验证。`);
  return {
    outputs: {
      out: tensor(
        x.shape.filter((_, i) => i !== axis),
        x.axes.filter((_, i) => i !== axis),
        x.dtype,
      ),
    },
    notes: [
      `索引轴：${params.dim} = ${dim}；索引值：${params.index} = ${index}。`,
      `取出轴 ${axis} 上的单个元素，移除该轴。`,
    ],
    warnings,
  };
};

export const indexingModules: ModuleDefinition[] = [
  {
    type: 'DictionaryGet',
    label: '字典字段读取',
    category: '索引与切片',
    description: '从字典中读取嵌套字段，例如 agent.position；输出可以是 Tensor 或子字典。',
    inputs: input,
    outputs: output,
    fields: [text('path', '字段路径', '')],
    inferValue: ({ in: source }, params) => {
      const path = parseFieldPath(params.path);
      let value: ValueSpec = source;
      for (const key of path) {
        requireThat(
          value && isDictionarySpec(value),
          `读取 ${key} 时需要字典，当前为${shapeText(value)}。`,
        );
        requireThat(Object.hasOwn(value.fields, key), `字典中不存在字段：${key}`);
        value = value.fields[key];
      }
      return {
        outputs: { out: clone(value) },
        notes: [
          `读取 ${path.map((key) => `[${JSON.stringify(key)}]`).join('')} → ${shapeText(value)}`,
        ],
      };
    },
  },
  {
    type: 'TensorIndex',
    label: '索引与切片',
    category: '索引与切片',
    description:
      '沿一个轴取值：整数索引 Index 移除该轴，切片 Slice 保留该轴。支持负轴、负索引及 history_steps - 1 等超参数表达式。',
    inputs: input,
    outputs: output,
    fields: selectionFields,
    getFields: (params) =>
      selectionFields.filter(
        (f) =>
          ['mode', 'dim'].includes(f.key) ||
          (params.mode === 'slice'
            ? [
                'start',
                'step',
                'stopFromInput',
                params.stopFromInput === true ? 'stopLabel' : 'stop',
              ].includes(f.key)
            : f.key === 'index'),
      ),
    getPorts: (params) => ({
      inputs:
        params.mode === 'slice' && params.stopFromInput === true
          ? [...input, { id: 'stop', label: String(params.stopLabel || 'stop') }]
          : input,
      outputs: output,
    }),
    infer: (inputs, params, context) => {
      requireThat(params.stopFromInput !== true, '结束位置来自连线时需要 stop 整数输入。');
      return inferTensorSelection(inputs, params, context);
    },
    inferValue: (values, params, context) => {
      const x = tensorValue(values.in);
      if (params.mode !== 'slice' || params.stopFromInput !== true)
        return inferTensorSelection({ in: x }, params, context);
      const stop = values.stop;
      requireThat(stop && isIntegerSpec(stop), '切片 stop 接口需要局部整数，例如 T。');
      requireThat(
        typeof stop.value === 'number' && Number.isSafeInteger(stop.value),
        '切片 stop 需要已确定的安全整数，符号尺寸暂不能作为结束边界。',
      );
      const result = inferTensorSelection(
        { in: x },
        { ...params, stop: String(stop.value) },
        context,
      );
      return {
        ...result,
        notes: [
          ...(result.notes ?? []),
          `结束边界来自 ${params.stopLabel || 'stop'}：${stop.value}`,
        ],
      };
    },
  },
];
