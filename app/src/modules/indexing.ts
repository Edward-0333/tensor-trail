import type { ModuleDefinition, ValueSpec } from '../core/types';
import { isDictionarySpec, clone, shapeText } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { checkDictionaryKey } from '../core/dictionary';
import { text, input, output } from './primitives';
import { requireThat, axisIndex, tensor } from './math';

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
    label: '张量整数索引',
    category: '索引与切片',
    description: '沿一个轴取出单个元素并移除该轴；支持负索引和 history_steps - 1 等超参数表达式。',
    inputs: input,
    outputs: output,
    fields: [
      { ...text('dim', '索引轴', '0'), maxLength: 256 },
      { ...text('index', '索引值', '0'), maxLength: 256 },
    ],
    infer: ({ in: x }, params, context) => {
      const dim = resolveIntegerExpression(params.dim, context?.hyperparameters);
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
    },
  },
];
