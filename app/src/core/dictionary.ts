import type { DictionarySpec, Dim, TensorSpec } from './types';

export const DICTIONARY_SCHEMA_LIMIT = 64_000;
export const DICTIONARY_DTYPES = [
  'float32',
  'float16',
  'float64',
  'bfloat16',
  'int64',
  'int32',
  'int16',
  'int8',
  'uint8',
  'bool',
];
export const emptyDictionary = (): DictionarySpec => ({ kind: 'dict', fields: {} });

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);

export function checkDictionaryKey(key: string) {
  check(key.trim().length > 0 && key.length <= 128, '字段名需要 1–128 个字符。');
  check(!['__proto__', 'constructor', 'prototype'].includes(key), '该字段名为保留名称。');
}

function checkShape(value: unknown): asserts value is Dim[] {
  check(Array.isArray(value) && value.length <= 8, 'Tensor 形状需要 0–8 个维度，[] 表示标量。');
  check(
    value.every((dim) =>
      typeof dim === 'number'
        ? Number.isSafeInteger(dim) && dim >= 0
        : typeof dim === 'string' && /^[A-Za-z][A-Za-z0-9_]*$/.test(dim),
    ),
    '维度需要非负整数或符号，例如 B, A, T, 2。',
  );
}

export function dictionaryShape(text: string): Dim[] {
  const trimmed = text.trim();
  check(trimmed.length > 0, '请填写形状；标量填写 []。');
  const content =
    trimmed.startsWith('[') && trimmed.endsWith(']') ? trimmed.slice(1, -1).trim() : trimmed;
  const shape =
    content === ''
      ? []
      : content.split(',').map((s) => {
          const dim = s.trim();
          return /^\d+$/.test(dim) ? Number(dim) : dim;
        });
  checkShape(shape);
  return shape;
}

/** Describes an input contract, not tensor values or executable Python. */
export function parseDictionarySchema(text: unknown): DictionarySpec {
  check(
    typeof text === 'string' && text.length <= DICTIONARY_SCHEMA_LIMIT,
    `字典结构描述最多 ${DICTIONARY_SCHEMA_LIMIT} 个字符。`,
  );
  let source: unknown;
  try {
    source = JSON.parse(text);
  } catch {
    throw new Error('字典结构描述格式无效。');
  }
  let count = 0;
  const walk = (value: unknown, depth: number): DictionarySpec | TensorSpec => {
    check(++count <= 256 && depth <= 8, '字典最多 256 个条目、8 层嵌套。');
    check(record(value), '字段必须是字典或 Tensor 描述。');
    if (value.kind === 'dict') {
      check(
        Object.keys(value).every((key) => ['kind', 'fields'].includes(key)) && record(value.fields),
        '字典描述需要 kind="dict" 和 fields。',
      );
      const fields = Object.fromEntries(
        Object.entries(value.fields).map(([key, child]) => {
          checkDictionaryKey(key);
          return [key, walk(child, depth + 1)];
        }),
      );
      return { kind: 'dict', fields };
    }
    check(
      Object.keys(value).every((key) => ['shape', 'axes', 'dtype'].includes(key)),
      'Tensor 描述只包含 shape、dtype 和可选的 axes。',
    );
    checkShape(value.shape);
    check(
      typeof value.dtype === 'string' && DICTIONARY_DTYPES.includes(value.dtype),
      'Tensor dtype 无效。',
    );
    const axes = value.axes ?? value.shape.map((_, i) => `axis${i}`);
    check(
      Array.isArray(axes) &&
        axes.length === value.shape.length &&
        axes.every(
          (axis) => typeof axis === 'string' && axis.trim().length > 0 && axis.length <= 128,
        ),
      '轴名称数量需要与 Tensor 维度一致。',
    );
    return { shape: [...value.shape], axes: [...axes], dtype: value.dtype };
  };
  const result = walk(source, 0);
  check('kind' in result && result.kind === 'dict', '输入的根类型需要是字典。');
  return result;
}
