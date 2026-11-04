import type { Hyperparameter, Hyperparameters } from './types';

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
export function checkHyperparameterName(name: string) {
  check(
    /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name) &&
      !['__proto__', 'prototype', 'constructor', 'self'].includes(name),
    '超参数名称需要是 1–64 字符的标识符，且不能使用保留名称。',
  );
}
export function validateHyperparameters(value: unknown): asserts value is Hyperparameters {
  check(!!value && typeof value === 'object' && !Array.isArray(value), '模型超参数必须是对象。');
  check(Object.keys(value).length <= 128, '最多支持 128 个模型超参数。');
  for (const [name, raw] of Object.entries(value)) {
    checkHyperparameterName(name);
    check(!!raw && typeof raw === 'object' && !Array.isArray(raw), `${name} 的定义无效。`);
    const parameter = raw as Record<string, unknown>;
    check(
      Object.keys(parameter).every((key) => ['type', 'value', 'source'].includes(key)),
      `${name} 有未知字段。`,
    );
    const v = parameter.value;
    check(
      parameter.type === 'int'
        ? Number.isSafeInteger(v)
        : parameter.type === 'float'
          ? typeof v === 'number' && Number.isFinite(v)
          : parameter.type === 'bool'
            ? typeof v === 'boolean'
            : parameter.type === 'string'
              ? typeof v === 'string' && v.length <= 1999
              : false,
      `${name} 的值与类型不匹配，或超出范围。`,
    );
    check(
      parameter.source === undefined ||
        (typeof parameter.source === 'string' && parameter.source.length <= 256),
      `${name} 的来源说明过长。`,
    );
  }
}

export function parseHyperparameterValue(
  type: Hyperparameter['type'],
  text: string,
): Hyperparameter['value'] {
  if (type === 'string') return text;
  if (type === 'bool') {
    check(['true', 'false'].includes(text), '布尔值需要是 true 或 false。');
    return text === 'true';
  }
  check(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text.trim()), '请填写数值。');
  const value = Number(text);
  check(
    type === 'int' ? Number.isSafeInteger(value) : Number.isFinite(value),
    '数值与类型不匹配，或超出范围。',
  );
  return value;
}

/** Small arithmetic grammar. Never evaluates Python/JavaScript or arbitrary property access. */
export function resolveIntegerExpression(
  expression: unknown,
  hyperparameters: Hyperparameters = {},
): number {
  check(
    typeof expression === 'string' && expression.length > 0 && expression.length <= 256,
    '索引表达式需要 1–256 个字符。',
  );
  const tokens: string[] = [];
  const pattern =
    /\s*((?:self\.)?[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\/\/|[()+\-*/%])/y;
  let offset = 0;
  while (offset < expression.length) {
    if (!expression.slice(offset).trim()) break;
    pattern.lastIndex = offset;
    const match = pattern.exec(expression);
    check(match, `表达式中有不支持的内容：${expression.slice(offset, offset + 20)}`);
    tokens.push(match[1]);
    offset = pattern.lastIndex;
    check(tokens.length <= 128, '索引表达式过长。');
  }
  let cursor = 0;
  const finite = (v: number) => {
    check(
      Number.isFinite(v) && Math.abs(v) <= Number.MAX_SAFE_INTEGER,
      '表达式结果超出安全数值范围。',
    );
    return v;
  };
  function atom(depth: number): number {
    check(depth <= 16, '表达式嵌套最多 16 层。');
    const token = tokens[cursor++];
    if (token === '+' || token === '-') return (token === '-' ? -1 : 1) * atom(depth + 1);
    if (token === '(') {
      const value = sum(depth + 1);
      check(tokens[cursor++] === ')', '表达式缺少右括号。');
      return value;
    }
    check(token !== undefined, '表达式不完整。');
    if (/^[\d.]/.test(token)) return finite(Number(token));
    check(/^(self\.)?[A-Za-z_][A-Za-z0-9_]*$/.test(token), `无效的表达式符号：${token}`);
    const name = token.replace(/^self\./, '');
    check(Object.hasOwn(hyperparameters, name), `未定义超参数：${name}`);
    const parameter = hyperparameters[name];
    check(
      ['int', 'float'].includes(parameter.type) && typeof parameter.value === 'number',
      `超参数 ${name} 需要是数值类型。`,
    );
    return finite(parameter.value);
  }
  function product(depth: number): number {
    let value = atom(depth);
    while (['*', '/', '//', '%'].includes(tokens[cursor])) {
      const op = tokens[cursor++];
      const right = atom(depth);
      check(op === '*' || right !== 0, '表达式不能除以零。');
      value = finite(
        op === '*'
          ? value * right
          : op === '/'
            ? value / right
            : op === '//'
              ? Math.floor(value / right)
              : value - Math.floor(value / right) * right,
      );
    }
    return value;
  }
  function sum(depth: number): number {
    let value = product(depth);
    while (['+', '-'].includes(tokens[cursor])) {
      const op = tokens[cursor++];
      const right = product(depth);
      value = finite(op === '+' ? value + right : value - right);
    }
    return value;
  }
  const result = sum(0);
  check(cursor === tokens.length, '表达式包含多余符号；只支持数值、超参数、括号和基本算术。');
  check(Number.isSafeInteger(result), '索引表达式必须得到整数。');
  return result;
}
