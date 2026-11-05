export type FormulaNode =
  | { kind: 'input' }
  | { kind: 'number'; value: number }
  | { kind: 'unary'; operator: '+' | '-'; operand: FormulaNode }
  | {
      kind: 'binary';
      operator: '+' | '-' | '*' | '/' | '%';
      left: FormulaNode;
      right: FormulaNode;
    };

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** A small elementwise grammar; formulas are data, never executable Python or JavaScript. */
export function parseFunctionFormula(expression: unknown, argument: unknown = 'x'): FormulaNode {
  check(
    typeof argument === 'string' &&
      /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(argument) &&
      !['math', 'pi', '__proto__', 'prototype', 'constructor'].includes(argument),
    '输入变量需为 1–64 字符的名称，不能使用 math、pi 等保留名称。',
  );
  check(
    typeof expression === 'string' && expression.trim().length > 0 && expression.length <= 1000,
    '请填写 1–1000 字符的数学表达式。',
  );
  const pattern =
    /\s*(math\.pi|[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[()+\-*/%])/y;
  const tokens: string[] = [];
  let offset = 0;
  while (offset < expression.length && expression.slice(offset).trim()) {
    pattern.lastIndex = offset;
    const match = pattern.exec(expression);
    check(match, `公式中有不支持的内容：${expression.slice(offset, offset + 20)}`);
    tokens.push(match[1]);
    offset = pattern.lastIndex;
    check(tokens.length <= 128, '公式最多支持 128 个符号。');
  }
  let cursor = 0,
    usesInput = false;
  function atom(depth: number): FormulaNode {
    check(depth <= 24, '公式嵌套最多 24 层。');
    const token = tokens[cursor++];
    check(token !== undefined, '数学公式不完整。');
    if (token === '+' || token === '-')
      return { kind: 'unary', operator: token, operand: atom(depth + 1) };
    if (token === '(') {
      const value = sum(depth + 1);
      check(tokens[cursor++] === ')', '公式缺少右括号。');
      return value;
    }
    if (token === argument) {
      usesInput = true;
      return { kind: 'input' };
    }
    if (token === 'math.pi' || token === 'pi') return { kind: 'number', value: Math.PI };
    if (/^(?:\d|\.)/.test(token)) {
      const value = Number(token);
      check(Number.isFinite(value), '公式中的数值必须有限。');
      return { kind: 'number', value };
    }
    throw new Error(`无法识别 ${token}；公式输入变量为 ${argument}。`);
  }
  function product(depth: number): FormulaNode {
    let value = atom(depth);
    while (['*', '/', '%'].includes(tokens[cursor])) {
      const operator = tokens[cursor++] as '*' | '/' | '%';
      value = { kind: 'binary', operator, left: value, right: atom(depth) };
    }
    return value;
  }
  function sum(depth: number): FormulaNode {
    let value = product(depth);
    while (['+', '-'].includes(tokens[cursor])) {
      const operator = tokens[cursor++] as '+' | '-';
      value = { kind: 'binary', operator, left: value, right: product(depth) };
    }
    return value;
  }
  const formula = sum(0);
  check(cursor === tokens.length, '公式包含多余内容；支持 +、-、*、/、%、括号与 math.pi。');
  check(usesInput, `公式需要引用输入变量 ${argument}。`);
  // Validate fully known constant subexpressions without needing any tensor values.
  evaluate(formula);
  return formula;
}

function evaluate(node: FormulaNode, input?: number): number | undefined {
  if (node.kind === 'input') return input;
  if (node.kind === 'number') return node.value;
  if (node.kind === 'unary') {
    const value = evaluate(node.operand, input);
    return value === undefined ? undefined : node.operator === '-' ? -value : value;
  }
  const left = evaluate(node.left, input),
    right = evaluate(node.right, input);
  check(!['/', '%'].includes(node.operator) || right !== 0, '公式不能除以零或对零取模。');
  if (left === undefined || right === undefined) return undefined;
  let value: number;
  switch (node.operator) {
    case '+':
      value = left + right;
      break;
    case '-':
      value = left - right;
      break;
    case '*':
      value = left * right;
      break;
    case '/':
      value = left / right;
      break;
    case '%': {
      // Python / torch.remainder use the divisor's sign, unlike JavaScript's %.
      value = left % right;
      if (value !== 0 && value < 0 !== right < 0) value += right;
      break;
    }
  }
  check(Number.isFinite(value), '公式计算结果超出有限数值范围。');
  return value;
}

/** Scalar reference for validating the formula's math; the canvas still infers metadata only. */
export function evaluateFunctionFormula(formula: FormulaNode, input: number): number {
  check(Number.isFinite(input), '输入必须是有限数值。');
  return evaluate(formula, input)!;
}
