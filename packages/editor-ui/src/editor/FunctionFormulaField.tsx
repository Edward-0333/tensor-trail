import { useEffect, useState } from 'react';
import type { ModelNode } from '@tensor-trail/model-core/core/types';
import { parseFunctionFormula } from '@tensor-trail/model-core/core/functionFormula';
import { useEditor } from './store';

export function FunctionFormulaField({ node }: { node: ModelNode }) {
  const value = String(node.params.expression ?? '');
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [node.id, value]);
  let error = '';
  try {
    parseFunctionFormula(draft, node.params.argument);
  } catch (e) {
    error = (e as Error).message;
  }
  return (
    <div className="function-formula-field">
      <label>
        <span>数学公式 · 输出 =</span>
        <textarea
          aria-label="数学公式"
          value={draft}
          rows={4}
          maxLength={1000}
          spellCheck={false}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            if (draft !== value) useEditor.getState().setParam(node.id, 'expression', draft);
          }}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
              event.preventDefault();
              event.currentTarget.blur();
            }
          }}
        />
      </label>
      <p className="microcopy">
        输入端口对应变量 <code>{String(node.params.argument)}</code>；只填写等号右侧。离开输入框或按
        Ctrl+Enter 应用。
      </p>
      <p className="microcopy">
        支持数值、math.pi（π）、+ − * / % 和括号；% 按 Python/PyTorch 的取模规则计算。
      </p>
      <p className={error ? 'diagnostic error' : 'microcopy'} role="status">
        {error || '公式有效，逐元素计算并保持输入形状。'}
      </p>
    </div>
  );
}
