import type { ModelNode } from '@tensor-trail/model-core/core/types';
import {
  parseCalculations,
  defaultSlice,
  calculationLabel,
  type Calculation,
} from '@tensor-trail/model-core/modules/calculation';
import { useEditor } from './store';

export function CalculationEditor({ node }: { node: ModelNode }) {
  let ops: Calculation[] = [];
  try {
    ops = parseCalculations(node.params.operations);
  } catch {
    /* Offer a fresh valid operation below. */
  }
  const commit = (next: Calculation[]) => {
    try {
      const text = JSON.stringify(next);
      parseCalculations(text);
      useEditor.getState().setParam(node.id, 'operations', text);
    } catch (error) {
      useEditor.getState().toast((error as Error).message);
    }
  };
  const update = (index: number, value: Calculation) =>
    commit(ops.map((op, i) => (i === index ? value : op)));
  const move = (index: number, by: number) => {
    const next = [...ops];
    [next[index], next[index + by]] = [next[index + by], next[index]];
    commit(next);
  };
  return (
    <div className="calculation-editor">
      <p className="microcopy">从上到下依次计算，仅推导形状和类型。</p>
      {!ops.length && <p className="diagnostic error">步骤格式无效，请添加一个运算以重新设置。</p>}
      {ops.map((op, index) => (
        <section
          className="calculation-step-editor"
          key={`${node.id}-${index}-${JSON.stringify(op)}`}
        >
          <div className="calculation-step-title">
            <strong>
              {index + 1}. {calculationLabel(op)}
            </strong>
            <button
              aria-label={`上移第 ${index + 1} 步`}
              disabled={index === 0}
              onClick={() => move(index, -1)}
            >
              ↑
            </button>
            <button
              aria-label={`下移第 ${index + 1} 步`}
              disabled={index === ops.length - 1}
              onClick={() => move(index, 1)}
            >
              ↓
            </button>
            <button
              aria-label={`删除第 ${index + 1} 步`}
              disabled={ops.length === 1}
              onClick={() => commit(ops.filter((_, i) => i !== index))}
            >
              ×
            </button>
          </div>
          {op.op !== 'invert' && (
            <label>
              轴 {op.op === 'any' && '（留空表示全部）'}
              <input
                aria-label={`第 ${index + 1} 步的轴`}
                type="number"
                defaultValue={op.dim ?? ''}
                key={String(op.dim)}
                onBlur={(e) => {
                  const v = e.target.value;
                  if (op.op === 'any') update(index, { ...op, dim: v === '' ? null : Number(v) });
                  else update(index, { ...op, dim: v === '' ? -1 : Number(v) });
                }}
              />
            </label>
          )}
          {op.op === 'any' && (
            <label className="calculation-checkbox">
              <input
                aria-label={`第 ${index + 1} 步保留维度`}
                type="checkbox"
                checked={op.keepdim}
                onChange={(e) => update(index, { ...op, keepdim: e.target.checked })}
              />
              保留被归约的轴
            </label>
          )}
          {op.op === 'slice' && (
            <div className="slice-fields">
              {(['start', 'stop'] as const).map((key) => (
                <label key={key}>
                  {key === 'start' ? '开始' : '结束（不含）'}
                  <input
                    aria-label={`第 ${index + 1} 步${key === 'start' ? '开始' : '结束'}`}
                    type="number"
                    defaultValue={op[key]}
                    onBlur={(e) => update(index, { ...op, [key]: e.target.value })}
                  />
                </label>
              ))}
              <label>
                步长
                <input
                  aria-label={`第 ${index + 1} 步步长`}
                  type="number"
                  min="1"
                  defaultValue={op.step}
                  onBlur={(e) =>
                    update(index, {
                      ...op,
                      step: Math.max(1, Math.trunc(Number(e.target.value)) || 1),
                    })
                  }
                />
              </label>
            </div>
          )}
        </section>
      ))}
      <select
        aria-label="添加计算运算"
        value=""
        disabled={ops.length >= 16}
        onChange={(e) => {
          const value: Calculation =
            e.target.value === 'invert'
              ? { op: 'invert' }
              : e.target.value === 'any'
                ? { op: 'any', dim: null, keepdim: false }
                : { ...defaultSlice };
          commit([...ops, value]);
        }}
      >
        <option value="" disabled>
          ＋ 添加运算
        </option>
        <option value="slice">切片 [..., :2]</option>
        <option value="invert">取反 ~（bool / 整数）</option>
        <option value="any">any() 归约</option>
      </select>
    </div>
  );
}
