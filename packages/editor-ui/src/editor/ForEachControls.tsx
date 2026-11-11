import type { NodeResult } from '@tensor-trail/model-core/core/types';
import { isTensorSequenceSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { loopIteration } from '@tensor-trail/model-core/core/forEach';

export function ForEachControls({
  result,
  selected,
  onSelect,
}: {
  result?: NodeResult;
  selected?: number;
  onSelect: (index: number) => void;
}) {
  const iterations = result?.iterations;
  if (!iterations)
    return <div className="foreach-controls">循环输入待连接或修正；尚无迭代可查看。</div>;
  if (!iterations.length)
    return (
      <div className="foreach-controls">
        循环次数：0。输入为空，跳过循环体并保留初始列表；下方未执行。仅静态推导。
      </div>
    );
  const index = loopIteration(result, selected);
  const initial = result!.inputs.initial;
  const prefix = initial && isTensorSequenceSpec(initial) ? initial.items.length : 0;
  const input = result!.inputs.items;
  const item = input && isTensorSequenceSpec(input) ? input.items[index] : undefined;
  const output = iterations[index].outputs.out;
  return (
    <div className="foreach-controls" data-testid="foreach-controls">
      <strong>循环次数：{iterations.length}</strong>
      <label>
        当前迭代
        <select
          aria-label="当前迭代"
          value={index}
          onChange={(event) => onSelect(Number(event.target.value))}
        >
          {iterations.map((_, i) => (
            <option key={i} value={i}>
              {i}（第 {i + 1}/{iterations.length} 次）
            </option>
          ))}
        </select>
      </label>
      <code>
        {shapeText(item)} → {shapeText(output)}
      </code>
      <span>
        {result?.status === 'error'
          ? '循环存在错误，尚无有效的完整收集结果'
          : output
            ? `列表元素数：${prefix + index} → ${prefix + index + 1}`
            : '本轮未得到有效输出'}
      </span>
      <small>静态推导；切换查看不执行 Python，也不修改图或循环结果。</small>
    </div>
  );
}
