import type { NodeResult } from '@tensor-trail/model-core/core/types';
import { isTensorSequenceSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { loopIteration } from '@tensor-trail/model-core/core/forEach';

export function ForLoopControls({ result, selected, count, onSelect }: {
  result?: NodeResult; selected?: number; count: number; onSelect: (index: number) => void;
}) {
  const iterations = result?.iterations;
  if (!iterations) return <div className="foreach-controls">循环输入待连接或修正；尚无迭代可查看。</div>;
  if (!iterations.length) return <div className="foreach-controls">循环次数：0。跳过循环体，保留初始状态；仅静态推导。</div>;
  const index = loopIteration(result, selected);
  const current = iterations[index];
  return <div className="foreach-controls" data-testid="for-loop-controls">
    <strong>For · {count} 次</strong>
    <label>当前迭代 <select aria-label="当前迭代" value={index} onChange={e => onSelect(Number(e.target.value))}>
      {iterations.map((_, i) => <option key={i} value={i}>{i}（第 {i + 1}/{count} 次）</option>)}
    </select></label>
    {result?.status === 'error' ? <span>循环存在错误，尚无有效的最终状态。</span> :
      Object.entries(current.outputs).map(([name, after]) => {
        const before = index === 0 ? result?.inputs[name] : iterations[index - 1].outputs[name];
        return <code key={name}>{name}: {before && isTensorSequenceSpec(before) && isTensorSequenceSpec(after)
          ? `list ${before.items.length} → ${after.items.length}`
          : `${shapeText(before)} → ${shapeText(after)}`}</code>;
      })}
    <small><span className="loop-feedback-legend" aria-hidden="true" />虚线：输出 → 下一轮同名输入，不代表数值或 shape 不变。
      点击虚线后可拖动线段和文字。仅静态推导，不执行 Python，不表示网络权重共享。</small>
  </div>;
}
