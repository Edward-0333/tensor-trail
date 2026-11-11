import {
  DEFAULT_TENSOR_PALETTE,
  gradLabel,
  tensorInk,
} from '@tensor-trail/model-core/core/tensorAppearance';
import { useEditor } from './store';

export function TensorAppearancePanel() {
  const state = useEditor();
  const palette = state.project.tensorPalette ?? DEFAULT_TENSOR_PALETTE;
  return (
    <section className="tensor-appearance-panel" aria-label="全项目张量配色">
      <div className="section-label">统一背景色 · 全项目</div>
      <p className="microcopy">
        按 requires_grad 状态着色，应用于主图及所有子图的 Tensor。背景色不再逐节点设置。
      </p>
      {(
        [
          ['on', '需要求导'],
          ['off', '不对该张量求导'],
          ['unknown', '状态未知'],
        ] as const
      ).map(([key, label]) => (
        <label className="tensor-palette-row" key={key}>
          <span
            className="tensor-palette-preview"
            style={{ background: palette[key], color: tensorInk(palette[key]) }}
          >
            {gradLabel(key)}
          </span>
          <span>{label}</span>
          <input
            type="color"
            aria-label={`${gradLabel(key)} 统一背景色`}
            value={palette[key]}
            disabled={!state.hasProject}
            onChange={(event) => state.setTensorPaletteColor(key, event.target.value)}
          />
        </label>
      ))}
      <button
        className="soft-button"
        disabled={!state.hasProject}
        onClick={state.resetTensorPalette}
      >
        恢复默认背景配色
      </button>
      <p className="microcopy">
        设置随项目保存，支持撤销。单个节点的边框色仍可在“节点属性”中自定义。
      </p>
      <div className="section-label">状态含义</div>
      <p className="microcopy">
        grad:on 仅表示 requires_grad=True，不保证该张量连接到最终 loss，也不代表已经执行 backward
        或保留了 .grad。
      </p>
      <p className="microcopy">
        未采集或无法静态确定的状态显示 grad:?。字典和 Python 整数不使用张量背景色。
      </p>
    </section>
  );
}
