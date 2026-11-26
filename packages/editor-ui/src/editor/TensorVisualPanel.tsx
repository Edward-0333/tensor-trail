import { useState } from 'react';
import type {
  ModelNode,
  NodeResult,
  NodeTensorVisual,
  TensorDepthStyle,
  TensorVisualTone,
} from '@tensor-trail/model-core/core/types';
import {
  buildTensorGeometryDiagram,
  canvasTensorVisual,
  defaultTensorVisualWidth,
  type GraphTensorTones,
} from '@tensor-trail/model-core/core/tensorGeometry';
import {
  minTensorVisualWidth,
  maxTensorVisualWidth,
  minTensorBoxWidth,
  maxTensorBoxWidth,
  minTensorBoxHeight,
  maxTensorBoxHeight,
  minTensorBoxDepth,
  maxTensorBoxDepth,
  tensorDepthStyles,
  tensorVisualModes,
  tensorVisualTones,
} from '@tensor-trail/model-core/core/tensorVisual';
import { useEditor } from './store';

const modes = { off: '关闭', block: '3D/4D 积木' };
const names: Record<TensorVisualTone, string> = {
  auto: '自动',
  blue: '蓝',
  violet: '紫',
  teal: '青',
  amber: '橙',
  rose: '玫红',
  slate: '灰',
};
const styles: Record<TensorDepthStyle, string> = {
  auto: '自动',
  solid: '实心体积',
  slices: '分层切片',
};
const sliders = [
  ['width', '卡片宽度', '积木宽度', minTensorVisualWidth, maxTensorVisualWidth],
  ['boxWidth', '块宽 W', '积木块宽', minTensorBoxWidth, maxTensorBoxWidth],
  ['boxHeight', '块高 H', '积木块高', minTensorBoxHeight, maxTensorBoxHeight],
  ['boxDepth', '块深 Z', '积木块深', minTensorBoxDepth, maxTensorBoxDepth],
] as const;

export function TensorVisualPanel({
  node,
  result,
  graphTones,
  referenceMaxDim,
}: {
  node: ModelNode;
  result?: NodeResult;
  graphTones: GraphTensorTones;
  referenceMaxDim?: number;
}) {
  const setVisual = useEditor((s) => s.setNodeTensorVisual);
  const visual = node.tensorVisual;
  const globalTensorBlocks = useEditor((s) => s.globalTensorBlocks);
  const mode = canvasTensorVisual(node, globalTensorBlocks)?.mode ?? 'off';
  const save = (patch: Partial<NodeTensorVisual>) => {
    const next = { mode, ...visual, ...patch };
    for (const key of Object.keys(patch) as (keyof NodeTensorVisual)[]) {
      if (next[key] === undefined) delete next[key];
    }
    return setVisual(node.id, next);
  };
  const [axisError, setAxisError] = useState(false);
  const diagram =
    mode === 'block' &&
    buildTensorGeometryDiagram(node, result, graphTones, {
      mode,
      referenceMaxDim,
      observeInput: node.type === 'SetAttribute',
    });
  const box = diagram ? diagram.output : undefined;
  const spec =
    diagram && (diagram.observeInput ? result?.inputs : result?.outputs)?.[diagram.output.port];
  const tensor = spec && 'shape' in spec ? spec : undefined;
  const inferredAxes = tensor?.shape.map((_, i) => tensor.axes[i] || `axis${i}`).join(', ');
  const automatic = {
    width: diagram ? diagram.width : defaultTensorVisualWidth,
    boxWidth: box?.wPx,
    boxHeight: box?.hPx,
    boxDepth: box && box.depthPx + (box.depth.sliceCount - 1) * box.sliceGapPx,
  };
  const axes = visual?.axisLabels?.join(', ') ?? '';
  return (
    <details className="tensor-visual-panel" open={mode !== 'off'}>
      <summary>
        张量积木 <small>{visual ? modes[mode] : '跟随画布默认'}</small>
      </summary>
      <div className="tv-options" role="group" aria-label="积木模式">
        {tensorVisualModes.map((m) => (
          <button key={m} aria-pressed={mode === m} onClick={() => save({ mode: m })}>
            {modes[m]}
          </button>
        ))}
        {visual && <button onClick={() => setVisual(node.id, null)}>恢复默认</button>}
      </div>
      {mode !== 'off' && (
        <>
          {tensor && (
            <div className="tv-shape">
              [{tensor.shape.join(', ')}] · {inferredAxes || '标量'}
            </div>
          )}
          <div className="tv-swatches" role="group" aria-label="积木语义配色">
            {tensorVisualTones.map((tone) => (
              <button
                key={tone}
                title={names[tone]}
                aria-label={names[tone]}
                aria-pressed={!visual?.color && (visual?.tone ?? 'auto') === tone}
                onClick={() => save({ tone, color: undefined })}
              >
                {tone === 'auto' ? (
                  names.auto
                ) : (
                  <i style={{ background: `var(--tv-${tone}-front)` }} />
                )}
              </button>
            ))}
          </div>
          <div className="tv-color-row">
            <label>
              <input
                type="color"
                aria-label="自定义颜色"
                value={visual?.color ?? '#579fea'}
                onInput={(e) => save({ color: e.currentTarget.value })}
              />{' '}
              自定义颜色
            </label>
            {visual?.color && (
              <>
                <code>{visual.color}</code>
                <button className="tv-reset" onClick={() => save({ color: undefined })}>
                  清除
                </button>
              </>
            )}
          </div>
          <div className="tv-options" role="group" aria-label="深度显示方式">
            {tensorDepthStyles.map((style) => (
              <button
                key={style}
                aria-pressed={(visual?.depthStyle ?? 'auto') === style}
                onClick={() => save({ depthStyle: style })}
              >
                {styles[style]}
              </button>
            ))}
          </div>
          <div className="tv-dimensions">
            {sliders.map(([key, label, aria, min, max]) => (
              <div className="tv-slider" key={key}>
                <label htmlFor={node.id + '-tv-' + key}>{label}</label>
                <span>{visual?.[key] === undefined ? '自动' : visual[key] + ' px'}</span>
                {visual?.[key] !== undefined && (
                  <button
                    className="tv-reset"
                    aria-label={label + '恢复自动'}
                    onClick={() => save({ [key]: undefined })}
                  >
                    自动
                  </button>
                )}
                <input
                  type="range"
                  id={node.id + '-tv-' + key}
                  min={min}
                  max={max}
                  value={visual?.[key] ?? Math.round(Math.max(min, automatic[key] ?? min))}
                  aria-label={aria}
                  onInput={(e) => save({ [key]: Number(e.currentTarget.value) })}
                />
              </div>
            ))}
          </div>
          <label className="param-field">
            <span>轴显示名</span>
            <input
              key={axes}
              aria-label="自定义轴名"
              defaultValue={axes}
              placeholder={inferredAxes || '例如: queue, query, head, dim'}
              onInput={() => setAxisError(false)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.nativeEvent.isComposing) e.currentTarget.blur();
              }}
              onBlur={(e) => {
                const value = e.currentTarget.value.trim();
                if (value !== axes)
                  setAxisError(
                    !save({
                      axisLabels: value ? value.split(/[,，]/).map((s) => s.trim()) : undefined,
                    }),
                  );
              }}
            />
            {axisError && <small role="alert">请输入至多 8 个轴名，每个为 1–32 字。</small>}
          </label>
          <label className="param-field">
            <span>积木说明</span>
            <input
              key={visual?.caption ?? ''}
              aria-label="积木说明"
              maxLength={240}
              defaultValue={visual?.caption ?? ''}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.nativeEvent.isComposing) e.currentTarget.blur();
              }}
              onBlur={(e) => {
                const caption = e.currentTarget.value;
                if (caption !== (visual?.caption ?? '')) save({ caption: caption || undefined });
              }}
            />
          </label>
        </>
      )}
    </details>
  );
}
