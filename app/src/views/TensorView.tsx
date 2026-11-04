import { useState } from 'react';
import type { NodeResult, TensorSpec } from '../core/types';
import { isTensorSpec, shapeText } from '../core/types';

export function TensorAxes({ spec }: { spec: TensorSpec }) {
  return (
    <div className="tensor-axes">
      {spec.shape.map((dimension, i) => (
        <div className={`axis axis-${i % 4}`} key={i}>
          <span className="axis-index">AXIS {i}</span>
          <strong>{dimension}</strong>
          <span>{spec.axes[i] ?? `axis${i}`}</span>
        </div>
      ))}
    </div>
  );
}

export function TensorView({ result }: { result?: NodeResult }) {
  const [chosen, setChosen] = useState('out');
  if (!result) return <p className="muted">连接输入后，在这里查看 Tensor。</p>;
  const entries = Object.entries(result.outputs);
  const port = entries.some(([key]) => key === chosen) ? chosen : entries[0]?.[0];
  const spec = result.outputs[port];
  return (
    <section className="tensor-inspector">
      <div className="section-label">
        {spec && !isTensorSpec(spec) ? '输入结构' : 'TENSOR 观察'}
      </div>
      {!!entries.length && (
        <div className="port-tabs">
          {entries.map(([key]) => (
            <button
              key={key}
              className={port === key ? 'active' : ''}
              onClick={() => setChosen(key)}
            >
              {key}
            </button>
          ))}
        </div>
      )}
      {spec && !isTensorSpec(spec) ? (
        <>
          <div className="container-value-kind">{shapeText(spec)}</div>
          <pre className="container-value-text">{spec.text || '未填写内容说明'}</pre>
          <p className="microcopy">容器输入不要求具体形状；文字仅用于架构说明。</p>
        </>
      ) : spec ? (
        <>
          <div className="tensor-shape">{shapeText(spec)}</div>
          <div className="tensor-meta">
            {spec.dtype} <span>·</span> {spec.shape.length} 维 Tensor
          </div>
          <TensorAxes spec={spec} />
          <p className="microcopy">轴与尺寸示意，不代表真实数值。</p>
        </>
      ) : (
        <div className="empty-tensor">
          暂时没有可用输出
          <br />
          <small>查看下方提示或补全输入连接</small>
        </div>
      )}
      {Object.entries(result.inputs).length > 0 && (
        <div className="input-summary">
          {Object.entries(result.inputs).map(([key, value]) => (
            <div key={key}>
              <span>输入 · {key}</span>
              <code>{shapeText(value)}</code>
            </div>
          ))}
        </div>
      )}
      {result.messages.map((message, i) => (
        <div className={`diagnostic ${result.status}`} key={i}>
          {message}
        </div>
      ))}
      {result.notes.map((note, i) => (
        <p className="learning-note" key={i}>
          {note}
        </p>
      ))}
    </section>
  );
}
