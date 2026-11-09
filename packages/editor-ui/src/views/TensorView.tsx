import { useState } from 'react';
import type { NodeResult, TensorSpec, Port } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSpec, isIntegerSpec, shapeText } from '@tensor-trail/model-core/core/types';
import { DictionaryTree } from './DictionaryTree';

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

export function TensorView({ result, outputPorts }: { result?: NodeResult; outputPorts?: Port[] }) {
  const [chosen, setChosen] = useState('out');
  if (!result) return <p className="muted">连接输入后，在这里查看 Tensor。</p>;
  const entries = Object.entries(result.outputs);
  const port = entries.some(([key]) => key === chosen) ? chosen : entries[0]?.[0];
  const spec = result.outputs[port];
  return (
    <section className="tensor-inspector">
      <div className="section-label">
        {spec && isIntegerSpec(spec)
          ? '整数尺寸'
          : spec && !isTensorSpec(spec)
            ? '输入结构'
            : 'TENSOR 观察'}
      </div>
      {!!entries.length && (
        <div className="port-tabs">
          {entries.map(([key]) => (
            <button
              key={key}
              className={port === key ? 'active' : ''}
              onClick={() => setChosen(key)}
            >
              {outputPorts?.find((p) => p.id === key)?.label ?? key}
            </button>
          ))}
        </div>
      )}
      {spec && isDictionarySpec(spec) ? (
        <>
          <div className="container-value-kind">{shapeText(spec)}</div>
          <DictionaryTree value={spec} />
          <p className="microcopy">字段结构描述，不包含真实张量数值。</p>
        </>
      ) : spec && isIntegerSpec(spec) ? (
        <>
          <div className="tensor-shape">{spec.value}</div>
          <div className="tensor-meta">
            {typeof spec.value === 'number' ? 'int · 整数尺寸' : 'int · 符号尺寸（尚未确定数值）'}
          </div>
          <p className="microcopy">读取张量的轴长度，不是张量元素或零维 Tensor；不参与梯度计算。</p>
        </>
      ) : spec && !isTensorSpec(spec) ? (
        <>
          <div className="container-value-kind">{shapeText(spec)}</div>
          {'text' in spec ? (
            <>
              <pre className="container-value-text">{spec.text || '未填写内容说明'}</pre>
              <p className="microcopy">容器输入不要求具体形状；文字仅用于架构说明。</p>
            </>
          ) : (
            <p className="microcopy">
              Python {spec.kind === 'none' ? 'None 空值' : '布尔条件'}，不是 Tensor。
            </p>
          )}
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
