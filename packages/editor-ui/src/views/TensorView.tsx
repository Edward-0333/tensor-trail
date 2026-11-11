import { useState } from 'react';
import type { NodeResult, TensorSpec, Port } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSpec, isIntegerSpec, isIntegerListSpec, isControlSpec, isSizeSpec, isTensorSequenceSpec, isModuleSpec, isDeviceSpec } from '@tensor-trail/model-core/core/types';
import { DictionaryTree } from './DictionaryTree';
import { isScalarTensor, valueDisplayText } from './valueDisplayText';
import { gradLabel, gradSourceLabel, tensorGradState } from '@tensor-trail/model-core/core/tensorAppearance';

export function TensorAxes({ spec }: { spec: TensorSpec }) {
  if (isScalarTensor(spec)) return (
    <div className="scalar-tensor-details">
      <p className="microcopy">零维标量张量，包含 1 个元素；没有维度轴，不是空张量。</p>
      <code>shape: torch.Size([])</code>
    </div>
  );
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

export function TensorView({ result, outputPorts, observeInput = false }: { result?: NodeResult; outputPorts?: Port[]; observeInput?: boolean }) {
  const [chosen, setChosen] = useState('out');
  if (!result) return <p className="muted">连接输入后，在这里查看 Tensor。</p>;
  const values = observeInput ? result.inputs : result.outputs;
  const entries = Object.entries(values);
  const port = entries.some(([key]) => key === chosen) ? chosen : entries[0]?.[0];
  const spec = values[port];
  return (
    <section className="tensor-inspector">
      <div className="section-label">
        {observeInput ? '属性写入值（输入 Tensor）' : spec && isDeviceSpec(spec) ? '设备元数据' : spec && isModuleSpec(spec) ? '模块引用' : spec && isIntegerSpec(spec)
          ? '整数值'
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
      {spec && (isIntegerListSpec(spec) || isControlSpec(spec)) ? (
        <><code>{valueDisplayText(spec)}</code><p className="microcopy">{isControlSpec(spec)
          ? '条件控制依赖，不是 Tensor 或观测到的布尔值。两条分支仅静态检查，不表示已执行或同时执行。'
          : 'Python list[int]，每项为配置声明的整数。容器不是 Tensor，没有 shape、dtype 或梯度。'}</p></>
      ) : spec && isDeviceSpec(spec) ? (
        <>
          <div className="container-value-kind">DEVICE</div>
          <code>{spec.reference}</code>
          <p className="microcopy">torch.device 设备元数据，不是 Tensor；没有 shape、dtype 或 requires_grad。</p>
          <p className="microcopy">这里只记录设备来源表达式，实际设备未采集，不推断为 cuda:0 或 cpu。灰色虚线表示设备依赖，不传递张量元素或梯度。</p>
        </>
      ) : spec && isModuleSpec(spec) ? (
        <>
          <div className="container-value-kind">{spec.moduleType}</div>
          <code>{spec.reference}</code>
          <div className="tensor-sequence-items">
            {Object.entries(spec.parameters).map(([name, parameter]) => (
              <div className="tensor-sequence-item" key={name}>
                <code>{name} · Parameter {valueDisplayText(parameter)}</code>
                <span>{parameter.dtype} · {gradLabel(tensorGradState(parameter))} · {gradSourceLabel(parameter)}</span>
              </div>
            ))}
          </div>
          <p className="microcopy">模块对象不是 Tensor，没有 shape 或 requires_grad。以上是其注册参数的静态规格；不含真实权重数值。</p>
          <p className="microcopy">用 GetParameter 显式读取参数。此处不执行 forward，也不连接真实 Python 实例。</p>
        </>
      ) : spec && isDictionarySpec(spec) ? (
        <>
          <div className="container-value-kind">{valueDisplayText(spec)}</div>
          <DictionaryTree value={spec} />
          <p className="microcopy">字段结构描述，不包含真实张量数值。</p>
        </>
      ) : spec && isIntegerSpec(spec) ? (
        <>
          <div className="tensor-shape">{spec.value}</div>
          <div className="tensor-meta">
            {typeof spec.value === 'number' ? 'int · Python 整数' : 'int · 符号整数（尚未确定数值）'}
          </div>
          <p className="microcopy">Python 整数或符号整数，可来自尺寸、计数、索引或标量读取；不是零维 Tensor，不参与梯度计算。符号不代表已采集实际数值。</p>
        </>
      ) : spec && isSizeSpec(spec) ? (
        <>
          <div className="container-value-kind">{valueDisplayText(spec)}</div>
          <p className="microcopy">torch.Size 是由整数尺寸组成的 tuple-like 元数据，不是 Tensor；不参与梯度计算。</p>
          <p className="microcopy">数值从上游形状推导；符号尺寸保持未定，不代表已执行 Python。</p>
        </>
      ) : spec && isTensorSequenceSpec(spec) ? (
        <>
          <div className="container-value-kind">{valueDisplayText(spec)}</div>
          {spec.items.length === 0 ? (
            <p className="microcopy">空{spec.kind === 'list' ? '列表 []' : '元组 ()'}，尚无 Tensor 元素；不是形状为 [0] 的张量。</p>
          ) : (
            <div className="tensor-sequence-items">
              {spec.items.map((item, index) => (
                <div className="tensor-sequence-item" key={index}>
                  <code>[{index}] {valueDisplayText(item)}</code>
                  <span>{item.dtype} · {gradLabel(tensorGradState(item))} · {gradSourceLabel(item)}</span>
                </div>
              ))}
            </div>
          )}
          <p className="microcopy">容器本身不是 Tensor，没有 requires_grad；元素分别保留形状与梯度标注，不进行 stack 或 concat。</p>
        </>
      ) : spec && !isTensorSpec(spec) ? (
        <>
          <div className="container-value-kind">{valueDisplayText(spec)}</div>
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
          <div className="tensor-shape">{valueDisplayText(spec)}</div>
          <div className="tensor-meta">
            {spec.dtype} <span>·</span> {spec.shape.length} 维 {spec.parameterRef ? 'nn.Parameter' : spec.bufferRef ? 'Tensor（注册 buffer）' : 'Tensor'}
          </div>
          {spec.parameterRef && <p className="microcopy">注册参数：<code>{spec.parameterRef.module}.{spec.parameterRef.name}</code>；参数身份和 requires_grad 是两个不同属性。</p>}
          {spec.bufferRef && <p className="microcopy">注册 buffer：<code>{spec.bufferRef.reference}</code>；不是 Tensor 子类或 Parameter，{spec.bufferRef.persistent ? '随 state_dict 保存' : '不随 state_dict 保存'}。当前数值未采集，不保证仍等于构造初值。</p>}
          <TensorAxes spec={spec} />
          <div className="tensor-autograd-summary">
            <code>{gradLabel(tensorGradState(spec))}</code> · {gradSourceLabel(spec)}
          </div>
          <p className="microcopy">requires_grad 状态不等于已有梯度；不保证连接到 loss，或执行了 backward。</p>
          <p className="microcopy">轴与尺寸示意，不代表真实数值。</p>
        </>
      ) : (
        <div className="empty-tensor">
          暂时没有可用输出
          <br />
          <small>查看下方提示或补全输入连接</small>
        </div>
      )}
      {!observeInput && Object.entries(result.inputs).length > 0 && (
        <div className="input-summary">
          {Object.entries(result.inputs).map(([key, value]) => (
            <div key={key}>
              <span>输入 · {key}</span>
              <code>{valueDisplayText(value)}</code>
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
