import { useEffect } from 'react';
import { Handle, Position, useUpdateNodeInternals, type Node, type NodeProps } from '@xyflow/react';
import { Box, ChevronRight, AlertCircle, ArrowUpRight } from 'lucide-react';
import type { ModelNode, NodeResult, Port } from '../core/types';
import { shapeText } from '../core/types';

export type TrailNode = Node<
  {
    model: ModelNode;
    result?: NodeResult;
    inputs: Port[];
    outputs: Port[];
    category: string;
    reverse: boolean;
    active: boolean;
    dimmed: boolean;
    open: () => void;
  },
  'module'
>;
export const tone = (category: string) =>
  category === 'Transformer' || category === '注意力算子'
    ? 'purple'
    : category === '卷积'
      ? 'amber'
      : category === '循环网络'
        ? 'blue'
        : category === '输入输出' || category === '内部'
          ? 'teal'
          : 'slate';

export function ModuleNode({ data, selected }: NodeProps<TrailNode>) {
  const { model, result, inputs, outputs } = data;
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(model.id);
  }, [model.id, data.reverse, inputs.length, outputs.length, updateNodeInternals]);
  const bad = result?.status === 'error' || result?.status === 'blocked';
  const rows = Math.max(inputs.length, outputs.length, 1);
  return (
    <div
      className={`module-node tone-${tone(data.category)} ${data.reverse ? 'flow-left' : ''} ${selected ? 'selected' : ''} ${bad ? 'invalid' : ''} ${data.active ? 'step-active' : ''} ${data.dimmed ? 'dimmed' : ''}`}
      data-testid={`node-${model.id}`}
    >
      <div className="node-heading">
        <div className="node-symbol">
          <Box size={15} />
        </div>
        <div>
          <span className="node-kind">{model.type}</span>
          <strong title={model.label}>{model.label}</strong>
        </div>
        {model.subgraph && <ArrowUpRight className="node-expand-icon" size={15} />}
      </div>
      <div className="node-ports" style={{ minHeight: rows * 27 + 12 }}>
        {Array.from({ length: rows }, (_, i) => {
          const ip = inputs[i],
            op = outputs[i];
          const spec = op
            ? result?.outputs[op.id]
            : i === 0
              ? (result?.outputs.out ?? result?.inputs.in)
              : undefined;
          return (
            <div className="node-port-row" key={i}>
              {ip && (
                <>
                  <Handle
                    id={ip.id}
                    type="target"
                    position={data.reverse ? Position.Right : Position.Left}
                    style={{ top: 13.5, ...(data.reverse ? { right: -9 } : { left: -9 }) }}
                    aria-label={`输入 ${ip.id}`}
                  />
                  <span className="port-id input-id">{ip.id}</span>
                </>
              )}
              <code title={shapeText(spec)}>
                {spec ? shapeText(spec) : bad ? '待修正' : i === 0 ? '等待输入' : '·'}
              </code>
              {op && (
                <>
                  <span className="port-id output-id">{op.id}</span>
                  <Handle
                    id={op.id}
                    type="source"
                    position={data.reverse ? Position.Left : Position.Right}
                    style={{ top: 13.5, ...(data.reverse ? { left: -9 } : { right: -9 }) }}
                    aria-label={`输出 ${op.id}`}
                  />
                </>
              )}
            </div>
          );
        })}
      </div>
      <div className="node-footer">
        {bad ? (
          <span className="node-error">
            <AlertCircle size={12} />
            {result.status === 'blocked' ? '等待输入或上游修正' : '形状不兼容'}
          </span>
        ) : model.subgraph ? (
          <button className="nodrag" onClick={data.open}>
            展开 {model.subgraph.nodes.length} 个节点 <ChevronRight size={12} />
          </button>
        ) : (
          <span>
            {result?.status === 'warning' ? '符号约束待验证' : result ? '形状已推导' : '等待连接'}
          </span>
        )}
      </div>
    </div>
  );
}
