import { Fragment } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { Port, PortSide } from '../core/types';

const positions: Record<PortSide, Position> = {
  left: Position.Left,
  right: Position.Right,
  top: Position.Top,
  bottom: Position.Bottom,
};

export function PortHandles({
  inputs,
  outputs,
  inputSide,
  outputSide,
  horizontalArea,
  labels = true,
}: {
  inputs: Port[];
  outputs: Port[];
  inputSide: PortSide;
  outputSide: PortSide;
  horizontalArea?: { top: number; height: number };
  labels?: boolean;
}) {
  const handles = [
    ...inputs.map((port) => ({ port, side: inputSide, kind: 'target' as const })),
    ...outputs.map((port) => ({ port, side: outputSide, kind: 'source' as const })),
  ];
  return (
    <>
      {handles.map((handle) => {
        const { port, side, kind } = handle;
        const group = handles.filter((other) => other.side === side);
        const fraction = (group.indexOf(handle) + 1) / (group.length + 1);
        const horizontal = side === 'left' || side === 'right';
        const offset = horizontal
          ? {
              top: horizontalArea
                ? horizontalArea.top + horizontalArea.height * fraction
                : `${fraction * 100}%`,
            }
          : { left: `${fraction * 100}%` };
        return (
          <Fragment key={`${kind}-${port.id}`}>
            <Handle
              id={port.id}
              type={kind}
              position={positions[side]}
              style={offset}
              aria-label={`${kind === 'target' ? '输入' : '输出'} ${port.id}`}
            />
            {labels && (
              <span
                className={`node-port-label port-label-${side}`}
                style={offset}
                title={port.label}
              >
                {port.id}
              </span>
            )}
          </Fragment>
        );
      })}
    </>
  );
}
