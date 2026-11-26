import { Fragment } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { Port, PortSide } from '@tensor-trail/model-core/core/types';

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
  inputLabels,
  outputLabels,
}: {
  inputs: Port[];
  outputs: Port[];
  inputSide: PortSide;
  outputSide: PortSide;
  horizontalArea?: { top: number; height: number };
  labels?: boolean;
  inputLabels?: Record<string, string>;
  outputLabels?: Record<string, string>;
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
              title={port.label}
            />
            {labels && (
              <span
                className={`node-port-label port-label-${side}`}
                data-port-id={port.id}
                style={offset}
                title={port.label}
              >
                {(kind === 'target' ? inputLabels : outputLabels)?.[port.id] ?? port.id}
              </span>
            )}
          </Fragment>
        );
      })}
    </>
  );
}

// An absolutely positioned grid child uses its named grid area as its containing
// block. Handles stay on the outer card while their labels own space in the same band.
export function SubmodulePortSlots({ inputs, outputs, inputSide, outputSide }: {
  inputs: Port[]; outputs: Port[]; inputSide: PortSide; outputSide: PortSide;
}) {
  const groups: Record<PortSide, { port: Port; kind: 'target' | 'source' }[]> = {
    left: [], right: [], top: [], bottom: [],
  };
  groups[inputSide].push(...inputs.map(port => ({ port, kind: 'target' as const })));
  groups[outputSide].push(...outputs.map(port => ({ port, kind: 'source' as const })));
  const rows = Math.max(groups.left.length, groups.right.length);
  const labels = (side: PortSide) => groups[side].map(({ port, kind }, index) => {
    const lateral = side === 'left' || side === 'right';
    return <span className="submodule-port-label" key={`${kind}-${port.id}`}
      data-side={side} data-port-id={port.id} title={port.id}
      style={lateral ? { gridColumn: side === 'left' ? 1 : 2, gridRow: index + 1 } : undefined}>
      {port.id}
    </span>;
  });
  return <>
    {(['top', 'bottom'] as const).map(side => groups[side].length > 0 &&
      <div key={side} className="submodule-port-band" data-band={side} style={{
        gridArea: side, gridTemplateColumns: `repeat(${groups[side].length}, minmax(0, 1fr))`,
      }}>{labels(side)}</div>)}
    {rows > 0 && <div className="submodule-port-band" data-band="sides" style={{ gridArea: 'sides' }}>
      {labels('left')}{labels('right')}
    </div>}
    {(['left', 'right', 'top', 'bottom'] as const).flatMap(side =>
      groups[side].map(({ port, kind }, index) => {
        const lateral = side === 'left' || side === 'right';
        const offset = `${100 * (index + 0.5) / (lateral ? rows : groups[side].length)}%`;
        return <Handle key={`${kind}-${port.id}`} id={port.id} type={kind} position={positions[side]}
          style={{ gridArea: lateral ? 'sides' : side, ...(lateral ? { top: offset } : { left: offset }) }}
          aria-label={`${kind === 'target' ? '输入' : '输出'} ${port.id}`} title={port.label} />;
      }))}
  </>;
}
