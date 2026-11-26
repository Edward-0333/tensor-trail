import { EdgeLabelRenderer, Position, useReactFlow, type EdgeProps } from '@xyflow/react';
import { useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { useEditor } from '../editor/store';
import type { TrailEdge } from './EditableEdge';

import { tensorFaceColor } from './tensorFaceColor';

const directions = {
  [Position.Left]: [-1, 0],
  [Position.Right]: [1, 0],
  [Position.Top]: [0, -1],
  [Position.Bottom]: [0, 1],
} as const;

/** Anchors have no routed geometry. Only the target stub and chip participate in hit testing. */
export function AnchorEdge({
  data,
  id,
  selected,
  sourceX,
  sourceY,
  targetX,
  targetY,
  targetPosition,
}: EdgeProps<TrailEdge>) {
  const { setEdges, setNodes } = useReactFlow();
  const inspected = useEditor((state) => state.selectedEdgeId === id);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const chip = useRef<HTMLButtonElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    setWidth(chip.current?.offsetWidth ?? 0);
  }, [data?.sourceLabel]);
  const marker = 'anchor-arrow-' + useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const [dx, dy] = directions[targetPosition];
  const dockX = targetX + dx * 16,
    dockY = targetY + dy * 16;
  const cx = dockX + (dx * width) / 2,
    cy = dockY + dy * 10;
  const arc = Math.min(100, Math.max(24, Math.hypot(sourceX - cx, sourceY - cy) / 6));
  const controlX = (sourceX + cx) / 2 + (sourceY === cy ? 0 : dx * arc * 0.5);
  const controlY = (sourceY + cy) / 2 - arc;
  const tone = data?.sourceTone ?? 'blue';
  const stroke = tensorFaceColor(tone, 'stroke');
  const revealed = Boolean(hovered || focused || inspected || selected || data?.sourceSelected);
  return (
    <>
      <defs>
        <marker
          id={marker}
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M 0 0 L 8 4 L 0 8 Z" fill={stroke} />
        </marker>
      </defs>
      <path
        className="edge-anchor-hover-wire"
        data-revealed={revealed}
        d={`M ${sourceX} ${sourceY} Q ${controlX} ${controlY} ${cx} ${cy}`}
        stroke={stroke}
        markerEnd={`url(#${marker})`}
      />
      <path
        className="edge-anchor-stub"
        d={`M ${dockX} ${dockY} L ${targetX} ${targetY}`}
        stroke={stroke}
        strokeWidth="1.5"
        markerEnd={`url(#${marker})`}
      />
      <EdgeLabelRenderer>
        <button
          ref={chip}
          type="button"
          className="edge-anchor-chip nodrag nopan"
          aria-pressed={!!selected || inspected}
          title={data?.annotation || data?.sourceLabel}
          data-revealed={revealed}
          style={
            {
              '--chip-side': tensorFaceColor(tone, 'side'),
              '--chip-top': tensorFaceColor(tone, 'top'),
              '--chip-stroke': stroke,
              transform: `translate(${dockX}px, ${dockY}px) translate(${(dx - 1) * 50}%, ${(dy - 1) * 50}%)`,
            } as CSSProperties
          }
          onPointerEnter={() => setHovered(true)}
          onPointerLeave={() => setHovered(false)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            setNodes((nodes) =>
              nodes.map((node) => (node.selected ? { ...node, selected: false } : node)),
            );
            setEdges((edges) => edges.map((edge) => ({ ...edge, selected: edge.id === id })));
            data?.select();
          }}
        >
          <span aria-hidden="true">●</span> {data?.sourceLabel}
        </button>
      </EdgeLabelRenderer>
    </>
  );
}
