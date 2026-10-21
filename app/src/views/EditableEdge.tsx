import { useEffect, useRef, useState, type PointerEvent } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  useReactFlow,
  useViewport,
  type Edge,
  type EdgeProps,
} from '@xyflow/react';
import { manualRoute, routeInsertIndex, routeMidpoint, routePath, type Point } from './edgeRouting';

export type TrailEdge = Edge<
  {
    waypoints?: Point[];
    changeRoute: (points?: Point[]) => void;
  },
  'editable'
>;

interface Drag {
  pointerId: number;
  element: HTMLButtonElement;
  index: number;
  start: Point;
  points: Point[];
  moved: boolean;
}

export function EditableEdge(props: EdgeProps<TrailEdge>) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected } =
    props;
  const { screenToFlowPosition } = useReactFlow();
  const { zoom } = useViewport();
  const [draft, setDraft] = useState<Point[] | null>(null);
  const drag = useRef<Drag | null>(null);
  const [automaticPath, automaticX, automaticY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 6,
    offset: 24,
  });
  const pins = draft ?? data?.waypoints;
  const route = pins?.length
    ? manualRoute(
        { x: sourceX, y: sourceY },
        { x: targetX, y: targetY },
        sourcePosition,
        targetPosition,
        pins,
      )
    : undefined;
  const path = route ? routePath(route) : automaticPath;
  const label = route ? routeMidpoint(route) : { x: automaticX, y: automaticY };
  const controls = pins?.length ? pins : [{ x: automaticX, y: automaticY }];

  function cancel() {
    const current = drag.current;
    drag.current = null;
    setDraft(null);
    if (current?.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
  }
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && drag.current) {
        event.preventDefault();
        cancel();
      }
    };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, []);

  function start(event: PointerEvent<HTMLButtonElement>, index: number) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      pointerId: event.pointerId,
      element: event.currentTarget,
      index,
      start: { x: event.clientX, y: event.clientY },
      points: controls.map((p) => ({ ...p })),
      moved: false,
    };
  }
  function move(event: PointerEvent<HTMLButtonElement>) {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    if (
      !current.moved &&
      Math.hypot(event.clientX - current.start.x, event.clientY - current.start.y) < 2
    )
      return;
    current.moved = true;
    current.points[current.index] = screenToFlowPosition({ x: event.clientX, y: event.clientY });
    setDraft(current.points.map((p) => ({ ...p })));
  }
  function finish(event: PointerEvent<HTMLButtonElement>) {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.stopPropagation();
    drag.current = null;
    setDraft(null);
    if (current.moved) data?.changeRoute(current.points);
  }

  return (
    <>
      <g
        onDoubleClick={(event) => {
          event.stopPropagation();
          const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
          const next = [...(data?.waypoints ?? [])];
          next.splice(route ? routeInsertIndex(route, point) : 0, 0, point);
          data?.changeRoute(next);
        }}
      >
        <BaseEdge
          id={id}
          path={path}
          markerStart={props.markerStart}
          markerEnd={props.markerEnd}
          interactionWidth={props.interactionWidth}
          style={{
            ...props.style,
            ...(selected ? { stroke: 'var(--accent)', strokeWidth: 2 } : {}),
          }}
          label={props.label}
          labelX={label.x}
          labelY={label.y - (selected ? 14 / zoom : 0)}
          labelStyle={props.labelStyle}
          labelShowBg={props.labelShowBg}
          labelBgStyle={props.labelBgStyle}
          labelBgPadding={props.labelBgPadding}
          labelBgBorderRadius={props.labelBgBorderRadius}
        />
      </g>
      {selected && (
        <EdgeLabelRenderer>
          {controls.map((point, index) => (
            <button
              key={index}
              className="edge-control-point nodrag nopan"
              aria-label={`连线控制点 ${index + 1}`}
              title={pins?.length ? '拖动调整线路；双击删除控制点' : '拖动调整线路'}
              style={{
                transform: `translate(-50%, -50%) translate(${point.x}px, ${point.y}px)`,
                width: 14 / zoom,
                height: 14 / zoom,
                borderWidth: 1.5 / zoom,
              }}
              onPointerDown={(event) => start(event, index)}
              onPointerMove={move}
              onPointerUp={finish}
              onPointerCancel={cancel}
              onLostPointerCapture={cancel}
              onClick={(event) => event.stopPropagation()}
              onDoubleClick={(event) => {
                event.stopPropagation();
                if (pins?.length) data?.changeRoute(pins.filter((_, i) => i !== index));
              }}
            />
          ))}
          <div
            className="edge-route-toolbar nodrag nopan"
            style={{
              transform: `translate(${label.x}px, ${label.y + 18 / zoom}px) scale(${1 / zoom}) translate(-50%, 0)`,
            }}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <span>拖动圆点 · 双击线加点</span>
            {data?.waypoints?.length ? (
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  data.changeRoute();
                }}
              >
                恢复自动
              </button>
            ) : null}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
