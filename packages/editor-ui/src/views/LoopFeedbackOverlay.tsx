import { useEffect, useId, useRef, useState, type PointerEvent } from 'react';
import { EdgeLabelRenderer, useReactFlow, useViewport } from '@xyflow/react';
import type { LoopFeedbackLayout } from '@tensor-trail/model-core/core/types';
import {
  manualRoute, moveRouteSegment, projectLabelPosition, routeLabelAnchor,
  routePath, simplifyRoute, snapSegmentMovement, type Point,
} from './edgeRouting';
import type { FeedbackRoute } from './loopFeedback';

interface Props {
  routes: FeedbackRoute[];
  layouts?: Record<string, LoopFeedbackLayout>;
  selected?: string;
  onSelect: (state?: string) => void;
  onChange: (state: string, layout?: LoopFeedbackLayout) => void;
}

interface Drag {
  element: SVGPathElement | SVGTextElement;
  pointerId: number;
  kind: 'segment' | 'label';
  index: number;
  points: Point[];
  start: Point;
  flowStart: Point;
  label: FeedbackRoute['label'];
  layout: LoopFeedbackLayout;
  moved: boolean;
}

function FeedbackWire({ route, layout, selected, onSelect, onChange }: {
  route: FeedbackRoute;
  layout?: LoopFeedbackLayout;
  selected: boolean;
  onSelect: Props['onSelect'];
  onChange: Props['onChange'];
}) {
  const arrow = `loop-feedback-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const { screenToFlowPosition } = useReactFlow();
  const { zoom } = useViewport();
  const drag = useRef<Drag | null>(null);
  const [draft, setDraft] = useState<LoopFeedbackLayout>();
  const points = draft ? simplifyRoute(manualRoute(route.points[0], route.points.at(-1)!,
    'right', route.targetSide, draft.waypoints)) : route.points;
  const label = draft ? routeLabelAnchor(points, draft.labelPosition) : route.label;
  // Keep capturing segments mounted until pointerup, even when bends collapse.
  const hitPoints = drag.current?.points ?? points;

  function cancel() {
    const current = drag.current;
    drag.current = null;
    setDraft(undefined);
    if (current?.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
  }

  useEffect(() => {
    if (!selected) cancel();
  }, [selected]);
  useEffect(() => {
    if (!selected) return;
    const keydown = (event: KeyboardEvent) => {
      if ((event.target as Element | null)?.closest?.('input, textarea, select, [contenteditable="true"]')) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        if (drag.current) cancel();
        else onSelect();
      } else if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(event.key.toLowerCase())) {
        cancel(); // Undo/redo is handled by the existing editor shortcut.
      }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [selected, onSelect]);

  function start(event: PointerEvent<SVGPathElement | SVGTextElement>, kind: Drag['kind'], index = 0) {
    if (event.button !== 0 || drag.current) return;
    event.preventDefault();
    event.stopPropagation();
    onSelect(route.state);
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      element: event.currentTarget, pointerId: event.pointerId, kind, index, points,
      start: { x: event.clientX, y: event.clientY },
      flowStart: screenToFlowPosition({ x: event.clientX, y: event.clientY }), label,
      layout: { targetSide: route.targetSide, waypoints: points.slice(1, -1),
        ...(layout?.labelPosition !== undefined ? { labelPosition: layout.labelPosition } : {}) },
      moved: false,
    };
  }

  function move(event: PointerEvent<SVGPathElement | SVGTextElement>) {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    const dx = event.clientX - current.start.x, dy = event.clientY - current.start.y;
    const horizontal = current.points[current.index].y === current.points[current.index + 1].y;
    const distance = current.kind === 'label' ? Math.hypot(dx, dy) : Math.abs(horizontal ? dy : dx);
    if (!current.moved && distance < 2) return;
    const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
    const delta = { x: point.x - current.flowStart.x, y: point.y - current.flowStart.y };
    current.moved = true;
    if (current.kind === 'label') {
      current.layout = { ...current.layout, labelPosition: projectLabelPosition(current.points,
        { x: current.label.x + delta.x, y: current.label.y + delta.y }, current.layout.labelPosition) };
    } else {
      const movement = snapSegmentMovement(current.points, current.index, delta, event.altKey ? 0 : 12 / zoom);
      current.layout = { ...current.layout, waypoints: moveRouteSegment(current.points, current.index,
        movement.delta, 'right', current.layout.targetSide, movement.coordinate) };
    }
    setDraft(current.layout);
  }

  function finish(event: PointerEvent<SVGPathElement | SVGTextElement>) {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    event.stopPropagation();
    cancel();
    if (current.moved) onChange(route.state, current.layout);
  }

  return <>
    <g className={`loop-feedback-route ${selected ? 'selected' : ''}`} data-feedback-state={route.state}
      data-source={route.source} data-target={route.target}>
      <title>{`${route.state}：本轮输出 → 下一轮同名输入。值和 shape 可改变；点击选中、拖动线段或文字，仅调整布局。`}</title>
      <defs><marker id={arrow} viewBox="0 0 10 10" refX={9} refY={5}
        markerWidth={7} markerHeight={7} orient="auto-start-reverse">
        <path d="M 1 1 L 9 5 L 1 9" fill="none"
          stroke={selected ? 'var(--accent)' : 'var(--category-teal)'} strokeWidth={1.5} />
      </marker></defs>
      <path className="loop-feedback-line" d={routePath(points)}
        markerEnd={`url(#${arrow})`} strokeDasharray="7 6" />
      {hitPoints.slice(1).map((point, index) => {
        const from = hitPoints[index];
        if (from.x === point.x && from.y === point.y) return null;
        return <path key={index} className={`loop-feedback-hit nodrag nopan ${from.y === point.y ? 'horizontal' : 'vertical'}`}
          data-segment-index={index} aria-label={`${route.state} 回传线段 ${index + 1}`}
          d={routePath([from, point])} fill="none" stroke="transparent" strokeWidth={16 / zoom}
          onPointerDown={event => start(event, 'segment', index)} onPointerMove={move}
          onPointerUp={finish} onPointerCancel={cancel} onLostPointerCapture={cancel}
          onClick={event => { event.stopPropagation(); onSelect(route.state); }} />;
      })}
      <text className="loop-feedback-label nodrag nopan" x={label.x} y={label.y - 8}
        textAnchor="middle" transform={!label.horizontal ? `rotate(-90 ${label.x} ${label.y})` : undefined}
        aria-label={`${route.state} 回传文字，沿线拖动`}
        onPointerDown={event => start(event, 'label')} onPointerMove={move}
        onPointerUp={finish} onPointerCancel={cancel} onLostPointerCapture={cancel}
        onClick={event => event.stopPropagation()}>
        {route.state} → 下一轮
      </text>
    </g>
    {selected && <EdgeLabelRenderer>
      <div className="edge-route-toolbar loop-feedback-toolbar nodrag nopan"
        style={{ transform: `translate(${label.x}px, ${label.y + 18 / zoom}px) scale(${1 / zoom}) translate(-50%, 0)` }}
        onPointerDown={event => event.stopPropagation()} onClick={event => event.stopPropagation()}>
        <span>拖动线段 / 文字 · Alt 暂不吸附 · Esc 取消</span>
        <button disabled={!layout} onClick={() => onChange(route.state)}>恢复自动走线</button>
      </div>
    </EdgeLabelRenderer>}
  </>;
}

export function LoopFeedbackOverlay({ routes, layouts, selected, onSelect, onChange }: Props) {
  if (!routes.length) return null;
  return <svg className="loop-feedback-overlay" width={1} height={1}
    role="group" aria-label="循环状态回传：本轮输出指向下一轮同名输入；可选中拖动布局，不代表数值不变">
    {routes.map(route => <FeedbackWire key={route.state} route={route} layout={layouts?.[route.state]}
      selected={selected === route.state} onSelect={onSelect} onChange={onChange} />)}
  </svg>;
}
