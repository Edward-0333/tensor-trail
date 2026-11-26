import type { ModelEdge } from '@tensor-trail/model-core/core/types';
import type { ProjectedEdge } from '@tensor-trail/model-core/core/canvasProjection';
import { useEditor } from '../editor/store';
import { AnchorEdge } from './AnchorEdge';
import { Fragment, useEffect, useId, useMemo, useRef, useState, type PointerEvent } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  useReactFlow,
  useViewport,
  type Edge,
  type EdgeProps,
} from '@xyflow/react';
import {
  automaticRoute,
  manualRoute,
  moveRouteSegment,
  snapSegmentMovement,
  simplifyRoute,
  routeInsertIndex,
  routeLabelAnchor,
  projectLabelPosition,
  routePath,
  type Point,
} from './edgeRouting';
import {
  useAutomaticWireRoute,
  useRenderedEdge,
  useWirePreview,
  useWireSelection,
} from './EdgeJunctions';
import {
  highlightedDrawing,
  nearestSection,
  pointHighlighted,
  selectWireSection,
  wireHighlightStarts,
} from './wireHighlight';
import {
  wireDrawing,
  sharedSections,
  sharedDragTargets,
  moveSharedTargets,
  type WireDragTarget,
  type WireRouteChange,
} from './sharedWires';
import { mergeNearbyWire, mergeSharedWireChanges, movedSection } from './edgeMerging';

export type TrailEdge = Edge<
  {
    presentation?: ModelEdge['presentation'];
    annotation?: string;
    edgeOps?: ProjectedEdge['edgeOps'];
    sourceTone?: ProjectedEdge['sourceTone'];
    sourceLabel?: string;
    sourceSelected?: boolean;
    selectedNodeId?: string | null;
    activeNodeId?: string;
    valueKey?: string;
    waypoints?: Point[];
    labelPosition?: number;
    changeRoute: (points?: Point[]) => void;
    changeSharedRoutes: (changes: WireRouteChange[]) => void;
    changeLabelPosition: (position?: number) => void;
    select: () => void;
  },
  'editable'
>;

interface Drag {
  pointerId: number;
  element: HTMLButtonElement | SVGPathElement;
  kind: 'point' | 'segment';
  route: Point[];
  flowStart: Point;
  index: number;
  start: Point;
  points: Point[];
  targets: WireDragTarget[];
  sharedChanges?: WireRouteChange[];
  moved: boolean;
}

interface LabelDrag {
  pointerId: number;
  element: Element;
  route: Point[];
  flowStart: Point;
  screenStart: Point;
  anchor: Point;
  initialPosition: number;
  position: number;
  moved: boolean;
}

export function EditableEdge(props: EdgeProps<TrailEdge>) {
  // Switching modes unmounts the routed edge and deregisters its bus and junction geometry.
  return props.data?.presentation === 'anchor' ? <AnchorEdge {...props} /> : <RoutedEditableEdge {...props} />;
}

function RoutedEditableEdge(props: EdgeProps<TrailEdge>) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected } =
    props;
  const { screenToFlowPosition, setEdges, setNodes } = useReactFlow();
  const { zoom } = useViewport();
  const [draft, setDraft] = useState<Point[] | null>(null);
  const [snapTarget, setSnapTarget] = useState<'self' | 'peer' | null>(null);
  const drag = useRef<Drag | null>(null);
  const labelDrag = useRef<LabelDrag | null>(null);
  const [labelDraft, setLabelDraft] = useState<number | null>(null);
  const wirePreview = useWirePreview(id);
  const wireSelection = useWireSelection(id, selected);
  const arrowId = 'wire-arrow-' + useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const slide = data?.presentation === 'slide';
  const edgeOps = data?.edgeOps ?? [];
  const pipeline = edgeOps.length > 0;
  const draggedStep = useRef(false);
  const capsule = slide || !!data?.annotation;
  const labelKey = [data?.edgeOps?.map(step => step.id).join(' '), data?.annotation, props.label].filter(Boolean).join('\n');
  const highlightColor = capsule && props.style?.stroke === 'var(--danger)' ? 'var(--danger)' : 'var(--accent)';
  const pins = draft ?? wirePreview.points ?? data?.waypoints;
  const rawRoute = useMemo(
    () =>
      pins?.length
        ? manualRoute(
            { x: sourceX, y: sourceY },
            { x: targetX, y: targetY },
            sourcePosition,
            targetPosition,
            pins,
          )
        : automaticRoute(
            { x: sourceX, y: sourceY },
            { x: targetX, y: targetY },
            sourcePosition,
            targetPosition,
          ),
    [pins, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition],
  );
  const baseGeometry = useMemo(
    () => ({
      id,
      source: props.source,
      sourcePort: props.sourceHandleId ?? '',
      points: simplifyRoute(rawRoute),
      sourceSide: sourcePosition,
      targetSide: targetPosition,
      automatic: !pins?.length,
    }),
    [
      id,
      props.source,
      props.sourceHandleId,
      rawRoute,
      sourcePosition,
      targetPosition,
      pins?.length,
    ],
  );
  const route = useAutomaticWireRoute(baseGeometry);
  const path = routePath(route);
  const label = routeLabelAnchor(route, labelDraft ?? data?.labelPosition);
  const toolbar = routeLabelAnchor(route);
  const controls = pins ?? [];
  // Keep the capturing hit area mounted until pointerup, even if a new elbow appears.
  const rendered = useMemo(
    () => ({
      id,
      source: props.source,
      sourcePort: props.sourceHandleId ?? '',
      points: route,
      color: props.style?.stroke,
      valueKey: data?.valueKey,
      selected,
      sourceSide: sourcePosition,
      targetSide: targetPosition,
      labelDragging: !!labelDrag.current,
      label: labelKey
          ? {
              text: labelKey,
              x: label.x,
              y: label.y,
              horizontal: label.horizontal,
            }
          : undefined,
    }),
    [
      id,
      props.source,
      props.sourceHandleId,
      route,
      props.style?.stroke,
      data?.valueKey,
      selected,
      sourcePosition,
      targetPosition,
      !!labelDrag.current,
      labelKey,
      label.x,
      label.y,
      label.horizontal,
    ],
  );
  const { edges: renderedEdges, getRenderedEdges } = useRenderedEdge(rendered);
  const drawing = useMemo(() => wireDrawing(rendered, renderedEdges), [rendered, renderedEdges]);
  const hitRoute = drag.current?.route ?? sharedSections(rendered, renderedEdges);
  const starts = useMemo(
    () => wireHighlightStarts(renderedEdges, wireSelection.selection),
    [renderedEdges, wireSelection.selection],
  );
  const stroke = highlightedDrawing(route, drawing.sharedLength, starts.get(id));
  const labelHighlighted = pointHighlighted(rendered, label, starts);

  function selectSection(index: number) {
    wireSelection.select(selectWireSection(rendered, getRenderedEdges(), index));
    data?.select();
    setNodes((nodes) => nodes.map((node) => (node.selected ? { ...node, selected: false } : node)));
    // An already selected edge may own pointer capture for a segment/label drag.
    // Keep that DOM node mounted instead of requesting a new React Flow selection.
    if (selected && !getRenderedEdges().some(edge => edge.id !== id && edge.selected)) return;
    setEdges((edges) => edges.map((edge) => ({ ...edge, selected: edge.id === id })));
  }

  function cancel() {
    const current = drag.current;
    drag.current = null;
    setDraft(null);
    wirePreview.clearPreview();
    setSnapTarget(null);
    if (current?.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
    cancelLabel();
  }
  function cancelLabel() {
    const current = labelDrag.current;
    labelDrag.current = null;
    setLabelDraft(null);
    if (current?.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
  }
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && (drag.current || labelDrag.current)) {
        event.preventDefault();
        cancel();
      }
    };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, []);

  function startLabel(event: PointerEvent<SVGTextElement | HTMLDivElement>) {
    if (event.button !== 0 || drag.current || labelDrag.current) return;
    event.preventDefault();
    event.stopPropagation();
    draggedStep.current = false;
    if (!pipeline) selectSection(nearestSection(sharedSections(rendered, getRenderedEdges()), label));
    // Capture on the pressed button so a stationary pointer still produces its native click.
    const element = (pipeline && (event.target as Element).closest?.('button')) || event.currentTarget;
    element.setPointerCapture(event.pointerId);
    labelDrag.current = {
      pointerId: event.pointerId,
      element,
      route,
      flowStart: screenToFlowPosition({ x: event.clientX, y: event.clientY }),
      screenStart: { x: event.clientX, y: event.clientY },
      anchor: { x: label.x, y: label.y },
      initialPosition: label.position,
      position: label.position,
      moved: false,
    };
    setLabelDraft(label.position);
  }
  function moveLabel(event: PointerEvent<SVGTextElement | HTMLDivElement>) {
    const current = labelDrag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    if (
      !current.moved &&
      Math.hypot(event.clientX - current.screenStart.x, event.clientY - current.screenStart.y) < 2
    )
      return;
    current.moved = true;
    const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
    current.position = projectLabelPosition(
      current.route,
      {
        x: current.anchor.x + point.x - current.flowStart.x,
        y: current.anchor.y + point.y - current.flowStart.y,
      },
      current.position,
    );
    setLabelDraft(current.position);
  }
  function finishLabel(event: PointerEvent<SVGTextElement | HTMLDivElement>) {
    const current = labelDrag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.stopPropagation();
    labelDrag.current = null;
    setLabelDraft(null);
    if (current.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
    draggedStep.current = current.moved;
    if (current.moved && Math.abs(current.position - current.initialPosition) > 1e-6)
      data?.changeLabelPosition(current.position);
  }

  function start(
    event: PointerEvent<HTMLButtonElement | SVGPathElement>,
    index: number,
    kind: Drag['kind'],
  ) {
    if (event.button !== 0 || labelDrag.current) return;
    event.preventDefault();
    event.stopPropagation();
    if (kind === 'segment') selectSection(index);
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      pointerId: event.pointerId,
      element: event.currentTarget,
      index,
      kind,
      route: hitRoute,
      targets:
        kind === 'segment'
          ? sharedDragTargets(rendered, hitRoute, index, getRenderedEdges(), event.altKey)
          : [],
      flowStart: screenToFlowPosition({ x: event.clientX, y: event.clientY }),
      start: { x: event.clientX, y: event.clientY },
      points: controls.map((p) => ({ ...p })),
      moved: false,
    };
  }
  function move(event: PointerEvent<HTMLButtonElement | SVGPathElement>) {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    const dx = event.clientX - current.start.x,
      dy = event.clientY - current.start.y;
    const distance =
      current.kind === 'segment'
        ? Math.abs(current.route[current.index].y === current.route[current.index + 1].y ? dy : dx)
        : Math.hypot(dx, dy);
    if (!current.moved && distance < 2) return;
    current.moved = true;
    const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
    setSnapTarget(null);
    if (current.kind === 'segment') {
      const movement = snapSegmentMovement(
        current.route,
        current.index,
        { x: point.x - current.flowStart.x, y: point.y - current.flowStart.y },
        event.altKey ? 0 : 12 / zoom,
      );
      if (movement.snapped) setSnapTarget('self');
      current.points = moveRouteSegment(
        current.route,
        current.index,
        movement.delta,
        sourcePosition,
        targetPosition,
        movement.coordinate,
      );
      if (current.targets.length > 1) {
        current.sharedChanges = moveSharedTargets(
          current.targets,
          movement.delta,
          movement.coordinate,
        );
        if (!event.altKey)
          current.sharedChanges = mergeSharedWireChanges(
            current.sharedChanges,
            current.targets,
            getRenderedEdges(),
            movement.delta,
            12 / zoom,
            8 / zoom,
          );
        current.points = current.sharedChanges.find((change) => change.id === id)!.waypoints;
        wirePreview.setPreview(current.sharedChanges);
      } else if (!event.altKey) {
        const moved = simplifyRoute(
          manualRoute(
            { x: sourceX, y: sourceY },
            { x: targetX, y: targetY },
            sourcePosition,
            targetPosition,
            current.points,
          ),
        );
        const merged = mergeNearbyWire(
          { ...rendered, points: moved },
          movedSection(moved, current.route, current.index, movement.delta),
          getRenderedEdges(),
          sourcePosition,
          targetPosition,
          // A nearby peer must not pull an already straightened section apart.
          movement.snapped ? 0 : 12 / zoom,
          8 / zoom,
        );
        if (merged) {
          current.points = merged.waypoints;
          setSnapTarget('peer');
        }
      }
    } else current.points[current.index] = point;
    setDraft(current.points.map((p) => ({ ...p })));
  }
  function finish(event: PointerEvent<HTMLButtonElement | SVGPathElement>) {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.stopPropagation();
    drag.current = null;
    setDraft(null);
    wirePreview.clearPreview();
    setSnapTarget(null);
    if (current.element.hasPointerCapture(current.pointerId))
      current.element.releasePointerCapture(current.pointerId);
    if (current.moved) {
      if (current.sharedChanges) data?.changeSharedRoutes(current.sharedChanges);
      else data?.changeRoute(current.points);
    }
  }

  const labelEvents = {
    onPointerDown: startLabel,
    onPointerMove: moveLabel,
    onPointerUp: finishLabel,
    onPointerCancel: cancelLabel,
    onLostPointerCapture: cancelLabel,
    onClick: (event: React.MouseEvent) => event.stopPropagation(),
    onDoubleClick: (event: React.MouseEvent) => {
      event.stopPropagation();
      if (data?.labelPosition !== undefined) data.changeLabelPosition();
    },
  };

  return (
    <>
      <g
        onDoubleClick={(event) => {
          event.stopPropagation();
          const point = screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
          });
          // Preserve the current polyline when first pinning an automatic route.
          const next = [...(data?.waypoints ?? route.slice(1, -1))];
          const insertionRoute = manualRoute(
            { x: sourceX, y: sourceY },
            { x: targetX, y: targetY },
            sourcePosition,
            targetPosition,
            next,
          );
          next.splice(routeInsertIndex(insertionRoute, point), 0, point);
          data?.changeRoute(next);
        }}
      >
        <defs>
          <marker
            id={arrowId}
            viewBox="-10 -10 20 20"
            refX="0"
            refY="0"
            markerWidth={slide ? 20 : 16}
            markerHeight={slide ? 20 : 16}
            orient="auto-start-reverse"
          >
            <polyline points="-5,-4 0,0 -5,4 -5,-4" fill={highlightColor} stroke={highlightColor} />
          </marker>
        </defs>
        <BaseEdge
          id={id}
          className="wire-routed wire-normal"
          path={stroke.normal}
          markerStart={props.markerStart}
          markerEnd={stroke.normal && !stroke.highlighted ? props.markerEnd : undefined}
          interactionWidth={0}
          style={{
            ...props.style,
            strokeLinejoin: 'miter',
            strokeLinecap: 'butt',
          }}
        />
        {stroke.highlighted && (
          <BaseEdge
            className="wire-routed wire-highlighted"
            path={stroke.highlighted}
            markerEnd={props.markerEnd ? `url(#${arrowId})` : undefined}
            interactionWidth={0}
            style={{
              ...props.style,
              stroke: highlightColor,
              strokeWidth: slide ? 2.5 : 2,
              strokeLinejoin: 'miter',
              strokeLinecap: 'butt',
              pointerEvents: 'none',
            }}
          />
        )}
        <path
          className="react-flow__edge-interaction"
          d={path}
          fill="none"
          stroke="transparent"
          strokeOpacity={0}
          strokeWidth={props.interactionWidth ?? 24}
          onClick={(event) => {
            if (event.shiftKey || event.metaKey || event.ctrlKey) {
              wireSelection.clear();
              return;
            }
            event.stopPropagation();
            const point = screenToFlowPosition({
              x: event.clientX,
              y: event.clientY,
            });
            selectSection(nearestSection(sharedSections(rendered, getRenderedEdges()), point));
          }}
        />
        {selected &&
          hitRoute.slice(1).map((point, index) => {
            const startPoint = hitRoute[index];
            if (Math.hypot(point.x - startPoint.x, point.y - startPoint.y) < 1e-6) return null;
            const horizontal = point.y === startPoint.y;
            return (
              <path
                key={index}
                className={`edge-segment-hit nodrag nopan ${horizontal ? 'horizontal' : 'vertical'}`}
                data-segment-index={index}
                aria-label={`连线${horizontal ? '水平' : '竖直'}段 ${index + 1}`}
                data-shared-count={
                  sharedDragTargets(rendered, hitRoute, index, renderedEdges).length
                }
                d={`M ${startPoint.x} ${startPoint.y} L ${point.x} ${point.y}`}
                fill="none"
                stroke="transparent"
                strokeWidth={14 / zoom}
                style={{ strokeOpacity: drag.current ? 0 : undefined }}
                onPointerDown={(event) => start(event, index, 'segment')}
                onPointerMove={move}
                onPointerUp={finish}
                onPointerCancel={cancel}
                onLostPointerCapture={cancel}
                onClick={(event) => event.stopPropagation()}
              />
            );
          })}
        {!pipeline && !capsule && props.label != null && drawing.showLabel && (
          <text
            className={`react-flow__edge-text edge-side-label nodrag nopan ${labelDrag.current ? 'dragging' : ''}`}
            x={label.x + (label.horizontal ? 0 : 8 / zoom)}
            y={label.y - (label.horizontal ? 8 / zoom : 0)}
            data-anchor-x={label.x}
            data-anchor-y={label.y}
            data-label-position={label.position}
            textAnchor={label.horizontal ? 'middle' : 'start'}
            dominantBaseline={label.horizontal ? 'auto' : 'central'}
            aria-label="连线文字，沿连线拖动"
            style={{
              ...props.labelStyle,
              ...(labelHighlighted ? { fill: 'var(--accent)', fontWeight: 600 } : {}),
            }}
            {...labelEvents}
          >
            <title>沿连线拖动文字；双击恢复默认位置</title>
            {props.label}
          </text>
        )}
      </g>
      {pipeline && <EdgeLabelRenderer>
        <div className="edge-op-pipeline nodrag nopan" data-label-position={label.position}
          data-multi={edgeOps.length > 1 || undefined}
          style={{ transform: `translate(${label.x}px, ${label.y}px) translate(-50%, -50%)` }}
          aria-label="内嵌算子，沿连线拖动" {...labelEvents}>
          <div className="edge-op-steps">
            {edgeOps.map((step, index) => <Fragment key={step.id}>
              {index > 0 && <span aria-hidden="true" className="edge-op-chevron">↓</span>}
              <button type="button" className={`edge-op-step tone-${step.tone}`}
                data-status={step.status} data-active={data?.activeNodeId === step.id}
                title={[...new Set([step.label.trim(), step.detail ?? step.badge, step.shapeText, step.status].filter(Boolean))].join(' · ')}
                aria-pressed={data?.selectedNodeId === step.id}
                onKeyDown={event => {
                  if (event.key === 'Enter' || event.key === ' ') event.stopPropagation();
                }}
                onClick={event => {
                  event.stopPropagation();
                  if (event.detail && draggedStep.current) return;
                  wireSelection.clear();
                  setEdges(edges => edges.map(edge => edge.selected ? { ...edge, selected: false } : edge));
                  useEditor.getState().select(step.id);
                }}>{step.badge}</button>
            </Fragment>)}
          </div>
          {(data?.annotation || props.label != null) && <small>
            {data?.annotation}{data?.annotation && props.label != null && ' · '}{props.label}
          </small>}
        </div>
      </EdgeLabelRenderer>}
      {!pipeline && capsule && labelKey && drawing.showLabel && (
        <EdgeLabelRenderer>
          <div className="edge-slide-label nodrag nopan" aria-label="连线文字，沿连线拖动"
            data-label-position={label.position} data-highlighted={labelHighlighted}
            style={{ transform: `translate(${label.x}px, ${label.y}px) ${label.horizontal
              ? 'translate(-50%, calc(-100% - 10px))' : 'translate(10px, -50%)'}` }}
            title="沿连线拖动文字；双击恢复默认位置" {...labelEvents}>
            {data?.annotation && <strong>{data.annotation}</strong>}
            {props.label != null && <span>{props.label}</span>}
          </div>
        </EdgeLabelRenderer>
      )}
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
              onPointerDown={(event) => start(event, index, 'point')}
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
          {wireSelection.selection?.edgeId === id && <div
            className="edge-route-toolbar nodrag nopan"
            style={{
              transform: `translate(${toolbar.x + (toolbar.horizontal ? 0 : 8 / zoom)}px, ${toolbar.y + 18 / zoom}px) scale(${1 / zoom}) translate(${toolbar.horizontal ? '-50%' : '0'}, 0)`,
            }}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <span role={snapTarget ? 'status' : undefined}>
              {drag.current && drag.current.targets.length > 1
                ? `拖动公共线段 · ${drag.current.targets.length} 条分支同步移动`
                : snapTarget === 'self'
                  ? '松开拉直折线 · Alt 暂不吸附'
                  : snapTarget === 'peer'
                    ? '松开合并同源连线 · Alt 暂不合并'
                    : '拖动公共段同步分支 · Alt 拖动单条线'}
            </span>
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
          </div>}
        </EdgeLabelRenderer>
      )}
    </>
  );
}
