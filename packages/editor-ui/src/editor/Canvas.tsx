import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  ControlButton,
  MiniMap,
  MarkerType,
  ConnectionLineType,
  SelectionMode,
  ReactFlow,
  useNodesState,
  useEdgesState,
  useNodesInitialized,
  useReactFlow,
  useViewport,
  ViewportPortal,
} from '@xyflow/react';
import { Maximize } from 'lucide-react';
import type { Graph, GraphResult, ModelNode } from '@tensor-trail/model-core/core/types';
import { layoutIfRegions } from '../views/ifRegionLayout';
import { layoutRegions, measuredNodeSizes } from '../views/layoutRegions';
import { constrainLayoutMoves, isLayoutRegion, layoutRegionOf, layoutRegionOutputOwners, layoutRegionRect } from '@tensor-trail/model-core/core/layoutRegions';
import { isCodeRegion } from '@tensor-trail/model-core/core/ifRegions';
import { isCountContract, valueEdgeLabel, valueEdgeStyle } from '../views/valueEdgeStyle';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules/index';
import { categoryColor, ModuleNode, type TrailNode } from '../views/ModuleNode';
import { useEditor } from './store';
import { MODULE_DRAG_TYPE, readModuleDrag } from './moduleDrag';
import { EditableEdge, type TrailEdge } from '../views/EditableEdge';
import { EdgeJunctions, EdgeRoutesProvider } from '../views/EdgeJunctions';
import {
  nearbyCalculation,
  type NodeBounds,
} from '@tensor-trail/model-core/core/calculationGroups';
import {
  effectiveValueNode,
  isValueNode,
  linkedValueNodes,
} from '@tensor-trail/model-core/core/valueLinks';
import { removeValueNodes } from '@tensor-trail/model-core/editing/valueLinkEditing';
import { localParameterColors } from '../views/localParameterColors';
import { levelInputs } from '../views/levelInputs';
import { finalOutputValues } from '../views/finalOutputs';
import { createRoutes } from '../views/routeStore';
import { layoutLoopFeedback, loopFeedback } from '../views/loopFeedback';
import { LoopFeedbackOverlay } from '../views/LoopFeedbackOverlay';
import {
  nodeWirePreview,
  nodeDragRoutes,
  snapNodesToWires,
  type NodePosition,
  type WireSnapGuide,
} from './nodeWireSnap';

const nodeTypes = { module: ModuleNode };
const edgeTypes = { editable: EditableEdge };
const ariaLabelConfig = {
  'controls.ariaLabel': '画布缩放控制',
  'controls.zoomIn.ariaLabel': '放大画布',
  'controls.zoomOut.ariaLabel': '缩小画布',
  'controls.fitView.ariaLabel': '适配全图',
  'minimap.ariaLabel': '架构缩略图，可平移或缩放',
};
function flowsLeft(graph: Graph, id: string): boolean {
  const current = graph.nodes.find((n) => n.id === id)!;
  const next = graph.edges
    .filter((e) => e.source === id)
    .map((e) => graph.nodes.find((n) => n.id === e.target)!)
    .find((n) => n.position.x !== current.position.x);
  if (next) return next.position.x < current.position.x;
  const previous = graph.edges
    .filter((e) => e.target === id)
    .map((e) => graph.nodes.find((n) => n.id === e.source)!)
    .find((n) => n.position.x !== current.position.x);
  return !!previous && previous.position.x > current.position.x;
}
export function Canvas({
  graph,
  result,
  step,
  theme,
  owner,
}: {
  graph: Graph;
  result?: GraphResult;
  step: number;
  theme: 'dark' | 'light';
  owner?: ModelNode;
}) {
  const selected = useEditor((s) => s.selected),
    selectedIds = useEditor((s) => s.selectedIds),
    pasteCount = useEditor((s) => s.pasteCount),
    path = useEditor((s) => s.path);
  const initialized = useNodesInitialized();
  const { fitView, fitBounds, screenToFlowPosition, getNodes } = useReactFlow<TrailNode>();
  const knownRegions = useRef(new Set(graph.nodes.filter(isLayoutRegion).map(n => n.id)));
  useEffect(() => {
    const regions = graph.nodes.filter(isLayoutRegion);
    const added = regions.find(n => !knownRegions.current.has(n.id));
    knownRegions.current = new Set(regions.map(n => n.id));
    if (!added) return;
    const frame = requestAnimationFrame(() => {
      void fitBounds(layoutRegionRect(graph, added, useEditor.getState().nodeSizes), { padding: .25, duration: 250 });
    });
    return () => cancelAnimationFrame(frame);
  }, [graph, fitBounds]);
  const feedback = useMemo(() => loopFeedback(graph, owner), [graph, owner]);
  const [selectedFeedback, setSelectedFeedback] = useState<string>();
  const fitGraph = useCallback(() => {
    const { bounds } = layoutLoopFeedback(feedback, getNodes(), owner?.loopFeedbackRoutes);
    if (bounds) void fitBounds(bounds, { padding: 0.18 });
    else void fitView({ padding: 0.18, maxZoom: 1.1 });
  }, [feedback, getNodes, fitBounds, fitView, owner?.loopFeedbackRoutes]);
  const viewport = useViewport();
  const renderedRoutes = useMemo(createRoutes, []);
  const wireDrag = useRef<{
    initial: NodePosition[];
    routes: ReturnType<typeof renderedRoutes.getSnapshot>;
    guides: WireSnapGuide[];
  } | null>(null);
  const [wireGuides, setWireGuides] = useState<WireSnapGuide[]>([]);
  const alignDrag = (moved: TrailNode[], disabled: boolean) => {
    const session = wireDrag.current;
    if (!session)
      return { positions: moved.map((n) => ({ id: n.id, position: n.position })), guides: [] };
    const hasRegion = moved.some(n => isLayoutRegion(n.data.model) || layoutRegionOf(graph, n.id));
    const snapped = snapNodesToWires(
      graph,
      session.routes,
      session.initial,
      moved,
      viewport.zoom,
      session.guides,
      disabled || hasRegion,
    );
    snapped.positions = constrainLayoutMoves(graph, snapped.positions, measuredNodeSizes(getNodes()));
    session.guides = snapped.guides;
    return snapped;
  };
  const [mergePreview, setMergePreview] = useState<{ moving: string; target: string } | null>(null);
  const bounds = (node: TrailNode): NodeBounds => ({
    id: node.id,
    position: node.position,
    width: node.measured?.width ?? 0,
    height: node.measured?.height ?? 0,
  });
  const mergeTarget = (node: TrailNode, moved: TrailNode[], altKey: boolean) =>
    !altKey && moved.length === 1
      ? nearbyCalculation(graph, bounds(node), getNodes().map(bounds), registry)
      : undefined;
  const [dropActive, setDropActive] = useState(false);
  useEffect(() => {
    const clearDrop = () => setDropActive(false);
    window.addEventListener('dragend', clearDrop);
    window.addEventListener('drop', clearDrop);
    return () => {
      window.removeEventListener('dragend', clearDrop);
      window.removeEventListener('drop', clearDrop);
    };
  }, []);
  const fitted = useRef(false);
  useEffect(() => {
    if (!initialized || fitted.current) return;
    // Wait for measured cards before fitting a newly opened graph.
    const frame = requestAnimationFrame(() => {
      fitGraph();
      fitted.current = true;
    });
    return () => cancelAnimationFrame(frame);
  }, [initialized, fitGraph]);
  const derived = useMemo(() => {
    const localColors = localParameterColors(graph);
    const finalOutputs = finalOutputValues(graph, result, owner);
    const regionOutputs = layoutRegionOutputOwners(graph);
    const inputBoundaries = new Map(levelInputs(graph, path.length === 0).map(input => [input.id, input]));
    const linked = new Set(
      graph.nodes.flatMap((node) => (node.valueSource ? [node.id, node.valueSource] : [])),
    );
    const highlighted = new Set(
      [...selectedIds, ...(selected ? [selected] : [])].flatMap((id) => {
        const selectedNode = graph.nodes.find((node) => node.id === id);
        if (!selectedNode || !isValueNode(selectedNode.type)) return [];
        const members = linkedValueNodes(graph, id);
        return members.length > 1 ? members.map((member) => member.id) : [];
      }),
    );
    return graph.nodes.map((stored) => {
      const model = effectiveValueNode(graph, stored),
        nodePorts = ports(model, registry);
      const outputRegion = regionOutputs.get(model.id);
      return {
        id: model.id,
        type: 'module' as const,
        position: model.position,
        selected: model.id === selected,
        deletable: !!stored.valueSource || !['Inlet', 'Outlet'].includes(model.type),
        className: [
          isCodeRegion(model) ? 'if-region-flow' : isLayoutRegion(model) ? 'layout-region-flow' : '',
          linked.has(model.id) ? 'value-linked' : '',
          highlighted.has(model.id) ? 'value-link-highlight' : '',
        ]
          .filter(Boolean)
          .join(' '),
        data: {
          model,
          inputBoundary: inputBoundaries.get(model.id),
          finalOutput: finalOutputs.has(model.id),
          regionOutput: outputRegion ? { id: outputRegion.id, label: outputRegion.label, color: String(outputRegion.params.color) } : undefined,
          localParameterColor: localColors.get(model.id),
          result: result?.nodes[model.id],
          inputs: stored.valueSource ? [] : nodePorts.inputs,
          outputs: nodePorts.outputs,
          category: registry.get(model.type)?.category ?? '基础层',
          reverse: flowsLeft(graph, model.id),
          active: step >= 0 && result?.order[step] === model.id,
          dimmed: step >= 0 && (result?.order.indexOf(model.id) ?? 0) > step,
          open: () => {
            if (model.subgraph) useEditor.getState().navigate([...path, model.id]);
          },
        },
      };
    });
  }, [graph, result, owner, selected, selectedIds, path, step]);
  const [nodes, setNodes, onNodesChange] = useNodesState<TrailNode>(derived);
  useEffect(() => {
    useEditor.getState().setNodeSizes(measuredNodeSizes(nodes));
  }, [nodes]);
  const feedbackRoutes = layoutLoopFeedback(feedback, nodes, owner?.loopFeedbackRoutes).routes;
  useEffect(() => {
    if (selected || selectedIds.length) setSelectedFeedback(undefined);
  }, [selected, selectedIds]);
  useEffect(() => {
    setNodes((current) => {
      const selection = new Set(current.filter((n) => n.selected).map((n) => n.id));
      const measurements = new Map(current.map(node => [node.id, node.measured]));
      // Selection/data refreshes must not briefly unmount measurement-based overlays
      // and lose pointer capture while a return path is being dragged.
      return derived.map((node) => ({ ...node, measured: measurements.get(node.id), selected: selection.has(node.id) }));
    });
  }, [derived, setNodes]);
  useEffect(() => {
    setNodes((current) => {
      if (selected && current.some((n) => n.id === selected && n.selected)) return current;
      return current.map((node) => ({ ...node, selected: node.id === selected }));
    });
  }, [selected, setNodes]);
  useEffect(() => {
    if (pasteCount > 0)
      setNodes((current) =>
        current.map((node) => ({ ...node, selected: selectedIds.includes(node.id) })),
      );
    // Only a paste requests a group selection; canvas selection changes are read back below.
  }, [pasteCount, setNodes]);
  const rememberSelection = useCallback(({ nodes }: { nodes: TrailNode[] }) => {
    const ids = nodes.map((node) => node.id);
    if (JSON.stringify(ids) !== JSON.stringify(useEditor.getState().selectedIds))
      useEditor.setState({ selectedIds: ids });
  }, []);
  const derivedEdges: TrailEdge[] = useMemo(
    () =>
      graph.edges.map((edge) => {
        const countContract = isCountContract(graph.nodes.find(node => node.id === edge.target), edge.targetPort);
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourcePort,
          targetHandle: edge.targetPort,
          type: 'editable',
          data: {
            valueKey: JSON.stringify([
              graph.nodes.find((node) => node.id === edge.source)?.valueSource ?? edge.source,
              edge.sourcePort,
            ]),
            waypoints: edge.waypoints,
            labelPosition: edge.labelPosition,
            changeRoute: (points) => useEditor.getState().setEdgeWaypoints(edge.id, points),
            changeSharedRoutes: (changes) => useEditor.getState().setSharedWireWaypoints(changes),
            changeLabelPosition: (position) =>
              useEditor.getState().setEdgeLabelPosition(edge.id, position),
            select: () => { setSelectedFeedback(undefined); useEditor.getState().select(null); },
          },
          label: valueEdgeLabel(result?.nodes[edge.source]?.outputs[edge.sourcePort], countContract),
          animated: step >= 0 && result?.order[step] === edge.source,
          style: valueEdgeStyle(result?.nodes[edge.source]?.outputs[edge.sourcePort], result?.nodes[edge.source]?.status === 'error', countContract),
          labelStyle: {
            fill: 'var(--text-secondary)',
            fontSize: 10,
            fontFamily: 'Consolas, monospace',
          },
        };
      }),
    [graph, result, step],
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState<TrailEdge>(derivedEdges);
  useEffect(
    () =>
      setEdges((current) => {
        const selection = new Set(current.filter((edge) => edge.selected).map((edge) => edge.id));
        return derivedEdges.map((edge) => ({ ...edge, selected: selection.has(edge.id) }));
      }),
    [derivedEdges, setEdges],
  );
  const startNodeDrag = () => {
    setMergePreview(null); setWireGuides([]);
    wireDrag.current = {
      initial: getNodes().map(n => ({ id: n.id, position: { ...n.position } })),
      routes: renderedRoutes.getSnapshot(), guides: [],
    };
  };
  const updateDragPositions = (positions: NodePosition[]) => {
    const byId = new Map(positions.map(n => [n.id, n.position]));
    setNodes(current => current.map(n => byId.has(n.id) ? { ...n, position: byId.get(n.id)! } : n));
  };
  const dragNodes = (event: { altKey?: boolean }, node: TrailNode, moved: TrailNode[]) => {
    const snapped = alignDrag(moved, Boolean(event.altKey));
    updateDragPositions(snapped.positions); setWireGuides(snapped.guides);
    if (wireDrag.current) {
      const preview = nodeWirePreview(graph, wireDrag.current.routes, wireDrag.current.initial, snapped.positions);
      if (preview.length) renderedRoutes.setPreview('node-drag', preview);
      else renderedRoutes.clearPreview('node-drag');
    }
    const position = snapped.positions.find(n => n.id === node.id)?.position ?? node.position;
    const target = mergeTarget({ ...node, position }, moved, Boolean(event.altKey));
    setMergePreview(target ? { moving: node.id, target } : null);
  };
  const stopNodeDrag = (event: { altKey?: boolean }, node: TrailNode, moved: TrailNode[]) => {
    const snapped = alignDrag(moved, Boolean(event.altKey));
    const position = snapped.positions.find(n => n.id === node.id)?.position ?? node.position;
    const target = mergeTarget({ ...node, position }, moved, Boolean(event.altKey));
    const session = wireDrag.current;
    const routeChanges = session ? nodeDragRoutes(graph, session.routes, session.initial,
      snapped.positions, viewport.zoom, Boolean(event.altKey)) : [];
    setMergePreview(null); setWireGuides([]); wireDrag.current = null;
    renderedRoutes.clearPreview('node-drag'); updateDragPositions(snapped.positions);
    useEditor.getState().finishNodeDrag(snapped.positions, target ? [node.id, target] : undefined, routeChanges);
  };
  return (
    <EdgeRoutesProvider routes={renderedRoutes}>
      <ReactFlow<TrailNode, TrailEdge>
        className={dropActive ? 'module-drop-active' : undefined}
        colorMode={theme}
        ariaLabelConfig={ariaLabelConfig}
        nodes={layoutRegions(layoutIfRegions(nodes)).map((node) => ({
          ...node,
          className: [
            node.className,
            mergePreview?.target === node.id ? 'calculation-merge-target' : '',
          ]
            .filter(Boolean)
            .join(' '),
        }))}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionLineType={ConnectionLineType.Step}
        onDragOver={(event) => {
          if (!event.dataTransfer.types.includes(MODULE_DRAG_TYPE)) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = 'copy';
          setDropActive(true);
        }}
        onDragLeave={(event) => {
          if (
            !(event.relatedTarget instanceof Node) ||
            !event.currentTarget.contains(event.relatedTarget)
          )
            setDropActive(false);
        }}
        onDrop={(event) => {
          setDropActive(false);
          const module = readModuleDrag(event.dataTransfer);
          if (!module) return;
          event.preventDefault();
          const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
          useEditor.getState().add(module.type, module.variantId, position);
        }}
        onNodesChange={onNodesChange}
        onNodeClick={(_, node) => { setSelectedFeedback(undefined); useEditor.setState({ selected: node.id }); }}
        onSelectionChange={rememberSelection}
        onEdgeClick={() => { setSelectedFeedback(undefined); useEditor.getState().select(null); }}
        onPaneClick={() => { setSelectedFeedback(undefined); useEditor.getState().select(null); }}
        onSelectionStart={() => { setSelectedFeedback(undefined); useEditor.getState().select(null); }}
        onNodeDoubleClick={(_, node) => node.data.open()}
        onNodeDragStart={startNodeDrag}
        onNodeDrag={dragNodes}
        onNodeDragStop={stopNodeDrag}
        onSelectionDragStart={startNodeDrag}
        onSelectionDrag={(event, moved) => { if (moved.length) dragNodes(event, moved[0], moved); }}
        onSelectionDragStop={(event, moved) => { if (moved.length) stopNodeDrag(event, moved[0], moved); }}
        onBeforeDelete={async ({ nodes: removed, edges: removedEdges }) => {
          if (!removed.length && !removedEdges.length) return false;
          const state = useEditor.getState();
          // React Flow normally removes incident edges first. Keep them available for
          // shared-value promotion and commit the whole delete as one undoable edit.
          const explicitEdges = new Set(
            removedEdges.filter((edge) => !removed.length || edge.selected).map((edge) => edge.id),
          );
          if (
            state.edit((current, project) => {
              removeValueNodes(
                project.graph,
                state.path,
                removed.map((node) => node.id),
              );
              current.edges = current.edges.filter((edge) => !explicitEdges.has(edge.id));
            }, explicitEdges.size > 0 || removed.some(node => !isLayoutRegion(node.data.model)))
          )
            state.select(null);
          return false;
        }}
        onEdgesChange={onEdgesChange}
        onConnect={(connection) => {
          if (
            connection.source &&
            connection.target &&
            connection.sourceHandle &&
            connection.targetHandle
          )
            useEditor.getState().connect({
              source: connection.source,
              sourcePort: connection.sourceHandle,
              target: connection.target,
              targetPort: connection.targetHandle,
            });
        }}
        minZoom={0.04}
        maxZoom={2}
        panOnDrag={[1]}
        selectionOnDrag
        selectionMode={SelectionMode.Partial}
        deleteKeyCode={['Backspace', 'Delete']}
        defaultEdgeOptions={{
          interactionWidth: 24,
          markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--edge)', width: 16, height: 16 },
        }}
        elevateEdgesOnSelect
      >
        <EdgeJunctions />
        <ViewportPortal><LoopFeedbackOverlay routes={feedbackRoutes} layouts={owner?.loopFeedbackRoutes}
          selected={selectedFeedback}
          onSelect={state => {
            if (state === selectedFeedback) return;
            setSelectedFeedback(state);
            if (state) {
              useEditor.getState().select(null);
              setNodes(current => current.map(node => ({ ...node, selected: false })));
              setEdges(current => current.map(edge => ({ ...edge, selected: false })));
            }
          }}
          onChange={(state, layout) => { useEditor.getState().setLoopFeedbackLayout(state, layout); }}
        /></ViewportPortal>
        {dropActive && <div className="canvas-drop-hint">松开以添加模块</div>}
        {mergePreview && (
          <div className="canvas-merge-hint" role="status">
            松开合并计算 · 按住 Alt 仅移动
          </div>
        )}
        {!!wireGuides.length && !mergePreview && (
          <>
            <svg className="node-wire-guides" aria-hidden="true">
              {wireGuides.map((g) => (
                <line
                  key={g.key}
                  x1={g.from.x * viewport.zoom + viewport.x}
                  y1={g.from.y * viewport.zoom + viewport.y}
                  x2={g.to.x * viewport.zoom + viewport.x}
                  y2={g.to.y * viewport.zoom + viewport.y}
                />
              ))}
            </svg>
            <div className="canvas-merge-hint" role="status">
              连线已对齐 · 松开固定位置 · Alt 暂不吸附
            </div>
          </>
        )}
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--grid)" />
        <Controls showInteractive={false} showFitView={!feedback.length}>
          {!!feedback.length && <ControlButton className="react-flow__controls-fitview"
            onClick={fitGraph} title="适配全图" aria-label="适配全图"><Maximize /></ControlButton>}
        </Controls>
        <MiniMap
          style={{ width: 124, height: 84 }}
          nodeColor={(node) => node.className?.includes('if-region-flow') ? 'transparent' : categoryColor(String(node.data.category))}
          maskColor="var(--minimap-mask)"
          pannable
          zoomable
        />
      </ReactFlow>
    </EdgeRoutesProvider>
  );
}
