import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  MarkerType,
  SelectionMode,
  ReactFlow,
  useNodesState,
  useEdgesState,
  useNodesInitialized,
  useReactFlow,
} from '@xyflow/react';
import type { Graph, GraphResult } from '../core/types';
import { shapeText } from '../core/types';
import { ports } from '../core/graph';
import { registry } from '../modules';
import { categoryColor, ModuleNode, type TrailNode } from '../views/ModuleNode';
import { useEditor } from './store';
import { MODULE_DRAG_TYPE, readModuleDrag } from './moduleDrag';
import { EditableEdge, type TrailEdge } from '../views/EditableEdge';

const nodeTypes = { module: ModuleNode };
const edgeTypes = { editable: EditableEdge };
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
}: {
  graph: Graph;
  result?: GraphResult;
  step: number;
  theme: 'dark' | 'light';
}) {
  const selected = useEditor((s) => s.selected),
    path = useEditor((s) => s.path);
  const initialized = useNodesInitialized();
  const { fitView, screenToFlowPosition } = useReactFlow();
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
      void fitView({ padding: 0.18, maxZoom: 1.1 });
      fitted.current = true;
    });
    return () => cancelAnimationFrame(frame);
  }, [initialized, fitView]);
  const derived = useMemo(
    () =>
      graph.nodes.map((model) => ({
        id: model.id,
        type: 'module' as const,
        position: model.position,
        selected: model.id === selected,
        deletable: !['Inlet', 'Outlet'].includes(model.type),
        data: {
          model,
          result: result?.nodes[model.id],
          ...ports(model, registry),
          category: registry.get(model.type)?.category ?? '基础层',
          reverse: flowsLeft(graph, model.id),
          active: step >= 0 && result?.order[step] === model.id,
          dimmed: step >= 0 && (result?.order.indexOf(model.id) ?? 0) > step,
          open: () => {
            if (model.subgraph) useEditor.getState().navigate([...path, model.id]);
          },
        },
      })),
    [graph, result, selected, path, step],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState<TrailNode>(derived);
  useEffect(() => {
    setNodes((current) => {
      const selection = new Set(current.filter((node) => node.selected).map((node) => node.id));
      return derived.map((node) => ({ ...node, selected: selection.has(node.id) }));
    });
  }, [derived, setNodes]);
  useEffect(() => {
    setNodes((current) => {
      if (selected && current.some((node) => node.id === selected && node.selected)) return current;
      return current.map((node) => ({ ...node, selected: node.id === selected }));
    });
  }, [selected, setNodes]);
  const derivedEdges: TrailEdge[] = useMemo(
    () =>
      graph.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourcePort,
        targetHandle: edge.targetPort,
        type: 'editable',
        data: {
          waypoints: edge.waypoints,
          changeRoute: (points) => useEditor.getState().setEdgeWaypoints(edge.id, points),
        },
        label: result?.nodes[edge.source]?.outputs[edge.sourcePort]
          ? shapeText(result.nodes[edge.source].outputs[edge.sourcePort])
          : '待推导',
        animated: step >= 0 && result?.order[step] === edge.source,
        style: {
          stroke: result?.nodes[edge.source]?.status === 'error' ? 'var(--danger)' : 'var(--edge)',
          strokeWidth: 1.5,
        },
        labelStyle: {
          fill: 'var(--text-secondary)',
          fontSize: 10,
          fontFamily: 'Consolas, monospace',
        },
        labelBgStyle: { fill: 'var(--canvas)', fillOpacity: 0.96 },
        labelBgPadding: [5, 3] as [number, number],
      })),
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
  return (
    <ReactFlow<TrailNode, TrailEdge>
      className={dropActive ? 'module-drop-active' : undefined}
      colorMode={theme}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
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
      onNodeClick={(_, node) => useEditor.getState().select(node.id)}
      onEdgeClick={() => useEditor.getState().select(null)}
      onPaneClick={() => useEditor.getState().select(null)}
      onSelectionStart={() => useEditor.getState().select(null)}
      onNodeDoubleClick={(_, node) => node.data.open()}
      onNodeDragStop={(_, __, moved) => {
        const positions = new Map(moved.map((node) => [node.id, node.position]));
        useEditor.getState().edit((current) => {
          for (const node of current.nodes) {
            const position = positions.get(node.id);
            if (position) node.position = { ...position };
          }
        }, false);
      }}
      onNodesDelete={(deleted) => useEditor.getState().removeNodes(deleted.map((n) => n.id))}
      onEdgesDelete={(deleted) => useEditor.getState().removeEdges(deleted.map((e) => e.id))}
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
      minZoom={0.15}
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
      {dropActive && <div className="canvas-drop-hint">松开以添加模块</div>}
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--grid)" />
      <Controls showInteractive={false} />
      <MiniMap
        style={{ width: 124, height: 84 }}
        nodeColor={(node) => categoryColor(String(node.data.category))}
        maskColor="var(--minimap-mask)"
        pannable
        zoomable
      />
    </ReactFlow>
  );
}
