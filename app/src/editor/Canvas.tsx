import { useEffect, useMemo } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  MarkerType,
  ReactFlow,
  useNodesState,
  useEdgesState,
  type Edge,
} from '@xyflow/react';
import type { Graph, GraphResult } from '../core/types';
import { shapeText } from '../core/types';
import { ports } from '../core/graph';
import { registry } from '../modules';
import { ModuleNode, type TrailNode } from '../views/ModuleNode';
import { useEditor } from './store';

const nodeTypes = { module: ModuleNode };
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
}: {
  graph: Graph;
  result?: GraphResult;
  step: number;
}) {
  const selected = useEditor((s) => s.selected),
    path = useEditor((s) => s.path);
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
  useEffect(() => setNodes(derived), [derived, setNodes]);
  const derivedEdges: Edge[] = useMemo(
    () =>
      graph.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourcePort,
        targetHandle: edge.targetPort,
        type: 'smoothstep',
        label: result?.nodes[edge.source]?.outputs[edge.sourcePort]
          ? shapeText(result.nodes[edge.source].outputs[edge.sourcePort])
          : '待推导',
        animated: step >= 0 && result?.order[step] === edge.source,
        style: {
          stroke: result?.nodes[edge.source]?.status === 'error' ? '#df7781' : '#a5b5c5',
          strokeWidth: 1.5,
        },
        labelStyle: { fill: '#586b80', fontSize: 10, fontFamily: 'Consolas, monospace' },
        labelBgStyle: { fill: '#f8fafc', fillOpacity: 0.95 },
        labelBgPadding: [5, 3] as [number, number],
      })),
    [graph, result, step],
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(derivedEdges);
  useEffect(() => setEdges(derivedEdges), [derivedEdges, setEdges]);
  return (
    <ReactFlow<TrailNode>
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeClick={(_, node) => useEditor.getState().select(node.id)}
      onPaneClick={() => useEditor.getState().select(null)}
      onNodeDoubleClick={(_, node) => node.data.open()}
      onNodeDragStop={(_, node) => useEditor.getState().move(node.id, node.position)}
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
      fitView
      fitViewOptions={{ padding: 0.18, maxZoom: 1.1 }}
      minZoom={0.15}
      maxZoom={2}
      deleteKeyCode={['Backspace', 'Delete']}
      defaultEdgeOptions={{
        interactionWidth: 24,
        markerEnd: { type: MarkerType.ArrowClosed, color: '#a5b5c5', width: 16, height: 16 },
      }}
      elevateEdgesOnSelect
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#d6dee7" />
      <Controls showInteractive={false} />
      <MiniMap nodeColor="#b8cad8" maskColor="rgba(245,248,251,.7)" pannable zoomable />
    </ReactFlow>
  );
}
