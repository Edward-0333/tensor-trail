import {
  createContext,
  useContext,
  useLayoutEffect,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from 'react';
import { EdgeLabelRenderer, useViewport } from '@xyflow/react';
import { branchJunctions, type RoutedEdge } from './edgeRouting';
import { prioritizeWires, type WireRouteChange } from './sharedWires';
import { junctionHighlighted, wireHighlightStarts, type WireSelection } from './wireHighlight';

// Transient rendered geometry, scoped to the current canvas. Never changes the model
// or its undo history; previews and node/port movement update the same junction view.
function createRoutes() {
  const routes = new Map<string, RoutedEdge>();
  const listeners = new Set<() => void>();
  let snapshot: RoutedEdge[] = [];
  let preview = new Map<string, WireRouteChange['waypoints']>();
  let previewOwner: string | null = null;
  const previewListeners = new Set<() => void>();
  let selection: WireSelection | null = null;
  const selectionListeners = new Set<() => void>();
  const notify = () => {
    snapshot = [...routes.values()];
    listeners.forEach((listener) => listener());
  };
  return {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot: () => snapshot,
    getSelection: () => selection,
    subscribeSelection(listener: () => void) {
      selectionListeners.add(listener);
      return () => {
        selectionListeners.delete(listener);
      };
    },
    select(selectionValue: WireSelection | null) {
      selection = selectionValue;
      selectionListeners.forEach((listener) => listener());
    },
    clearSelection(id: string) {
      if (selection?.edgeId !== id) return;
      selection = null;
      selectionListeners.forEach((listener) => listener());
    },
    getPreview: () => preview,
    subscribePreview(listener: () => void) {
      previewListeners.add(listener);
      return () => {
        previewListeners.delete(listener);
      };
    },
    setPreview(owner: string, changes: WireRouteChange[]) {
      previewOwner = owner;
      preview = new Map(changes.map((change) => [change.id, change.waypoints]));
      previewListeners.forEach((listener) => listener());
    },
    clearPreview(owner: string) {
      if (previewOwner !== owner) return;
      previewOwner = null;
      preview = new Map();
      previewListeners.forEach((listener) => listener());
    },
    publish(route: RoutedEdge) {
      if (JSON.stringify(routes.get(route.id)) === JSON.stringify(route)) return;
      routes.set(route.id, route);
      notify();
    },
    remove(id: string) {
      // Geometry can unmount while React Flow updates selection/z-order.
      // Selection is cleared by deselection, not by transient route removal.
      if (routes.delete(id)) notify();
    },
  };
}
const RoutesContext = createContext<ReturnType<typeof createRoutes> | null>(null);

export function EdgeRoutesProvider({ children }: { children: ReactNode }) {
  const routes = useMemo(createRoutes, []);
  return <RoutesContext.Provider value={routes}>{children}</RoutesContext.Provider>;
}

export function useRenderedEdge(route: RoutedEdge) {
  const routes = useContext(RoutesContext)!;
  useLayoutEffect(() => {
    routes.publish(route);
  }, [routes, route]);
  useLayoutEffect(() => () => routes.remove(route.id), [routes, route.id]);
  const edges = useSyncExternalStore(routes.subscribe, routes.getSnapshot);
  return { edges, getRenderedEdges: routes.getSnapshot };
}

export function useWirePreview(id: string) {
  const routes = useContext(RoutesContext)!;
  const preview = useSyncExternalStore(routes.subscribePreview, routes.getPreview);
  useLayoutEffect(() => () => routes.clearPreview(id), [routes, id]);
  return {
    points: preview.get(id),
    setPreview: (changes: WireRouteChange[]) => routes.setPreview(id, changes),
    clearPreview: () => routes.clearPreview(id),
  };
}

export function useWireSelection(id: string, selected?: boolean) {
  const routes = useContext(RoutesContext)!;
  const selection = useSyncExternalStore(routes.subscribeSelection, routes.getSelection);
  useLayoutEffect(() => {
    if (!selected) routes.clearSelection(id);
  }, [routes, id, selected]);
  return {
    selection,
    select: routes.select,
    clear: () => routes.clearSelection(id),
  };
}

export function EdgeJunctions() {
  const routes = useContext(RoutesContext)!;
  const edges = useSyncExternalStore(routes.subscribe, routes.getSnapshot);
  const selection = useSyncExternalStore(routes.subscribeSelection, routes.getSelection);
  const starts = useMemo(() => wireHighlightStarts(edges, selection), [edges, selection]);
  const junctions = useMemo(() => branchJunctions(prioritizeWires(edges)), [edges]);
  const { zoom } = useViewport();
  return (
    <EdgeLabelRenderer>
      {junctions.map((point) => (
        <div
          key={point.key}
          className="edge-junction"
          role="img"
          aria-label="同一输出端口的分支连接点"
          data-source={point.source}
          data-source-port={point.sourcePort}
          data-flow-x={point.x}
          data-flow-y={point.y}
          data-highlighted={junctionHighlighted(point, edges, starts) || undefined}
          style={{
            transform: `translate(-50%, -50%) translate(${point.x}px, ${point.y}px)`,
            width: 8 / zoom,
            height: 8 / zoom,
            background: junctionHighlighted(point, edges, starts)
              ? 'var(--accent)'
              : (point.color ?? 'var(--edge)'),
          }}
        />
      ))}
    </EdgeLabelRenderer>
  );
}
