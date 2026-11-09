import {
  createContext,
  useCallback,
  useContext,
  useLayoutEffect,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from 'react';
import { EdgeLabelRenderer, useViewport } from '@xyflow/react';
import { branchJunctions, type RoutedEdge } from './edgeRouting';
import { createRoutes } from './routeStore';
import { prioritizeWires, type WireRouteChange } from './sharedWires';
import { junctionHighlighted, wireHighlightStarts } from './wireHighlight';
import { createWirePlanner, type WireGeometry } from './automaticWires';

// Transient rendered geometry, scoped to the current canvas. Never changes the model
// or its undo history; previews and node/port movement update the same junction view.
const RoutesContext = createContext<ReturnType<typeof createRoutes> | null>(null);
function createAutomaticLayout() {
  const raw = createRoutes();
  const plan = createWirePlanner();
  let last: ReturnType<typeof raw.getSnapshot>;
  let planned = new Map<string, RoutedEdge['points']>();
  return {
    ...raw,
    getPlan: () => {
      const current = raw.getSnapshot();
      if (current !== last) {
        last = current;
        planned = plan(current as WireGeometry[]);
      }
      return planned;
    },
  };
}
const AutomaticContext = createContext<ReturnType<typeof createAutomaticLayout> | null>(null);

export function EdgeRoutesProvider({
  children,
  routes,
}: {
  children: ReactNode;
  routes: ReturnType<typeof createRoutes>;
}) {
  const automatic = useMemo(createAutomaticLayout, []);
  return (
    <RoutesContext.Provider value={routes}>
      <AutomaticContext.Provider value={automatic}>{children}</AutomaticContext.Provider>
    </RoutesContext.Provider>
  );
}

export function useAutomaticWireRoute(geometry: WireGeometry) {
  const store = useContext(AutomaticContext)!;
  useLayoutEffect(() => {
    store.publish(geometry);
  }, [store, geometry]);
  useLayoutEffect(() => () => store.remove(geometry.id), [store, geometry.id]);
  const getRoute = useCallback(
    () => store.getPlan().get(geometry.id) ?? geometry.points,
    [store, geometry.id, geometry.points],
  );
  return useSyncExternalStore(store.subscribe, getRoute);
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
