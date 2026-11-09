import type { RoutedEdge } from './edgeRouting';
import type { WireRouteChange } from './sharedWires';
import type { WireSelection } from './wireHighlight';

// Geometry reads stay immediate; one notification publishes a complete React layout batch.
export function createRoutes() {
  const routes = new Map<string, RoutedEdge>();
  const listeners = new Set<() => void>();
  let snapshot: RoutedEdge[] = [];
  let preview = new Map<string, WireRouteChange['waypoints']>();
  let previewOwner: string | null = null;
  const previewListeners = new Set<() => void>();
  let selection: WireSelection | null = null;
  const selectionListeners = new Set<() => void>();
  let notificationQueued = false;
  const notify = () => {
    snapshot = [...routes.values()];
    if (notificationQueued) return;
    notificationQueued = true;
    queueMicrotask(() => {
      notificationQueued = false;
      listeners.forEach((listener) => listener());
    });
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
