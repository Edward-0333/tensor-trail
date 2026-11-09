import { describe, expect, it, vi } from 'vitest';
import { createRoutes } from '../src/views/routeStore';
import type { RoutedEdge } from '../src/views/edgeRouting';

const edge = (id: string, y = 0): RoutedEdge => ({
  id,
  source: 'input',
  sourcePort: 'out',
  points: [
    { x: 0, y },
    { x: 100, y },
  ],
});

describe('画布几何更新批次', () => {
  it('同时发布多条分支时，命中测试立即读取完整几何，界面只接收一次通知', async () => {
    const store = createRoutes();
    const snapshots: RoutedEdge[][] = [];
    store.subscribe(() => snapshots.push(store.getSnapshot()));
    for (let i = 0; i < 83; i++) store.publish(edge(`edge-${i}`, i));
    expect(store.getSnapshot()).toHaveLength(83);
    expect(snapshots).toHaveLength(0);
    await Promise.resolve();
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]).toHaveLength(83);
  });
  it('选择导致的卸载重挂不会发布临时缺线状态', async () => {
    const store = createRoutes();
    store.publish(edge('a'));
    await Promise.resolve();
    const listener = vi.fn();
    store.subscribe(listener);
    store.remove('a');
    store.publish(edge('a', 40));
    expect(store.getSnapshot()[0].points[0].y).toBe(40);
    await Promise.resolve();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot()).toHaveLength(1);
  });
  it('无变化不通知，后续拖动仍继续发布；取消订阅不留下回调', async () => {
    const store = createRoutes();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    store.publish(edge('a'));
    await Promise.resolve();
    store.publish(edge('a'));
    await Promise.resolve();
    expect(listener).toHaveBeenCalledTimes(1);
    store.publish(edge('a', 20));
    await Promise.resolve();
    expect(listener).toHaveBeenCalledTimes(2);
    store.remove('a');
    unsubscribe();
    await Promise.resolve();
    expect(listener).toHaveBeenCalledTimes(2);
    expect(store.getSnapshot()).toHaveLength(0);
  });
});
