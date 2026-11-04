import { describe, expect, it } from 'vitest';
import {
  clipboardText,
  readClipboard,
  copySelection,
  duplicateSelection,
} from '../src/editor/clipboard';
import { example } from '../src/modules/examples';

describe('图形剪贴板协议', () => {
  it('往返保留模块、子图、端口方向与背景色', () => {
    const g = example().graph;
    g.nodes[0].tensorColor = '#dbeafe';
    g.nodes[0].portLayout = { output: 'bottom' };
    const fragment = copySelection(g, ['input', 'transformer']);
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
  });
  it('普通文字不作为模型解析', () => {
    expect(readClipboard('普通文本')).toBeNull();
    expect(readClipboard('{"nodes":[]}')).toBeNull();
  });
  it('拒绝带错误连接或非法节点的片段', () => {
    const fragment = copySelection(example().graph, ['input', 'transformer']);
    fragment.edges[0].target = 'missing';
    expect(() => readClipboard(clipboardText(fragment))).toThrow('不存在');
    fragment.edges = [];
    fragment.nodes[0].id = '__proto__';
    expect(() => readClipboard(clipboardText(fragment))).toThrow('ID');
  });
  it('重复粘贴得到独立标识并同步平移控制点', () => {
    const fragment = copySelection(example().graph, ['input', 'transformer']);
    fragment.edges[0].waypoints = [{ x: 150, y: 100 }];
    const first = duplicateSelection(fragment, 32),
      second = duplicateSelection(fragment, 64);
    expect(first.nodes[0].id).not.toBe(second.nodes[0].id);
    expect(first.edges[0].waypoints).toEqual([{ x: 182, y: 132 }]);
    expect(second.edges[0].waypoints).toEqual([{ x: 214, y: 164 }]);
    expect(fragment.edges[0].waypoints).toEqual([{ x: 150, y: 100 }]);
    expect(first.nodes[1].subgraph).not.toBe(second.nodes[1].subgraph);
  });
});
