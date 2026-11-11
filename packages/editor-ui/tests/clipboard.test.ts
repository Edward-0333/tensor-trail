import { describe, expect, it } from 'vitest';
import {
  clipboardText,
  readClipboard,
  copySelection,
  duplicateSelection,
} from '@tensor-trail/model-core/editing/clipboard';
import { example } from '@tensor-trail/model-core/modules/examples';
import { clone, type Graph, type ModelNode } from '@tensor-trail/model-core/core/types';
import { parseProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules/index';

function linkedCopy(source: ModelNode, id: string): ModelNode {
  return { ...clone(source), id, valueSource: source.id };
}

function boundaryGraph(): Graph {
  return {
    nodes: [
      registry.create('Inlet', { port: 'data' }, 'data'),
      registry.create('Outlet', { port: 'result' }, 'result'),
    ],
    edges: [{ id: 'pass', source: 'data', sourcePort: 'out', target: 'result', targetPort: 'in' }],
  };
}

describe('图形剪贴板协议', () => {
  it('往返保留模块、子图、端口方向与边框色', () => {
    const g = example().graph;
    g.nodes[0].tensorBorderColor = '#dbeafe';
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

  it('边界节点可复制并携带内部连接，剪贴板在子模块作用域校验', () => {
    const graph = boundaryGraph();
    const fragment = copySelection(graph, ['data', 'result']);
    expect(fragment.nodes.map((node) => node.type)).toEqual(['Inlet', 'Outlet']);
    expect(fragment.edges).toEqual(graph.edges);
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
    expect(() =>
      parseProject(
        JSON.stringify({ schemaVersion: 1, name: '根层', graph: fragment, variants: [] }),
        registry,
      ),
    ).toThrow('根图不能');
  });

  it('仅复制模块输出时临时添加校验输入，但不将它留在片段中', () => {
    const fragment = copySelection(boundaryGraph(), ['result']);
    const restored = readClipboard(clipboardText(fragment))!;
    expect(restored).toEqual(fragment);
    expect(restored.nodes).toHaveLength(1);
    expect(restored.nodes[0].type).toBe('Outlet');
    expect(restored.edges).toEqual([]);
  });

  it('临时输入不覆盖已有同名普通节点，也不删掉真实节点', () => {
    const input = registry.create('Input', {}, 'clipboard_input');
    const output = registry.create('Outlet', { port: 'out' }, 'result');
    const fragment = {
      nodes: [input, output],
      edges: [
        { id: 'pass', source: input.id, sourcePort: 'out', target: output.id, targetPort: 'in' },
      ],
    };
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
  });

  it('单独复制值副本时读取源当前配置，并清除片段外引用', () => {
    const source = registry.create('Input', { shape: '2, 3', axes: 'batch, feature' }, 'source');
    const copy = linkedCopy(source, 'copy');
    source.label = '最新名称';
    source.params.shape = '4, 5';
    copy.tensorBorderColor = '#abcdef';
    const graph = { nodes: [source, copy], edges: [] };
    const fragment = copySelection(graph, ['copy']);
    expect(fragment.nodes[0].valueSource).toBeUndefined();
    expect(fragment.nodes[0].label).toBe('最新名称');
    expect(fragment.nodes[0].params.shape).toBe('4, 5');
    expect(fragment.nodes[0].tensorBorderColor).toBe('#abcdef');
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
    expect(copy.valueSource).toBe('source');
    expect(copy.params.shape).toBe('2, 3');
  });

  it('同时复制源和副本保留组内引用，重复粘贴为各自独立的共享组', () => {
    const source = registry.create('Input', {}, 'source');
    const graph = { nodes: [source, linkedCopy(source, 'copy')], edges: [] };
    const fragment = copySelection(graph, ['source', 'copy']);
    expect(fragment.nodes[1].valueSource).toBe('source');
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
    const first = duplicateSelection(fragment, 32);
    const second = duplicateSelection(fragment, 64);
    expect(first.nodes[1].valueSource).toBe(first.nodes[0].id);
    expect(second.nodes[1].valueSource).toBe(second.nodes[0].id);
    expect(first.nodes[1].valueSource).not.toBe(second.nodes[1].valueSource);
    expect(fragment.nodes[1].valueSource).toBe('source');
    expect(readClipboard(clipboardText(first))).toEqual(first);
  });

  it.each(['Inlet', 'Outlet'])('仅复制多个 %s 副本时生成独立且唯一的边界端口', (type) => {
    const source = registry.create(type, { port: 'data' }, 'source');
    const reserved = registry.create(type, { port: 'data_2' }, 'reserved');
    const graph = {
      nodes: [source, linkedCopy(source, 'a'), linkedCopy(source, 'b'), reserved],
      edges: [],
    };
    const fragment = copySelection(graph, ['a', 'b', 'reserved']);
    expect(fragment.nodes.every((node) => node.valueSource === undefined)).toBe(true);
    expect(new Set(fragment.nodes.map((node) => node.params.port)).size).toBe(3);
    expect(fragment.nodes.find((node) => node.id === 'reserved')!.params.port).toBe('data_2');
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
    expect(graph.nodes[1].params.port).toBe('data');
    expect(graph.nodes[2].params.port).toBe('data');
  });

  it('源与边界副本一起复制时只保留同一父接口名称', () => {
    const graph = boundaryGraph();
    graph.nodes.push(
      linkedCopy(graph.nodes[0], 'data_copy'),
      linkedCopy(graph.nodes[1], 'result_copy'),
    );
    const fragment = copySelection(
      graph,
      graph.nodes.map((node) => node.id),
    );
    expect(fragment.nodes.find((node) => node.id === 'data_copy')!.params.port).toBe('data');
    expect(fragment.nodes.find((node) => node.id === 'data_copy')!.valueSource).toBe('data');
    expect(fragment.nodes.find((node) => node.id === 'result_copy')!.params.port).toBe('result');
    expect(readClipboard(clipboardText(fragment))).toEqual(fragment);
    const duplicated = duplicateSelection(fragment, 32);
    expect(duplicated.nodes[2].valueSource).toBe(duplicated.nodes[0].id);
    expect(duplicated.nodes[3].valueSource).toBe(duplicated.nodes[1].id);
    expect(readClipboard(clipboardText(duplicated))).toEqual(duplicated);
  });

  it('临时输入不能补全原本悬空的共享值或连线引用', () => {
    const outlet = registry.create('Outlet', { port: 'out' }, 'result');
    const missingSource = registry.create('Inlet', { port: 'in' }, 'copy');
    missingSource.valueSource = 'clipboard_input';
    expect(() =>
      readClipboard(clipboardText({ nodes: [outlet, missingSource], edges: [] })),
    ).toThrow('不存在于当前画布');
    expect(() =>
      readClipboard(
        clipboardText({
          nodes: [outlet],
          edges: [
            {
              id: 'invalid',
              source: 'clipboard_input',
              sourcePort: 'out',
              target: 'result',
              targetPort: 'in',
            },
          ],
        }),
      ),
    ).toThrow('不存在');
  });

  it('外部边界片段仍拒绝重复实际端口及无效节点', () => {
    const inlet = registry.create('Inlet', { port: 'same' }, 'first');
    const duplicate = registry.create('Inlet', { port: 'same' }, 'second');
    expect(() => readClipboard(clipboardText({ nodes: [inlet, duplicate], edges: [] }))).toThrow(
      '边界端口重复',
    );
    const outlet = registry.create('Outlet', { port: 'out' }, '__proto__');
    expect(() => readClipboard(clipboardText({ nodes: [outlet], edges: [] }))).toThrow('ID');
  });

  it('复制包含内部共享值的整个子模块时，内部引用仍留在自己的作用域', () => {
    const child = registry.create('Submodule', {}, 'child');
    child.subgraph!.nodes.push(linkedCopy(child.subgraph!.nodes[0], 'data_alias'));
    const fragment = copySelection({ nodes: [child], edges: [] }, ['child']);
    const duplicated = duplicateSelection(fragment, 32);
    expect(duplicated.nodes[0].id).not.toBe('child');
    expect(duplicated.nodes[0].subgraph!.nodes[1].valueSource).toBe('data');
    expect(readClipboard(clipboardText(duplicated))).toEqual(duplicated);
  });
});
