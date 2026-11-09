import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Graph } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules/index';
import { example } from '@tensor-trail/model-core/modules/examples';
import { clipboardText } from '@tensor-trail/model-core/editing/clipboard';
import { createModuleClipboard } from '../src/editor/moduleClipboard';
import { useEditor } from '../src/editor/store';

beforeEach(() => {
  useEditor.setState({
    project: example(),
    hasProject: true,
    canvasVersion: 0,
    path: [],
    selected: 'input',
    selectedIds: ['input'],
    clipboard: null,
    clipboardOrigin: null,
    pasteCount: 0,
    past: [],
    future: [],
    notice: '',
  });
});

afterEach(() => vi.unstubAllGlobals());

function externalFragment(): Graph {
  const node = registry.create('Input', { shape: '7, 8' }, 'external_input');
  node.label = '来自外部画布';
  return { nodes: [node], edges: [] };
}

describe('模块剪贴板动作', () => {
  it('没有 DOM 或系统剪贴板能力也能立即复制和粘贴', () => {
    vi.stubGlobal('window', undefined);
    vi.stubGlobal('document', undefined);
    vi.stubGlobal('navigator', undefined);
    const actions = createModuleClipboard();
    const before = useEditor.getState().project;
    expect(actions.ownsShortcut()).toBe(false);
    const copied = actions.copy();
    expect(copied?.nodes.map((node) => node.id)).toEqual(['input']);
    expect(actions.ownsShortcut()).toBe(true);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
    expect(actions.paste()).toBe(true);
    expect(useEditor.getState().project.graph.nodes).toHaveLength(before.graph.nodes.length + 1);
  });

  it('本地动作不请求浏览器剪贴板读写权限', () => {
    const readText = vi.fn(() => {
      throw new Error('不应请求系统读取');
    });
    const writeText = vi.fn(() => {
      throw new Error('不应请求系统写入');
    });
    vi.stubGlobal('navigator', { clipboard: { readText, writeText } });
    const actions = createModuleClipboard();
    actions.copy();
    expect(actions.paste()).toBe(true);
    expect(readText).not.toHaveBeenCalled();
    expect(writeText).not.toHaveBeenCalled();
  });

  it('快速连续粘贴两次各新增一份，并能分别撤销和重做', () => {
    const actions = createModuleClipboard();
    actions.copy();
    const before = useEditor.getState().project;
    expect(actions.paste()).toBe(true);
    const first = useEditor.getState().project;
    expect(actions.paste()).toBe(true);
    const second = useEditor.getState().project;
    expect(first.graph.nodes).toHaveLength(before.graph.nodes.length + 1);
    expect(second.graph.nodes).toHaveLength(before.graph.nodes.length + 2);
    expect(second.graph.nodes.at(-1)!.id).not.toBe(first.graph.nodes.at(-1)!.id);
    expect(useEditor.getState().pasteCount).toBe(2);
    expect(useEditor.getState().past).toHaveLength(2);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(first);
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
    useEditor.getState().redo();
    useEditor.getState().redo();
    expect(useEditor.getState().project).toEqual(second);
  });

  it('keydown 与原生 copy 连续复制只更新缓冲，不产生图编辑', () => {
    const actions = createModuleClipboard();
    const before = useEditor.getState().project;
    const first = actions.copy();
    const second = actions.copy();
    expect(second).toEqual(first);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
    expect(actions.paste()).toBe(true);
    expect(useEditor.getState().project.graph.nodes).toHaveLength(before.graph.nodes.length + 1);
  });

  it.each(['', '  \n\t'])('原生粘贴没有可用文本时回退内部缓冲：%j', (text) => {
    const actions = createModuleClipboard();
    actions.copy();
    const count = useEditor.getState().project.graph.nodes.length;
    expect(actions.pasteText(text)).toBe(true);
    expect(useEditor.getState().project.graph.nodes).toHaveLength(count + 1);
  });

  it.each(['普通文本', '{"nodes":[]}'])('非图形文本不触发旧模块粘贴：%s', (text) => {
    const actions = createModuleClipboard();
    actions.copy();
    const before = useEditor.getState().project;
    const copied = useEditor.getState().clipboard;
    expect(actions.pasteText(text)).toBe(false);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().clipboard).toEqual(copied);
    expect(useEditor.getState().past).toHaveLength(0);
  });

  it('损坏的图形剪贴板协议报告错误，不回退粘贴旧模块', () => {
    const actions = createModuleClipboard();
    actions.copy();
    const before = useEditor.getState().project;
    useEditor.setState({ notice: '' });
    const malformed = clipboardText(externalFragment()).slice(0, -1);
    expect(actions.pasteText(malformed)).toBe(false);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
    expect(useEditor.getState().notice).toContain('粘贴失败');
  });

  it.each(['Inlet', 'Outlet'])('默认复制 %s 为独立边界，需要显式连接新的父模块接口', (type) => {
    const actions = createModuleClipboard();
    actions.copy();
    useEditor.getState().navigate(['transformer']);
    const module = useEditor.getState().project.graph.nodes.find((node) => node.id === 'transformer')!;
    const boundary = module.subgraph!.nodes.find((node) => node.type === type)!;
    const originalPorts = ports(module, registry);
    useEditor.getState().select(boundary.id);
    const historyLength = useEditor.getState().past.length;
    expect(actions.copy()?.nodes.map((node) => node.id)).toEqual([boundary.id]);
    expect(actions.ownsShortcut()).toBe(true);
    expect(useEditor.getState().pasteCount).toBe(0);
    expect(actions.paste()).toBe(true);
    const updated = useEditor.getState().project.graph.nodes.find((node) => node.id === 'transformer')!;
    expect(updated.subgraph!.nodes.at(-1)).toMatchObject({ type });
    expect(updated.subgraph!.nodes.at(-1)!.valueSource).toBeUndefined();
    const side = type === 'Inlet' ? 'inputs' : 'outputs';
    expect(ports(updated, registry)[side]).toHaveLength(originalPorts[side].length + 1);
    expect(useEditor.getState().past).toHaveLength(historyLength + 1);
  });

  it('释放快捷键所有权不丢内部缓冲，并允许导入外部有效图', () => {
    const actions = createModuleClipboard();
    actions.copy();
    actions.paste();
    const copied = useEditor.getState().clipboard;
    actions.release();
    expect(actions.ownsShortcut()).toBe(false);
    expect(useEditor.getState().clipboard).toEqual(copied);
    const before = useEditor.getState().project;
    const external = externalFragment();
    expect(actions.pasteText(clipboardText(external))).toBe(true);
    expect(useEditor.getState().clipboard).toEqual(external);
    expect(useEditor.getState().pasteCount).toBe(1);
    const pasted = useEditor.getState().project.graph.nodes.at(-1)!;
    expect(pasted.label).toBe('来自外部画布');
    expect(pasted.params.shape).toBe('7, 8');
    expect(pasted.id).not.toBe('external_input');
    useEditor.getState().undo();
    expect(useEditor.getState().project).toEqual(before);
  });

  it('空选择复制失败后失焦释放所有权，仍可接收新外部片段', () => {
    const actions = createModuleClipboard();
    actions.copy();
    useEditor.getState().navigate(['transformer']);
    useEditor.getState().select(null);
    expect(actions.copy()).toBeNull();
    actions.release();
    expect(actions.pasteText(clipboardText(externalFragment()))).toBe(true);
    const child = useEditor
      .getState()
      .project.graph.nodes.find((node) => node.id === 'transformer')!.subgraph!;
    expect(child.nodes.at(-1)!.label).toBe('来自外部画布');
  });

  it('输入框复制文字后丢弃本地图形缓冲，空文本也不能复活旧图', () => {
    const actions = createModuleClipboard();
    actions.copy();
    actions.discard();
    const before = useEditor.getState().project;
    expect(actions.ownsShortcut()).toBe(false);
    expect(useEditor.getState().clipboard).toBeNull();
    expect(actions.paste()).toBe(false);
    expect(actions.pasteText('')).toBe(false);
    expect(actions.pasteText('用户复制的文字')).toBe(false);
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
    expect(actions.pasteText(clipboardText(externalFragment()))).toBe(true);
  });

  it('空缓冲粘贴返回失败并提示，不产生空撤销记录', () => {
    const actions = createModuleClipboard();
    const before = useEditor.getState().project;
    expect(actions.paste()).toBe(false);
    expect(useEditor.getState().notice).not.toBe('');
    expect(useEditor.getState().project).toBe(before);
    expect(useEditor.getState().past).toHaveLength(0);
  });
});
