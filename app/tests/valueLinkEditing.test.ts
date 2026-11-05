import { beforeEach, describe, expect, it } from 'vitest';
import type { Project } from '../src/core/types';
import { graphAt, ports } from '../src/core/graph';
import { inferProject, resultAt } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { registry } from '../src/modules';
import { useEditor } from '../src/editor/store';
import { clipboardText, copySelection, readClipboard } from '../src/editor/clipboard';
import { createLinkedPaste } from '../src/editor/valueLinkEditing';

function fixture(): Project {
  const module = registry.create('Submodule', {}, 'unit');
  module.subgraph!.nodes.push(registry.create('Outlet', { port: 'out' }, 'return'));
  module.subgraph!.edges.push({ id: 'inside', source: 'data', sourcePort: 'out', target: 'return', targetPort: 'in' });
  return {
    schemaVersion: 1, name: 'linked values', variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,3', axes: 'batch,feature' }, 'x'),
        registry.create('TensorVariable', {}, 'v'),
        registry.create('Output', {}, 'y'),
        module,
      ],
      edges: [
        { id: 'input', source: 'x', sourcePort: 'out', target: 'v', targetPort: 'in', waypoints: [{ x: 70, y: 80 }], labelPosition: 0.4 },
        { id: 'output', source: 'v', sourcePort: 'out', target: 'y', targetPort: 'in' },
        { id: 'call', source: 'x', sourcePort: 'out', target: 'unit', targetPort: 'data', waypoints: [{ x: 80, y: 90 }] },
      ],
    },
  };
}

const state = () => useEditor.getState();
const graph = () => graphAt(state().project.graph, state().path);
const node = (id: string) => graph().nodes.find((n) => n.id === id)!;
function pasteOf(id: string): string {
  state().select(id);
  state().copy();
  const before = graph().nodes.length;
  state().paste('requested');
  expect(graph().nodes.length, state().notice).toBe(before + 1);
  return state().selected!;
}
function inferred(id: string) {
  const result = resultAt(inferProject(state().project, registry), state().path)!;
  expect(result.issues).toEqual([]);
  expect(result.nodes[id]?.status, result.nodes[id]?.messages.join(';')).toBe('ok');
  return result.nodes[id].outputs.out;
}
function reloadable() {
  expect(parseProject(serializeProject(state().project), registry)).toEqual(state().project);
}

beforeEach(() => {
  state().load(fixture());
  useEditor.setState({ clipboard: null, clipboardOrigin: null, pasteCount: 0, past: [], future: [], notice: '' });
});

describe('同画布共享值副本的编辑', () => {
  it('Input 副本共享参数、名称和轴含义，但位置、颜色和端口方位各自保留', () => {
    const alias = pasteOf('x');
    expect(node(alias).valueSource).toBe('x');
    const originalPosition = { ...node('x').position };
    state().setParam(alias, 'shape', '4,7');
    state().renameNode(alias, 'features');
    state().setAxis(alias, 1, 'channel');
    for (const id of ['x', alias]) {
      expect(node(id)).toMatchObject({ label: 'features', params: { shape: '4,7', axes: 'batch, channel' } });
      expect(inferred(id)).toMatchObject({ shape: [4, 7], axes: ['batch', 'channel'] });
    }
    state().move(alias, { x: 480, y: 210 });
    state().setTensorColor(alias, '#123456');
    state().setPortSide(alias, 'output', 'bottom');
    expect(node(alias)).toMatchObject({ position: { x: 480, y: 210 }, tensorColor: '#123456', portLayout: { output: 'bottom' } });
    expect(node('x').position).toEqual(originalPosition);
    expect(node('x').tensorColor).toBeUndefined();
    expect(node('x').portLayout).toBeUndefined();
    state().setParam('x', 'shape', '5,9');
    expect(inferred(alias)).toMatchObject({ shape: [5, 9] });
    reloadable();
  });

  it.each(['v', 'y'])('复制已连接的 %s 直接继承推导值，不需要重复输入连线', (id) => {
    const alias = pasteOf(id);
    expect(node(alias).valueSource).toBe(id);
    expect(graph().edges.some((e) => e.target === alias)).toBe(false);
    expect(ports(node(alias), registry).inputs).toEqual([]);
    expect(inferred(alias)).toEqual(inferred(id));
    state().setParam('x', 'shape', '8,11');
    expect(inferred(alias)).toMatchObject({ shape: [8, 11] });
  });

  it('继续复制副本仍直接引用最初来源，不形成引用链', () => {
    const first = pasteOf('v'), second = pasteOf(first);
    expect(node(first).valueSource).toBe('v');
    expect(node(second).valueSource).toBe('v');
    state().renameNode(second, 'positions');
    expect(['v', first, second].map((id) => node(id).label)).toEqual(['positions', 'positions', 'positions']);
    expect(inferred(second)).toEqual(inferred('v'));
    reloadable();
  });

  it('计算模块的复制保持独立，不变成共享值', () => {
    state().add('Identity', undefined, { x: 180, y: 100 });
    const original = state().selected!;
    const alias = pasteOf(original);
    expect(node(alias).valueSource).toBeUndefined();
  });
});

describe('剪贴板来源及可移植片段', () => {
  it('换到含同 ID 的另一项目粘贴也不链接该项目的节点', () => {
    state().select('x'); state().copy();
    const previousDocument = state().documentId;
    state().load(fixture());
    expect(state().documentId).not.toBe(previousDocument);
    state().paste();
    const pasted = state().selected!;
    expect(node(pasted).valueSource).toBeUndefined();
    state().setParam(pasted, 'shape', '6,7');
    expect(node('x').params.shape).toBe('2,3');
  });

  it('同项目跨层粘贴不链接，即使子图存在同 ID 同类型节点', () => {
    state().edit((g) => g.nodes.find((n) => n.id === 'unit')!.subgraph!.nodes.push(registry.create('Input', { shape: '9,9' }, 'x')));
    state().select('x'); state().copy();
    state().navigate(['unit']); state().paste();
    const pasted = state().selected!;
    expect(node(pasted).valueSource).toBeUndefined();
    expect(node(pasted).params.shape).toBe('2,3');
    expect(node('x').params.shape).toBe('9,9');
    reloadable();
  });

  it('替换剪贴板对象后旧来源信息不能误认外部片段为本地复制', () => {
    state().select('x'); state().copy();
    const external = readClipboard(clipboardText(state().clipboard!))!;
    useEditor.setState({ clipboard: external });
    state().paste();
    expect(node(state().selected!).valueSource).toBeUndefined();
  });

  it('来源和副本一起导出时重映射内部引用；单独导出副本则独立化', () => {
    const alias = pasteOf('x');
    const portable = readClipboard(clipboardText(copySelection(graph(), ['x', alias])))!;
    const pasted = createLinkedPaste({ nodes: [], edges: [] }, portable, 64, false);
    const owner = pasted.nodes.find((n) => n.valueSource === undefined)!;
    const linked = pasted.nodes.find((n) => n.valueSource !== undefined)!;
    expect(owner.id).not.toBe('x');
    expect(linked.valueSource).toBe(owner.id);
    expect(readClipboard(clipboardText(pasted))).toEqual(pasted);
    const solo = copySelection(graph(), [alias]);
    expect(solo.nodes[0].valueSource).toBeUndefined();
    expect(solo.nodes[0].params).toEqual(node('x').params);
    expect(readClipboard(clipboardText(solo))).toEqual(solo);
  });

  it('同次复制来源和副本回原画布时都直接共享原来源，副本不附带输入线', () => {
    const alias = pasteOf('v');
    state().selectMany(['x', 'v', alias]); state().copy(); state().paste('requested');
    const pasted = state().selectedIds.map(node);
    expect(pasted.filter((n) => n.type === 'TensorVariable').map((n) => n.valueSource)).toEqual(['v', 'v']);
    expect(pasted.filter((n) => n.type === 'Input').map((n) => n.valueSource)).toEqual(['x']);
    expect(graph().edges.some((e) => state().selectedIds.includes(e.target))).toBe(false);
    for (const n of pasted) expect(inferred(n.id)).toEqual(inferred(n.valueSource!));
  });
});

describe('解除链接和删除来源', () => {
  it('解除副本链接复制原输入并清除旧路线，其余共享组保持原样', () => {
    const alias = pasteOf('v'), sibling = pasteOf('v'), other = pasteOf('x');
    state().unlinkValue(alias);
    expect(node(alias).valueSource).toBeUndefined();
    expect(node(sibling).valueSource).toBe('v');
    expect(node(other).valueSource).toBe('x');
    const input = graph().edges.find((e) => e.target === alias)!;
    expect(input).toMatchObject({ source: 'x', sourcePort: 'out', targetPort: 'in' });
    expect(input.id).not.toBe('input');
    expect(input.waypoints).toBeUndefined();
    expect(input.labelPosition).toBeUndefined();
    expect(inferred(alias)).toEqual(inferred('v'));
    state().renameNode(alias, 'independent');
    expect(node('v').label).not.toBe('independent');
    expect(node(sibling).label).toBe(node('v').label);
    reloadable();
  });

  it('解除来源自身时提拔一个副本并复制输入，原节点独立保留', () => {
    const first = pasteOf('v'), second = pasteOf('v'), other = pasteOf('x');
    state().unlinkValue('v');
    expect(node('v').valueSource).toBeUndefined();
    expect(node(first).valueSource).toBeUndefined();
    expect(node(second).valueSource).toBe(first);
    expect(node(other).valueSource).toBe('x');
    expect(graph().edges.filter((e) => e.target === 'v')).toHaveLength(1);
    expect(graph().edges.filter((e) => e.target === first)).toHaveLength(1);
    expect(inferred(second)).toEqual(inferred('v'));
    state().renameNode('v', 'independent');
    state().renameNode(second, 'remaining group');
    expect(node('v').label).toBe('independent');
    expect(node(first).label).toBe('remaining group');
    reloadable();
  });

  it('同时删除来源和第一个副本时，只提拔未删除的副本并迁移上下游连线', () => {
    const removed = pasteOf('v'), promoted = pasteOf('v'), remaining = pasteOf('v');
    state().removeNodes(['v', removed]);
    expect(graph().nodes.some((n) => n.id === 'v' || n.id === removed)).toBe(false);
    expect(node(promoted).valueSource).toBeUndefined();
    expect(node(remaining).valueSource).toBe(promoted);
    expect(graph().edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ source: 'x', target: promoted }),
      expect.objectContaining({ source: promoted, target: 'y' }),
    ]));
    expect(inferred(remaining)).toEqual(inferred('y'));
    expect(graph().edges.find((e) => e.id === 'input')?.waypoints).toBeUndefined();
    reloadable();
  });
});

describe('子图边界共享值', () => {
  it.each(['data', 'return'])('复制边界 %s 不增加父模块端口，副本仍获得当前推导结果', (id) => {
    const originalPorts = ports(node('unit'), registry);
    state().navigate(['unit']);
    const alias = pasteOf(id);
    expect(node(alias).valueSource).toBe(id);
    expect(inferred(alias)).toEqual(inferred(id));
    const module = state().project.graph.nodes.find((n) => n.id === 'unit')!;
    expect(ports(module, registry)).toEqual(originalPorts);
    expect(inferProject(state().project, registry).nodes.unit.outputs.out).toMatchObject({ shape: [2, 3] });
    reloadable();
  });

  it.each([false, true])('解除 Inlet（来源自身=%s）创建唯一输入端口并保留父级输入', (detachSource) => {
    state().navigate(['unit']);
    const alias = pasteOf('data');
    const detached = detachSource ? 'data' : alias;
    state().unlinkValue(detached);
    const newPort = String(node(detached).params.port);
    expect(newPort).not.toBe('data');
    expect(node(detached).valueSource).toBeUndefined();
    const module = state().project.graph.nodes.find((n) => n.id === 'unit')!;
    expect(ports(module, registry).inputs.map((p) => p.id).sort()).toEqual(['data', newPort].sort());
    const parentInputs = state().project.graph.edges.filter((e) => e.target === 'unit');
    expect(parentInputs).toHaveLength(2);
    expect(parentInputs.map((e) => e.source)).toEqual(['x', 'x']);
    expect(parentInputs.find((e) => e.targetPort === newPort)?.waypoints).toBeUndefined();
    expect(inferred('data')).toEqual(inferred(alias));
    reloadable();
  });

  it('解除 Outlet 副本创建新输出，复制内部输入但保持原输出不变', () => {
    state().navigate(['unit']);
    const alias = pasteOf('return');
    state().unlinkValue(alias);
    const newPort = String(node(alias).params.port);
    expect(newPort).not.toBe('out');
    expect(graph().edges.find((e) => e.target === alias)).toMatchObject({ source: 'data' });
    const module = state().project.graph.nodes.find((n) => n.id === 'unit')!;
    expect(ports(module, registry).outputs.map((p) => p.id).sort()).toEqual(['out', newPort].sort());
    const outputs = inferProject(state().project, registry).nodes.unit.outputs;
    expect(outputs[newPort]).toEqual(outputs.out);
    reloadable();
  });

  it('边界副本可删除，原边界仍受保护且父接口不受影响', () => {
    state().navigate(['unit']);
    const alias = pasteOf('data');
    state().removeNodes(['data', alias]);
    expect(node('data')).toBeDefined();
    expect(node(alias)).toBeUndefined();
    expect(inferred('data')).toMatchObject({ shape: [2, 3] });
    reloadable();
  });
});

describe('共享值的撤销与保存', () => {
  it('粘贴、修改、解除链接可逐次撤销重做；JSON 重载后依然保持同步', () => {
    const original = state().project;
    const alias = pasteOf('x');
    const linked = state().project;
    state().setParam(alias, 'shape', '7,8');
    const edited = state().project;
    state().unlinkValue(alias);
    const detached = state().project;
    expect(state().past).toHaveLength(3);
    state().undo(); expect(state().project).toEqual(edited);
    state().undo(); expect(state().project).toEqual(linked);
    state().undo(); expect(state().project).toEqual(original);
    state().redo(); state().redo(); state().redo();
    expect(state().project).toEqual(detached);
    state().load(parseProject(serializeProject(edited), registry));
    state().setParam(alias, 'shape', '12,13');
    expect(node('x').params.shape).toBe('12,13');
    expect(inferred(alias)).toEqual(inferred('x'));
    reloadable();
  });
});
