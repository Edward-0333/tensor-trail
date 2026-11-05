import { beforeEach, describe, expect, it } from 'vitest';
import { useEditor } from '../src/editor/store';
import { registry } from '../src/modules';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { linkedValueNodes } from '../src/core/valueLinks';
const s = () => useEditor.getState();
const node = (id: string) => s().project.graph.nodes.find((n) => n.id === id)!;
function copy(id: string) {
  s().select(id);
  s().copy();
  s().paste();
  return s().selected!;
}
beforeEach(() => {
  s().load({
    schemaVersion: 1,
    name: 'copy policy',
    variants: [],
    graph: {
      nodes: [
        registry.create('Input', { shape: '2,21' }, 'x'),
        registry.create('TensorSize', { axes: '1', names: 'T' }, 'size'),
        registry.create('LocalParameter', {}, 'T'),
        registry.create('TensorVariable', {}, 'v'),
        registry.create('Output', {}, 'out'),
        registry.create('ForwardInput', {}, 'data'),
        registry.create('IntegerParameter', { expression: '21' }, 'constant'),
      ],
      edges: [
        { id: 'size', source: 'x', sourcePort: 'out', target: 'size', targetPort: 'in' },
        { id: 'T', source: 'size', sourcePort: 'dim0', target: 'T', targetPort: 'in' },
        { id: 'v', source: 'x', sourcePort: 'out', target: 'v', targetPort: 'in' },
        { id: 'out', source: 'v', sourcePort: 'out', target: 'out', targetPort: 'in' },
      ],
    },
  });
  useEditor.setState({
    past: [],
    future: [],
    clipboard: null,
    clipboardOrigin: null,
    pasteCount: 0,
  });
});
describe('仅局部参数默认关联', () => {
  it('普通复制T仍共享来源、推导值和选择组', () => {
    const alias = copy('T');
    expect(node(alias).valueSource).toBe('T');
    expect(linkedValueNodes(s().project.graph, alias).map((n) => n.id)).toEqual(['T', alias]);
    s().setParam('x', 'shape', '2,12');
    expect(inferProject(s().project, registry).nodes[alias].outputs.out).toEqual({
      kind: 'int',
      value: 12,
    });
    expect(parseProject(serializeProject(s().project), registry)).toEqual(s().project);
  });
  it.each(['x', 'v', 'out', 'data', 'constant'])('普通复制 %s 不产生隐式值来源', (id) => {
    const alias = copy(id);
    expect(node(alias).valueSource).toBeUndefined();
    expect(linkedValueNodes(s().project.graph, alias)).toHaveLength(1);
    const label = node(id).label;
    s().renameNode(alias, '独立副本');
    expect(node(id).label).toBe(label);
    expect(parseProject(serializeProject(s().project), registry)).toEqual(s().project);
  });
  it('显式创建共享副本保留值同步，普通复制该副本则独立；不污染剪贴板', () => {
    s().select('T');
    s().copy();
    const before = s().project,
      clipboard = s().clipboard;
    expect(s().duplicateLinked('v')).toBe(true);
    const alias = s().selected!;
    expect(node(alias).valueSource).toBe('v');
    expect(s().clipboard).toBe(clipboard);
    expect(inferProject(s().project, registry).nodes[alias].outputs.out).toEqual(
      inferProject(s().project, registry).nodes.v.outputs.out,
    );
    s().undo();
    expect(s().project).toEqual(before);
    s().redo();
    expect(node(alias).valueSource).toBe('v');
    const independent = copy(alias);
    expect(node(independent).valueSource).toBeUndefined();
    expect(node(independent).label).toBe(node('v').label);
  });
  it('复制一组普通节点保留组内显式连线，并独立于原组', () => {
    s().selectMany(['x', 'v']);
    s().copy();
    s().paste();
    const [input, variable] = s().selectedIds;
    expect(node(input).valueSource).toBeUndefined();
    expect(node(variable).valueSource).toBeUndefined();
    expect(s().project.graph.edges).toContainEqual(
      expect.objectContaining({ source: input, target: variable }),
    );
    s().setParam(input, 'shape', '4,8');
    expect(node('x').params.shape).toBe('2,21');
    expect(inferProject(s().project, registry).nodes[variable].outputs.out).toMatchObject({
      shape: [4, 8],
    });
  });
  it('不允许为计算节点创建值关联', () => {
    const before = s().project;
    expect(s().duplicateLinked('size')).toBe(false);
    expect(s().project).toBe(before);
  });
  it('局部参数跨项目粘贴保持独立', () => {
    s().select('T');
    s().copy();
    s().newProject();
    s().paste();
    expect(node(s().selected!).valueSource).toBeUndefined();
  });
});
