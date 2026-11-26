import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { buildModel, validateModel } from '@tensor-trail/model-core/agent';
import { registry } from '@tensor-trail/model-core/modules';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { Inspector } from '../src/editor/Inspector';
import { useEditor } from '../src/editor/store';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';

// Static React rendering uses the server snapshot. Read the current test document.
vi.mock('../src/editor/store', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/editor/store')>();
  type State = ReturnType<typeof actual.useEditor.getState>;
  const read = (selector?: (s: State) => unknown) =>
    selector ? selector(actual.useEditor.getState()) : actual.useEditor.getState();
  return { ...actual, useEditor: Object.assign(read, actual.useEditor) };
});

const editor = useEditor.getState;
function renderNode(model: ModelNode, result?: NodeResult) {
  const props: NodeProps<TrailNode> = {
    id: model.id,
    type: 'module',
    selected: true,
    dragging: false,
    zIndex: 0,
    selectable: true,
    deletable: true,
    draggable: true,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
    data: {
      model,
      result,
      ...ports(model, registry),
      category: registry.get(model.type)!.category,
      reverse: false,
      active: false,
      dimmed: false,
      open: () => {},
    },
  };
  return renderToStaticMarkup(
    createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
  );
}

describe('semantic canvas editing', () => {
  beforeEach(() => {
    const built = buildModel({
      name: '两条胶水链',
      nodes: [
        { id: 'x', type: 'Input' },
        { id: 'a', type: 'Unsqueeze' },
        { id: 'b', type: 'Contiguous' },
        { id: 's', type: 'SemanticOp' },
        { id: 'c', type: 'Unsqueeze' },
        { id: 'd', type: 'Contiguous' },
        { id: 'y', type: 'Output' },
      ],
      connections: [
        ['x.out', 'a.in'],
        ['a.out', 'b.in'],
        ['b.out', 's.x'],
        ['s.out', 'c.in'],
        ['c.out', 'd.in'],
        ['d.out', 'y.in'],
      ],
    });
    expect(built.diagnostics).toEqual([]);
    useEditor.setState({
      project: built.project!,
      hasProject: true,
      path: [],
      past: [],
      future: [],
      selected: null,
      selectedIds: [],
    });
  });

  it('shows the legacy ordered steps for empty summaries and a semantic card for a described group', () => {
    editor().mergeCalculationNodes(['a', 'b'], '', 'ignored unless described');
    const group = editor().selected!;
    const model = () => editor().project.graph.nodes.find((n) => n.id === group)!;
    expect(renderNode(model())).toContain('<ol class="calculation-group-steps">');
    expect(renderNode({ ...model(), params: {} })).toContain(
      '<ol class="calculation-group-steps">',
    );
    expect(editor().setParam(group, 'summary', '在末尾增加单例轴')).toBe(true);
    expect(editor().setParam(group, 'formula', 'x.unsqueeze(-1).contiguous()')).toBe(true);
    const html = renderNode(model());
    expect(html).not.toContain('calculation-group-steps');
    for (const text of [
      '折叠 · 2 步',
      '在末尾增加单例轴',
      'x.unsqueeze(-1).contiguous()',
      '双击查看内部操作',
      '输入 in1',
      '输出 out1',
    ])
      expect(html).toContain(text);
    const inspector = renderToStaticMarkup(
      createElement(Inspector, { graph: editor().project.graph }),
    );
    for (const text of ['语义摘要 (summary)', '精简表达式 (formula)', '拆开组合'])
      expect(inspector).toContain(text);
    editor().undo();
    expect(model().params.formula).toBe('ignored unless described');
    editor().redo();
    expect(model().params.formula).toBe('x.unsqueeze(-1).contiguous()');
  });

  it('edits a full-length Unicode summary and also accepts it as the label of a merged group', () => {
    const summary = '𠮷'.repeat(240);
    editor().select('s');
    expect(editor().setParam('s', 'summary', summary)).toBe(true);
    expect(editor().setParam('s', 'summary', summary + '𠮷')).toBe(false);
    const html = renderToStaticMarkup(createElement(Inspector, { graph: editor().project.graph }));
    const field = html.match(/<input[^>]+aria-label="语义说明 \(summary\)"[^>]*>/)?.[0];
    expect(field).toContain(summary);
    expect(field).not.toMatch(/maxlength/i);
    editor().mergeCalculationNodes(['a', 'b'], summary, summary);
    const group = editor().project.graph.nodes.find(n => n.type === 'CalculationGroup');
    expect(group).toMatchObject({ label: summary, params: { summary, formula: summary } });
    expect(validateModel(editor().project).ok).toBe(true);
  });

  it('renders declared semantic ports, summary, formula and each inferred output shape', () => {
    const model = registry.create('SemanticOp', {
      summary: '提取最近位置',
      formula: 'min(xy, mask)',
      inputs: 'xy,mask',
      outputs: 'd,t',
    });
    const tensor = { shape: [2, 4], axes: ['b', 'xy'], dtype: 'float32' };
    const html = renderNode(model, {
      inputs: {},
      outputs: { d: tensor, t: { ...tensor, shape: [2], axes: ['b'], dtype: 'int64' } },
      messages: [],
      notes: [],
      status: 'ok',
    });
    for (const text of [
      '语义步骤',
      '提取最近位置',
      'min(xy, mask)',
      'd [2, 4]',
      't [2]',
      'semantic-card',
    ])
      expect(html).toContain(text);
    for (const id of ['xy', 'mask', 'd', 't']) expect(html).toContain(`data-port-id="${id}"`);
    expect(renderNode(registry.create('SemanticOp'))).not.toContain('semantic-card-formula');
  });

  it.each(['merge', 'auto', 'expand'])(
    'records %s as a single undoable and redoable change',
    (action) => {
      if (action === 'expand') editor().autoSimplifyCurrentGraph();
      const before = editor().project,
        count = editor().past.length;
      if (action === 'merge')
        editor().mergeCalculationNodes(['b', 'a'], '整理布局', 'contiguous(unsqueeze(x))');
      if (action === 'auto') editor().autoSimplifyCurrentGraph();
      if (action === 'expand') editor().expandAllCalculationsInCurrentGraph();
      const after = editor().project;
      expect(after.graph.nodes.filter((n) => n.type === 'CalculationGroup')).toHaveLength(
        action === 'auto' ? 2 : action === 'merge' ? 1 : 0,
      );
      expect(editor().past).toHaveLength(count + 1);
      expect(validateModel(after)).toMatchObject({
        ok: true,
        diagnostics: [],
        outputs: validateModel(before).outputs,
      });
      editor().undo();
      expect(editor().project).toEqual(before);
      editor().redo();
      expect(editor().project).toEqual(after);
    },
  );

  it('offers multi-selection merging only for an eligible set and skips empty history entries', () => {
    const inspector = () =>
      renderToStaticMarkup(createElement(Inspector, { graph: editor().project.graph }));
    editor().selectMany(['a', 'b']);
    expect(inspector()).toContain('合并为语义计算块');
    editor().selectMany(['a', 'c']);
    expect(inspector()).not.toContain('合并为语义计算块');
    editor().expandAllCalculationsInCurrentGraph();
    expect(editor().past).toHaveLength(0);
    editor().autoSimplifyCurrentGraph();
    editor().autoSimplifyCurrentGraph();
    expect(editor().past).toHaveLength(1);
  });
});
