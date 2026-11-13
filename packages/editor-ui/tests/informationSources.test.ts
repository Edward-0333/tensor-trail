import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it } from 'vitest';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { registry } from '@tensor-trail/model-core/modules';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { DEFAULT_TENSOR_PALETTE } from '@tensor-trail/model-core/core/tensorAppearance';
import type { Project, ValueSpec, ValueLineage } from '@tensor-trail/model-core/core/types';
import { LineageBadges, LineageDetails, sourceColor } from '../src/views/InformationSources';
import { InformationSourcePanel } from '../src/editor/InformationSourcePanel';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { useEditor } from '../src/editor/store';

const state = () => useEditor.getState();
const output = () => inferProject(state().project, registry).nodes.x.outputs.out;
const numbers = (v: ValueSpec) => v.lineage?.sources.map((s) => s.index);
const lineage = (value: Partial<ValueLineage> = {}): ValueLineage => ({
  sources: [],
  origins: [],
  initializations: [],
  unknown: false,
  evidence: 'inferred',
  ...value,
});
const value = (info: ValueLineage): ValueSpec => ({
  shape: [2, 3],
  axes: ['B', 'C'],
  dtype: 'float32',
  lineage: info,
});
beforeEach(() => {
  const x = registry.create('Input', { shape: '2,3', axes: 'batch,feature' }, 'x');
  x.label = 'arbitrary input';
  x.tensorAutograd = { requiresGrad: false, source: 'manual' };
  x.tensorBorderColor = '#123456';
  const p: Project = {
    schemaVersion: 1,
    name: 'sources',
    variants: [],
    graph: { nodes: [x], edges: [] },
  };
  state().load(p);
  useEditor.setState({ past: [], future: [], path: [] });
});

describe('Root input number badges', () => {
  it('automatically numbers on enable and supports undo, redo, reopen without changing shapes or geometry', () => {
    const original = structuredClone(state().project);
    expect(output().lineage).toBeUndefined();
    state().setLineageEnabled(true);
    expect(numbers(output())).toEqual([1]);
    expect(state().project.graph).toEqual(original.graph);
    expect(output()).toMatchObject({ shape: [2, 3], autograd: { requiresGrad: false } });
    state().undo();
    expect(output().lineage).toBeUndefined();
    state().redo();
    expect(numbers(output())).toEqual([1]);
    state().load(parseProject(serializeProject(state().project), registry));
    expect(numbers(output())).toEqual([1]);
    expect(state().project.graph.nodes[0].informationSource).toBeUndefined();
    state().setLineageEnabled(false);
    expect(output().lineage).toBeUndefined();
  });
  it('shares the same source ID through linked copies and renames, with a new number only after unlinking a root input', () => {
    state().setLineageEnabled(true);
    state().duplicateLinked('x');
    const id = state().selected!;
    let r = inferProject(state().project, registry);
    expect(r.nodes[id].outputs.out.lineage?.sources).toEqual(
      r.nodes.x.outputs.out.lineage?.sources,
    );
    state().renameNode('x', 'renamed');
    expect(output().lineage?.sources).toEqual([{ id: 'x', index: 1, label: 'renamed' }]);
    state().unlinkValue(id);
    r = inferProject(state().project, registry);
    expect(numbers(r.nodes.x.outputs.out)).toEqual([1]);
    expect(numbers(r.nodes[id].outputs.out)).toEqual([2]);
  });
  it('renders colored numeric badges independently of gradient backgrounds and IN identity', () => {
    state().setLineageEnabled(true);
    const model = state().project.graph.nodes[0],
      result = inferProject(state().project, registry).nodes.x;
    const props: NodeProps<TrailNode> = {
      id: 'x',
      type: 'module',
      selected: false,
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
        category: '输入输出',
        reverse: false,
        active: false,
        dimmed: false,
        open: () => {},
        inputBoundary: { id: 'x', label: 'x', index: 1, total: 1 },
      },
    };
    const html = renderToStaticMarkup(
      createElement(ReactFlowProvider, null, createElement(ModuleNode, props)),
    );
    expect(html).toContain('IN · 1/1');
    expect(html).toContain('class="lineage-source"');
    expect(html).toContain('data-source-id="x"');
    expect(html).toContain('data-source-index="1"');
    expect(html).toContain(`background-color:${sourceColor(1)}`);
    expect(html).toContain('>1</span>');
    expect(html).toContain('grad:off');
    expect(html).toContain(`--tensor-bg:${DEFAULT_TENSOR_PALETTE.off}`);
    expect(html).toContain('--tensor-border:#123456');
  });
  it('shows initialization history in details without inventing a source number for a parameter or constant', () => {
    const parameter = value(lineage({ origins: ['parameter'], initializations: ['random'] }));
    expect(renderToStaticMarkup(createElement(LineageBadges, { value: parameter }))).toBe('');
    const html = renderToStaticMarkup(createElement(LineageDetails, { value: parameter }));
    expect(html).toContain('模型参数');
    expect(html).toContain('随机初始化');
    expect(html).toContain('不表示当前数值仍是初始值');
    expect(html).toContain('未接入根层输入的数值信息');
    const zero = value(lineage({ origins: ['constant'], initializations: ['zeros'] }));
    expect(renderToStaticMarkup(createElement(LineageBadges, { value: zero }))).toBe('');
  });
  it('shows multiple original numbers together, but marks unknown paths and container summaries distinctly', () => {
    const v: ValueSpec = {
      kind: 'list',
      items: [],
      lineage: lineage({
        sources: [
          { id: 'a', index: 1, label: 'a' },
          { id: 'c', index: 3, label: 'c' },
        ],
        unknown: true,
      }),
    };
    const html = renderToStaticMarkup(createElement(LineageDetails, { value: v }));
    expect(html).toContain('>1</span>');
    expect(html).toContain('>3</span>');
    expect(html).toContain('>?</span>');
    expect(html).toContain('已知编号不是完整来源集合');
    expect(html).toContain('不表示各元素已经特征融合');
    expect(html).toContain('1 = a；3 = c');
  });
  it('shows resolved loop state origins in details without a question mark or invented root IDs', () => {
    const resolved = value(lineage({ origins: ['constant', 'metadata', 'state'] }));
    expect(renderToStaticMarkup(createElement(LineageBadges, { value: resolved }))).toBe('');
    const html = renderToStaticMarkup(createElement(LineageDetails, { value: resolved }));
    expect(html).toContain('模型状态');
    expect(html).toContain('常量');
    expect(html).not.toContain('lineage-unknown');
    expect(html).not.toContain('lineage-source');
    expect(html).toContain('历史更新或 checkpoint');
  });
  it('uses a root-only legend instead of a semantic declaration dropdown, including inside nested graphs', () => {
    useEditor.setState({ path: ['nested'] });
    const html = renderToStaticMarkup(
      createElement(InformationSourcePanel, { graph: state().project.graph }),
    );
    expect(html).toContain('根层输入编号');
    expect(html).toContain('arbitrary input');
    expect(html).toContain('开启输入来源追踪');
    expect(html).not.toContain('<select');
    expect(renderToStaticMarkup(createElement(LineageBadges, { value: output() }))).toBe('');
  });
  it('uses consistent filled colors for arbitrary input counts, not just three inputs', () => {
    expect(new Set(Array.from({ length: 20 }, (_, i) => sourceColor(i + 1))).size).toBe(20);
    expect(sourceColor(1)).not.toBe(sourceColor(2));
    expect(sourceColor(37)).toMatch(/^hsl\(/);
  });
});
