import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import type { ModelNode, NodeResult } from '@tensor-trail/model-core/core/types';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { ModuleGlyph } from '../src/views/ModuleGlyph';

function render(options: {
  reverse?: boolean;
  portLayout?: ModelNode['portLayout'];
  result?: NodeResult;
  selected?: boolean;
  active?: boolean;
  dimmed?: boolean;
} = {}) {
  const model = registry.create('MatMul', {}, 'matmul');
  model.label = 'torch.matmul(lidar2img, ego2lidar)';
  model.portLayout = options.portLayout;
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: {
      model, result: options.result, category: '注意力算子', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...ports(model, registry), open: () => {},
    },
  };
  return renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
}

describe('Compact MatMul display', () => {
  it('preserves the original matrix glyph and removes the large generic card chrome', () => {
    const html = render({ result: {
      status: 'ok', inputs: {},
      outputs: { out: { shape: [2, 30, 4, 4], axes: ['B', 'TN', 'row', 'col'], dtype: 'float32' } },
      messages: [], notes: [],
    } });
    expect(html).toContain('compact-operation-node matmul-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'MatMul' })));
    expect(html).toContain('<code>matmul()</code>');
    expect(html).toContain('aria-label="torch.matmul(lidar2img, ego2lidar): matmul()"');
    expect(html.match(/data-handleid=/g)).toHaveLength(3);
    for (const removed of ['node-heading', 'node-operation', 'node-footer', 'node-axis-labels', 'grad:'])
      expect(html).not.toContain(removed);
  });

  it('retains the ordered a/b inputs and labelled output', () => {
    const html = render();
    for (const id of ['a', 'b']) expect(html).toContain(`aria-label="输入 ${id}"`);
    expect(html).toContain('aria-label="输出 out"');
    for (const id of ['a', 'b', 'out'])
      expect(html).toMatch(new RegExp(`class="node-port-label [^"]+"[^>]*>${id}</span>`));
    expect(html.indexOf('data-handleid="a"')).toBeLessThan(html.indexOf('data-handleid="b"'));
  });

  it('supports reversed, vertical and same-side port layouts', () => {
    const reversed = render({ reverse: true });
    expect(reversed).toMatch(/react-flow__handle-right[^>]+aria-label="输入 a"/);
    expect(reversed).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const vertical = render({ portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 b"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    expect(vertical).toContain('padding-top:22px');
    expect(vertical).toContain('padding-bottom:22px');
    const sameSide = render({ portLayout: { input: 'left', output: 'left' } });
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(3);
  });

  it.each(['error', 'blocked'] as const)('preserves %s feedback, selection and step highlighting', status => {
    const html = render({
      selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['Incompatible matrix dimensions'], notes: [] },
    });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('Incompatible matrix dimensions');
    expect(html).toContain('title="matmul()');
  });
});
