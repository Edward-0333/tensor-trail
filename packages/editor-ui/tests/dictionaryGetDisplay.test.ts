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
  path?: string; reverse?: boolean; portLayout?: ModelNode['portLayout'];
  result?: NodeResult; selected?: boolean; active?: boolean; dimmed?: boolean;
} = {}) {
  const model = registry.create('DictionaryGet', { path: options.path ?? '["query_feat"]' }, 'get');
  model.label = 'outs field';
  model.portLayout = options.portLayout;
  const before = structuredClone(model);
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false,
    zIndex: 0, selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, result: options.result, category: '索引与切片', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false,
      ...ports(model, registry), open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(model).toEqual(before);
  return html;
}

describe('Compact DictionaryGet', () => {
  it('adds a dictionary-to-value glyph without a header, footer or duplicate tensor details', () => {
    const html = render({ result: { status: 'ok', inputs: {}, outputs: {
      out: { shape: [2, 1040, 256], axes: ['batch', 'query', 'embedding'], dtype: 'float32' },
    }, messages: [], notes: [] } });
    expect(html).toContain('compact-operation-node field-read-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'DictionaryGet' })));
    expect(html).toContain('按键读取字典字段');
    expect(html).toContain('按键路径读取字段，不复制或融合数据');
    for (const removed of ['node-heading', 'node-footer', 'node-axis-labels', 'node-port-label', '组合子网络', 'grad:', '[2, 1040, 256]'])
      expect(html).not.toContain(removed);
  });

  it.each([
    ['["query_feat"]', '["query_feat"]'],
    ['agent.position', '["agent"]["position"]'],
    ['["temporal_ego_states",0]', '["temporal_ego_states"][0]'],
    ['["0"]', '["0"]'],
    ['["name.with.dots"]', '["name.with.dots"]'],
  ])('shows the full typed path for %s', (path, expression) => {
    const html = render({ path });
    expect(html).toContain(renderToStaticMarkup(createElement('code', null, expression)));
  });

  it('retains full long paths in the tooltip and escapes field names', () => {
    const key = 'a'.repeat(128);
    const html = render({ path: JSON.stringify([key, '<script>']) });
    expect(html).toContain(`title="outs field · [&quot;${key}&quot;][&quot;&lt;script&gt;&quot;]`);
    expect(html).not.toContain('<script>');
  });

  it('keeps invalid paths visible and editable instead of throwing', () => {
    for (const path of ['', '[', '[]']) {
      const html = render({ path, result: { status: 'error', inputs: {}, outputs: {},
        messages: ['字段路径无效'], notes: [] } });
      expect(html).toContain('<code>选择字典字段</code>');
      expect(html).toContain('字段路径无效');
      expect(html).toContain('按键读取字典字段');
    }
  });

  it('retains accessible ports in standard, reverse, vertical and same-side layouts', () => {
    const normal = render();
    expect(normal.match(/data-handleid=/g)).toHaveLength(2);
    expect(normal).toContain('aria-label="输入 in"');
    expect(normal).toContain('aria-label="输出 out"');
    const reverse = render({ reverse: true });
    expect(reverse).toMatch(/react-flow__handle-right[^>]+aria-label="输入 in"/);
    expect(reverse).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const vertical = render({ portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 in"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    const sameSide = render({ portLayout: { input: 'left', output: 'left' } });
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(2);
    expect(sameSide).toContain('min-height:42px');
  });

  it.each(['error', 'blocked'] as const)('preserves %s feedback and selection/stepping styles', status => {
    const html = render({ selected: true, active: true, dimmed: true,
      result: { status, inputs: {}, outputs: {}, messages: ['字典不存在该字段'], notes: [] } });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('字典不存在该字段');
    expect(html).toContain('aria-label="字典字段读取：[&quot;query_feat&quot;]"');
  });
});
