import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { ModelNode, NodeResult, Params } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { ModuleGlyph } from '../src/views/ModuleGlyph';
import { isCountContract, valueEdgeLabel, valueEdgeStyle } from '../src/views/valueEdgeStyle';

function render(type: string, params: Params = {}, options: {
  result?: NodeResult; reverse?: boolean; portLayout?: ModelNode['portLayout'];
  selected?: boolean; active?: boolean; dimmed?: boolean;
} = {}) {
  const model = registry.create(type, params, 'op');
  model.portLayout = options.portLayout;
  const before = structuredClone(model);
  const props: NodeProps<TrailNode> = {
    id: model.id, type: 'module', selected: options.selected ?? false, dragging: false, zIndex: 0,
    selectable: true, deletable: true, draggable: true, isConnectable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
    data: { model, result: options.result, ...ports(model, registry), category: '数学运算', reverse: options.reverse ?? false,
      active: options.active ?? false, dimmed: options.dimmed ?? false, open: () => {} },
  };
  const html = renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
  expect(model).toEqual(before);
  return html;
}
describe('Compact selection preparation operators', () => {
  it.each([
    ['IntegerSubtract', 'a - b'], ['IntegerMinimum', 'min(a, b)'], ['IntegerMaximum', 'max(a, b)'],
  ])('shows %s as a compact operation', (type, label) => {
    const html = render(type);
    expect(html).toContain('compact-operation-node feature-operation-node');
    expect(html).toContain(`<code>${label}</code>`);
    expect(html).not.toContain('node-footer');
  });
  it('shows a record-filter glyph and dim badge, not ambiguous code-only slicing', () => {
    const html = render('BooleanIndexAxis');
    expect(html).toContain('compact-operation-node axis-filter-node');
    expect(html).toContain(renderToStaticMarkup(createElement(ModuleGlyph, { type: 'BooleanIndexAxis' })));
    expect(html).toContain('<strong>按轴筛选</strong>');
    expect(html).toContain('class="axis-filter-dim">dim=1</code>');
    expect(html).toContain('不是逐元素乘法置零或连续区间切片');
    expect(html).not.toContain('axis-filter-contract');
    expect(html).not.toContain('node-footer');
    expect(html).not.toContain('node-heading');
    expect(html).toContain('>x</span>');
    expect(html).toContain('>mask</span>');
  });
  it.each(['0','-1','axis','<script>',''])('renders axis %s safely without a misleading @ expression', dim => {
    const html = render('BooleanIndexAxis', { dim });
    expect(html).toContain(`dim=${dim === '<script>' ? '&lt;script&gt;' : dim || '?'}`);
    expect(html).not.toContain('mask @');
    expect(html).not.toContain('<script>');
  });
  it('labels declared counts explicitly and retains all four ports', () => {
    const html = render('BooleanIndexAxis', { declared_count: true });
    expect(html).toContain('True 数量契约（非实测）');
    expect(html.match(/data-handleid=/g)).toHaveLength(4);
    expect(html).toContain('min-height:72px');
    expect(html).toContain('data-port-id="count"');
    expect(html).toContain('>契约</span>');
    expect(html).toContain('不是 Python 索引的第三个数据参数');
    expect(html).toContain('数量契约 <code>?</code>');
    const resolved = render('BooleanIndexAxis', { declared_count: true }, {
      result: { status: 'ok', inputs: { count: { kind: 'int', value: 720 } }, outputs: {}, messages: [], notes: [] },
    });
    expect(resolved).toContain('数量契约 <code>720</code>');
  });
  it('keeps real port IDs and supports reverse, vertical and same-side layouts', () => {
    const reverse = render('BooleanIndexAxis', { declared_count: true }, { reverse: true });
    expect(reverse).toMatch(/react-flow__handle-right[^>]+aria-label="输入 mask"/);
    expect(reverse).toMatch(/react-flow__handle-left[^>]+aria-label="输出 out"/);
    const vertical = render('BooleanIndexAxis', { declared_count: true }, { portLayout: { input: 'top', output: 'bottom' } });
    expect(vertical).toMatch(/react-flow__handle-top[^>]+aria-label="输入 count"/);
    expect(vertical).toMatch(/react-flow__handle-bottom[^>]+aria-label="输出 out"/);
    const sameSide = render('BooleanIndexAxis', { declared_count: true }, { portLayout: { input: 'left', output: 'left' } });
    expect(sameSide.match(/react-flow__handle-left/g)).toHaveLength(4);
    expect(sameSide).toContain('min-height:90px');
  });
  it('retains error messages, selection, active and dimmed states', () => {
    const html = render('BooleanIndexAxis', {}, { selected: true, active: true, dimmed: true,
      result: { status: 'error', inputs: {}, outputs: {}, messages: ['需要 bool 掩码'], notes: [] },
    });
    expect(html).toContain('selected invalid step-active dimmed');
    expect(html).toContain('需要 bool 掩码');
  });
  it('uses only dtype/device and size inputs when zero creation is expanded', () => {
    const html = render('NewZeros', { metadata_inputs: true, shape: 'd0,d1,d2,1' });
    expect(html).toContain('<code>new_zeros()</code>');
    expect(html).toContain('dtype（类型元数据）');
    expect(html).toContain('device（设备元数据）');
    expect(html).not.toContain('参考 Tensor');
    expect(html.match(/data-handleid=/g)).toHaveLength(6);
  });
});

describe('Count contract edge presentation', () => {
  const count = { kind: 'int' as const, value: 720 };
  it('marks only declared BooleanIndexAxis count inputs, not ordinary integer data or masks', () => {
    const node = registry.create('BooleanIndexAxis', { declared_count: true });
    expect(isCountContract(node, 'count')).toBe(true);
    expect(isCountContract(node, 'mask')).toBe(false);
    expect(isCountContract(node, 'in')).toBe(false);
    expect(isCountContract(registry.create('BooleanIndexAxis'), 'count')).toBe(false);
    expect(isCountContract(registry.create('NewZeros'), 'd0')).toBe(false);
    expect(isCountContract(undefined, 'count')).toBe(false);
  });
  it('uses grey dashed contracts and retains red errors and normal integer wire styles', () => {
    expect(valueEdgeStyle(count, false, true)).toMatchObject({ stroke: 'var(--text-secondary)', strokeDasharray: '5 4' });
    expect(valueEdgeStyle(count, true, true)).toMatchObject({ stroke: 'var(--danger)', strokeDasharray: '5 4' });
    expect(valueEdgeStyle(count)).toMatchObject({ stroke: 'var(--edge)' });
    expect(valueEdgeStyle(count)).not.toHaveProperty('strokeDasharray');
    expect(valueEdgeStyle(undefined, false, true)).toHaveProperty('strokeDasharray', '5 4');
  });
  it('labels counts as declarations rather than measured true values', () => {
    expect(valueEdgeLabel(count, true)).toBe('数量契约 · 720（非实测）');
    expect(valueEdgeLabel(undefined, true)).toBe('数量契约 · ?（非实测）');
    expect(valueEdgeLabel(count)).not.toContain('契约');
    expect(valueEdgeLabel({ shape: [1040], axes: ['query'], dtype: 'bool' }, true)).toBe('数量契约 · ?（非实测）');
  });
});
