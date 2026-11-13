import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { example } from '@tensor-trail/model-core/modules/examples';
import { HierarchyNavigation, hierarchyLabel } from '../src/editor/HierarchyNavigation';

describe('compact hierarchy navigation', () => {
  it('removes generated forward suffixes but preserves module identifiers and special level meaning', () => {
    expect(hierarchyLabel('主模型 · forward()')).toBe('主模型');
    expect(hierarchyLabel('feed_forward · forward()')).toBe('feed_forward');
    expect(hierarchyLabel('weight · 参数初始化（非 forward）')).toBe('weight · 参数初始化');
    expect(hierarchyLabel('decoder · 循环体')).toBe('decoder · 循环体');
    expect(hierarchyLabel('reweight()')).toBe('reweight()');
  });

  it('keeps the parent button outside the scrolling breadcrumbs and disables it at root', () => {
    const levels = resolveForwardView(example().graph).levels;
    const html = renderToStaticMarkup(createElement(HierarchyNavigation, { levels, onNavigate: () => {} }));
    expect(html).toContain('aria-label="上一层级" disabled=""');
    expect(html.indexOf('hierarchy-up-button')).toBeLessThan(html.indexOf('<nav'));
    expect(html).toContain('aria-current="page"');
    expect(html).toContain('主模型');
    expect(html).not.toContain('forward');
  });

  it('shows every ancestor with a full title and one active level, without changing the underlying titles', () => {
    const view = resolveForwardView(example().graph, ['transformer', 'attention']);
    const before = structuredClone(view.levels);
    const html = renderToStaticMarkup(createElement(HierarchyNavigation, { levels: view.levels, onNavigate: () => {} }));
    expect(html).not.toContain('disabled');
    expect(html).not.toContain('forward()');
    expect(html.match(/aria-current="page"/g)).toHaveLength(1);
    expect(html.match(/class="breadcrumb-label"/g)).toHaveLength(3);
    expect(html).toContain('title="返回');
    expect(view.levels).toEqual(before);
    expect(view.current.title).toContain('forward()');
  });
});
