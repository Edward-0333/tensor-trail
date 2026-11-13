import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { parseDictionarySchema } from '@tensor-trail/model-core/core/dictionary';
import { registry } from '@tensor-trail/model-core/modules';
import { DictionaryInputEditor } from '../src/editor/DictionaryInputEditor';
import { DictionaryTree } from '../src/views/DictionaryTree';

const schema = parseDictionarySchema({ kind: 'dict', fields: {
  all_cls_scores: { kind: 'list', items: [1, 4, 16, 24, 32, 48].map(p => ({
    shape: [2, 1040, p, 17], dtype: 'float32', autograd: { requiresGrad: true, source: 'manual' },
  })) }, empty: { kind: 'tuple', items: [] },
} });

it('expands dictionary Tensor lists into indexed shapes and per-element gradient labels', () => {
  const html = renderToStaticMarkup(createElement(DictionaryTree, { value: schema }));
  expect(html).toContain('all_cls_scores');
  expect(html).toContain('<details>');
  expect(html).toContain('[0]');
  expect(html).toContain('[5]');
  expect(html).toContain('[2, 1040, 48, 17]');
  expect(html).toContain('grad:on');
  expect(html).toContain('空元组 ()');
});

it('does not mislabel sequence fields as None in the dictionary input editor', () => {
  const node = registry.create('ForwardInput', { schema });
  const html = renderToStaticMarkup(createElement(DictionaryInputEditor, { node }));
  expect(html).toContain('展开 6 个 Tensor');
  expect(html).toContain('[2, 1040, 48, 17]');
  expect(html).not.toContain('<code>None</code>');
});
