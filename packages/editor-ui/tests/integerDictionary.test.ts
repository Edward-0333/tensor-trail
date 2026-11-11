import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { parseDictionarySchema } from '@tensor-trail/model-core/core/dictionary';
import { registry } from '@tensor-trail/model-core/modules/index';
import { DictionaryPathField } from '../src/editor/IndexingFields';
import { DictionaryInputEditor } from '../src/editor/DictionaryInputEditor';
import { DictionaryTree } from '../src/views/DictionaryTree';

const schema = JSON.stringify({
  kind: 'dict',
  fields: {
    temporal_ego_states: {
      kind: 'dict',
      keyType: 'int',
      fields: { 0: { shape: [1, 1, 21], dtype: 'float32' } },
    },
  },
});

describe('Integer dictionary editor', () => {
  it('offers numeric lookup paths and selects the current integer key', () => {
    const node = registry.create('DictionaryGet', { path: '["temporal_ego_states",0]' });
    const html = renderToStaticMarkup(
      createElement(DictionaryPathField, {
        node,
        input: parseDictionarySchema(schema),
      }),
    );
    expect(html).toContain('value="[&quot;temporal_ego_states&quot;,0]" selected=""');
    expect(html).not.toContain('value="[&quot;temporal_ego_states&quot;,&quot;0&quot;]"');
  });

  it('shows the integer key type and uses [0], not ["0"], in tensor editor labels', () => {
    const node = registry.create('ForwardInput', { schema });
    const html = renderToStaticMarkup(createElement(DictionaryInputEditor, { node }));
    expect(html).toContain('value="int" selected=""');
    expect(html).toContain('形状 [&quot;temporal_ego_states&quot;][0]');
    const tree = renderToStaticMarkup(
      createElement(DictionaryTree, {
        value: parseDictionarySchema(schema),
      }),
    );
    expect(tree).toContain('dict[int]');
    expect(tree).toContain('0 (int)');
  });

  it('keeps old string-key dictionaries and their dropdown values unchanged', () => {
    const input = parseDictionarySchema(
      '{"kind":"dict","fields":{"0":{"shape":[1],"dtype":"bool"}}}',
    );
    const node = registry.create('DictionaryGet', { path: '["0"]' });
    const html = renderToStaticMarkup(createElement(DictionaryPathField, { node, input }));
    expect(html).toContain('value="[&quot;0&quot;]" selected=""');
  });
});
