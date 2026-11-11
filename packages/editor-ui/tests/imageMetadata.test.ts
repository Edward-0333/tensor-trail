import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ValueSpec } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { TensorView } from '../src/views/TensorView';

const render = (value: ValueSpec) =>
  renderToStaticMarkup(
    createElement(TensorView, {
      result: { status: 'ok', inputs: {}, outputs: { out: value }, messages: [], notes: [] },
    }),
  );

describe('Non-Tensor image metadata display', () => {
  it('shows torch.Size as metadata, not as a boolean, Tensor or gradient state', () => {
    const html = render({ kind: 'size', dims: [256, 704] });
    expect(html).toContain('torch.Size([256, 704])');
    expect(html).toContain('不是 Tensor');
    expect(html).not.toContain('布尔条件');
    expect(html).not.toContain('tensor-autograd-summary');
  });
  it('explains the known per-item update while retaining the opaque input disclaimer', () => {
    const result = registry.get('UpdateImageMetaInputShape')!.inferValue!(
      {
        img_metas: { kind: 'forward', text: 'list[dict]' },
        input_shape: { kind: 'size', dims: [256, 704] },
      },
      {},
    );
    const html = render(result.outputs.out);
    expect(html).toContain('torch.Size([256, 704])');
    expect(html).toContain('非完整运行时数据');
    expect(html).toContain('不创建新列表');
    expect(html).not.toContain('tensor-autograd-summary');
  });
});
