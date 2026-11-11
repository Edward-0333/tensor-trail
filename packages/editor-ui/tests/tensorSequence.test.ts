import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ValueSpec } from '@tensor-trail/model-core/core/types';
import { TensorView } from '../src/views/TensorView';

const render = (value: ValueSpec) =>
  renderToStaticMarkup(
    createElement(TensorView, {
      result: { status: 'ok', inputs: {}, outputs: { out: value }, messages: [], notes: [] },
    }),
  );

describe('Tensor sequence inspection', () => {
  it('shows each element shape, dtype and gradient state, not a container Tensor', () => {
    const html = render({
      kind: 'tuple',
      items: [64, 32, 16, 8].map((h, i) => ({
        shape: [60, 256, h, 176 / 2 ** i],
        axes: ['BN', 'C', 'H', 'W'],
        dtype: 'float32',
        ...(i < 2 ? { autograd: { requiresGrad: i === 0, source: 'runtime' as const } } : {}),
      })),
    });
    expect(html).toContain('tuple · 4 个 Tensor');
    for (const shape of [
      '[60, 256, 64, 176]',
      '[60, 256, 32, 88]',
      '[60, 256, 16, 44]',
      '[60, 256, 8, 22]',
    ])
      expect(html).toContain(shape);
    expect(html).toContain('float32');
    expect(html).toContain('grad:on');
    expect(html).toContain('grad:off');
    expect(html).toContain('grad:?');
    expect(html).not.toContain('tensor-autograd-summary');
    expect(html).toContain('容器本身不是 Tensor');
  });
  it.each(['tuple', 'list'] as const)(
    'shows an empty %s as a container, not a shape [0]',
    (kind) => {
      const html = render({ kind, items: [] });
      expect(html).toContain(`${kind} · 0 个 Tensor`);
      expect(html).toContain(kind === 'list' ? '空列表 []' : '空元组 ()');
      expect(html).toContain('不是形状为 [0]');
      expect(html).not.toContain('tensor-autograd-summary');
      expect(html).not.toContain('tensor-axes');
    },
  );
});
