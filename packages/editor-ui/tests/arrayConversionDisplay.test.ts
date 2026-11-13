import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ValueSpec } from '@tensor-trail/model-core/core/types';
import { TensorView } from '../src/views/TensorView';
import { valueDisplayText } from '../src/views/valueDisplayText';
import { valueEdgeStyle } from '../src/views/valueEdgeStyle';

const render = (value: ValueSpec) => renderToStaticMarkup(createElement(TensorView, {
  result: { status: 'ok', inputs: {}, outputs: { out: value }, messages: [], notes: [] },
}));
describe('NumPy array and dtype display', () => {
  it('does not show gradient or tensor role for a NumPy array', () => {
    const array: ValueSpec = { kind: 'ndarray', dims: [2, 30, 4, 4],
      axes: ['B', 'TN', 'row', 'col'], dtype: 'float32' };
    const html = render(array);
    expect(html).toContain('np.ndarray [2, 30, 4, 4]');
    expect(html).toContain('不是 PyTorch Tensor');
    expect(html).not.toContain('tensor-autograd-summary');
    expect(html).not.toContain('grad:');
    expect(valueEdgeStyle(array)).not.toHaveProperty('strokeDasharray');
  });
  it('displays dtype separately, with a metadata wire rather than a Tensor data wire', () => {
    const dtype: ValueSpec = { kind: 'dtype', value: 'float32' };
    expect(valueDisplayText(dtype)).toBe('torch.float32');
    expect(render(dtype)).toContain('DTYPE');
    expect(render(dtype)).not.toContain('tensor-autograd-summary');
    expect(valueEdgeStyle(dtype)).toMatchObject({ strokeDasharray: '5 4' });
  });
});
