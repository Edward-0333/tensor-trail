import type { ModelNode, Params } from './types';

export const tensorIndexDefaults: Params = {
  mode: 'index',
  dim: '0',
  index: '0',
  start: '',
  stop: '2',
  stopFromInput: false,
  stopLabel: 'stop',
  step: 1,
};

/** Both old operators become one node without changing their selection semantics. */
export function normalizeTensorSelection(type: string, params: Params) {
  if (!['TensorIndex', 'Slice'].includes(type)) return { type, params };
  return {
    type: 'TensorIndex',
    params: {
      ...tensorIndexDefaults,
      ...params,
      ...(type === 'Slice'
        ? {
            mode: 'slice',
            dim: typeof params.dim === 'number' ? String(params.dim) : (params.dim ?? '-1'),
          }
        : {}),
    },
  };
}

export function migrateTensorSelection(node: ModelNode) {
  if (!['TensorIndex', 'Slice'].includes(node.type)) return;
  Object.assign(node, normalizeTensorSelection(node.type, node.params));
  if (['切片 Slice', '张量整数索引'].includes(node.label)) node.label = '索引与切片';
}
