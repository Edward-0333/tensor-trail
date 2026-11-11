import type { Recipe, RecipeGraph } from '../../src/agent/types';

export const featureShapes = [
  [60, 256, 64, 176],
  [60, 256, 32, 88],
  [60, 256, 16, 44],
  [60, 256, 8, 22],
];
export const reshapedFeatures = featureShapes.map(([, c, h, w]) => [2, 30, c, h, w]);

export function reshapeBody(): RecipeGraph {
  return {
    nodes: [
      { id: 'item', type: 'Inlet', params: { port: 'item' } },
      { id: 'B', type: 'Inlet', params: { port: 'B' } },
      { id: 'index', type: 'Inlet', params: { port: 'index' } },
      { id: 'size', type: 'TensorSize', params: { axes: '0,1,2,3', names: 'BN,C,H,W' } },
      { id: 'divide', type: 'IntegerTruncDivide' },
      { id: 'view', type: 'View', params: { shape: 'd0,d1,d2,d3,d4' } },
      { id: 'collect', type: 'Outlet', params: { port: 'out' } },
    ],
    connections: [
      ['item.out', 'size.in'],
      ['size.dim0', 'divide.a'],
      ['B.out', 'divide.b'],
      ['item.out', 'view.in'],
      ['B.out', 'view.d0'],
      ['divide.out', 'view.d1'],
      ['size.dim1', 'view.d2'],
      ['size.dim2', 'view.d3'],
      ['size.dim3', 'view.d4'],
      ['view.out', 'collect.in'],
    ],
  };
}

export function loopRecipe(): Recipe {
  return {
    name: 'Four feature levels',
    nodes: [
      ...featureShapes.map((shape, i) => ({
        id: `input${i}`,
        type: 'Input',
        params: { shape: shape.join(','), axes: 'BN,C,H,W', dtype: 'float32' },
      })),
      { id: 'pack', type: 'TensorSequencePack' },
      { id: 'empty', type: 'TensorSequencePack', params: { kind: 'list', input_count: 0 } },
      { id: 'B', type: 'IntegerParameter', params: { expression: '2' } },
      { id: 'loop', type: 'ForEachTensor', position: { x: 800, y: 100 }, subgraph: reshapeBody() },
      { id: 'result', type: 'Output' },
    ],
    connections: [
      ...featureShapes.map((_, i): [string, string] => [`input${i}.out`, `pack.item${i}`]),
      ['pack.out', 'loop.items'],
      ['empty.out', 'loop.initial'],
      ['B.out', 'loop.B'],
      ['loop.out', 'result.in'],
    ],
  };
}
