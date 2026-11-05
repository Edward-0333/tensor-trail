import type { ModuleDefinition } from '../core/types';

export const submodule: ModuleDefinition = {
  type: 'Submodule',
  label: '子模块',
  category: '网络结构',
  description:
    '封装另一个神经网络模块的 forward()。双击进入内部画布；父级传入的数据会显示在内部输入节点中。初始仅创建 data 输入，内部流程和输出可逐步补齐。',
  fields: [],
  inputs: [{ id: 'data', label: 'data' }],
  outputs: [],
  composite: true,
  build: () => ({
    nodes: [
      {
        id: 'data',
        type: 'Inlet',
        definitionVersion: 1,
        label: 'data',
        params: { port: 'data' },
        position: { x: 0, y: 0 },
      },
    ],
    edges: [],
  }),
};
