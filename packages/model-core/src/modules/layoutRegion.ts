import type { ModuleDefinition } from '../core/types';
import { checkLayoutRegion } from '../core/layoutRegions';
import { num, select, text } from './primitives';

export const layoutRegion: ModuleDefinition = {
  type: 'LayoutRegion', label: '布局分区', category: '网络结构',
  description: '圆角浅色布局区域；成员移入后只能在区域内拖动，解除归属后可移出。移动分区携带成员，删除分区保留模块。不改变计算或代码作用域。',
  fields: [
    select('sizing', '尺寸模式', 'auto', ['auto', 'manual']),
    num('width', '宽度', 720, 360), num('height', '高度', 400, 200),
    text('color', '分区颜色', '#38b2ac'),
    { ...text('note', '分区注释', ''), maxLength: 240 },
    { ...text('members', '成员 ID', ''), maxLength: 51000 },
    { ...text('outputs', '分区输出成员 ID', ''), maxLength: 51000 },
  ],
  getFields: () => [],
  inputs: [], outputs: [],
  inferValue: (_, params) => {
    checkLayoutRegion({ id: '', type: 'LayoutRegion', label: '', params, position: { x: 0, y: 0 }, definitionVersion: 1 });
    return { outputs: {}, notes: ['仅组织画布；不执行运算、不产生新输入或改变梯度。'] };
  },
};
