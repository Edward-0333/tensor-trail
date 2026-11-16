import type { ModuleDefinition } from '../core/types';
import { checkLayoutRegion } from '../core/layoutRegions';
import { num, select, text } from './primitives';

export const layoutRegion: ModuleDefinition = {
  type: 'LayoutRegion', label: '布局分区', category: '网络结构',
  description: '圆角浅色布局区域；自动适配时随成员拖动向外扩大，手动尺寸时限制成员拖动。可用1-5星标注当前层级内的相对重要性，不跨层比较或自动继承。移动分区携带成员，删除分区保留模块。不改变计算或代码作用域。',
  fields: [
    select('sizing', '尺寸模式', 'auto', ['auto', 'manual']),
    { ...num('width', '宽度', 720, 360), integer: false },
    { ...num('height', '高度', 400, 200), integer: false },
    text('color', '分区颜色', '#38b2ac'),
    { ...text('note', '分区注释', ''), maxLength: 240 },
    { ...num('importance', '本层重要性（0=未评级，1-5星）', 0, 0), max: 5, integer: true },
    { ...text('importance_note', '本层评级依据', ''), maxLength: 240 },
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
