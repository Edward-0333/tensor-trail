import type { ModuleDefinition } from '../core/types';
import { text } from './primitives';

const condition = text('condition', 'if 条件（仅显示，不执行）', 'self.pretrain');
const note = text('note', '范围说明', '仅条件为真时执行；静态展示，未执行条件判断。');
export const ifRegion: ModuleDefinition = {
  type: 'IfRegion',
  label: 'if 条件范围框',
  category: '分支运算',
  description:
    '虚线框包围同层条件体，顶部显示 if 条件。仅分组展示，不执行条件、不改变节点输入输出或推导顺序。边框随成员移动和尺寸自动调整；删除框不删除内部节点。',
  fields: [
    condition,
    note,
    { ...text('members', '框内节点 ID（逗号分隔）', ''), maxLength: 51000 },
  ],
  getFields: () => [condition, note],
  inputs: [],
  outputs: [],
  inferValue: (_, params) => {
    if (typeof params.condition !== 'string' || !params.condition.trim())
      throw new Error('请填写 if 条件；条件只用于显示，不会执行。');
    return {
      outputs: {},
      notes: [
        String(params.note),
        '范围框不产生 Tensor，内部路径仅做静态形状检查，不代表已执行或条件为真。',
      ],
    };
  },
};
