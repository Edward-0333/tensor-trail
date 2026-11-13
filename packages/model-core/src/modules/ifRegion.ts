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

const loopStatement = text('statement', '循环语句（仅显示，不执行）', 'for item in items');
const loopNote = text('note', '展示范围', '每轮执行框内代码；当前只展示一轮的数据流，不模拟迭代、状态反馈或最终结果。');
export const loopRegion: ModuleDefinition = {
  type: 'LoopRegion',
  label: '循环范围框',
  category: '网络结构',
  description: '在父画布用虚线框展示循环体，让调用参数准备直接可见。仅标注代码范围，不执行或展开多轮，不产生循环最终输出；需要自动状态递推时使用 ForLoop。',
  fields: [loopStatement, loopNote, { ...text('members', '框内节点 ID（逗号分隔）', ''), maxLength: 51000 }],
  getFields: () => [loopStatement, loopNote],
  inputs: [], outputs: [],
  inferValue: (_, params) => {
    if (typeof params.statement !== 'string' || !/^(for|while)\s+\S/.test(params.statement.trim()))
      throw new Error('请填写以 for 或 while 开始的循环语句；只用于显示，不执行。');
    return { outputs: {}, notes: [String(params.note),
      '循环范围框不改变数据流，不模拟次数、状态反馈或梯度；成员只按其当前输入规格推导这一轮。'] };
  },
};
