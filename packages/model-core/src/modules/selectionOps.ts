import type { Dim, ModelNode, ModuleDefinition, Params } from '../core/types';
import { tensorValue } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { axisIndex, requireThat, same, tensor } from './math';
import { loopInteger } from './loopOps';
import { bool, output, text } from './primitives';

export const selectionLabels: Record<string, string> = {
  IntegerSubtract: 'a - b',
  IntegerMinimum: 'min(a, b)',
  IntegerMaximum: 'max(a, b)',
};
export function selectionCode(node: ModelNode): string | undefined {
  if (selectionLabels[node.type])
    return `out = ${selectionLabels[node.type]}\n# Python 整数计算，不是 Tensor 运算；仅计算声明的配置值。`;
  if (node.type === 'BooleanIndexAxis') return `selection = [slice(None)] * x.ndim
selection[${node.params.dim}] = mask
out = x[tuple(selection)]
# 一维 bool mask 选择指定轴，例如 dim=1 对应 x[:, mask]。
# 不广播 mask，不合并 batch 轴，不是 masked_select 或 detach。
# count 是可选的 True 数量契约，不是新增 Python 参数或真实掩码计数。
# 未声明数量时保留符号；声明数量必须由调用方的算法或运行数据保证。`;
}
const selectionPorts = (p: Params) => ({
  inputs: [{ id: 'in', label: '输入 Tensor' }, { id: 'mask', label: '一维 bool 掩码' },
    ...(p.declared_count === true ? [{ id: 'count', label: 'True 数量契约（非实测）' }] : [])],
  outputs: output,
});

export const selectionOps: ModuleDefinition[] = [
  {
    type: 'BooleanIndexAxis', label: '指定轴布尔索引', category: '索引与切片',
    description: '用一维 bool 掩码筛选指定轴，其余轴保持不变。可显式声明 True 数量契约，不假装读取了实际掩码。',
    fields: [text('dim', '筛选轴', '1'), text('count_symbol', '未知数量符号', 'N_selected'),
      bool('declared_count', '连接 True 数量契约（不是实际计数）', false)],
    ...selectionPorts({}), getPorts: selectionPorts,
    inferValue: (values, p, context) => {
      const x = tensorValue(values.in), mask = tensorValue(values.mask), warnings: string[] = [];
      const dim = axisIndex(resolveIntegerExpression(p.dim, context?.hyperparameters), x.shape.length);
      requireThat(mask.dtype === 'bool' && mask.shape.length === 1, '筛选轴需要一维 bool 掩码，不支持广播。');
      same(mask.shape[0], x.shape[dim], '掩码长度需要等于筛选轴长度', warnings);
      let count: Dim;
      if (p.declared_count === true) {
        count = loopInteger(values.count);
        requireThat(typeof count !== 'number' || count >= 0, 'True 数量契约不能为负。');
        if (typeof count === 'number' && typeof mask.shape[0] === 'number')
          requireThat(count <= mask.shape[0], 'True 数量契约不能大于掩码长度。');
      } else {
        requireThat(typeof p.count_symbol === 'string' && /^[A-Za-z][A-Za-z0-9_]*$/.test(p.count_symbol), '数量符号无效。');
        count = mask.shape[0] === 0 ? 0 : p.count_symbol;
      }
      const shape = [...x.shape];
      shape[dim] = count;
      return { outputs: { out: tensor(shape, [...x.axes], x.dtype) }, warnings,
        notes: [`仅筛选轴 ${dim}，保持 batch 和其他轴；这是高级索引，不自动 detach。`,
          p.declared_count === true
            ? `count_nonzero(mask) = ${count} 为调用方声明的数量契约，尚未执行或采集掩码计数。`
            : `输出数量 ${count} 取决于 mask；不从掩码形状猜测 True 数量。`] };
    },
  },
  ...Object.keys(selectionLabels).map((type): ModuleDefinition => ({
    type, label: `整数 ${selectionLabels[type]}`, category: '数学运算',
    description: '计算两个 Python 整数，不创建 Tensor；符号输入保留表达式。',
    fields: [], inputs: [{ id: 'a', label: 'a' }, { id: 'b', label: 'b' }], outputs: output,
    inferValue: ({ a, b }) => {
      const left = loopInteger(a), right = loopInteger(b);
      const numeric = typeof left === 'number' && typeof right === 'number';
      const value: Dim = numeric
        ? type === 'IntegerSubtract' ? left - right
          : type === 'IntegerMinimum' ? Math.min(left, right) : Math.max(left, right)
        : type === 'IntegerSubtract' ? `(${left}) - (${right})`
          : `${type === 'IntegerMinimum' ? 'min' : 'max'}(${left}, ${right})`;
      requireThat(typeof value !== 'number' || Number.isSafeInteger(value), '整数结果超出安全范围。');
      return { outputs: { out: { kind: 'int', value } },
        notes: ['仅计算声明的整数规格，不执行模型训练或 Python 分支。'] };
    },
  })),
];
