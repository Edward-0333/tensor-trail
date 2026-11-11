import type {
  ControlSpec,
  Dim,
  ModelNode,
  ModuleDefinition,
  TensorSpec,
  ValueSpec,
} from '../core/types';
import { clone, isControlSpec, isIntegerListSpec, isIntegerSpec, tensorValue } from '../core/types';
import { resolveIntegerExpression } from '../core/hyperparameters';
import { canRequireGrad } from '../core/tensorAppearance';
import { requireThat, tensor } from './math';
import { requireRealTensor } from './sorting';
import { bool, input, output, text } from './primitives';

const noGrad = { requiresGrad: false, source: 'inferred' as const };
const named = (...names: string[]) => names.map((id) => ({ id, label: id }));
export function loopInteger(v: ValueSpec | undefined): Dim {
  requireThat(v && isIntegerSpec(v), '需要 Python 整数或符号整数，不是 Tensor。');
  requireThat(
    typeof v.value === 'number' ? Number.isSafeInteger(v.value) : v.value.trim().length > 0,
    '整数无效。',
  );
  return v.value;
}
function indexCheck(index: Dim, size: Dim, end = false) {
  if (typeof index === 'number' && typeof size === 'number')
    requireThat(index >= (end ? 0 : -size) && index < size + (end ? 1 : 0), '索引或切片起点越界。');
}
function plain(x: TensorSpec): TensorSpec {
  const out = tensor([...x.shape], [...x.axes], x.dtype);
  if (!canRequireGrad(x.dtype) || x.autograd?.requiresGrad === false) out.autograd = { ...noGrad };
  return out;
}
function control(v: ValueSpec | undefined, branch?: boolean): ControlSpec {
  requireThat(
    v && isControlSpec(v) && v.branch === branch,
    '需要匹配的条件控制端口，不能当作 Tensor 或实际布尔值。',
  );
  return v;
}
function startRows(x: TensorSpec, start: Dim): Dim {
  requireThat(x.shape.length >= 1, '行切片需要至少一维 Tensor。');
  indexCheck(start, x.shape[0], true);
  return typeof start === 'number' && typeof x.shape[0] === 'number'
    ? x.shape[0] - start
    : `max(${x.shape[0]}-${start},0)`;
}

export const loopOperationLabels: Record<string, string> = {
  TensorRowAt: 'x[i]',
  Argmax: 'argmax(-1)',
  TensorItemInt: '.item()',
  ScalarIndexWrite: 'x[index] = value',
  TensorEqualInt: 'x == value',
  IntegerListGet: 'list[index]',
  IntegerAddOne: 'i + 1',
  ZeroTailColumn: 'maps[start:, ind] = 0',
  SliceRowsFrom: 'x[start:]',
  WriteTailRows: 'x[start:] = rows',
  MergeBranches: '合并分支状态',
};
export function loopOpsCode(n: ModelNode): string | undefined {
  const code: Record<string, string> = {
    IntegerList: `num_inds = [${n.params.values}]\n# Python 整数列表，不是 Tensor；来自配置声明。`,
    TensorRowAt: 'out = x[i]\n# i 为本轮位置。只推导选中行的规格，不读取真实数值。',
    Argmax: 'out = torch.argmax(x, dim=-1)\n# 返回 int64 索引 Tensor；同分取第一个位置。',
    TensorItemInt: `${n.params.cast ? 'out = int(x)' : 'out = x.item()'}\n# 符号整数 ${n.params.symbol}；实际数值未采集。${n.params.cast ? 'float Tensor 中保存的原 query 编号显式转 Python int。' : '将整数标量 Tensor 读为 Python int，不是新建可训练参数。'}`,
    ScalarIndexWrite: 'x[index] = value\n# 原地写入；图中的输出表示写入后的状态，不表示 clone。',
    TensorEqualInt: 'out = (x == value)\n# 输出 bool Tensor，用于统计已经分配到该时间戳的 query。',
    IntegerListGet: 'out = num_inds[index]\n# 配额来自 Python list[int]；动态索引的实际值未知。',
    IntegerAddOne: 'out = i + 1',
    IfCondition:
      'if count == quota:\n    # True 分支\n    ...\nelse:\n    # False 分支\n    ...\n# 两条路径仅静态展示；没有执行比较或采集分支结果。',
    BranchValue: `# ${n.params.branch ? 'True' : 'False'} 分支入口\nout = x\n# control 是条件依赖，不是 Tensor 数值。两侧只检查形状，不表示同时执行。`,
    ZeroTailColumn: 'maps[start:, ind] = 0\n# 仅清零剩余行的一列分数；不是删除该列或赋值 -inf。',
    SliceRowsFrom: 'out = x[start:]\n# 剩余行数依赖本轮 i；末轮允许为空。',
    WriteTailRows: 'x[start:] = rows\n# 只写回剩余行；已处理的前缀不改变。输出是更新后的状态。',
    MergeBranches:
      '# True: 使用更新后的 maps\n# False: 保留本轮 maps\n# 合并的是两条可能路径的 shape/dtype，不是 torch.where，也不是拼接。',
  };
  return code[n.type];
}

export const loopOps: ModuleDefinition[] = [
  {
    type: 'IntegerList',
    label: '整数列表',
    category: '输入输出',
    description: 'Python list[int] 配置值，不是张量序列。',
    fields: [text('values', '整数表达式（逗号分隔）', '720,60,60,60,60,40,40')],
    inputs: [],
    outputs: output,
    inferValue: (_, p, context) => {
      const entries = String(p.values)
        .split(',')
        .map((s) => s.trim());
      requireThat(
        entries.length <= 256 && entries.every(Boolean),
        '整数列表不能为空且最多 256 项。',
      );
      const values = entries.map((s) => resolveIntegerExpression(s, context?.hyperparameters));
      requireThat(
        values.every((v) => Number.isSafeInteger(v) && v >= 0),
        '配额需要非负安全整数。',
      );
      return { outputs: { out: { kind: 'int-list', values } } };
    },
  },
  {
    type: 'TensorRowAt',
    label: '动态取一行',
    category: '索引与切片',
    description: 'x[i]，i 来自整数输入。',
    fields: [],
    inputs: named('in', 'index'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in),
        i = loopInteger(v.index);
      requireThat(x.shape.length > 0, '取行需要至少一维 Tensor。');
      indexCheck(i, x.shape[0]);
      const out = plain(x);
      out.shape.shift();
      out.axes.shift();
      return { outputs: { out } };
    },
  },
  {
    type: 'Argmax',
    label: '最大值索引 Argmax',
    category: '逻辑与归约',
    description: '沿最后一轴返回最大值的位置，移除该轴。',
    fields: [],
    inputs: input,
    outputs: output,
    infer: ({ in: x }) => {
      requireRealTensor(x);
      requireThat(x.shape.length > 0 && x.shape.at(-1) !== 0, 'Argmax 需要非空归约轴。');
      return {
        outputs: {
          out: { ...tensor(x.shape.slice(0, -1), x.axes.slice(0, -1), 'int64'), autograd: noGrad },
        },
      };
    },
  },
  {
    type: 'TensorItemInt',
    label: '读取 Python 整数',
    category: '维度操作',
    description: '标量整数 Tensor.item() 或 int(标量 Tensor)，仅输出符号整数，实际数值未采集。',
    fields: [text('symbol', '整数符号名', 'value'), bool('cast', '显式 int(x) 转换', false)],
    inputs: input,
    outputs: output,
    inferValue: (v, p) => {
      const x = tensorValue(v.in);
      requireRealTensor(x);
      requireThat(x.shape.length === 0, '此操作要求零维标量 Tensor。');
      requireThat(
        p.cast === true || /^(u?int)/.test(x.dtype),
        'float Tensor.item() 返回 float，需开启显式 int 转换。',
      );
      requireThat(/^[A-Za-z][A-Za-z0-9_]*$/.test(String(p.symbol)), '整数符号名无效。');
      return {
        outputs: { out: { kind: 'int', value: String(p.symbol) } },
        notes: ['这里只保留符号整数；不是运行时数值，也不是尺寸读取。'],
      };
    },
  },
  {
    type: 'ScalarIndexWrite',
    label: '索引写入标量',
    category: '索引与切片',
    description: '一维 Tensor 的 x[index] = value，保留 dtype。',
    fields: [],
    inputs: named('in', 'index', 'value'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in);
      requireRealTensor(x);
      requireThat(x.shape.length === 1, '索引写入要求一维目标。');
      indexCheck(loopInteger(v.index), x.shape[0]);
      loopInteger(v.value);
      return { outputs: { out: plain(x) } };
    },
  },
  {
    type: 'TensorEqualInt',
    label: 'Tensor 与整数比较',
    category: '逻辑与归约',
    description: 'x == value，逐元素生成 bool Tensor。',
    fields: [],
    inputs: named('in', 'value'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in);
      loopInteger(v.value);
      return {
        outputs: { out: { ...tensor([...x.shape], [...x.axes], 'bool'), autograd: noGrad } },
      };
    },
  },
  {
    type: 'IntegerListGet',
    label: '读取整数列表元素',
    category: '索引与切片',
    description: '动态读取 num_inds[ind]；未观测 ind 时仅返回符号。',
    fields: [],
    inputs: named('in', 'index'),
    outputs: output,
    inferValue: (v) => {
      requireThat(v.in && isIntegerListSpec(v.in), '需要 list[int]，不是 Tensor。');
      const index = loopInteger(v.index);
      indexCheck(index, v.in.values.length);
      const value =
        typeof index === 'number'
          ? v.in.values[(index + v.in.values.length) % v.in.values.length]
          : `num_inds[${index}]`;
      return { outputs: { out: { kind: 'int', value } } };
    },
  },
  {
    type: 'IntegerAddOne',
    label: '整数加一',
    category: '数学运算',
    description: 'i + 1，支持静态整数和符号整数。',
    fields: [],
    inputs: input,
    outputs: output,
    inferValue: (v) => {
      const n = loopInteger(v.in),
        value = typeof n === 'number' ? n + 1 : `(${n} + 1)`;
      requireThat(typeof value !== 'number' || Number.isSafeInteger(value), '整数溢出。');
      return { outputs: { out: { kind: 'int', value } } };
    },
  },
  {
    type: 'IfCondition',
    label: '配额已满？',
    category: '分支运算',
    description:
      '比较 Python 整数 count == quota；输出静态条件和 True/False 控制依赖，不声称已执行分支。',
    fields: [],
    inputs: named('count', 'quota'),
    outputs: named('out', 'true', 'false'),
    inferValue: (v) => {
      const expression = `${loopInteger(v.count)} == ${loopInteger(v.quota)}`;
      return {
        outputs: {
          out: { kind: 'control', expression },
          true: { kind: 'control', expression, branch: true },
          false: { kind: 'control', expression, branch: false },
        },
      };
    },
  },
  {
    type: 'BranchValue',
    label: '分支入口',
    category: '分支运算',
    description: '带条件依赖的数据入口。静态展示可能路径，不执行 Python 条件。',
    fields: [bool('branch', 'True 分支', true)],
    inputs: named('in', 'control'),
    outputs: output,
    inferValue: (v, p) => {
      requireThat(typeof p.branch === 'boolean', '分支标志必须是 bool。');
      control(v.control, p.branch);
      return { outputs: { out: clone(tensorValue(v.in)) } };
    },
  },
  {
    type: 'ZeroTailColumn',
    label: '剩余行的指定列清零',
    category: '索引与切片',
    description: 'maps[start:, ind] = 0，保留已处理前缀。',
    fields: [],
    inputs: named('in', 'start', 'column'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in);
      requireThat(x.shape.length === 2, '清零指定列需要二维矩阵。');
      startRows(x, loopInteger(v.start));
      indexCheck(loopInteger(v.column), x.shape[1]);
      return { outputs: { out: plain(x) } };
    },
  },
  {
    type: 'SliceRowsFrom',
    label: '读取剩余行',
    category: '索引与切片',
    description: 'x[start:]，起点来自整数输入，允许空后缀。',
    fields: [],
    inputs: named('in', 'start'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in),
        out = plain(x);
      out.shape[0] = startRows(x, loopInteger(v.start));
      return { outputs: { out } };
    },
  },
  {
    type: 'WriteTailRows',
    label: '写回剩余行',
    category: '索引与切片',
    description: 'x[start:] = rows，不覆盖已处理前缀，输出表示更新后的同一变量。',
    fields: [],
    inputs: named('in', 'start', 'rows'),
    outputs: output,
    inferValue: (v) => {
      const x = tensorValue(v.in),
        rows = tensorValue(v.rows);
      const shape = [startRows(x, loopInteger(v.start)), ...x.shape.slice(1)];
      requireThat(
        x.dtype === rows.dtype && JSON.stringify(shape) === JSON.stringify(rows.shape),
        '写回行的 shape/dtype 与目标后缀不一致。',
      );
      return { outputs: { out: plain(x) } };
    },
  },
  {
    type: 'MergeBranches',
    label: '合并分支状态',
    category: '分支运算',
    description: '合并同形状、同 dtype 的两个可能分支；不是同时执行或 torch.where。',
    fields: [],
    inputs: named('condition', 'true', 'false'),
    outputs: output,
    inferValue: (v) => {
      control(v.condition);
      const a = tensorValue(v.true),
        b = tensorValue(v.false);
      requireThat(
        a.dtype === b.dtype && JSON.stringify(a.shape) === JSON.stringify(b.shape),
        '分支的 shape/dtype 必须相同。',
      );
      const out = plain(a);
      if (b.autograd?.requiresGrad !== false && canRequireGrad(b.dtype)) delete out.autograd;
      return { outputs: { out }, notes: ['未观测条件结果，输出仅表示分支合流的规格。'] };
    },
  },
];
