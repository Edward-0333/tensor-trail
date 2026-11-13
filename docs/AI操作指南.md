---
title: "Tensor Trail AI 操作指南"
created: 2026-10-19
last_updated: 2026-11-12
authors:
  human: "用户"
  ai: "Codex"
  ai_model: "GPT-6"
contributions:
  human: ["提出让 AI 阅读架构并搭建模型的需求"]
  ai: ["实现协议、API、CLI 与画布接入，编写说明"]
source_files:
  - "packages/model-core/src/agent/index.ts"
  - "packages/model-core/src/agent/canvas.ts"
  - "packages/cli/scripts/agent.mjs"
review_status: "unreviewed"
tags: [项目, Tensor, AI, API]
---

# AI 操作指南

本版提供纯 JavaScript/TypeScript API、命令行和 `window.tensorTrail` 画布 API。外部 AI 负责理解架构并生成 JSON，程序共用原有模块注册表进行构建和形状校验。无需在本程序中配置模型密钥。

**尚未提供 MCP、HTTP REST 服务或内置大模型聊天框。** 当前仅做形状推导，不运行真实张量、训练或 PyTorch 前向。普通聊天模型需由具备文件、命令行或页面调用能力的工具连接程序。

## 调用边界与实参准备

绘制 `module(expr1, expr2, ...)` 时，括号里的表达式属于调用方，不属于 `module` 的内部实现。
先在调用侧画出创建张量、乘法、索引、reshape、detach 等实参准备，再连接到封装模块的输入端口。
模块内部从形参 Inlet 开始，不重复创建这些实参，不把参数准备藏进被调用模块。

例如 `ego_cross_attn(ego_feat.new_ones(B,1,3)*0.5, ego_feat, curr_query_pos.detach(), curr_query_feat.detach())`：
调用侧应直接显示 `GetDtype/GetDevice -> NewOnes -> *0.5`、原始 `ego_feat` 和两条 `Detach` 路径；
注意力内部从 `query_points/query_feat/key_points/key_feat` 开始。

额外的循环封装也不应隐藏用户要求观察的调用参数。需要在父画布显示时，优先将循环展开为同层范围，
或采用明确的内部引用视图。**显示在封装外不等于移到循环前执行**：循环内的参数准备仍每轮发生，
尤其不能把依赖更新状态的 detach、索引等操作只计算一次。首轮预览必须明确标注，
不将第一轮 shape 或结果当作所有轮次的结果，不伪造尚未绘制的状态反馈和最终输出。

## 快速开始

在项目根目录执行。依赖统一从根锁文件安装；本机也可用 `node scripts/npm.cjs` 替代 npm。

```sh
npm ci
npm run build:agent
node packages/cli/scripts/agent.mjs describe
node packages/cli/scripts/agent.mjs schema recipe
node packages/cli/scripts/agent.mjs build examples/cnn-transformer.recipe.json --out model.json
node packages/cli/scripts/agent.mjs validate model.json
node packages/cli/scripts/agent.mjs patch model.json examples/swiglu.patch.json --out swiglu-model.json
```

用网页“打开”按钮打开生成的 model.json；或点击 **AI 构建**，粘贴 Recipe，依次“检查形状”“应用到画布”。“修改当前模型”接收 Patch，整批作为一次操作，可一次撤销。面板新建时保留“我的变体”。

面板可下载模块目录、JSON Schema、当前快照和验证报告。快照包装为 `{apiVersion, revision, project}`，CLI validate/patch 与普通文件导入需要其中的 project 正文，不接受整个快照对象。

![AI 构建工作区](assets/ai-builder.jpg)

## Recipe v1

```json
{
  "recipeVersion": 1,
  "name": "我的 Transformer",
  "nodes": [
    { "id": "input", "type": "Input", "params": { "shape": "8, 100, 64" } },
    { "id": "encoder", "type": "Transformer", "params": { "d_model": 64, "heads": 8, "d_ff": 128 } },
    { "id": "output", "type": "Output" }
  ],
  "connections": [["input.out", "encoder.in"], ["encoder.out", "output.in"]]
}
```

- recipeVersion 可省略，默认 1。name、nodes、connections 必填。
- type 使用模块目录的准确名称，大小写敏感。label 为可选显示名称。id 在同一图中唯一，允许 1–100 个字母、数字、下划线和连字符；禁止 __proto__、constructor、prototype。
- 端点为 `节点ID.端口ID`。通常为 in/out；Add、Mul、MatMul 为 a/b → out；LSTM 有 out/hidden/cell。以目录为准。
- params 可省略，缺失值取注册表默认值并列入 defaultsApplied；不会根据上游尺寸自动改通道数或 d_model。未知字段、未知参数、非法类型/范围/枚举会被拒绝。
- 自动生成位置和组合模块内部子图，输出完整项目仍为 schemaVersion: 1。节点可提供 `position: {x,y}`，两者必须为有限数值；未指定的位置自动安排。默认严格模式要求顶层至少一个输入（Input 或 ForwardInput）和 Output，必需输入必须连接；草稿模式允许尚未补齐这些部分。两种模式都禁止环路、悬空端点和同一输入端口多条边。
- Input.shape 是逗号分隔文本。默认 axes 适用于三维 [batch, sequence, feature]；其他维度明确指定 axes，或设 `axes: ""` 自动命名。符号尺寸如 `B, T, 64` 可能产生尚不能确认的约束警告。
- Transformer 默认 [B,T,D]，Conv1d 为 [B,C,L]，通常用 Permute 转换。完整示例见 examples/cnn-transformer.recipe.json。

### Q/K/V 三输入注意力

`MultiheadAttention` 提供 `q`、`k`、`v` 三个输入和 `out` 输出。参数为 `d_model`（Q 与输出特征宽度，默认 64）、`heads`（默认 8）、`kdim`、`vdim`（默认 0 表示跟随 d_model）、`causal`（默认 false）。Q/K/V 是投影前输入，模块内部包含各自的线性投影。

Q `[B,Tq,d_model]`、K `[B,Tk,kdim]`、V `[B,Tk,vdim]`，输出 `[B,Tq,d_model]`；三路 batch 和浮点 dtype 必须一致，K/V 序列长度必须相同。Q 长度可独立变化。连接相同来源可做自注意力，连接不同来源可做交叉注意力。`causal` 标记按序列索引屏蔽未来位置，未实现外部 mask 或 KV Cache 位置偏移。

示例 `examples/cross-attention.recipe.json` 展示不同序列长度与 K/V 特征宽度：

```sh
node packages/cli/scripts/agent.mjs describe MultiheadAttention
node packages/cli/scripts/agent.mjs build examples/cross-attention.recipe.json --out cross-attention.json
node packages/cli/scripts/agent.mjs validate cross-attention.json
```

配方中的连接为 `query.out → attention.q`、`key.out → attention.k`、`value.out → attention.v`、`attention.out → output.in`。展开诊断路径 `attention/q_heads`、`attention/k_heads`、`attention/v_heads` 和 `attention/scores` 可观察多头与矩形注意力分数形状。缺失任一路输入或尺寸不兼容会阻断输出。

## 布局分区

`LayoutRegion` 是纯布局区域，圆角浅色填充，没有计算端口；不要用它代替模型调用、`IfRegion` 或 `LoopRegion`。分区及成员保持同一层级，只有显式加入的节点受边界约束。不要自动为用户已有图分区。

通过 API 新增空分区，再用 `assign_region` 移入成员。保留其他节点位置与快照；被显式移入的节点重新排在分区内，已有成员不重排。示例假定 `q`、`k`、`v` 是当前图中的节点 ID：

```js
const result = patchModel(project, {
  operations: [
    { op: 'add_node', node: { id: 'qkv_region', type: 'LayoutRegion',
      label: 'Q/K/V 投影', position: { x: 200, y: 300 },
      params: { sizing: 'auto', width: 900, height: 400, color: '#38b2ac' } } },
    { op: 'assign_region', ids: ['q', 'k', 'v'], region: 'qkv_region' },
  ],
}, { mode: 'draft', layout: 'preserve' });
```

`assign_region` 的 `region: null` 解除归属而不移动节点；指定另一分区则转移归属。`fit_region` 自动适配，`resize_region` 设置手动宽高并可选修改左上角 `position`，不会裁剪成员。`set_position` 移动分区时携带成员，移动成员时限制在区域内；内部手工线路随整体平移，跨界连线重新走线。每项操作都支持 `path`。

`set_params` 可设置颜色、尺寸模式和 `members`；设置成员会执行与 `assign_region` 相同的归属/移入逻辑，优先使用专用操作。节点“所属分区”由分区 `params.members` 计算，不另写 `regionId`。每个节点只属于一个分区，不嵌套分区，不把 if/for 虚线框本身归入分区。删除分区保留成员。

可用 `rename_node` 修改分区名称，`set_params` 设置 `note`（最多 240 字符）与 `outputs`（逗号分隔的本分区成员 ID）。例如 `{op:'set_params', id:'vision_region', params:{note:'提取多视角视觉特征，预测占用点及类别。', outputs:'outs'}}`。注释显示在标题下，输出成员显示独立徽标，不用绿色边框伪造 forward 最终返回身份；这些编辑不改变已有几何信息。标记仅对应指定节点，不传播给区外关联副本；解除成员归属或删除成员时清理，复制分区时重映射 ID。

用户要求分区输入不向外拉长线时，对原输入使用 `duplicate_linked` 在区外使用点附近创建关联副本，再用 `disconnect/connect` 重接对应的外部使用边；保留区内唯一原输入，不增加 forward 形参，不绘制主节点到副本的连线。只处理用户指定分区的输入使用，不将普通中间张量自动全部拆成副本。

纯 SDK 没有浏览器节点测量值，按默认卡片尺寸安排位置；打开编辑器后使用实际测量尺寸安全扩展边界，不改变模型计算。显式 `layout` 保留已有分区及成员位置，只重排未归属节点。

## Forward 输入

### if 条件范围框

`IfRegion` 是画布中的虚线范围框，不是函数或执行引擎。`condition` 写条件表达式（不含 `if` 和冒号），`members` 为同层节点 ID 的逗号分隔文本，`note` 为说明。条件只显示，不执行 Python、不改变数据边或推导顺序；所有内部形状均为静态检查，不表示两个条件同时成立。示例参数：`{"condition":"self.pretrain","members":"cast,normalize,match,set","note":"预训练时每次 forward 刷新分组"}`。

范围根据成员节点的实际尺寸自动计算，不改变旧节点位置或连线。画布多选节点后点击工具栏「if 框」；也可从库拖入空框，在右侧勾选成员。通过移动内部节点调整范围；非空框自身不整体拖动。支持严格树形嵌套：LoopRegion 可以直接包含 IfRegion，每个节点或子框只能有一个直接父框，不允许自引用、环和重复归属。父框包含子框标题和边界。删除内层框保留代码并提升到父框，复制框递归包含成员。API 使用普通 `add_node` / `set_params` 操作，无需直接编辑项目 JSON。

同一模块属性有多次赋值时，`SetAttribute` / `GetAttribute` 可设置 `version`，例如 `initial` 和 `pretrain`。每次 SET 使用不同版本，GET 显式选版本。它们仍然表示同一个 Python 属性，不是两个属性或参数；版本只消除静态绑定歧义，不模拟条件、执行顺序或最终状态。未写版本的旧单次赋值保持兼容。

`ForwardInput` 无输入端口，输出端口为 `out`。画布以简单矩形显示 `label`，默认“Forward输入”；无需填写形状。参数 `text` 是可选备注（默认空字符串，最多 1999 字符），保存在右侧属性中，不在画布另加内容框。示例为 `examples/structured-input.recipe.json`。

结果中的 `ValueSpec`：Tensor 保持 `{shape, axes, dtype}`；Forward 输入为 `{kind: "forward", text}`。读取形状前需判断 `"shape" in value`。Forward 输入可通过 Identity、Output 和自定义子图边界传递，直接连接张量算子会报告类型错误。

旧项目和旧 Recipe 中的 `DictInput`、`ListInput` 兼容转换为 `ForwardInput`，模块目录和新 Schema 只公开 ForwardInput。转换保留自定义名称、备注与连线；旧默认名称改为“Forward输入”。

`node packages/cli/scripts/agent.mjs build examples/structured-input.recipe.json --out forward-input.json`

## 切片与计算模块

模块目录新增 `Slice` 和 `Compute`，Input.dtype 新增 `bool`。原有 `Concat` 的类型和 a/b 端口不变，界面名称及图形为圆形 Cat。

- `Slice` 的参数为 `dim`（默认 -1）、`start`（默认空字符串）、`stop`（默认字符串 "2"）、`step`（默认 1 的正整数）。边界以字符串传入，空字符串表示对应端点省略；默认对应 `[..., :2]`。
- `Compute.params.operations` 是 JSON 数组序列化后的字符串，必须有 1–16 步。各步为 `{"op":"invert"}`、`{"op":"any","dim":null,"keepdim":false}` 或 `{"op":"slice","dim":-1,"start":"","stop":"2","step":1}`。不接受任意函数名或代码。
- `invert` 只接受 bool/int64，保持形状；`any` 输出 bool，dim=null 归约全部轴，dim 为整数则只归约指定轴。keepdim=true 将被归约轴保留为长度 1，否则删除这些轴，全部删除后形状为标量 `[]`。
- 所有运算仍为形状和类型推导，不执行 PyTorch 或数值计算。组合中的错误会注明失败步骤；原模型和画布在应用前保持不变。

完整示例为 `examples/calculation.recipe.json`，可以直接 build、validate 并在网页导入；其中输出应为 `[2,3]`、dtype 为 bool。网页逐轴编辑仍序列化为 Input.params.axes 的逗号分隔文本，旧 Recipe 保持兼容。

### 序列读取与已有权重

`TensorSequenceGet` 用 `in → out` 读取 Tensor 的 list/tuple 元素；`index` 是整数表达式字符串（默认 `"0"`，支持负索引）。例如 `mlvl_feats[0]` 返回完整的 `[B,N,C,H,W]` Tensor，不移除 batch 轴，也不执行 TensorIndex 或 stack。空序列、越界和非 Tensor 序列均报错；已有梯度标注原样保留。

`LearnedParameter` 新增 `initialization`（默认 `"normal_0.02"`，保持旧模型行为）与 `source_note`。用 `initialization: "existing"` 表示读取已有 `nn.Parameter`，如 `self.init_points.weight`；在备注里记录实际构造和初始化来源。不会在每次 forward 中重新随机初始化，也不是调用 Embedding 查表。仅记录规格，不虚构权重数值或运行时梯度状态。扩维和按 batch 复制可继续连接 `Unsqueeze` 与带动态整数端口的 `Repeat`。

`EmbeddingXYZInit` 是可展开的 **构造阶段说明**，不是 forward 调用：`EmbeddingWeight` 创建 `nn.Embedding(Q,3)` 并展示注册的 `.weight`；两个 `UniformInitColumns` 原地初始化 x 为 U(0,1.1)、y/z 为 U(0,1)。端口返回完整权重状态，不把模块对象伪装成 Tensor，也不新增参数副本。输出的 `requires_grad` 来自构造声明（默认 true，来源为静态推导），不是运行采集或已存在梯度值；加载 checkpoint 可覆盖初始数值。真实冻结时应修改声明或使用运行快照。初始化内部的 no_grad 不改变参数本身的 requires_grad。

建议背景继续只表示梯度状态；蓝色边框 `#60a5fa` 配合 `Parameter` 名称标识模型权重，扩维/复制结果仍是普通 Tensor，不继承“模型参数”的身份。`set_tensor_border` 可通过 API 应用此约定，颜色本身不改变数值或梯度语义。

## API 与返回结果

源码入口 packages/model-core/src/agent/index.ts，构建后的 ESM 入口 packages/model-core/dist/tensor-trail-agent.js，不依赖 React、DOM 或 localStorage。修改注册表后重新 build:agent；构建产物不提交。

```js
import { describeModules, getSchemas, buildModel, validateModel, patchModel }
  from './packages/model-core/dist/tensor-trail-agent.js';
const catalog = describeModules();   // 默认不含内部边界模块
const all = describeModules(true);   // 包括 Block、Inlet、Outlet
const schemas = getSchemas();        // recipe / patch，Draft 2020-12
const draft = buildModel(recipe);
const checked = validateModel(draft.project); // 也接受 Recipe
const changed = patchModel(draft.project, patch);
```

构建、验证、修改只生成草稿，没有文件或画布副作用。

| 字段 | 含义 |
| --- | --- |
| apiVersion | 当前为 1 |
| mode | 本次校验模式：strict 或 draft，不写入项目文件 |
| ok | 没有 error 为 true；warning 不会使其为 false |
| diagnostics | code、severity、path、message；参数错误另有 parameter |
| nodes | 各层级节点的 path、type、status、inputs、outputs |
| outputs | 按顶层 Output 节点 ID 索引的最终 TensorSpec |
| defaultsApplied | 补全的 path、parameter、value；只计 Recipe 显式节点，不重复计算生成的内部算子 |
| project | 完整可导入项目；结构错误时可能没有此字段 |

结构合法但形状不正确时仍返回 project 供继续修改；不要仅看 project 存在就应用。诊断路径如 `["encoder", "attention", "q"]` 定位 Q 投影；入口约束失败时可能停在组合模块路径，内部推导尚未发生。

| 常见 code | 处理方式 |
| --- | --- |
| UNKNOWN_MODULE / UNKNOWN_PARAMETER | 查询模块目录，修正名称 |
| INVALID_PARAMETER | 修正类型、范围或枚举 |
| STRUCTURE_ERROR / INVALID_ENDPOINT | 检查节点、端口、重复连接和环路 |
| SHAPE_ERROR / UPSTREAM_BLOCKED | 根据 path 修正形状或先修复上游 |
| MISSING_INPUT / MISSING_OUTPUT | 补齐顶层输入输出 |
| INCOMPLETE_NODE | 草稿中的未连接节点及其依赖；未推导出输出，不伪造张量 |
| RUNTIME_NOT_LOADED | 草稿仅检查静态图；运行快照未加载，原引用仍保留 |
| SHAPE_WARNING | 保留并解释不能验证的符号约束 |
| CUSTOMIZED_COMPOSITE | 用 path 修改内部算子 |
| REVISION_CONFLICT | 重新读取最新模型并重做修改 |
| WARNINGS_REQUIRE_ACK | 检查警告后显式允许应用 |
| LIMIT_EXCEEDED | 缩小模型或操作批次 |

## Patch v1

```json
{
  "patchVersion": 1,
  "operations": [
    { "op": "set_params", "id": "encoder", "params": { "norm": "RMSNorm", "ffn": "SwiGLU" } },
    { "op": "rename_node", "id": "encoder", "label": "我的 Transformer" }
  ]
}
```

| 操作 | 必填字段 | 作用 |
| --- | --- | --- |
| add_node | node | 按 Recipe 节点格式新增 |
| replace_node | id, node | 原子替换同一 ID 的类型与参数；保留默认位置，完整事务须修复受影响端口 |
| remove_node | id | 删除节点及关联边；保护 Inlet/Outlet |
| link_value | id, source | 把已有值节点恢复为同类型主节点的关联副本，保留 ID、位置、分区与输出连线 |
| set_params | id, params | 合并参数；未手工修改的组合模块重新生成 |
| rename_node | id, label | 修改显示名 |
| set_position | id, position | 移动节点，分区携带成员，成员受边界约束；不触发祖先 customized |
| assign_region | ids, region | 将同层节点移入分区；region 为 null 时解除归属 |
| fit_region | id | 将布局分区设为自动适配成员范围 |
| resize_region | id, width, height | 设置手动尺寸，可传 position；不能裁剪成员 |
| set_tensor_border | id, color | 设置本节点边框（`#RRGGBB`）；`null` 恢复默认，不改背景、梯度或布局，关联副本仅改本地颜色 |
| connect / disconnect | from, to | 连接/断开 node.port 端点 |
| layout | 无 | 重排目标图的未归属节点，保留布局分区及成员 |
| rename_project | name | 修改项目名 |

patchVersion 可省略，默认 1。除 rename_project 外，每项都可传 `path: ["encoder", "attention"]` 指定子图，省略为顶层，id 相对于该子图。

```json
{
  "operations": [
    { "op": "set_params", "path": ["encoder", "attention"], "id": "q", "params": { "in_features": 64, "out_features": 64 } }
  ]
}
```

内部语义编辑会标记所有祖先 customized，之后父级 set_params 拒绝重建，保护手工结构；应继续改内部算子。重命名和布局不会触发标记。

`link_value` 用于关联已有节点，而 `duplicate_linked` 会新建副本。它不按名称猜测来源，必须明确指定同层 `source`；源是副本时直接引用其主节点。目标不能有独立输入边、不能已经拥有其他副本，也不能是 Inlet/Outlet（避免隐式合并父图接口）。例如恢复输入副本：`{op: 'link_value', id: 'img_metas_near_head', source: 'img_metas'}`。它保持现有输出边、手工走线和分区归属，参数等共享元数据以主节点为准。

整批操作作用于克隆模型，最后统一检查，允许断开旧边、插入节点、重新连线。失败不会改变源模型。Patch 默认保留已有布局，仅安排未指定 position 的新节点；显式 layout 操作仍重排目标图。第三参数 `{layout: 'auto'}` 恢复添加节点后重排所在图的旧行为。Patch 保留 variants；build 新项目 variants 为空。

## 草稿与保留布局

适用于 SparseWorld 等按代码逐段补图的流程。校验策略是一次调用的选项，不改变静态图的 schema，也不把草稿当成完整模型。

```js
buildModel(recipe, { mode: 'draft' });
validateModel(project, { mode: 'draft' });
patchModel(project, patch, { mode: 'draft', layout: 'preserve' });
```

- `mode` 默认 `strict`，可选 `draft`。草稿允许缺少顶层输入/输出、尚未连接的输入，以及仅由这些缺失造成的下游阻塞；诊断为 warning，节点 trace 标为 incomplete，不生成虚假的输出。
- 环路、重复 ID、错误端口、未知模块/参数、已知形状或 dtype 错误始终是 error。`ok: true` 在 draft 下仅表示可继续编辑，不表示模型完整或可训练。
- 子图仍必须有合法边界。逐步展开请用 `Submodule`，至少保留一个 Inlet，可以暂不定义 Outlet；不要为了通过校验伪造返回值。
- `layout` 默认 `preserve`。旧节点位置、端口布局、现有连线拐点和标签位置不动；新节点有 position 时直接使用，否则确定性地放到已有布局右侧。生成式组合模块改参数后，保留仍存在的节点/连线几何信息。
- `set_position` 对未归属节点只改变其位置；布局分区携带成员，成员受边界约束，相关线路按上面的分区规则处理，不将祖先标成 customized。显式 `layout` 操作优先于保留布局，但保留已有分区及成员；`layout: 'auto'` 会在添加节点的子图中重排未归属节点。
- 原地修改的节点、连线和参数使用 Patch；不必手工编辑持久化 JSON。只有模块目录没有对应语义时才实现新的算子、形状规则并重新构建。

CLI 对应命令：

```sh
node packages/cli/scripts/agent.mjs validate ../SparseWorld.tensor-trail.json --draft
node packages/cli/scripts/agent.mjs patch ../SparseWorld.tensor-trail.json examples/sparseworld-incremental.patch.json --draft --layout preserve --out ../SparseWorld.preview.tensor-trail.json
node packages/cli/scripts/agent.mjs schema patchOptions
```

示例 Patch 仅在 `extract_feat` 子图添加 `img_normalized_copy` 展示节点，并连接 `img_after_normalize.out`，不添加新的模型计算。重复应用同一 ID 会被拒绝。`--out` 只在所选模式下无 error 时写出；覆盖现有文件仍需显式 `--force`。CLI 不会同步已经打开的桌面画布，文件写出后需重新打开。

画布 API 可以即时应用并一次撤销，仍必须使用最新 revision 并确认 warning：

```js
const api = window.tensorTrail;
const snapshot = api.getModel();
const changed = api.patchModel(snapshot.project, patch, {
  mode: 'draft', layout: 'preserve',
});
// 先阅读 changed.diagnostics；确认所有提示后再执行：
if (changed.ok) {
  api.applyModel({
    project: changed.project,
    expectedRevision: snapshot.revision,
    mode: 'draft',
    allowWarnings: true,
  });
}
```

界面的“AI 构建”也提供“草稿模式”和“保留已有布局”开关；修改开关会使上次预览失效，需要重新检查。“应用”只修改画布，保存仍沿用当前文件流程。

运行快照和静态图继续分离。独立 SDK/CLI 不自动读取侧边文件；草稿缺少活动快照时给出 `RUNTIME_NOT_LOADED`，仅对临时副本使用静态推导，返回的项目保留原 active/snapshots 引用。严格模式仍报错。SDK 可传 `{mode: 'draft', runtimeSnapshot: loadedSnapshot}`，画布 API 自动使用已加载的运行快照，不把它嵌回主 JSON。提供了无效或绑定失效的快照仍报错，不能以草稿模式绕过。运行快照改变也会使画布 revision 失效。

`replace_node` 可在同一事务中替换类型和参数，必须保留相同 ID；未指定位置时保留旧位置，显式端口布局与边框也保留。关联副本需先解除或移除。替换 Inlet/Outlet 时必须同步调整父图连线，否则整批拒绝；`remove_node` 仍不允许直接删除边界。示例：`{"op":"replace_node","path":["head"],"id":"scale","node":{"id":"scale","type":"GetAttribute","params":{"owner":"[\"head\"]","name":"points_scale"}}}`。

## 自定义组合和新算子

Recipe 的 Block 必须提供 subgraph；内置组合模块也可显式覆盖 subgraph，都会标记 customized。子图仍为 nodes/connections，至少有一个 Inlet 和一个 Outlet。

```json
{
  "id": "my_block", "type": "Block", "label": "我的归一化模块",
  "subgraph": {
    "nodes": [
      { "id": "input", "type": "Inlet", "params": { "port": "in" } },
      { "id": "norm", "type": "RMSNorm", "params": { "features": 64 } },
      { "id": "output", "type": "Outlet", "params": { "port": "out" } }
    ],
    "connections": [["input.out", "norm.in"], ["norm.out", "output.in"]]
  }
}
```

Inlet.params.port 定义外部输入端口，内部输出为 out；Outlet.params.port 定义外部输出端口，内部输入为 in。同图同方向边界端口必须唯一。

协议只组合已注册模块，不执行传入代码。RoPE、GQA 等新算子先在 src/modules/ 增加定义和形状规则，再注册并重新构建；目录与 Schema 自动更新。视觉效果通过 src/views/ 扩展。

## 操作当前画布

以下用于有页面 JavaScript 调用能力的工具适配器或开发者工具。

```js
const api = window.tensorTrail;
const snapshot = api.getModel();
const draft = api.patchModel(snapshot.project, {
  operations: [{ op: 'rename_project', name: 'AI 搭建的模型' }]
});
if (draft.ok && !draft.diagnostics.some(d => d.severity === 'warning')) {
  const applied = api.applyModel({
    project: draft.project,
    expectedRevision: snapshot.revision
  });
  console.log(applied.ok, applied.diagnostics);
}
```

画布 API 同时提供 describeModules、getSchemas、buildModel、validateModel。getModel 返回独立快照和不透明 revision；任意模型变更、撤销重做、导入或页面刷新都会改变 revision。applyModel 必须带读取时的 revision；冲突时重新读模型并重做修改，不能给旧草稿换新 revision 绕过检查。

有警告时须检查后显式传 `allowWarnings: true`。成功应用进入当前会话的编辑历史，需使用 Ctrl+S 保存到文件，清空重做记录、回到顶层，可一次撤销。直接 API 应用使用完整 project，包含 variants；新建时想保留变体，先把 snapshot.project.variants 赋给草稿再 validate。UI 面板已处理。

**CLI 只操作文件，不会自动改变打开的网页。** 生成文件需要导入，或调用页面 API。没有远程写入端口；后续 MCP 适配层可包装同一套函数，复用推导规则。

## CLI 约定与限制

- describe 支持可选模块类型和 --internal；schema 支持 recipe/patch，省略则输出两份。
- build 接收 Recipe；validate 接收 Recipe 或完整项目；patch 接收完整项目与 Patch。build/patch 可用 --out 写出。
- `-` 代替一个输入文件表示 stdin，至多一个 stdin 输入。
- stdout 是单个 JSON；自动化直接调用 node，npm run agent 会混入 npm 日志。
- 退出码 0 为成功/无 error，1 为协议或模型校验失败，2 为命令参数、读写或 JSON 语法错误。
- --out 只在 ok:true 时写出完整 project，不写报告包装。默认拒绝覆盖，--force 显式允许覆盖；校验失败即使 force 也不写文件。warning 模型可导出，画布应用另需确认警告。
- 输出父目录应已存在。CLI 不联网、不调用大模型、不执行传入代码。
- JSON 最多 3 MB，项目含展开子图及 variants 合计最多 2000 节点，每图最多 8000 边，嵌套最多 12 层，Patch 为 1–500 个操作。
- Schema 校验字段；跨节点、深度、总量和数学约束由运行时验证。形状检查不代表训练或显存可行性验证。

## 可直接交给 AI 的指令

```text
为 Tensor Trail 搭建模型。
1. 先读本指南，用 describe 和 schema 查询真实模块；不要虚构 type、参数或端口。
2. 整理架构的输入尺寸、轴顺序、参数、分支和残差；缺失信息明确列出假设。
3. 生成 Recipe v1，包含 Input/Output 和稳定英文 ID；非自定义模块不手写子图和位置。
4. 调用 build，读取 diagnostics/defaultsApplied/nodes/outputs，修复错误并重新检查。
5. 修改现有模型先读取 project，再用 Patch；customized 模块深入内部修改。
6. CLI 输出文件；浏览器应用使用原快照 revision，冲突后重新读取并重做修改。
7. 报告最终尺寸、假设和警告，不声称执行了训练或 PyTorch 前向。
```

## 图像尺寸与元数据旁路

`TensorShapeSlice` 接收 `in: Tensor`，参数 `start="-2"`、`end=""` 对应
`img.shape[-2:]`；输出 `out` 是 `{kind:"size", dims:[H,W]}`，界面显示为
`torch.Size([H,W])`，不是 Tensor，不带梯度状态。可接 `ValueVariable` 命名为 `input_shape`。

`UpdateImageMetaInputShape` 接收 `img_metas` 和 `input_shape` 两个端口，表示
`for img_meta in img_metas: img_meta.update(input_shape=input_shape)`。
`img_metas` 使用 `ForwardInput` 的不透明占位或已有列表说明；`input_shape` 必须为两个尺寸的
`torch.Size`。输出 `out` 是原列表更新后的状态说明，不是新 Tensor 或新 Python 列表。
右侧显示已知字段更新，其他字段与列表长度不做推测；这不是完整元数据采集或 Python 执行器。
两个模块可通过草稿模式 API 添加，修改上游图像尺寸会重新推导 H、W。

## ResNet-50 命名模块

`CodeFunction` 的 `operation="ResNet50Backbone"` 对应 SparseWorld 当前的固定
MMDetection ResNet-50 配置。输入端口 `in` 为浮点 NCHW 图像；`c2/c3/c4/c5`
分别表示实际返回 tuple 的 `img_feats[0]/[1]/[2]/[3]`，可连接四个 `TensorVariable`。
这只是将 tuple 元素分别展示，并非四次 backbone 调用或额外张量堆叠。

通道分别为 256/512/1024/2048，总步长分别为 4/8/16/32，空间大小按向上取整推导。
模板只显示网络名称及输入输出契约，不展开 ResNet 内部，也不执行 Python。
右侧代码注明 `use_grid_mask=False` 和 tuple 不触发字典转列表的当前路径。
FPN 与后续 batch/view 恢复不属于这个模块；输出梯度状态不凭参数可训练性推测。

## FPN 子图与二维插值

FPN 可通过 API 用 `Submodule`、`Conv2d`、`Add` 和 `CodeFunction` 组合，不需要单独硬编码 FPN 输出。
`operation="Interpolate2DNearest"` 接收 `in` 与 `like` 两路 NCHW Tensor，输出为
`F.interpolate(in, size=like.shape[2:], mode="nearest")`；只从 `like` 读取高宽，源 N/C/dtype 保留。
不要替换为固定放大两倍，奇数高宽需要精确对齐；这个节点不负责相加。

SparseWorld 当前 FPN 为四路 1x1 lateral 卷积，三次从粗到细的最近邻插值与 Add，
以及四路 3x3/padding=1 输出卷积；通道均为 256，无 BN、激活或额外金字塔层。
下一次上采样接前一次 Add 的结果，而不是未融合的 lateral 或最终 3x3 输出。
子模块的 c2-c5/p2-p5 端口分别展示输入/输出 tuple 元素，不表示四次 FPN 调用。

## tuple/list 返回值与空列表

`BooleanIndexAxis` 表示一维 bool 掩码筛选指定轴，例如 `dim="1"` 对应 `x[:, mask]`，
输入为 `in` 和 `mask`，不广播掩码、不合并 batch。默认输出数量为 `count_symbol`；
`declared_count=true` 增加整数 `count` 端口，用于有算法保证的 True 数量契约，不能将其说成软件实际计数。
例如按配额分组后的当前帧 query 数为 720；该节点不执行分组或读取真实掩码。

`NewZeros(metadata_inputs=true)` 可分别连接 `GetDtype.out → dtype` 和 `GetDevice.out → device`，
不再连接参考 Tensor 的 `in`。尺寸仍由 `shape` 中的 `d0`–`d7` 指定；创建独立的全零 Tensor，默认 grad:off、无原始输入来源编号。
旧图未设置该参数时仍使用原来的参考 Tensor 输入。

`IntegerSubtract`、`IntegerMinimum`、`IntegerMaximum` 使用 `a,b → out`，输出 Python 整数而非 Tensor。
它们与 `IntegerAddOne` 可展开 `max(1,min(epoch-finetune_epoch+1,limit))`，只计算声明的整数。
训练/推理条件使用互斥的 `IfRegion` 范围框展示；不要将两条输出合并成同时执行的数据流。

`TensorSequencePack` 用 `kind="tuple"` 或 `"list"`、`input_count=0..16` 配置容器。
输入端口为 `item0`、`item1` 等，输出 `out` 是保留各元素 shape/dtype/梯度标注的有序容器，
不是 Tensor，不要求元素同形，也不是 `stack` 或 `concat`。右侧观察器可查看每个元素。
将 `out` 接入 `Outlet`，父模块就通过一个端口返回整个 tuple/list，再用 `ValueVariable` 接收。
这表达已有返回值的结构，不表示源码额外复制或计算特征。

`kind="list", input_count=0` 没有输入端口，表示 `[]` 初始化，可接 `ValueVariable`。
不要从已有特征画一条虚假的输入依赖；空列表尚不包含后续循环 append 的 Tensor，
也不是形状 `[0]` 的 Tensor。容器本身没有 `requires_grad`，元素各自保留标注。
目前仅支持 Tensor 元素，不推导 Python 对象身份或 list 原地修改；不改变已有不透明 `list[dict]` 元数据。

## 关联副本与就近连接

已知的设备和类型可以直接用无输入的 `DeviceConstant(device="cuda:0")`、
`DtypeConstant(dtype="float32")` 表示，画布只显示 `DEVICE cuda:0` / `DTYPE float32`。
`context` 记录原 `.device/.dtype` 表达式、确定值的依据及适用配置。
这是固定值展示，不是运行时读取、数据转换或新的模型输入；切换精度或设备时必须同步更新。
未知值仍保留 `GetDevice/GetDtype`，不能为减少连线猜测设备或类型。
两类常量支持 `duplicate_linked`，可让每个使用点有一个就近引用；不需要张量到元数据值的连线。

通过 `duplicate_linked` 创建已有值节点在同一画布中的关联副本，例如：

```json
{"op":"duplicate_linked","path":["extract_feat"],"source":"B","id":"B_near_loop","position":{"x":4810,"y":100}}
```

`source` 和新 `id` 必须属于同一 `path`；若 source 本身是副本，会直接关联原始主节点，不产生引用链。
支持已有可关联值类型，例如 LocalParameter、IntegerParameter、TensorVariable 和 Inlet；不支持计算算子。
副本没有独立输入端口，不需要主节点到副本的连线；输出可就近连接消费者。
局部参数主节点可以只有输入、没有输出连线，仍然负责接收计算结果并为关联副本提供值。
不要为了保留主节点的直接使用点而拉长线：定义放在计算处，远处使用全部改为就近关联副本。
只作为副本值来源的主节点不是无用节点，不能清理删除。已有单用途副本太远时可直接移近使用点。
用 `disconnect/connect` 将原长线替换为副本到消费者的短线。它主要减少长线和交叉，不一定减少边的总数。
改名或改参数会同步主节点，位置独立。Inlet 副本不会增加父模块的输入端口。
循环中的副本共享当前迭代的值，不能把外层同名参数当成循环体内的尺寸值。
此操作同样支持 `mode:"draft",layout:"preserve"`，失败时整个补丁回滚；不直接编辑项目 JSON。

### 末尾返回结构的就近引用

补图时从当前 forward 的 `return` 倒查。如果其前方只有变量引用、`DictionaryPack`、
`TensorSequencePack` 或 `DictionaryUpdate`，可为已经形成的张量/结果列表创建就近关联副本，
一对一连接末尾打包端口。此规则不局限于局部参数；计算处的主节点可以没有直接使用连线。
不要改为独立常量或复制计算，也不要把普通中间计算链全部改成关联副本。

软件自动用加粗绿色边框标出本层原样返回的值及其关联副本，不需要写入自定义边框颜色。
倒查只穿过引用和纯容器组装，遇到计算、索引、detach、条件选择、列表追加或子模块调用即停止。
字典更新按字段排除被覆盖的旧值；仅部分字段保留下来的旧字典本身不标为完整最终输出。
循环体状态 Outlet 不作 forward 最终返回值处理；缺少有效推导结果时不猜测。
该身份与梯度、模型参数及输入来源独立：绿色不等于 no_grad，也不禁止其他分支继续使用该值。
只改本次涉及的尾部连线，用 `layout:"preserve"` 保留其他布局、路径和运行快照。

## ForEach 循环子图

`ForEachTensor` 将 Tensor tuple/list 的每个元素送入同一个子图，逐轮推导后按序收集为 list。
外层输入 `items` 是已知元素的 tuple/list，`initial` 是初始列表（通常由 `TensorSequencePack` 创建 `[]`）。
子图必须包含 `Inlet(port="item")` 和唯一 `Outlet(port="out")`，后者表示本轮待 append 的单个 Tensor。
可选 `Inlet(port="index")` 接收从 0 开始的整数迭代索引；其他 Inlet（例如 `B`）自动变为外层捕获端口，
各轮接收相同值。`items/initial/out` 为外层保留名，不能作为子图 Inlet。

这不是带状态的 RNN，不应将上一轮特征反馈到下一轮。初始列表不进入循环体，收集只发生在外层；
不支持 break/continue、逐轮条件过滤、修改捕获值或循环携带状态。不同迭代允许不同 shape/dtype。
空输入跳过循环体并保留初始列表。为避免静态推导膨胀，每个循环最多 16 次，收集结果最多 256 个元素，
一次完整推导最多 20000 次节点访问；这些是软件限制，不是 Python 限制。

双击进入循环体后，通过“当前迭代”选择器查看各轮的尺寸与列表长度变化。
选择器是临时视图状态，不写项目文件，不产生撤销记录，也不执行 Python。
所有迭代都会被验证，即使当前显示的是第一轮；API 的循环节点 trace 包含各轮 `iterations[index].outputs`。
子节点 trace 默认展示第一轮；桌面内部使用 `resultAt(result,path,selections)` 查看指定迭代。
草稿补图继续使用 `patchModel(...,{mode:"draft",layout:"preserve"})`。

整数尺寸模块 `IntegerTruncDivide` 的 `a/b → out` 对应 `int(a / b)`，向零截断而非向下取整，
拒绝除数 0；符号输入保留表达式并提示约束。可用于 SparseWorld 的 `BN / B`，再连接 `View` 的尺寸端口。
`View` 只验证尺寸/元素数，不验证 PyTorch stride 是否允许真正的零拷贝 view。

`IntegerFloorDivide` 的 `a/b → out` 对应 Python `a // b`，向下取整（例如 `-7 // 3 = -3`），
与向零截断的 `IntegerTruncDivide` 区分。输入必须是局部整数或符号尺寸，拒绝除数 0；
符号结果保留整除表达式，不伪造数值。可用于 `C = GC // G`，画布以紧凑 `a // b` 节点显示。

## For 状态循环

需要在父画布直接观察调用实参时，可以改用 `LoopRegion`：
`statement` 填写 `for interval in range(num_fu_frames)` 等语句，`members` 指定同层节点 ID，
`note` 必须说明展示的是哪一轮及未绘制范围。虚线框只标注“框内操作每轮执行”，
不封装成员、不求值循环语句、不自动模拟多轮或生成最终输出；与 `ForLoop` 的状态递推不同。
删除框保留代码，复制框递归携带成员；边框随节点位置自适应。支持唯一直接父框的树形嵌套，不允许成员重复归属或范围环。

`ForLoop` 表示有界 `for index in range(N)`，参数 `iterations` 为 0-16 的已知整数，
`states` 为逗号分隔的 1-8 个状态名。循环体需要 `Inlet(index)` 和全部状态 Inlet；
其他 Inlet 是每轮相同的捕获输入。Outlet 必须恰好对应状态名，其输出自动传给下一轮同名 Inlet。
外层没有 index 输入，只接收初始状态和捕获值，并输出最终状态；不要画普通反馈边制造环路。

设置 `iterations_input: true` 时，外层增加整数控制端口 `iterations`，取代静态参数 `iterations`；
可连接已算出的 `num_fu_frames` 等局部整数，仍要求已知值在 0-16 内。此模式禁止同名循环体 Inlet，
控制次数不是特征输入，也不会自动成为循环体捕获值。旧图默认继续使用参数模式。
逐段补图时可暂不连接状态 Outlet，使用 draft 校验；已连通的第一轮节点仍能查看，
但循环不返回最终状态、不假装剩余代码是恒等操作，也不推测后续轮次结果。

状态支持 Tensor、Tensor list/tuple 和局部整数，允许 Tensor shape 与列表长度逐轮变化，但不能改变值种类。
使用紧凑 `TensorListAppend(items,item)` 表达每轮 `list.append(tensor)`，保留每个元素的形状、梯度和来源；
不是 stack/concat，收集结果最多 256 个 Tensor。零次迭代原样返回状态。
所有轮次按序静态推导，后续轮次出错时不返回部分状态；循环体可切换迭代查看，嵌套子模块保持选定轮次。
不执行 Python，不支持 break/continue、数据依赖循环次数或循环内 SetAttribute。
循环体不能脱离 ForLoop 保存为普通模块变体。

`NewOnes` 对应 `reference.new_ones(size)`，支持 `metadata_inputs: true`，由 `GetDtype/GetDevice`
提供元数据和 d0-d7 提供尺寸；不使用参考张量的数值，不继承来源编号，默认梯度关闭。
可接公式为 `x * 0.5` 的 `Function`，画出固定参考坐标；不能用只继承 device 的 `OnesScaled` 冒充。
`OPUSCrossAttentionContract` 声明 query_points `[B,Q,3]`、query_feat `[B,Q,D]`、
key_points `[B,K,P,3]`、key_feat `[B,K,D]` 四个输入，输出 `[B,Q,D]`；Q 与 K 可不同。
它封装距离偏置、位置编码、多头交叉注意力及残差，不执行网络；仅暴露第一个被使用的返回值，
第二个被 `_` 忽略的注意力权重不接入数据流。detach 保留来源编号，但切断对应输入梯度。

需要展开时使用 `OPUSCrossAttention`，不再停留在接口契约：
距离函数包含米制解码、Key 点集均值、广播差、L2 范数、取负与 no_grad 结果标记；
主图绘制 `exp(dist/40)`、Key 特征产生的 `Linear -> Softplus -> Permute`，
两路广播相乘得到 `[B,H,Q,K]` 偏置，再 flatten 成 `[B*H,Q,K]`。
位置分支的两次 `self.pos_embed` 表示同一层的两次调用，不是两套独立权重。
`query_points * 0` 仍是逐元素乘法，不能替换为 new_zeros 的元数据依赖。
`self.attention` 包装层可进入查看：Q/K 加位置、原始 Key 作 V、sequence-first 转置、
`nn.MultiheadAttention`、输出转置、两个 p=0 的 Dropout，以及原始 Query 的残差连接。
`TorchMultiheadAttention` 保留为标准神经网络层接口，不声称已展开其原生实现；
浮点 attn_mask 加到 logits，而不是布尔屏蔽；返回的 weights 默认沿头平均为 `[B,Q,K]`。
checkpoint 条件与 `use_reentrant=False` 记录在展开模块的代码说明中，两条执行路径共享相同前向数据流，
不重复画计算，也不把 checkpoint 当成 detach；DUMP 写文件不计入神经网络数据流。

新增的 `Exp`、`Softplus`、`L2Norm` 是紧凑数学节点。
`NoGradResult` 仅标注 no_grad 内已计算结果的上下文效果，不是源码新增的 detach，
不适用于直接透传输入、Parameter 或显式 requires_grad=True 的工厂例外。

SparseWorld 的 Decoder 示例使用 `query_points,query_feat,cls_scores,refine_pts` 四个状态，
`mlvl_feats,occ2img,img_metas` 为捕获输入。每轮先 detach 输入坐标，再调用独立的 `decoder_layers[index]`。
`OPUSDecoderLayerAt` 只按索引选择接口契约，`refines="1,4,16,24,32,48"` 声明各层输出点数；
index 是模块选择配置，不是额外的 forward 张量，也不作为融合的信息来源。
同一个循环体不表示各层共享网络参数。两个 append 收集本层输出，不能收集下一轮 detach 后的值。
初始化两个空列表在循环外完成。最终只将 query_feat 和两个列表接到 Decoder 的 return。

### 展开 Decoder 层

`OPUSDecoderLayer` 是 `OPUSTransformerDecoderLayer.forward` 的可展开子图模板，保留上述 5 个数据输入和
仅用于配置选择的 index。可用 `replace_node` 替换旧的层占位封装，保持同名端口与外层布局。
各轮 `refines`/`last_refines` 通过整数列表取值；`LinearByDimensions(in,in_features,out_features)`
仅用后两个整数指定该层 Linear 规格，不传播其信息来源，也不表示动态修改同一份真实权重。
shape、来源编号、梯度标注由内部连接传播；不要再保留一个整体契约直接生成输出，绕过内部流程。

`OPUSSamplingContract` 输出 `[B,Q,G,T*P,D/G]`，其中 P 为 `num_points`，不是 `num_refines`。
`AdaptiveMixingContract`、`OPUSSelfAttentionContract`、`MMCVFFNContract` 保留相应模块接口及约束；
复杂内部可后续用子图替换，不能将这些契约误称为已执行的网络。
`ReduceMean` 是通用单轴浮点求均值；`PointRangeTransform` 按 pc_range 对最后 xyz 轴做仿射编码/解码。
`refine_points` 使用米制点集中心加缩放后的位移，再归一化，没有 clamp。
MMCV FFN 自带 add_identity=True，不能用软件内不含残差的普通 FFN 模板直接替代。

## While 状态循环

`WhileLoop` 展示有界循环 `while i < num_points`，每轮固定 `i += 1`。
参数 `states` 是逗号分隔的 Tensor 状态名，例如 `maps,inds`。
循环体需有 `Inlet(i)`、`Inlet(num_points)` 和每个状态的 Inlet；其余 Inlet 是只读捕获值。
循环体的 Outlet 必须恰好是 `i` 和所有状态，分别表示下一轮计数器与下一轮状态。
外层只输出各状态，不暴露循环体中的单轮 `i`。反馈由循环语义管理，不能连接普通边制造环路。

画布只展示一轮符号模板 `i`，检查输出 `i + 1` 及状态 shape/dtype 不变量，不展开 1040 个子图。
初始 i 与上界为已知整数且条件不成立时跳过循环体，原样返回状态。
不支持 break/continue、任意 while 条件、变形状状态或运行时迭代回放。
模板输出只声明循环结束时的规格，不包含真实 Tensor 数值或实际分配结果；不得标为某次真实迭代。
循环体不允许脱离 WhileLoop 保存为普通变体；可复制整个循环节点。

开启 `lineageEnabled` 后，沿状态反馈迭代传播可能来源，最多 32 次分析，来源集合稳定后停止。
这不是执行真实 Tensor 循环；多次迭代的来源取保守并集，按每个状态的实际依赖区分，不合并未使用的捕获输入。
动态索引、标量写入、条件与分支合流均参与来源追踪；元数据仍不生成根输入编号。
零次迭代保持初始来源，符号上界包含零次可能；未知算子或达到分析上限仍保留 `?`。
循环体中的 SET 属性副作用尚不支持，请用状态 Inlet/Outlet 表达更新；允许 GET 外部已写入的属性。

`IfCondition` 产生未执行的条件及 True/False 控制依赖，画为菱形和虚线。
`BranchValue` 为每条可能路径传入 Tensor，`MergeBranches` 检查两个路径的 shape/dtype 一致。
静态检查两条路径不代表 Python 同时执行两条路径，也不是 `torch.where`。
`IntegerList` 是 Python `list[int]`，不是 Tensor 列表。
`TensorItemInt` 只把标量读取表达为符号 Python 整数；float 标量须显式启用 `int(x)`，不能伪装成整数 `.item()`。

SparseWorld 示例在 `packages/model-core/tests/fixtures/matchingLoop.ts`：按 `original_id` 写入 `inds`，
用更新后的 `inds` 统计配额，配额满时仅清零、重排并写回 `maps[i+1:]`，保留已处理前缀。
使用 `patchModel(...,{mode:"draft",layout:"preserve"})` 插入子图，局部值通过 `duplicate_linked` 就近引用。

## SparseWorld 预测与返回

`SparseWorldForecastStep` 是 `forward_backbone` 中一轮预测的详细模板；可将节点平铺到父画布的
`LoopRegion`，也可把其 `interval` Inlet 的 port 改成 `index` 后用于 `ForLoop`。
七个携带状态为 `curr_query_feat/curr_query_pos/curr_query_timestamp` 和四个结果列表。
真实代码各轮复用同一份网络参数；图中的节点仅表达调用，不实例化 PyTorch 模型。

当前工作区图沿用 `training=True, epoch=0, pretrain=True, batch=2`：一轮完整展开到 `return outputs`。
`SingleIterationPreview` 只校验绘图范围，不是模型新增代码。若循环次数改成多轮会明确报错，
不能把首轮形状当最终结果；可复用单轮模板进行真实状态递推的静态推导。
测试覆盖 1 到 6 轮，追加后的 Query 数为 `780/840/900/960/1000/1040`；点/语义列表不能 stack。

- `traj_head`、位置编码及 reg/cls/vel 分支可展开为 Linear、激活、LayerNorm。
- 新增时间戳固定为 0.5，旧时间戳不累加。`fused_ego_feat` 沿 Query 轴广播相加。
- 回归偏移先乘 0.5；XY 另加 `vel_offset * moving_mask`，速度项没有乘 0.5。
- `SparseWorldRefinePoints` 解码点集并求中心，加 48 个独立偏移后编码，不是每个旧点直接加偏移。
- `SparseWorldTransformPoints` 包含 `.cpu() -> linalg.inv -> .cuda()`、米制正向变换、平移、逆变换和编码。
  调用侧先把点展平成 `[B,Q*48,3]`；平移保持 `[B,1,3]`，之后恢复原 shape。
- `pred_traj_expand` 实际取 `[-GT_x,-GT_y,0]`。`zeros_like(pred_traj[:,:,:1])` 不使用预测轨迹数值，
  也不继承其梯度；来源标记只保留真实 GT 数值来源。掩码只检测归一化 X>=0，不是完整空间有效性。
- `SparseWorldForecastFinish` 绘制循环后额外一次轨迹和 `outputs.update`。`SelectValue` 接收已知 Python bool，
  只采用选中路径的值规格；IfRegion 仍只是视觉范围，静态检查另一分支不代表执行了它。
  当前 pretrain=True 不追加；微调未满6步时仅追加一次轨迹，不追加占据结果。
- 原始 `outputs['refine_pts']` 仍是循环前720个Query；循环重新绑定变量不修改已保存的字典字段。

运行快照只采集了 img 和 temporal_ego_states，不篡改为完整采集。
`TensorFieldContract` 为缺失的 `kwargs['temporal_trajs']` 提供明确的静态规格：本地轨迹文件抽查单样本
`[6,2] float32`，batch=2 声明 `[2,6,2]`，不是本次前向实测。
`BatchMetaStack` 保留 NumPy 原 dtype 为 unknown；`ArrayToTensorExplicit` 明确指定源码的 float32/CUDA。
`ReshapeLike` 仅继承参考 shape；`DictionaryUpdate` 浅层更新且保留各字段自己的来源和梯度描述。

## 开发验证

```sh
npm test
npm run build
npm run test:cli
```

测试覆盖混合网络、自动展开、深层诊断、自定义 Block、非法参数/连接、资源限制、原子修改、祖先保护、版本冲突和警告。store 测试覆盖一次撤销、变体保留与持久化；CLI 冒烟覆盖 stdin、退出码和失败时文件不变。

背景见 [架构方案](架构方案.md)；人工编辑和视图扩展见 [使用与扩展](使用与扩展.md)。
