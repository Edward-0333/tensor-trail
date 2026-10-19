# Tensor Trail

**搭建神经网络，沿着数据流理解每一步 Tensor 形状的变化。**

Tensor Trail 是本地运行的神经网络学习工作台。定义输入 Tensor，手动连接 Transformer、CNN、RNN 和维度操作；修改参数后，即时查看每个节点及连线的形状，定位不兼容的维度。

![Tensor Trail 深色工作台](docs/assets/ui-desktop.jpg)

## v0.1 功能

- 29 个可添加模块，覆盖线性层、卷积、RNN/GRU/LSTM、维度操作与分支合并。
- Transformer → Self-Attention / FFN → Q/K/V、拆头、矩阵乘法等多层子图展开。
- Pre-Norm / Post-Norm、LayerNorm / RMSNorm、Standard FFN / SwiGLU；可编辑内部算子并保存为独立变体。
- Tensor 轴含义、命名输入输出端口、实时约束检查与逐步高亮。
- 图编辑、撤销重做、JSON 导入导出、浏览器自动保存，以及宽屏和窄屏布局。
- 参考 TensorBloom 的深色工作台、可记忆的明暗主题、侧栏折叠、分类节点色标与当前层级概览；按 `/` 快速搜索模块。
- AI 构建：简化 JSON 配方、模块目录和 JSON Schema、构建/校验/批量修改 API、CLI，以及带版本冲突检查的画布应用入口。

内置混合网络展示：

```text
Input            [8, 100, 64]
  ↓ Transformer  [8, 100, 64]
  ↓ Permute      [8, 64, 100]
  ↓ Conv1d       [8, 128, 100]
  ↓ Permute      [8, 100, 128]
  ↓ LSTM         [8, 100, 32]
Output           [8, 100, 32]
```

## 运行

需要 Node.js 20 或更新版本与 npm。

```sh
git clone https://github.com/Edward-0333/tensor-trail.git
cd tensor-trail/app
npm ci
npm run dev -- --port 5178
```

访问终端显示的本地地址，通常为 `http://127.0.0.1:5178/`。Windows 也可以双击 `app/start.cmd`，首次运行会安装依赖并打开浏览器。

默认只监听本机。编辑结果保存在当前浏览器、当前地址的 localStorage；换电脑、浏览器或端口前请导出 JSON。

## 开发与验证

在 `app/` 中执行：

```sh
npm test
npm run build
npm run preview -- --port 5178
```

当前已通过 63 项自动测试、CLI 端到端检查、TypeScript 检查及生产构建，浏览器交互验证见[第一版验收](docs/第一版验收.md)。

```text
app/src/core/      图协议、TensorSpec、递归形状推导、序列化
app/src/modules/   算子注册、形状规则、Transformer 组合子图
app/src/agent/     Recipe/Patch、模块目录、Schema、构建与校验 API
app/scripts/       AI 命令行及端到端检查
app/examples/      AI 可读取的架构配方与修改示例
app/src/editor/    编辑会话、画布、参数面板、历史与保存
app/src/views/     节点与 Tensor 视图、模块专属可视化
app/tests/         形状契约与编辑事务测试
docs/             设计、使用指南和验证记录
```

核心引擎使用纯 TypeScript，与 React 画布分离。新增算子时注册模块定义与形状规则；新增 Transformer 结构时组合已有算子；新增视觉效果时注册详情视图。

- [详细架构方案](docs/架构方案.md)
- [使用与扩展指南](docs/使用与扩展.md)
- [应用运行说明](app/README.md)
- [AI 操作指南](docs/AI操作指南.md)

## 让 AI 搭建模型

AI 生成 Recipe JSON 后，可在页面顶部“AI 构建”中粘贴，检查形状并应用到画布；也可在 app/ 中运行：

```sh
npm run build:agent
node scripts/agent.mjs describe
node scripts/agent.mjs build examples/cnn-transformer.recipe.json --out model.json
node scripts/agent.mjs patch model.json examples/swiglu.patch.json --out next.json
```

生成的项目可用网页“导入”打开。页面 API 为 `window.tensorTrail`，提供读取、构建、校验、批量修改及携带 revision 的应用操作。CLI 只生成文件，不自动修改打开的网页。当前未接入内置大模型、MCP 或 HTTP REST 服务，接入步骤及可交给 AI 的工作指令见 [AI 操作指南](docs/AI操作指南.md)。

## 当前范围与后续方向

当前为**形状模式**，不运行真实张量计算、训练、梯度或注意力权重。符号维度仅做有限传播；自定义变体保存具体子图快照，暂不提供内部参数提升或任意节点封装界面。

后续计划包括 RoPE / GQA 等算子、更多 Transformer 变体、教学动画，以及用于数值前向和形状对照的 PyTorch 后端。具体支持范围见使用指南。

## 开源协议

Tensor Trail 的自有代码与文档采用 [MIT License](LICENSE)，版权署名为 `Copyright (c) 2026 Edward-0333`。允许使用、修改、分发和商业使用，包括闭源集成；复制或分发本软件的全部或重要部分时须保留版权与许可声明。软件按现状提供，不附带担保。

第三方依赖和参考项目遵循各自的许可证。

## 参考与贡献说明

交互与架构设计参考了 [TensorBloom](https://github.com/mdsunbeam/tensorbloom) 和 [NodeTorch](https://github.com/WilsonZheng0327/nodetorch)。本仓库包含独立实现的 Tensor Trail 代码，参考项目源码请到各自仓库获取。

项目需求与扩展方向由维护者提出，v0.1 的实现、文档与验证由 Codex（GPT-6）协助完成。算子契约与设计来源列于架构方案。
