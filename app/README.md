---
title: "Tensor Trail 应用"
created: 2026-10-18
authors:
  ai: "Codex"
  ai_model: "GPT-6"
source_files: ["docs/架构方案.md"]
review_status: "unreviewed"
---

# Tensor Trail 0.1

本地运行的神经网络形状学习工作台。React + TypeScript + React Flow + Zustand；核心引擎不依赖 UI。

Windows 可双击本目录的 `start.cmd`，自动启动并打开浏览器。若已有预览服务，可直接访问它。首次安装需要网络和 Node.js 20 或更新版本。

在本目录运行：

```sh
npm ci
npm run dev -- --port 5178
```

默认只监听 127.0.0.1。端口占用时开发服务器会选择下一个端口，以终端提示为准。开发服务停止后网页不能继续访问；项目数据仍保存在原浏览器对应地址的本地存储中。不同浏览器、地址或端口的本地存储不共享，换环境前导出 JSON。

```sh
npm test
npm run build
npm run preview -- --port 5178
npm run format
```

`build` 执行 TypeScript 检查，并生成网页与独立 Agent SDK；`preview` 查看网页产物。不要直接双击 dist/index.html，浏览器模块需要本地服务。

AI 操作见 [AI 操作指南](../docs/AI操作指南.md)。页面顶部“AI 构建”支持粘贴 JSON，校验后应用；命令行可以生成可导入项目：

```sh
npm run build:agent
node scripts/agent.mjs build examples/cnn-transformer.recipe.json --out model.json
npm run test:cli
```

目录说明：

- `src/core/`：图协议、递归推导、端口校验、保存格式。
- `src/modules/`：基本算子、Transformer/Attention/FFN 子图、示例。
- `src/agent/`：Recipe/Patch、模块目录和 Schema、构建与校验 API。
- `scripts/agent.mjs`：命令行；`examples/`：可运行的配方和修改示例。
- `src/editor/`：编辑会话、撤销重做、画布和属性面板。
- `src/views/`：通用 Tensor 视图、节点卡片、可注册的模块详情视图。
- `tests/`：数学形状契约、图行为与持久化事务测试。

完整说明见 [使用与扩展](../docs/使用与扩展.md)，设计见 [架构方案](../docs/架构方案.md)，验证记录见 [第一版验收](../docs/第一版验收.md)。
