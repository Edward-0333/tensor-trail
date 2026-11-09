---
title: "Tensor Trail 应用"
created: 2026-10-18
authors:
  ai: "Codex"
  ai_model: "GPT-6"
source_files: ["projects/neural-network-visualizer/docs/架构方案.md"]
review_status: "unreviewed"
---

# Tensor Trail 0.1

本地运行的神经网络形状学习工作台。React + TypeScript + React Flow + Zustand；核心引擎不依赖 UI。

Windows 可双击本目录的 `start.cmd`，转到根目录启动脚本并打开浏览器。工具链使用 Node.js 24.22.2 LTS、npm 11.19.1；本机也可由 `scripts/npm.cjs` 使用项目专用 Node，详见[根说明](../../README.md)。

在项目根目录运行（依赖仅在根安装）：

```sh
npm ci
npm run dev:web -- --port 5178
```

默认只监听 127.0.0.1。端口占用时开发服务器会选择下一个端口，以终端提示为准。开发服务停止后网页不能继续访问；每次打开网页均为空白工作区，请选择 JSON 项目文件。关闭或刷新前使用 Ctrl+S 保存编辑。

```sh
npm test
npm run build
npm run preview -- --port 5178
npm run typecheck
```

`build` 执行 TypeScript 检查，并生成网页与独立 Agent SDK；`preview` 查看网页产物。不要直接双击 dist/index.html，浏览器模块需要本地服务。

AI 操作见 [AI 操作指南](../../docs/AI操作指南.md)。页面顶部“AI 构建”支持粘贴 JSON，校验后应用；命令行可以生成可导入项目：

```sh
npm run build:agent
node packages/cli/scripts/agent.mjs build examples/cnn-transformer.recipe.json --out model.json
npm run test:cli
```

从左侧模块库拖动名称到画布添加模块；单击不会添加。选中模块后，可在右侧属性页分别设置输入、输出端口的方向。双击连线添加控制点，拖动控制点调整路径；双击控制点删除，或恢复自动布线。所有布局操作支持撤销、保存与恢复。

“输入输出”中的“Forward输入”以简单矩形和居中文字标记输入，无需形状。支持双击修改文字、背景色、端口方向、复制粘贴和保存，兼容旧字典与列表输入。当前不自动拆出内部 Tensor，张量算子仍需用 Tensor 输入单独描述。AI 示例见 `examples/structured-input.recipe.json`。

每次启动或刷新后，点击中央“打开 JSON 文件”或顶部“打开”选择项目，成功后关联该文件。Ctrl+S /“保存”写回本次打开的文件；Ctrl+Shift+S /“另存为”选择新位置。项目和文件关联仅在本次会话有效，关闭或刷新前请保存。新建项目首次保存选择一次位置。源文件写入建议使用桌面 Chrome 或 Edge，第一次写入可能需要授权。

目录说明（相对于项目根目录）：

- `apps/web/src/`：React 挂载入口与浏览器平台适配。
- `packages/model-core/src/`：图协议、形状规则、序列化、算子和 Agent API。
- `packages/editor-ui/src/`：共享 React 编辑器、画布、节点视图与文档编辑会话。
- `packages/platform-api/src/`：平台服务类型。
- `packages/cli/scripts/`：独立 Node CLI。
- `packages/*/tests/`：原有单元测试；`tests/`：跨包和入口集成验证。
- `examples/`：Recipe 与 Patch 示例。

完整说明见 [使用与扩展](../../docs/使用与扩展.md)，设计见 [架构方案](../../docs/架构方案.md)，验证记录见 [第一版验收](../../docs/第一版验收.md)。
