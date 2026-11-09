# 共享包目录

已按[实施台账 D2](../docs/Windows桌面版实施台账.md)拆分，网页和后续桌面入口使用同一份实现：

- `model-core`：图模型、算子、形状推导、纯图编辑算法与 Agent API。
- `platform-api`：平台服务接口与消息类型。
- `editor-ui`：共享 React 编辑器和界面状态。
- `cli`：Node 命令行入口。

`model-core` 不依赖 React、DOM 或 Electron。编辑器通过 `platform-api` 注入文件和主题服务；浏览器实现位于 `apps/web/src/platform/browser.ts`。各包的安装、测试与构建统一在根目录执行。
