# 贡献指南

## 安装与代码位置

使用仓库声明的 Node.js 和 npm 版本，分别见 [.node-version](.node-version) 和 [toolchain.json](toolchain.json)。在仓库根目录执行 `npm ci`，各工作区共用根目录的锁文件。

- 图协议、算子、形状推导和 Agent API 在 [packages/model-core](packages/model-core)。
- 画布、编辑状态和属性面板在 [packages/editor-ui](packages/editor-ui)。
- 平台服务接口在 [packages/platform-api](packages/platform-api)，网页和桌面端的实现分别在 [apps/web](apps/web) 和 [apps/desktop](apps/desktop)。
- 命令行入口和端到端检查在 [packages/cli](packages/cli)。

## 提交前验证

在仓库根目录执行：

```sh
npm test
npm run test:cli
npm run build
```

GitHub Actions 在 PR 和推送到 `main` 时运行这三项。涉及界面交互时，用 `npm run dev:web -- --port 5178` 启动网页并验证受影响的操作。涉及桌面窗口时，按需在图形桌面运行 `npm run test:desktop`，详见 [桌面运行说明](apps/desktop/README.md)。

## 复现与 PR

报告问题时写明操作系统、Node.js/npm 版本和使用的入口（网页、桌面或 CLI），按顺序列出复现步骤、预期结果和实际结果。需要项目文件时，优先提供最小可复现的 JSON 或可直接载入的内置示例名称。

PR 请说明改动原因、影响范围和实际执行的验证。界面改动可附截图或录屏，未验证的场景请注明。
