# 共享层拆分验收（D2／M1）

日期：2026-11-09。D2-01～D2-07 已完成。M1 交付单一 Git 工作区、可运行的网页版与独立 CLI／SDK；Electron 桌面入口从 D3 开始，本阶段未安装 Electron。未执行 git add、commit 或 push。

## 实际结构

| 位置 | 职责 |
| --- | --- |
| `packages/model-core/src/core/` | 图协议、序列化、超参数、形状推导 |
| `packages/model-core/src/modules/` | 算子与组合子图 |
| `packages/model-core/src/agent/` | Recipe／Patch、目录、Schema 与模型 API |
| `packages/model-core/src/editing/` | 从 editor 抽离的纯图复制、删除及关联值操作 |
| `packages/editor-ui/src/` | 共享 React 编辑器、画布、视图、状态与样式 |
| `packages/platform-api/src/` | 无 DOM／Electron 对象的平台接口 |
| `packages/cli/scripts/` | 独立 Node 命令行与 CLI 验证 |
| `apps/web/src/` | React 挂载与浏览器文件／主题适配 |
| `tests/modelEntries.test.ts` | 源码规则与独立 Node SDK 的跨入口兼容验证 |

原 46 个测试文件按实际依赖迁到 model-core（17 个）和 editor-ui（29 个），原 851 项测试均保留。映射在本机恢复点 `.publish/backups/windows-after-D1-20261109/mapping.json`，D1 阶段文件副本与哈希清单也在该恢复点，便于单独比较目录迁移和共享层拆分。

## 已解开的依赖

`agent/model.ts` 现在导入模型包内的 `editing/valueLinkEditing.ts`，不再反向引用编辑器。模型包以 ES2022 与 Node 类型独立检查；源码扫描未发现 React、Electron、编辑器、window、document 或本机存储依赖。

共享 `ProjectFiles` 负责文档关联、保存快照与 dirty 状态。浏览器句柄、权限、createWritable、提交／中止流均在 `apps/web/src/platform/browser.ts`。原文件测试继续检查权限拒绝、写入／关闭失败、保存取消、保存中继续编辑、切换与撤销恢复，不删减原行为要求。主题偏好也通过平台接口注入。

## 验证

| 命令／检查 | 结果 |
| --- | --- |
| `node scripts/npm.cjs test -- --reporter=dot` | 47 文件、853 项通过；含 2 项新增跨入口验证 |
| `node scripts/npm.cjs run build` | 根 TypeScript、无 DOM 的模型核心检查、网页及 SDK 构建通过 |
| `node scripts/npm.cjs run test:cli` | CLI 全部原检查通过，SDK 在独立 Node 进程运行 |
| 开发服务 HTTP | 根页、薄入口、浏览器适配器、共享 App 与模型模块均为 200，无导入错误 |
| 样式顺序 | 调整共享入口导入顺序后，生产 CSS 与原版逐字节一致 |

新增验证对比源码 API 与独立 CLI 所用 SDK 的推导结果、诊断和序列化项目，覆盖旧 schemaVersion=1、超参数、关联值、手动路线、标签位置与 PLUTO 字典输入。PLUTO 的输入字段示例本就没有最终 Output，两种入口均保留相同的 MISSING_OUTPUT 诊断，没有为让检查通过改变模型规则。

本阶段新增 Node 类型依赖以执行纯 Node 类型检查。运行库版本沿用原锁定版本；React 由共享编辑器声明 peer dependency，网页入口提供同一实例。`package-lock.json` 继续是唯一安装锁文件。

## 当前运行入口与后续

仍可双击根 `start.cmd` 或旧 `app/start.cmd`。命令从根目录执行：

```sh
npm ci
npm run dev:web -- --port 5178
npm test
npm run build
npm run test:cli
node packages/cli/scripts/agent.mjs describe
```

本机可用 `node scripts/npm.cjs` 替代 npm，继续使用项目专用 Node。网页保持空白启动、手动打开 JSON、Ctrl+S 保存回当前文件和另存为约定。画布交互与 Windows 原生文件体验将在 D3～D5 另行运行验收；本阶段不把单元测试等同于桌面交付。
