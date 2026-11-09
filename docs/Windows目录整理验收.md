# Windows 目录整理验收（D0／D1）

日期：2026-11-09。对应[实施台账](Windows桌面版实施台账.md)的 8 个任务。本文记录 D1 完成时的目录与验证结果；后续共享层进展见[实施台账](Windows桌面版实施台账.md)。未执行 git add、commit 或 push。

## 当前入口

```text
根目录                     唯一日常 Git 工作区、npm workspace、package-lock.json
apps/web/                  原 app 的网页源码、测试、Vite 与 CLI 临时入口
examples/                  原 app/examples 的 5 个 Recipe／Patch
apps/desktop/              说明文件，等待 D3 创建 Electron 入口
packages/                  说明文件，等待 D2 抽取共享包
python/                    既有 Python 工具，保留原位
start.cmd                  网页启动入口；旧 app/start.cmd 兼容转发
docs/local/                本机首页、旧发布记录和差异清单，忽略提交
.publish/                  迁移前 Git 副本与备份，忽略提交
.tools/                    本机项目专用 Node，忽略提交
```

原 `app/` 只保留兼容入口和未搬动的旧依赖／产物。新的构建、测试和开发服务不再使用其中的依赖。依赖由根锁文件重新安装；未复制 `node_modules`。旧产物暂留作比较，不作为有效构建结果。

## 基线与保护

- 迁移前 fetch 后，旧副本 HEAD 与 origin/main 均为 `8245aa2dd2cd81d52591ed4c667e4e8dbc163aba`，工作区干净。根仓库承接同一历史、main 分支和 origin。
- 当前根工作区中既有目录迁移，也有此前未提交的 Python 环境说明、依赖清单及本台账。原始文本与差异清单保存在本机备份；部分旧文档差异来自 Obsidian 链接的发布转换。
- `.publish/backups/windows-baseline-20261109/` 包含 186 个文件的逐字节副本、SHA-256 清单和 `history.bundle`（751874 字节）。文件备份总计 2589068 字节，包含 3 个用户文件。
- `data/`、`references/`、`.conda/`、`.tools/`、`.publish/`、`docs/local/` 与 `release/` 均有忽略规则；没有把用户资料或环境加入暂存。
- 3 个用户文件的 SHA-256 与迁移前一致。Python 环境仍位于 `.conda/tensor-trail-cpu/`，未移动或重建。
- 原网页有 89 个源码文件。除 AgentPanel 的示例导入路径调整外，其余 88 个源码文件与备份完全一致。46 个测试文件完整保留，只调整了两处测试文件中的示例导入路径。

## 工具链

登记在根 `.node-version` 与 `toolchain.json`：Node 24.22.2 LTS、npm 11.19.1、Electron 45.2.3、electron-builder 26.18.0。Electron 的 npm 元数据要求 Node ≥22.12；本阶段仅准备 Node，Electron 依赖留到 D3 安装。

本机 `.tools/` 中的 Windows x64 Node 压缩包按[官方发行清单](https://nodejs.org/dist/index.json)和对应 SHASUMS256 校验；未更改全局 Node 20.15.1、全局 PATH 或其他 Conda 环境。`scripts/npm.cjs` 仅为子进程设置项目 Node 路径。

其他电脑安装 Node 24.22.2 LTS 后，可直接在根目录执行 `npm ci`。`.tools/` 不随 Git 分发。本机使用 `node scripts/npm.cjs ci` 和同一根锁文件安装成功。

## 验证结果

以下命令从项目根目录执行。`node scripts/npm.cjs` 是本机对项目 Node/npm 的调用入口，已安装所需 Node 的其他机器可直接用 npm。

| 检查 | 结果 |
| --- | --- |
| `node scripts/npm.cjs test -- --reporter=dot` | 46 文件，851 项通过 |
| `node scripts/npm.cjs run build` | TypeScript、网页和 Agent SDK 构建通过 |
| `node scripts/npm.cjs run test:cli` | describe、schema、build、结构化输入、validate、patch、stdin、错误退出与输出保护通过 |
| `.conda\tensor-trail-cpu\python.exe -m unittest discover -s python -p test_tensor_trail_runtime.py` | 7 项通过 |
| `.conda\tensor-trail-cpu\python.exe -m pip check` | No broken requirements found |
| 网页生产产物比较 | JS 与 CSS 均与迁移前产物逐字节一致 |
| 开发服务 | 原 5178 旧 Vite 进程经身份核对后，已切换为 `apps/web/` 的新服务 |
| HTTP 检查 | `/`、主入口、AgentPanel 与根示例 JSON 均为 200，无 Vite 导入错误 |

Windows 默认 GBK 下，基线 Python 测试最初有两项在读取中文 JSON 时失败。已为测试中的文本读写明确 `encoding='utf-8'`，生产 Python 工具未改行为；修正后 7 项通过，无需依赖系统编码开关。

本阶段没有重新进行画布人工交互验收；网页产物一致、原测试通过和 HTTP 检查用于验证目录迁移。Windows 界面、原生文件操作及安装包仍由后续阶段单独验收。已有 Vite 大包提示仍在，本阶段不调整分包和业务实现。

## Git 与恢复

日常 Git 命令在根目录执行。`scripts/prepare-publication.cjs` 已改为提示退出，不能再把旧发布副本当作源码同步目标。旧副本及历史备份均保留。

恢复前先保存当前修改并停止本项目开发服务。根据备份中的 `manifest.json` 找回需要的文件；历史可用 `git bundle verify` 检查后，在独立恢复目录中从 `history.bundle` 克隆。源码副本在 `files/` 下，仍采用迁移前目录结构。不要直接把整个旧 Git 副本覆盖当前根工作区，也不要覆盖迁移后新增的用户编辑或 Conda 环境。

完整的本地文件清单及用户文件哈希位于 `docs/local/windows-migration-inventory.json`，不作为公开文档分发。
