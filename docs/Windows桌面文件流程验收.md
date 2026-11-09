# Windows 桌面文件流程验收（D3／D4）

日期：2026-11-09。D3-01～D3-04 和 D4-01～D4-06 已完成，共 10 项任务；项目累计完成 25／41 项实施任务，下一项为 D5-01。M2 还需要 D5 的实际交互与缩放验收，当前不标记 M2 或安装版完成。本轮未执行 git add、commit 或 push。

## 可运行产出

根 `start-desktop.cmd` 构建并打开独立 Windows 窗口；`start.cmd` 保留网页版。桌面普通运行使用 `apps/desktop/dist/` 下的 main、preload 和 renderer，本地资源由 `tensor-trail://app/` 提供，不依赖 Vite、浏览器标签页或 5178 服务。

桌面与网页复用同一共享编辑器和 TypeScript 模型核心。项目 JSON 仍为 schemaVersion=1，原算子、布局及形状规则没有改协议。窗口标题、原生菜单、主题设置和日志已接入；每次启动为空白，手动打开文件后才关联，不保存上次项目或文件路径。

## 文件与边界

- 主进程保存文件路径、内容指纹和文档归属，界面只获得随机 key 与文件名。其他文档的 key 不能直接用于当前项目写入。
- UTF-8、BOM、3 MB 限制和共享格式校验在主进程及编辑器入口执行。取消或无效文件不替换当前模型。
- 保存先写入同目录唯一临时文件，flush 后再检查源文件内容，最后替换；替换失败不会通过删除原件来回退。成功提交才更新文件关联和 saved 状态。
- 内容指纹用于检查外部程序改动；在准备写入和替换前各检查一次。它不是跨进程的文件锁。极短的并发替换窗口不作强互斥保证。
- 新建、打开、示例切换和关闭共用未保存检查。Ctrl+S 和菜单保存共用一个入口，先提交正在编辑的属性，再捕获保存快照；保存期间的新编辑仍标记未保存。
- 界面启用 sandbox、contextIsolation，关闭 nodeIntegration；preload 不暴露通用 IPC、Node 或进程接口。生产资源协议限制目录和类型，外部导航与新窗口均拒绝。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| `node scripts/npm.cjs test -- --reporter=dot` | 48 个文件、867 项通过；M1 的 853 项保留，新增 8 项主进程文件／校验和 6 项未保存确认／并发切换检查 |
| `node scripts/npm.cjs run build:web` | 根 TypeScript、无 DOM 核心检查和网页版构建通过 |
| `node scripts/npm.cjs run test:cli` | 原 CLI 检查通过；SDK 仍可独立在 Node 运行 |
| `node scripts/npm.cjs run test:desktop` | 完整桌面构建与真实 Electron 窗口验证通过 |
| 桌面主流程 | 25 项自动观察点通过，包括原生接口、快捷键、文件关联和关闭流程 |
| 相同 profile 重启 | 6 项通过：恢复主题、保持空白项目、无上次文件关联 |
| 开发入口 | 6 项通过：本地 Vite 界面、preload、空白启动、配置和退出；与生产本地资源模式分别检查 |
| 退出清理 | 各轮记录的主进程、渲染器及辅助进程 PID 在退出后均已结束；开发服务关闭 |

37 项窗口观察包含三个阶段的重复基础检查，不代表 37 个彼此独立的功能。窗口真实创建和渲染，文件实际读写临时项目；文件选择和确认对话框的响应由测试提供，未把它描述为人工操作 Windows 对话框。结果、首屏和加载后截图保存在本机根目录 `test-results/desktop-*`。

主流程覆盖：取消打开、无效 JSON、中文空格路径和 BOM、Ctrl+S 单次原路径写入、活动文本框内容、Ctrl+Shift+S 取消后保留关联、成功另存为、外部修改冲突与另存恢复、新建前保存／放弃／取消、取消关闭、关闭前保存、撤销恢复文件关联、AI 修改沿用源文件、菜单保存、文本撤销、限制文件 key、主题、单实例及退出。

新文件检查还覆盖只读、替换失败后保留原件与重试、未签发 key、非法消息和超限文件。保存或确认期间继续编辑、打开期间异步修改等状态由针对性测试覆盖。

## 本机运行处理

Electron 已锁定并安装为 45.2.3，下载器使用项目内 `electron_config_cache`，没有改用非官方镜像。Windows 运行时曾因工具沙箱的目录权限拒绝启动，仅为 `node_modules/electron/dist/` 的公开运行时授予沙箱读取／执行权限；真实窗口验证在普通用户进程环境运行，应用自身的 sandbox 和隔离保持开启。

恢复点为 `.publish/backups/windows-after-M1-20261109/`，包括本轮开始时 216 个文件的副本及哈希。此前的目录迁移与 Git 历史备份仍保留。用户的 `data/` 和 `.conda/` 未用于窗口测试或改动。

## 后续范围

D5 继续验证 Windows 100%／125%／150% 显示缩放、拖入模块、连线与标签拖动、层级进入返回，以及大图的实际延迟。当前首屏和模型渲染已检查，不据此提前勾选这些任务。

安装包、安装／卸载验证和另一台电脑还原环境在 D6／R1；Python 在 P1。当前开发版仍需要 Node 工具链进行构建，不能视为无需 Node 的最终安装版。

实现依据：[Electron IPC](https://www.electronjs.org/docs/latest/tutorial/ipc)、[Electron 安全建议](https://www.electronjs.org/docs/latest/tutorial/security)、[MenuItem 的快捷键注册选项](https://www.electronjs.org/docs/latest/api/menu-item)。
