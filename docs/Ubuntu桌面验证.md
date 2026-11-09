# Ubuntu 桌面端验证

日期：2026-11-09。基于 Windows 桌面开发版，复用 `apps/desktop` 主进程、preload 和共享编辑器，补齐 Linux 入口。

## 环境与交付

- Ubuntu 20.04.6 LTS，x86_64，X11 图形桌面。
- Node.js 22.23.4、npm 10.9.9、Electron 45.2.3。项目最低 Node 要求为 22.12，登记的开发工具链版本保持不变。
- `start-desktop.sh` 从任意工作目录启动；支持显式 Node、项目本地 Node、Conda 与 PATH。首次安装锁定依赖与 Linux Electron，后续检测依赖指纹。
- `--install-launcher` / `--remove-launcher` 管理当前用户的应用菜单入口，记录 Node 的绝对路径，菜单启动不要求激活 Conda。
- Linux 使用 PNG 窗口图标及稳定的 desktop ID；Windows 继续使用 ICO 和原启动入口。
- 原生项目文件会话、保存冲突提示、主题、菜单、快捷键、独立本地窗口与 Windows 共用实现。图协议和 PLUTO 项目内容不变。
- 桌面 renderer 保持 sandbox、contextIsolation，关闭 nodeIntegration；未加入 `--no-sandbox`。

## 已完成验证

| 检查 | 结果 |
| --- | --- |
| TypeScript + 桌面生产构建 | 通过 |
| 自动测试 | 53 个文件、910 项通过 |
| CLI 端到端 | 目录、Schema、构建、校验、补丁、标准输入和输出保护通过 |
| 真实 Electron 文件流程 | 完整流程、重启、开发服务三阶段通过，共 37 项观察 |
| 真实 Electron 画布交互 | 100%、125%、150% 缩放，各 23 项观察，共 69 项通过 |
| PLUTO 大图 | 实际项目副本读取、子图进入、拖动、保存重开通过，原文件未改变 |
| 应用菜单文件 | `desktop-file-validate` 通过 |

文件流程覆盖中文／空格路径、BOM、取消打开、无效项目、Ctrl+S、输入框未失焦时保存、另存为、源文件冲突恢复、关闭确认、原生菜单、主题与单实例。自动测试运行真实 Electron 窗口，但文件选择器和确认框使用受控响应；这不等同于逐一人工点击 GTK 文件选择器。

画布测试覆盖最小窗口、分层进入、复制粘贴、撤销重做、模块拖入、端口连接、端口方向、共享线同步拖动、Alt、Esc、连线文字、控制点、NAT 子图、保存重开、示例及模块／线路吸附。拖入模块后等待实际端口几何稳定，消除平台渲染时序差异导致的坐标失效。

PLUTO 图冷启动读取约 0.54–0.73 秒，后续读取约 0.37–0.40 秒；本次采样拖动帧间隔中位数约 16.7 ms。以上为当前机器实测，非性能承诺。

同时修复了上游目录迁移遗留的 CLI 引用：从已废弃的 `apps/web/dist-agent` 改为实际生成的 `packages/model-core/dist`。未复制构建产物来掩盖旧路径。

摘要：[ubuntu-desktop-20261109.json](validation/ubuntu-desktop-20261109.json)。截图及完整日志在本机忽略提交的 `test-results/desktop-*` 和 `test-results/experience-*`。

## 范围

当前交付与 Windows 版一样，是从源码构建运行的桌面开发版。未制作 deb / AppImage；未在 Wayland、ARM64 或其他 Ubuntu 版本实测。共享 Windows 分支保留，当前 Linux 机器没有重做 Windows 原生 GUI 验证。

使用方法见[桌面运行说明](../apps/desktop/README.md)。
