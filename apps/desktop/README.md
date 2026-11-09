# Windows / Ubuntu 桌面开发版

Windows 与 Ubuntu 共用同一套 Electron 主进程、编辑器、模型核心及项目 JSON。当前为从源码运行的桌面开发版；Ubuntu 入口不依赖浏览器或网页服务，尚未制作 deb / AppImage 安装包。Windows M2（D3～D5）已完成，安装包在 D6 制作。

## 启动

Windows 双击项目根目录的 [start-desktop.cmd](../../start-desktop.cmd)。脚本检查依赖和 Electron、构建当前源码，再启动独立窗口，命令窗口随后退出。网页版仍使用根 `start.cmd`。

开发环境使用 Node 24.22.2 LTS；本机启动脚本可使用 `.tools/` 中的项目专用 Node，未改变全局 Node 或 Conda。首次安装需要下载依赖与 Electron Windows 运行时，缓存位于 `.npm-cache/electron/`。

在项目根目录也可运行：

```sh
npm run desktop
npm run dev:desktop
npm run build:desktop
npm run test:desktop
npm run test:desktop:experience
```

本机可将 npm 替换为 `node scripts/npm.cjs`。`desktop` 使用本地构建资源；`dev:desktop` 使用独立的 5180 开发服务并支持界面热更新，main／preload 修改后需要重新启动命令。普通运行和后续安装版不依赖网页版的 5178 服务。

## Ubuntu 启动

在 Ubuntu 图形桌面的终端中，从项目根目录运行：

```sh
./start-desktop.sh
```

要求 Node.js >=22.12（项目登记版本见 `toolchain.json`）。脚本依次选择 `TENSOR_TRAIL_NODE`、`.tools/node-v<登记版本>-linux-<架构>/bin/node`、当前 Conda 环境中的 Node、PATH 中的 Node。使用 Conda 时先激活已安装 Node 的环境；也可明确指定：

```sh
TENSOR_TRAIL_NODE=/path/to/node ./start-desktop.sh
```

第一次启动按根 `package-lock.json` 安装依赖并下载 Linux Electron，随后构建并打开窗口。依赖指纹记录在 `node_modules/.tensor-trail-linux-lock`，锁文件或平台改变后重新安装。运行时下载缓存在 `.npm-cache/electron/`；使用已配置的 HTTP(S) 代理，保留上游校验。终端命令持续运行至窗口关闭，Ctrl+C 会停止桌面进程。代码修改后重启即可重新构建；开发热更新使用 `./start-desktop.sh --dev`。

安装到**当前用户**的 Ubuntu 应用菜单（无需 sudo）：

```sh
./start-desktop.sh --install-launcher
# 之后可在应用列表中搜索 Tensor Trail
./start-desktop.sh --remove-launcher
```

菜单入口位于 `$XDG_DATA_HOME/applications/io.github.edward0333.tensortrail.desktop`，默认 `~/.local/share/applications/`。记录安装时 Node 的真实路径，点击图标时无需激活 Conda。移动仓库、删除或移动 Node 后需要重新安装入口。菜单启动日志在 `$XDG_STATE_HOME/tensor-trail/launcher.log`，默认 `~/.local/state/tensor-trail/launcher.log`；失败时若已安装 Zenity 会显示错误提示。首次启动可能需要等待依赖下载与构建。

普通 Ubuntu 桌面通常已具备 GTK 3、NSS、GBM 和 ALSA 库。脚本会列出 `ldd` 检测到的缺失库。Ubuntu 20.04 / 22.04 缺失时可安装：

```sh
sudo apt-get install libgtk-3-0 libnss3 libgbm1 libasound2 libxss1 libatk-bridge2.0-0
```

Ubuntu 24.04 的部分包使用 t64 名称，例如 `libgtk-3-0t64`、`libasound2t64`、`libatk-bridge2.0-0t64`。启动脚本不执行系统安装，不更改全局 Node，也不关闭 Electron sandbox；请使用普通用户启动。SSH 无图形会话时不会自动设置 DISPLAY。

Ubuntu 24.04 及后续系统若提示 user namespace / sandbox 权限错误，应按照系统的 AppArmor 策略为可信 Electron 可执行文件配置对应规则；不应把 `--no-sandbox` 设为默认启动参数。当前 Ubuntu 20.04 X11 的实机结果见[Ubuntu 桌面验证](../../docs/Ubuntu桌面验证.md)。Wayland、ARM64 与其他 Ubuntu 版本需分别验证。

应用菜单按照 [freedesktop Desktop Entry](https://specifications.freedesktop.org/desktop-entry-spec/latest/exec-variables.html) 生成；运行时下载遵循 [Electron 安装机制](https://www.electronjs.org/docs/latest/tutorial/installation)。

## 项目文件

- 每次启动为空白界面，需要手动选择 JSON，不恢复上次项目或浏览器草稿。
- Ctrl+O 打开，Ctrl+N 新建，Ctrl+S 保存回当前文件，Ctrl+Shift+S 另存为。
- 首次保存选择位置；成功另存为后才切换文件关联。取消、写入失败和只读错误保留原关联及未保存状态。
- 新建、打开、探索示例和关闭时，未保存修改可选择保存、不保存或取消。保存失败时留在当前项目。
- 若其他程序修改或删除了源文件，普通保存会提示冲突；可以重新打开，或另存为保留画布编辑。
- AI 修改沿用当前文件，撤销恢复旧项目时也恢复其文件关联。文件路径和会话 ID 不写进模型 JSON。

标题栏显示当前文件及未保存标记。主题保存在 Windows 用户配置目录 `%APPDATA%\Tensor Trail\settings.json`，Ubuntu 为 `$XDG_CONFIG_HOME/Tensor Trail/settings.json`（默认 `~/.config/Tensor Trail/settings.json`）。诊断写入配置同目录 `app.log`；配置不包含项目内容或上次文件路径。

## 开发入口

| 位置 | 职责 |
| --- | --- |
| `src/main/index.ts` | 窗口、菜单、协议资源、IPC 校验及退出 |
| `src/main/files.ts` | 主进程文件会话、UTF-8／大小限制、冲突检查和原子替换 |
| `src/main/settings.ts` | 用户主题设置 |
| `src/preload/index.ts` | 只暴露限定的桌面方法，不暴露 ipcRenderer／文件系统 |
| `src/renderer/` | 装配共享编辑器与桌面适配 |
| `src/shared/protocol.ts` | 桌面接口、消息类型与结果包络 |
| `dist/` | 本地构建产物，忽略 Git 提交 |
| `resources/` | 应用 SVG、PNG、多尺寸 Windows ICO 与图标许可 |

渲染器启用 sandbox、contextIsolation，并关闭 nodeIntegration。生产界面使用 `tensor-trail://app/` 读取限定的本地资源；主进程仅接受本窗口主框架的消息，文件操作使用选择器产生的随机 key。

窗口和运行中的任务栏使用紫色网络图标，重启软件后生效；图标源文件与导出方法见 [图标资源](resources/README.md)。Ubuntu 使用 PNG 窗口图标及对应的应用菜单入口；Windows EXE 文件和安装程序的图标在 D6 打包时使用同一资源。

`test:desktop` 会启动真实 Electron 窗口，并使用独立 profile、生成的 JSON 和受控对话框响应验证流程。结果与截图在根 `test-results/desktop-*`，不修改用户项目。详细结果及未验证范围见[桌面文件流程验收](../../docs/Windows桌面文件流程验收.md)。

`test:desktop:experience` 在 100%／125%／150% 进程 DPI 下执行真实窗口画布动作，验证 900×640 最小窗口、大图适配、拖放／连线／共享线／文字和层级操作，以及模块按连线吸附、Alt、保存重开，以及手动线端点跟随，共 69 项观察。默认读取本机 `data/pluto.tensor-trail.json`；可追加 `-- --project <已有项目.json>` 选择含子图的项目，只编辑测试副本。报告和截图位于 `test-results/experience-*`。这不会更改 Windows 系统显示设置，完整范围及性能数据见[桌面体验验收](../../docs/Windows桌面体验验收.md)。

下一阶段为 D6-01：配置 Windows x64 打包目录，再制作安装包。安装／卸载、签名与自动更新尚未交付，Python 集成仍在 P1。
