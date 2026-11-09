# 桌面图标

沿用编辑器左上角的紫色网络标识，带深色圆角底，便于在明暗任务栏上辨认。

- `tensor-trail.svg`：可编辑源图。
- `tensor-trail.ico`：Windows 窗口图标，包含 16／24／32／48／64／128／256 像素。
- `tensor-trail.png`：256 像素预览及 Linux 窗口／应用菜单图标。
- `lucide-LICENSE`：Network 图形沿用项目已有的 Lucide 图标，保留其 ISC／MIT 许可。

已由 `src/main/index.ts` 接入窗口，并为 Windows 设置稳定的 AppUserModelID `io.github.edward0333.tensortrail`。Linux 的 desktop ID 为 `io.github.edward0333.tensortrail.desktop`，与用户应用菜单入口保持一致。资源路径相对桌面应用目录，生产资源和开发窗口均可读取。

在项目根目录重新导出：

```cmd
node_modules\electron\dist\electron.exe scripts\export-desktop-icons.cjs
```

此命令只使用隐藏渲染窗口，不引入新的图像依赖。普通构建直接使用已导出的图标。

D6 制作安装包时，需把本目录纳入分发，并将同一 ICO 配置为 Windows EXE 和安装程序图标；AppUserModelID 与打包 appId 保持一致。目前没有修改依赖内的 `electron.exe`。

2026-11-09 桌面构建通过；在隔离的真实 Windows 窗口上使用 WM_GETICON 读取大／小图标成功，并导出检查了实际图形。原生检查输出在本机被忽略的 `test-results/desktop-icon-check/`。

配置依据：[Electron 窗口 icon 选项](https://www.electronjs.org/docs/latest/api/structures/base-window-options)。
