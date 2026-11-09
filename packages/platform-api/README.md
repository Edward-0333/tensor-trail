# 平台服务契约

`EditorPlatform` 是共享编辑器与网页／桌面环境之间的边界，类型定义见 `src/index.ts`。

| 接口 | 约定 |
| --- | --- |
| `files.pickOpen` | 可选；由用户选择文件，取消可返回 undefined 或抛 AbortError |
| `files.pickSave(name)` | 可选；选择保存目标，选择本身不能覆盖文件 |
| `ProjectFile.readText()` | 读取 UTF-8 项目文本；适配层在读取前执行 3 MB 限制 |
| `ProjectFile.writeText(text, stillCurrent, version)` | 写入前检查文档仍有效；桌面传递文档 ID／快照版本；完成提交才返回 true，取消返回 false，失败抛错 |
| `files.download` | 网页缺少源文件写入能力时，仅在用户明确另存为时导出副本 |
| `preferences.getTheme/setTheme` | 仅处理主题，不恢复项目或文件关联 |
| `lifecycle` | 桌面注入文档状态、菜单动作、未保存确认与关闭应答；网页可省略 |
| `runtime` | 可选，当前两端未接入 Python；后续 P1 扩展 |

不要在这些类型中传递 FileSystemHandle、Electron 对象或本机绝对路径。网页文件句柄、权限、流提交／中止均留在浏览器适配器。桌面使用主进程会话、随机文件 key 与限定 IPC；key、documentId 和 revision 不进入项目 JSON。

入口先调用共享编辑器的 `configureEditor(platform)`，再挂载 `App`；返回值用于注销编辑订阅，避免开发热更新重复注册。

菜单与键盘均进入共享保存动作，先提交正在编辑的字段，再读取保存快照。桌面的新建、打开、示例和关闭检查未保存修改；保存取消或失败时不继续切换。浏览器省略 lifecycle 后保留原有文件与导航行为。
