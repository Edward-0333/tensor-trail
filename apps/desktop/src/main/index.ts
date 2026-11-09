import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  nativeTheme,
  net,
  protocol,
  screen,
  session,
} from 'electron';
import type { IpcMainInvokeEvent, MenuItemConstructorOptions } from 'electron';
import path from 'node:path';
import * as fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import type { EditorAction } from '@tensor-trail/platform-api';
import { CHANNEL, type Reply } from '../shared/protocol';
import { DocumentFiles, FileFault } from './files';
import { Settings } from './settings';
import * as check from './validation';

app.setName('Tensor Trail');
if (process.platform === 'win32') app.setAppUserModelId('io.github.edward0333.tensortrail');
if (process.platform === 'linux') app.setDesktopName('io.github.edward0333.tensortrail.desktop');
if (!app.isPackaged && process.env.TENSOR_TRAIL_PROFILE_DIR)
  app.setPath('userData', path.resolve(process.env.TENSOR_TRAIL_PROFILE_DIR));
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'tensor-trail',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
    },
  },
]);
const root = path.resolve(__dirname, '../renderer');
const files = new DocumentFiles();
let window: BrowserWindow | undefined;
let settings: Settings;
let allowClose = false;
let closeRequest: string | undefined;
let rendererAlive = true;
let menuState = '';
const development = !app.isPackaged && process.env.TENSOR_TRAIL_DEV_URL;
const page = development || 'tensor-trail://app/index.html';
if (development) {
  const url = new URL(development);
  if (
    url.protocol !== 'http:' ||
    url.hostname !== '127.0.0.1' ||
    !url.port ||
    url.username ||
    url.password
  )
    throw new Error('The desktop development URL must use the local IPv4 loopback server.');
}
const productionCsp =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'";
function log(error: unknown) {
  const entry = `${new Date().toISOString()} ${(error as Error)?.stack || String(error)}\n`;
  void fs.appendFile(path.join(app.getPath('userData'), 'app.log'), entry).catch(() => {});
}
function trusted(event: IpcMainInvokeEvent) {
  if (
    !window ||
    event.sender !== window.webContents ||
    event.senderFrame !== window.webContents.mainFrame
  )
    throw new FileFault('INVALID_SENDER', '桌面请求来源无效。');
  const actual = new URL(event.senderFrame.url);
  const expected = new URL(page);
  if (
    actual.protocol !== expected.protocol ||
    actual.host !== expected.host ||
    actual.pathname !== expected.pathname
  )
    throw new FileFault('INVALID_SENDER', '桌面页面来源无效。');
}
function handle(name: string, count: number, action: (...args: unknown[]) => unknown) {
  ipcMain.handle(name, async (event, ...args): Promise<Reply<unknown>> => {
    try {
      trusted(event);
      if (args.length !== count) check.invalid();
      return { ok: true, value: await action(...args) };
    } catch (error) {
      log(error);
      const code = (error as NodeJS.ErrnoException).code;
      const message =
        error instanceof FileFault
          ? error.message
          : ['EPERM', 'EACCES', 'EBUSY'].includes(code || '')
            ? '文件被占用、只读或没有写入权限，请重试或另存为。'
            : code === 'ENOENT'
              ? '文件或文件夹已不存在，请重新选择。'
              : '桌面操作失败，请重试；诊断已写入应用日志。';
      return {
        ok: false,
        error: { code: error instanceof FileFault ? error.code : code || 'DESKTOP_ERROR', message },
      };
    }
  });
}
function send(action: EditorAction) {
  if (window && !window.webContents.isDestroyed()) window.webContents.send(CHANNEL.action, action);
}
function menu() {
  const busy = files.state?.busy ?? true;
  const canSave = Boolean(files.state?.hasProject) && !busy;
  const next = `${busy}:${canSave}`;
  if (next === menuState) return;
  menuState = next;
  const command = (
    label: string,
    type: 'new' | 'open' | 'save' | 'save-as' | 'undo' | 'redo',
    accelerator: string,
    enabled = true,
  ): MenuItemConstructorOptions => ({
    label,
    accelerator,
    registerAccelerator: false,
    enabled,
    click: () => send({ type }),
  });
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: '文件',
        submenu: [
          command('新建项目', 'new', 'CmdOrCtrl+N', !busy),
          command('打开 JSON…', 'open', 'CmdOrCtrl+O', !busy),
          { type: 'separator' },
          command('保存', 'save', 'CmdOrCtrl+S', canSave),
          command('另存为…', 'save-as', 'CmdOrCtrl+Shift+S', canSave),
          { type: 'separator' },
          { label: '退出', click: () => window?.close() },
        ],
      },
      {
        label: '编辑',
        submenu: [
          command('撤销', 'undo', 'CmdOrCtrl+Z'),
          command('重做', 'redo', 'CmdOrCtrl+Shift+Z'),
          { type: 'separator' },
          { role: 'cut', label: '剪切' },
          { role: 'copy', label: '复制' },
          { role: 'paste', label: '粘贴' },
          { role: 'selectAll', label: '全选' },
        ],
      },
      {
        label: '视图',
        submenu: [
          { role: 'resetZoom', label: '实际大小' },
          { role: 'zoomIn', label: '放大界面' },
          { role: 'zoomOut', label: '缩小界面' },
          { role: 'togglefullscreen', label: '全屏' },
          ...(!app.isPackaged ? [{ role: 'toggleDevTools' as const, label: '开发者工具' }] : []),
        ],
      },
      {
        label: '帮助',
        submenu: [
          {
            label: '关于 Tensor Trail',
            click: () => {
              void dialog.showMessageBox(window!, {
                type: 'info',
                title: 'Tensor Trail',
                message: `Tensor Trail ${process.platform === 'linux' ? 'Linux' : process.platform === 'win32' ? 'Windows' : '桌面'} 开发版`,
                detail: `版本 ${app.getVersion()}\n神经网络架构与 Tensor 形状工作台`,
              });
            },
          },
        ],
      },
    ]),
  );
}
function registerIpc() {
  handle(CHANNEL.preferences, 0, () => ({ theme: settings.theme }));
  handle(CHANNEL.theme, 1, async (value) => {
    const theme = check.theme(value);
    await settings.setTheme(theme);
    nativeTheme.themeSource = theme;
  });
  handle(CHANNEL.state, 1, (value) => {
    files.update(check.documentState(value));
    const state = files.state!;
    const title = `${state.dirty ? '● ' : ''}${state.fileName || (state.hasProject ? state.projectName : '未打开项目')} — Tensor Trail`;
    if (window!.getTitle() !== title) window!.setTitle(title);
    menu();
  });
  handle(CHANNEL.open, 1, async (state) => {
    const version = check.version(state);
    files.current(version);
    const picked = await dialog.showOpenDialog(window!, {
      title: '打开 Tensor Trail 项目',
      properties: ['openFile'],
      filters: [{ name: 'JSON 项目', extensions: ['json'] }],
    });
    if (picked.canceled || !picked.filePaths[0]) return undefined;
    return files.issue(picked.filePaths[0], 'open', version);
  });
  handle(CHANNEL.saveAs, 2, async (name, state) => {
    const version = check.version(state);
    files.current(version);
    const suggested = check.text(name, 240).replace(/[<>:"/\\|?*]/g, '_');
    const picked = await dialog.showSaveDialog(window!, {
      title: '保存 Tensor Trail 项目',
      defaultPath: suggested,
      filters: [{ name: 'JSON 项目', extensions: ['json'] }],
      properties: ['showOverwriteConfirmation'],
    });
    if (picked.canceled || !picked.filePath) return undefined;
    return files.issue(picked.filePath, 'save', version);
  });
  handle(CHANNEL.read, 1, (key) => files.read(check.text(key)));
  handle(CHANNEL.write, 3, (key, text, state) =>
    files.write(check.text(key), check.text(text, 3_000_000), check.version(state)),
  );
  handle(CHANNEL.discard, 0, async () => {
    const result = await dialog.showMessageBox(window!, {
      type: 'question',
      title: '保存修改',
      message: '当前项目有未保存的修改。',
      detail: '继续前是否保存当前项目？',
      buttons: ['保存', '不保存', '取消'],
      defaultId: 0,
      cancelId: 2,
      noLink: true,
    });
    return (['save', 'discard', 'cancel'] as const)[result.response] || 'cancel';
  });
  handle(CHANNEL.close, 3, async (id, state, approved) => {
    if (check.text(id) !== closeRequest || typeof approved !== 'boolean') check.invalid();
    const replied = check.documentState(state);
    const current = files.state;
    closeRequest = undefined;
    if (
      !approved ||
      !current ||
      current.busy ||
      replied.documentId !== current.documentId ||
      replied.revision !== current.revision ||
      replied.dirty !== current.dirty
    )
      return;
    await settings.settled().catch(log);
    allowClose = true;
    window!.close();
  });
}
async function createWindow() {
  settings = new Settings(app.getPath('userData'));
  await settings.load();
  nativeTheme.themeSource = settings.theme;
  protocol.handle('tensor-trail', async (request) => {
    try {
      const url = new URL(request.url);
      const selected = path.resolve(root, '.' + decodeURIComponent(url.pathname));
      if (
        url.hostname !== 'app' ||
        !selected.startsWith(root + path.sep) ||
        url.search ||
        !['.html', '.js', '.css', '.svg', '.png', '.woff2'].includes(path.extname(selected))
      )
        return new Response('Not found', { status: 404 });
      const result = await net.fetch(pathToFileURL(selected).href);
      const headers = new Headers(result.headers);
      headers.set('Content-Security-Policy', productionCsp);
      return new Response(result.body, { status: result.status, headers });
    } catch {
      return new Response('Not found', { status: 404 });
    }
  });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) =>
    callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(
    (_contents, permission) => permission === 'clipboard-sanitized-write',
  );
  if (development)
    session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
      callback({
        responseHeaders: {
          ...details.responseHeaders,
          'Content-Security-Policy': [
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws://127.0.0.1:*; object-src 'none'",
          ],
        },
      });
    });
  const workArea = screen.getPrimaryDisplay().workAreaSize;
  window = new BrowserWindow({
    width: Math.min(1440, workArea.width),
    height: Math.min(940, workArea.height),
    minWidth: Math.min(900, workArea.width),
    minHeight: Math.min(640, workArea.height),
    show: true,
    title: 'Tensor Trail',
    icon: path.resolve(
      __dirname,
      '../../resources',
      process.platform === 'win32' ? 'tensor-trail.ico' : 'tensor-trail.png',
    ),
    backgroundColor: settings.theme === 'light' ? '#f5f5f7' : '#0f121a',
    webPreferences: {
      preload: path.resolve(__dirname, '../preload/index.cjs'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
      webviewTag: false,
    },
  });
  registerIpc();
  menu();
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, url) => {
    if (url !== page) event.preventDefault();
  });
  window.webContents.on('before-input-event', (event, input) => {
    if (
      input.type === 'keyDown' &&
      (input.key === 'F5' || ((input.control || input.meta) && input.key.toLowerCase() === 'r'))
    )
      event.preventDefault();
  });
  window.webContents.on('render-process-gone', (_event, details) => {
    rendererAlive = false;
    log(new Error(`Renderer exited: ${details.reason}`));
  });
  window.on('page-title-updated', (event) => event.preventDefault());
  window.on('close', (event) => {
    if (allowClose || !files.state || !rendererAlive) return;
    event.preventDefault();
    if (closeRequest || !files.state) return;
    closeRequest = randomUUID();
    send({ type: 'close', requestId: closeRequest });
  });
  window.on('closed', () => {
    window = undefined;
  });
  await window.loadURL(page);
}

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => {
    if (window?.isMinimized()) window.restore();
    window?.show();
    window?.focus();
  });
  app.on('window-all-closed', () => app.quit());
  void app
    .whenReady()
    .then(createWindow)
    .catch((error) => {
      log(error);
      dialog.showErrorBox('Tensor Trail 启动失败', (error as Error).message);
      app.exit(1);
    });
}
