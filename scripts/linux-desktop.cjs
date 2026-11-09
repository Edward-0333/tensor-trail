const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn, spawnSync } = require('node:child_process');
const { installLauncher, launcherPath } = require('./linux-launcher.cjs');
const root = path.resolve(__dirname, '..');
const args = new Set(process.argv.slice(2));
const fromMenu = args.has('--from-menu');
let logFd;
let logFile;
function run(scriptArgs) {
  const result = spawnSync(process.execPath, scriptArgs, {
    cwd: root,
    env: process.env,
    stdio: logFd === undefined ? 'inherit' : ['ignore', logFd, logFd],
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw Error(`命令失败（${result.status ?? result.signal}）：node ${scriptArgs.join(' ')}`);
}
function main() {
  if (process.platform !== 'linux') throw Error('此入口仅用于 Linux。');
  for (const arg of args)
    if (
      !['--dev', '--install-launcher', '--remove-launcher', '--from-menu', '--help'].includes(arg)
    )
      throw Error(`未知参数：${arg}`);
  if (args.has('--help')) {
    console.log(
      '用法：./start-desktop.sh [--dev | --install-launcher | --remove-launcher]\n默认构建并打开 Ubuntu 桌面窗口；--dev 启用界面热更新。',
    );
    return;
  }
  if (fromMenu) {
    const base =
      process.env.XDG_STATE_HOME && path.isAbsolute(process.env.XDG_STATE_HOME)
        ? process.env.XDG_STATE_HOME
        : path.join(os.homedir(), '.local/state');
    logFile = path.join(base, 'tensor-trail/launcher.log');
    fs.mkdirSync(path.dirname(logFile), { recursive: true });
    logFd = fs.openSync(logFile, 'a', 0o600);
    fs.writeSync(logFd, `\n${new Date().toISOString()} Tensor Trail startup\n`);
  }
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (!(major > 22 || (major === 22 && minor >= 12))) throw Error('需要 Node.js >=22.12。');
  // GUI launches do not inherit an activated Conda/nvm shell. Keep the selected
  // Node on PATH so npm scripts use exactly the same interpreter.
  process.env.PATH = path.dirname(process.execPath) + path.delimiter + (process.env.PATH || '');
  if (args.has('--remove-launcher')) {
    fs.rmSync(launcherPath(), { force: true });
    console.log('已移除当前用户的 Tensor Trail 应用菜单入口。');
    return;
  }
  if (args.has('--install-launcher')) {
    const target = installLauncher(root, process.execPath);
    spawnSync('update-desktop-database', [path.dirname(target)], { stdio: 'ignore' });
    console.log(
      `已安装应用菜单入口：${target}\n可在 Ubuntu 应用列表中搜索 Tensor Trail。移动仓库或 Node 后请重新安装入口。`,
    );
    return;
  }
  if (!process.env.DISPLAY && !process.env.WAYLAND_DISPLAY)
    throw Error('未检测到图形会话。请在 Ubuntu 桌面终端运行；仅 SSH 终端需要先配置图形显示。');
  if (process.getuid?.() === 0) throw Error('请以普通用户运行桌面端，不要使用 sudo。');
  const fingerprint = crypto
    .createHash('sha256')
    .update(fs.readFileSync(path.join(root, 'package-lock.json')))
    .update(`${process.platform}:${process.arch}`)
    .digest('hex');
  const stamp = path.join(root, 'node_modules/.tensor-trail-linux-lock');
  const installed = fs.existsSync(path.join(root, 'node_modules/.package-lock.json'));
  if (!installed || !fs.existsSync(stamp) || fs.readFileSync(stamp, 'utf8') !== fingerprint) {
    run(['scripts/npm.cjs', 'ci', '--ignore-scripts', '--no-audit', '--no-fund']);
    fs.writeFileSync(stamp, fingerprint);
  }
  run(['scripts/npm.cjs', 'run', 'setup:desktop']);
  // Explain missing system libraries before Chromium can fail without a window.
  const binary = require('electron');
  const probe = spawnSync('ldd', [binary], { encoding: 'utf8' });
  const missing = (probe.stdout || '').split('\n').filter((line) => line.includes('not found'));
  if (missing.length)
    throw Error(
      `缺少系统依赖：\n${missing.join('\n')}\n请参阅 apps/desktop/README.md 的 Ubuntu 依赖说明。`,
    );
  if (args.has('--dev')) run(['scripts/npm.cjs', 'run', 'dev:desktop']);
  else {
    run(['scripts/npm.cjs', 'run', 'build:desktop']);
    run(['scripts/desktop.cjs']);
  }
}
try {
  main();
} catch (error) {
  const message = `Tensor Trail 启动失败：${error.message}${logFile ? `\n详细日志：${logFile}` : ''}`;
  console.error(message);
  if (logFd !== undefined) fs.writeSync(logFd, message + '\n');
  if (fromMenu) {
    const notice = spawn('zenity', ['--error', '--title=Tensor Trail', `--text=${message}`], {
      stdio: 'ignore',
      detached: true,
    });
    notice.on('error', () => {});
    notice.unref();
  }
  process.exitCode = 1;
} finally {
  if (logFd !== undefined) fs.closeSync(logFd);
}
