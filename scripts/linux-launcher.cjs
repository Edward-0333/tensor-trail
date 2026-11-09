// Freedesktop entries are not shell programs: escape their string and Exec
// grammars separately, and never interpolate paths into a shell command.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const desktopId = 'io.github.edward0333.tensortrail.desktop';
function value(text) {
  return text
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}
function argument(text) {
  if (/[\n\r]/.test(text)) throw Error('启动器路径不能包含换行。');
  return value('"' + text.replace(/%/g, '%%').replace(/["`$\\]/g, '\\$&') + '"');
}
function desktopEntry(root, node) {
  return [
    '[Desktop Entry]',
    'Type=Application',
    'Name=Tensor Trail',
    'Comment=Neural network architecture editor',
    'Comment[zh_CN]=神经网络架构与张量形状工作台',
    `Exec=${[node, path.join(root, 'scripts/linux-desktop.cjs'), '--from-menu'].map(argument).join(' ')}`,
    `Path=${value(root)}`,
    `Icon=${value(path.join(root, 'apps/desktop/resources/tensor-trail.png'))}`,
    'Terminal=false',
    'Categories=Development;',
    'StartupNotify=true',
    'StartupWMClass=io.github.edward0333.tensortrail',
    '',
  ].join('\n');
}
function launcherPath(env = process.env, home = os.homedir()) {
  const data =
    env.XDG_DATA_HOME && path.isAbsolute(env.XDG_DATA_HOME)
      ? env.XDG_DATA_HOME
      : path.join(home, '.local/share');
  return path.join(data, 'applications', desktopId);
}
function installLauncher(root, node, destination = launcherPath()) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, desktopEntry(root, fs.realpathSync(node)), { mode: 0o644 });
  return destination;
}
module.exports = { desktopId, desktopEntry, launcherPath, installLauncher };
