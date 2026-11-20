const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');

const root = fs.realpathSync(path.resolve(__dirname, '..'));
const lockfile = path.join(root, 'package-lock.json');
const modules = path.join(root, 'node_modules');
const stamp = path.join(modules, `.tensor-trail-${process.platform}-lock`);
const fingerprint = createHash('sha256')
  .update(fs.readFileSync(lockfile))
  .update(`${process.platform}:${process.arch}`)
  .digest('hex');

function isCurrent() {
  try {
    return fs.existsSync(path.join(modules, '.package-lock.json')) &&
      fs.readFileSync(stamp, 'utf8') === fingerprint;
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

async function ensureDependencies() {
  if (isCurrent()) return;
  // Only one launcher may run npm ci in this checkout at a time.
  const windows = process.platform === 'win32';
  const id = createHash('sha256').update(windows ? root.toLowerCase() : root).digest('hex');
  const address = (windows ? '\\\\.\\pipe\\' : '\0') + 'tensor-trail-install-' + id;
  const lock = net.createServer();
  await new Promise((resolve, reject) => {
    lock.once('error', reject);
    lock.listen(address, resolve);
  });
  try {
    if (isCurrent()) return;
    fs.rmSync(stamp, { force: true });
    const result = spawnSync(process.execPath, [
      path.join(__dirname, 'npm.cjs'), 'ci', '--no-audit', '--no-fund', ...process.argv.slice(2),
    ], { cwd: root, stdio: 'inherit' });
    if (result.error) throw result.error;
    if (result.status !== 0) throw Error(`npm ci failed (${result.status ?? result.signal}).`);
    fs.writeFileSync(stamp, fingerprint);
  } finally {
    lock.close();
  }
}

ensureDependencies().catch(error => {
  console.error(error.code === 'EADDRINUSE'
    ? '另一个启动窗口正在安装依赖，请在安装完成后重试。'
    : error.message);
  process.exitCode = 1;
});
