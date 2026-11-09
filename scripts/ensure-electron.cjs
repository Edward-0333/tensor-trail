const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const packageDir = path.dirname(require.resolve('electron/package.json'));
const expected = require(path.join(packageDir, 'package.json')).version;
const platform =
  process.env.ELECTRON_INSTALL_PLATFORM || process.env.npm_config_platform || process.platform;
const relative =
  platform === 'win32'
    ? 'electron.exe'
    : ['darwin', 'mas'].includes(platform)
      ? 'Electron.app/Contents/MacOS/Electron'
      : 'electron';
const dist = process.env.ELECTRON_OVERRIDE_DIST_PATH || path.join(packageDir, 'dist');
try {
  // Requiring electron can itself start a download in newer releases. Inspect
  // metadata first so a first install uses our cache and only downloads once.
  if (
    fs.readFileSync(path.join(packageDir, 'path.txt'), 'utf8').trim() === relative &&
    fs.readFileSync(path.join(dist, 'version'), 'utf8').trim().replace(/^v/, '') === expected &&
    fs.existsSync(path.join(dist, relative))
  )
    process.exit(0);
} catch {
  /* First installation or a checkout copied from another platform. */
}
const env = { ...process.env, electron_config_cache: path.join(root, '.npm-cache', 'electron') };
if (
  !env.ELECTRON_GET_USE_PROXY &&
  (env.HTTPS_PROXY || env.HTTP_PROXY || env.https_proxy || env.http_proxy)
)
  env.ELECTRON_GET_USE_PROXY = '1';
console.log(`Installing Electron ${expected} (${platform}/${process.arch})...`);
const result = spawnSync(process.execPath, [path.join(packageDir, 'install.js')], {
  cwd: root,
  env,
  stdio: 'inherit',
  windowsHide: true,
});
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
