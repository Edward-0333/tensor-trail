const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const packageDir = path.dirname(require.resolve('electron/package.json'));
let executable;
let versionMatches = false;
try {
  executable = require('electron');
  const expected = require(path.join(packageDir, 'package.json')).version;
  versionMatches =
    fs
      .readFileSync(path.join(packageDir, 'dist', 'version'), 'utf8')
      .trim()
      .replace(/^v/, '') === expected;
} catch {
  /* First installation. */
}
if (executable && fs.existsSync(executable) && versionMatches) process.exit(0);
const result = spawnSync(process.execPath, [path.join(packageDir, 'install.js')], {
  cwd: root,
  env: { ...process.env, electron_config_cache: path.join(root, '.npm-cache', 'electron') },
  stdio: 'inherit',
  windowsHide: true,
});
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
