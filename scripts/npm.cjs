// Use an optional project-local Node without changing the global installation.
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { resolveNpmCli } = require('./npm-toolchain.cjs');
const root = path.resolve(__dirname, '..');
const version = fs.readFileSync(path.join(root, '.node-version'), 'utf8').trim();
const platform = process.platform === 'win32' ? 'win' : process.platform;
const local = path.join(root, '.tools', `node-v${version}-${platform}-${process.arch}`);
const localNode = path.join(local, process.platform === 'win32' ? 'node.exe' : 'bin/node');
const node = fs.existsSync(localNode) ? localNode : process.execPath;
const probe = spawnSync(node, ['-p', 'process.versions.node'], {
  encoding: 'utf8',
  windowsHide: true,
});
const [major, minor] = (probe.stdout || '').trim().split('.').map(Number);
if (probe.status !== 0 || !(major > 22 || (major === 22 && minor >= 12))) {
  console.error(`Node.js >=22.12 required; project version: ${version}. See README.md.`);
  process.exit(1);
}
let npmCli;
try {
  npmCli = resolveNpmCli(node);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
const env = { ...process.env };
const oldPathKey = Object.keys(env).find((key) => key.toLowerCase() === 'path') || 'PATH';
const oldPath = env[oldPathKey] || '';
delete env[oldPathKey];
env.PATH = path.dirname(node) + path.delimiter + oldPath;
env.npm_config_cache = path.join(root, '.npm-cache');
const result = spawnSync(node, [npmCli, ...process.argv.slice(2)], {
  cwd: root,
  env,
  stdio: 'inherit',
  windowsHide: true,
});
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
