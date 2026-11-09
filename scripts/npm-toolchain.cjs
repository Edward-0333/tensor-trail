const fs = require('node:fs');
const path = require('node:path');

function resolveNpmCli(node, env = process.env) {
  const executable = fs.realpathSync(node);
  const bin = path.dirname(executable);
  const candidates = [
    env.npm_execpath,
    path.join(bin, 'node_modules/npm/bin/npm-cli.js'), // Node Windows archive
    path.join(bin, '../lib/node_modules/npm/bin/npm-cli.js'), // Linux archive / Conda / nvm
    path.join(bin, '../share/nodejs/npm/bin/npm-cli.js'), // Ubuntu /usr/bin/node
    ...String(env.PATH || '')
      .split(path.delimiter)
      .filter(Boolean)
      .map((dir) => path.join(dir, 'npm')),
  ];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const resolved = fs.realpathSync(candidate);
      if (path.basename(resolved) === 'npm-cli.js' && fs.statSync(resolved).isFile())
        return resolved;
    } catch {
      /* Try the next installation layout. */
    }
  }
  throw Error('npm CLI not found. Install Node.js with npm; see README.md.');
}
module.exports = { resolveNpmCli };
