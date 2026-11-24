// File-backed stdio also captures JSON reports larger than spawnSync's default 1 MiB buffer.
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

exports.runAgent = function runAgent(args, input = '') {
  const cwd = path.resolve(__dirname, '../../..');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensor-trail-cli-io-'));
  const descriptors = [];
  try {
    const stdin = path.join(dir, 'stdin'), stdout = path.join(dir, 'stdout');
    fs.writeFileSync(stdin, input);
    descriptors.push(fs.openSync(stdin, 'r'));
    descriptors.push(fs.openSync(stdout, 'w'));
    const result = spawnSync(process.execPath, ['packages/cli/scripts/agent.mjs', ...args], {
      cwd, stdio: [descriptors[0], descriptors[1], 'inherit'], windowsHide: true,
    });
    if (result.error) throw result.error;
    return { status: result.status, stdout: fs.readFileSync(stdout, 'utf8') };
  } finally {
    for (const fd of descriptors) fs.closeSync(fd);
    fs.rmSync(dir, { recursive: true, force: true });
  }
};
