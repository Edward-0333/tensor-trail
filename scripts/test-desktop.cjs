const fs = require('node:fs');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const directory = path.join(root, 'test-results', `desktop-${Date.now()}`);
fs.mkdirSync(directory, { recursive: true });
const env = {
  ...process.env,
  TENSOR_TRAIL_PROFILE_DIR: path.join(directory, 'profile'),
  TENSOR_TRAIL_TEST_DIR: directory,
};
delete env.ELECTRON_RUN_AS_NODE;
delete env.TENSOR_TRAIL_DEV_URL;
function checkProcesses(phase) {
  const report = JSON.parse(fs.readFileSync(path.join(directory, `${phase}-result.json`), 'utf8'));
  for (const pid of report.processIds) {
    let alive = false;
    for (let attempt = 0; attempt < 10; attempt++) {
      try {
        process.kill(pid, 0);
        alive = true;
      } catch {
        alive = false;
        break;
      }
      // Bounded, test-only process shutdown check.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
    }
    if (alive) {
      console.error(`Desktop child process still alive: ${pid}`);
      process.exit(1);
    }
  }
}
for (const phase of ['full', 'restart']) {
  const result = spawnSync(require('electron'), [path.join(root, 'scripts/desktop-smoke.cjs')], {
    cwd: root, env: { ...env, TENSOR_TRAIL_TEST_PHASE: phase },
    stdio: 'inherit', windowsHide: false, timeout: 90000,
  });
  if (result.error) console.error(result.error.message);
  if (result.status !== 0) process.exit(result.status ?? 1);
  checkProcesses(phase);
}
async function development() {
  const { createServer } = await import('vite');
  const server = await createServer({ root: path.join(root, 'apps/desktop'),
    configFile: path.join(root, 'apps/desktop/vite.renderer.config.ts'), server: { port: 0 } });
  try {
    await server.listen();
    const port = server.httpServer.address().port;
    const child = spawn(require('electron'), [path.join(root, 'scripts/desktop-smoke.cjs')], {
      cwd: root, env: { ...env, TENSOR_TRAIL_TEST_PHASE: 'dev', TENSOR_TRAIL_DEV_URL: `http://127.0.0.1:${port}/` },
      stdio: 'inherit', windowsHide: false,
    });
    const code = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { child.kill(); reject(Error('Development window test timed out')); }, 90000);
      child.once('error', error => { clearTimeout(timeout); reject(error); });
      child.once('exit', value => { clearTimeout(timeout); resolve(value); });
    });
    if (code !== 0) throw Error(`Development window test failed: ${code}`);
    checkProcesses('dev');
  } finally { await server.close(); }
}
void development().catch(error => { console.error(error); process.exitCode = 1; });
