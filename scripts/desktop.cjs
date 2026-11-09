const path = require('node:path');
const { spawn } = require('node:child_process');
const root = path.resolve(__dirname, '..');

async function main() {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.TENSOR_TRAIL_DEV_URL;
  const dev = process.argv.includes('--dev');
  let server;
  if (dev) {
    const { createServer } = await import('vite');
    server = await createServer({
      root: path.join(root, 'apps/desktop'),
      configFile: path.join(root, 'apps/desktop/vite.renderer.config.ts'),
    });
    await server.listen();
    env.TENSOR_TRAIL_DEV_URL = 'http://127.0.0.1:5180/';
  }
  const detached = process.argv.includes('--detach');
  const child = spawn(require('electron'), [path.join(root, 'apps/desktop')], {
    cwd: root,
    env,
    detached,
    stdio: detached ? 'ignore' : 'inherit',
    windowsHide: false,
  });
  child.on('error', (error) => {
    console.error(error.message);
    void server?.close();
    process.exitCode = 1;
  });
  if (detached) {
    child.unref();
    return;
  }
  const stop = () => {
    child.kill();
    void server?.close();
  };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  child.once('exit', (code) => {
    void server?.close();
    process.exitCode = code ?? 1;
  });
}
void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
