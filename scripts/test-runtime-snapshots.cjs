// Run with Electron after build:desktop; only temporary projects are modified.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tensor-trail-runtime-'));
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile');
delete process.env.TENSOR_TRAIL_DEV_URL;
const fixture = path.join(directory, 'graph.tensor-trail.json');
const portable = path.join(directory, 'portable.tensor-trail.json');
fs.mkdirSync(path.join(directory, 'runs'));
const snapshots = [2, 3].map((batch) => ({
  format: 'tensor-trail.runtime-snapshot',
  version: 1,
  id: `r${batch}`,
  name: `batch ${batch}`,
  capturedAt: '2026-11-10T00:00:00Z',
  context: { batch, mode: 'test' },
  bindings: [
    {
      nodePath: ['x'],
      nodeType: 'Input',
      value: {
        shape: [batch, 1, 21],
        axes: ['batch', 'ego', 'feature'],
        dtype: 'float32',
        autograd: { requiresGrad: batch === 3, source: 'runtime' },
      },
    },
  ],
}));
for (const s of snapshots)
  fs.writeFileSync(path.join(directory, `runs/${s.id}.json`), JSON.stringify(s));
const project = {
  schemaVersion: 1,
  name: 'Runtime snapshot test',
  variants: [],
  graph: {
    nodes: [
      {
        id: 'x',
        label: 'x',
        type: 'Input',
        definitionVersion: 1,
        position: { x: 0, y: 0 },
        params: { shape: '1,1,21', axes: 'batch,ego,feature', dtype: 'float32' },
      },
    ],
    edges: [],
  },
  runtime: {
    active: 'r2',
    snapshots: [
      ...snapshots.map((s) => ({ id: s.id, name: s.name, file: `runs/${s.id}.json` })),
      { id: 'missing', name: 'missing run', file: 'runs/missing.json' },
    ],
  },
};
fs.writeFileSync(fixture, JSON.stringify(project));
let opening = fixture;
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [opening] });
dialog.showSaveDialog = async () => ({ canceled: false, filePath: portable });
dialog.showMessageBox = async () => ({ response: 1 });
dialog.showErrorBox = (title, body) => console.error(title, body);
const timeout = setTimeout(() => {
  console.error('Runtime snapshot test timeout');
  app.exit(1);
}, 60000);
require('../apps/desktop/dist/main/index.cjs');
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(check, message) {
  for (let i = 0; i < 200; i++) {
    if (await check()) return;
    await wait(50);
  }
  throw Error(message);
}
async function run() {
  await app.whenReady();
  await until(() => BrowserWindow.getAllWindows().length === 1, 'No window');
  const win = BrowserWindow.getAllWindows()[0],
    web = win.webContents;
  const js = (code) => web.executeJavaScript(code);
  await until(
    async () =>
      !web.isLoadingMainFrame() &&
      (await js('!!window.tensorTrail && !!document.querySelector(".app-shell")').catch(
        () => false,
      )),
    'No editor',
  );
  win.setSize(1440, 900);
  const click = (text) =>
    js(
      `(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(text)});if(!b)throw Error('No button');b.click();})()`,
    );
  const tensor = () => js(`document.querySelector('[data-testid="node-x"]')?.textContent ?? ''`);
  const choose = (id) =>
    js(
      `(()=>{const e=document.querySelector('[aria-label="当前运行快照"]');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(id)});e.dispatchEvent(new Event('change',{bubbles:true}));})()`,
    );
  await click('打开');
  await until(async () => /grad:off/.test(await tensor()), 'Sidecar not automatically loaded');
  assert.match(await tensor(), /2.*1.*21/);
  await click('运行数据');
  await choose('');
  await until(async () => /grad:\?/.test(await tensor()), 'Static contract not restored');
  assert.match(await tensor(), /1.*1.*21/);
  await choose('r3');
  await until(async () => /grad:on/.test(await tensor()), 'Second run not loaded');
  assert.match(await tensor(), /3.*1.*21/);
  const model = await js('window.tensorTrail.getModel().project');
  assert.equal(model.graph.nodes[0].params.shape, '1,1,21');
  assert.equal(model.runtime.embedded, undefined);
  await click('保存');
  await until(
    () => JSON.parse(fs.readFileSync(fixture, 'utf8')).runtime.active === 'r3',
    'Not saved',
  );
  for (const s of snapshots)
    assert.equal(
      fs.readFileSync(path.join(directory, `runs/${s.id}.json`), 'utf8'),
      JSON.stringify(s),
    );
  await click('导出当前运行单文件');
  await until(() => fs.existsSync(portable), 'Portable copy not exported');
  const exported = JSON.parse(fs.readFileSync(portable, 'utf8'));
  assert.equal(exported.runtime.embedded.id, 'r3');
  assert.equal(exported.runtime.snapshots.length, 1);
  assert.equal(JSON.parse(fs.readFileSync(fixture, 'utf8')).runtime.embedded, undefined);
  await choose('missing');
  await until(
    async () => await js('!!document.querySelector(".runtime-panel [role=alert]")'),
    'Missing run has no explicit error',
  );
  assert.match(await tensor(), /grad:\?/);
  assert.doesNotMatch(await tensor(), /grad:on/);
  await fs.promises.writeFile(
    path.join(directory, 'missing.png'),
    (await web.capturePage()).toPNG(),
  );
  fs.renameSync(path.join(directory, 'runs'), path.join(directory, 'runs-unavailable'));
  opening = portable;
  await click('打开');
  await until(
    async () => /grad:on/.test(await tensor()),
    'Portable embedded snapshot not restored',
  );
  assert.match(await tensor(), /3.*1.*21/);
  await choose('');
  await until(async () => /grad:\?/.test(await tensor()), 'Portable static view failed');
  await choose('r3');
  await until(
    async () => /grad:on/.test(await tensor()),
    'Portable snapshot lost after static switch',
  );
  await fs.promises.writeFile(
    path.join(directory, 'portable.png'),
    (await web.capturePage()).toPNG(),
  );
  console.log(
    JSON.stringify({
      pass: true,
      directory,
      checks: [
        'auto load',
        'static/run switch',
        'static contract unchanged',
        'read-only sidecars',
        'portable export',
        'missing run error',
        'portable reopen without sidecars',
      ],
    }),
  );
  clearTimeout(timeout);
  app.exit(0);
}
run().catch((error) => {
  console.error(error);
  clearTimeout(timeout);
  app.exit(1);
});
