// Run with Electron; never opens or changes the user's project/profile.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-focus-');
const fixture = path.join(directory, 'focus.json');
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile'); delete process.env.TENSOR_TRAIL_DEV_URL;
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [fixture] });
require('../apps/desktop/dist/main/index.cjs');
const wait = ms => new Promise(r => setTimeout(r, ms)), timer = setTimeout(() => app.exit(1), 120000);
let web;
async function until(fn, label) {
  for (let i = 0; i < 200; i++) { if (await fn()) return; await wait(75); }
  throw Error(label);
}
async function run() {
  const sdk = await import(pathToFileURL(path.join(root, 'packages/model-core/dist/tensor-trail-agent.js')).href);
  const built = sdk.buildModel({ name: '关联节点定位检查', nodes: [
    { id: 'source', type: 'Input', params: { shape: '2,256', axes: 'B,C' }, position: { x: 100, y: 200 } },
    { id: 'out', type: 'Output', position: { x: 600, y: 200 } },
    { id: 'region1', type: 'LayoutRegion', label: '主节点分区', params: { members: 'source,out' } },
    { id: 'region2', type: 'LayoutRegion', label: '循环内副本分区', position: { x: 3400, y: 2000 } },
    { id: 'loop', type: 'LoopRegion', params: { members: 'region2', statement: 'for i in range(6)' } },
  ], connections: [['source.out', 'out.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics));
  const linked = sdk.patchModel(built.project, { operations: [
    { op: 'duplicate_linked', source: 'source', id: 'alias', position: { x: 3400, y: 2000 } },
    { op: 'set_params', id: 'region2', params: { members: 'alias' } },
  ] }, { mode: 'draft', layout: 'preserve' });
  assert(linked.ok, JSON.stringify(linked.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(linked.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1800, 1100); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = source => web.executeJavaScript(source), model = () => js('window.tensorTrail.getModel().project');
  const selector = id => `.react-flow__node[data-id="${id}"]`;
  const rect = sel => js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)throw Error('Missing element');const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};})()`);
  const viewport = () => js(`document.querySelector('.react-flow__viewport').style.transform`);
  async function click(sel) {
    await js(`document.querySelector(${JSON.stringify(sel)}).scrollIntoView({block:'nearest'})`);
    const r = await rect(sel), point = { x: Math.round(r.x + r.w / 2), y: Math.round(r.y + r.h / 2) };
    web.sendInputEvent({ type: 'mouseMove', ...point });
    web.sendInputEvent({ type: 'mouseDown', button: 'left', clickCount: 1, ...point });
    web.sendInputEvent({ type: 'mouseUp', button: 'left', clickCount: 1, ...point });
    await wait(500);
  }
  async function linkedClick(id) {
    await js(`(()=>{const b=[...document.querySelectorAll('.value-link-members button')].find(b=>b.querySelector('code').textContent===${JSON.stringify(id)});if(!b)throw Error('Missing related node');b.dataset.focusTest='target';})()`);
    await click('[data-focus-test="target"]');
    await js(`document.querySelector('[data-focus-test="target"]')?.removeAttribute('data-focus-test')`);
  }
  async function centered(id) {
    const canvas = await rect('.react-flow'), node = await rect(selector(id));
    assert(Math.abs(node.x + node.w / 2 - canvas.x - canvas.w / 2) < 3, `Not horizontally centered: ${id}`);
    assert(Math.abs(node.y + node.h / 2 - canvas.y - canvas.h / 2) < 3, `Not vertically centered: ${id}`);
    const width = await js(`document.querySelector(${JSON.stringify(selector(id))}).offsetWidth`);
    assert(node.w / width >= 1 && node.w < canvas.w && node.h < canvas.h, 'Unreadable zoom or clipped node');
    const selected = await js(`[...document.querySelectorAll('.react-flow__node.selected')].map(n=>n.dataset.id)`);
    assert.deepEqual(selected, [id]);
    assert.equal(await js(`document.querySelector('.value-link-members [aria-current="true"] code')?.textContent`), id);
  }
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  await js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`);
  await until(async () => (await model()).name === linked.project.name, 'open'); await wait(800);
  const original = await model();
  await click('.level-inputs-bar [data-input-id="source"]');
  await linkedClick('alias'); await centered('alias');
  await js(`Array.from(document.querySelectorAll('.value-link-actions button')).find(b=>b.textContent.trim()==='跳转主节点').click()`);
  await wait(500); await centered('source');
  await linkedClick('alias'); await centered('alias');
  // Pan without changing selection, then click the already selected list entry.
  const canvas = await rect('.react-flow'), start = { x: Math.round(canvas.x + 80), y: Math.round(canvas.y + canvas.h / 2) };
  const beforePan = await viewport();
  web.sendInputEvent({ type: 'mouseMove', ...start });
  web.sendInputEvent({ type: 'mouseDown', button: 'middle', clickCount: 1, ...start });
  for (let i = 1; i <= 8; i++) {
    web.sendInputEvent({ type: 'mouseMove', x: start.x + 30 * i, y: start.y + 10 * i, modifiers: ['middleButtonDown'] });
    await wait(30);
  }
  web.sendInputEvent({ type: 'mouseUp', button: 'middle', clickCount: 1, x: start.x + 240, y: start.y + 80 });
  await wait(200); assert.notEqual(await viewport(), beforePan, 'Pan did not run');
  await linkedClick('alias'); await centered('alias');
  // Ordinary canvas selection must not move the viewport.
  await click('button[aria-label="适配全图"]'); const fitted = await viewport();
  await click(`${selector('source')} .tensor-node-label`);
  assert.equal(await viewport(), fitted, 'Canvas click unexpectedly moved viewport');
  // Both node identity and geometry remain untouched, including wires and regions.
  assert.deepEqual(await model(), original);
  assert.equal(fs.readFileSync(fixture, 'utf8'), JSON.stringify(linked.project));
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ ok: true, directory, cases: ['cross-region-link', 'loop-region-link', 'jump-to-owner', 'refocus-current', 'ordinary-click-stays', 'model-unchanged'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error, directory);
  if (web && !web.isDestroyed()) fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
  clearTimeout(timer); app.exit(1);
});
