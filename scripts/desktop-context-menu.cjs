// Isolated Electron smoke: never opens or writes the user's project/profile.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const directory = fs.mkdtempSync('/tmp/tensor-trail-context-'), fixture = path.join(directory, 'context.json');
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile'); delete process.env.TENSOR_TRAIL_DEV_URL;
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [fixture] });
dialog.showMessageBox = async () => ({ response: 1 });
require('../apps/desktop/dist/main/index.cjs');
const timer = setTimeout(() => app.exit(1), 120000), wait = ms => new Promise(r => setTimeout(r, ms));
let web;
async function until(fn, label) {
  for (let i = 0; i < 160; i++) { if (await fn()) return; await wait(75); }
  throw Error(label);
}
async function run() {
  const sdk = await import(pathToFileURL(path.resolve(__dirname, '../packages/model-core/dist/tensor-trail-agent.js')).href);
  const built = sdk.buildModel({ name: 'Canvas context smoke', nodes: [
    { id: 'a', type: 'Input', label: 'Image features', params: { shape: '2,256', axes: 'B,C' }, position: { x: 0, y: 0 } },
    { id: 'b', type: 'ReLU', position: { x: 450, y: 60 } },
    { id: 'out', type: 'Output', position: { x: 900, y: 0 } },
    { id: 'region', type: 'LayoutRegion', label: 'Target region', params: { width: 420, height: 300 }, position: { x: 0, y: 500 } },
  ], connections: [['a.out', 'b.in'], ['b.out', 'out.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1800, 1100); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = s => web.executeJavaScript(s).catch(error => { console.error('Renderer script:', s); throw error; });
  const model = () => js('window.tensorTrail.getModel().project');
  const selector = id => `.react-flow__node[data-id="${id}"]`;
  const bounds = s => js(`(()=>{const e=document.querySelector(${JSON.stringify(s)});if(!e)throw Error('Missing '+${JSON.stringify(s)});const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};})()`);
  const button = (label, scope = 'document') => js(`(()=>{const b=[...${scope}.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(label)}||e.querySelector('span')?.textContent===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Button '+${JSON.stringify(label)});b.click();})()`);
  const menuButton = label => button(label, 'document.querySelector(".canvas-context-menu")');
  const mouse = (type, p, extra = {}) => web.sendInputEvent({ type, x: Math.round(p.x), y: Math.round(p.y), ...extra });
  async function click(p, button = 'left') {
    mouse('mouseMove', p); mouse('mouseDown', p, { button, clickCount: 1 }); mouse('mouseUp', p, { button, clickCount: 1 }); await wait(220);
  }
  const center = async s => { const r = await bounds(s); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; };
  async function key(keyCode, modifiers = []) {
    web.sendInputEvent({ type: 'keyDown', keyCode, modifiers }); web.sendInputEvent({ type: 'keyUp', keyCode, modifiers }); await wait(200);
  }
  async function fit() { await js(`document.querySelector('button[aria-label="适配全图"]').click()`); await wait(500); }
  async function boxSelect() {
    await key('Escape'); await fit();
    const a = await bounds(selector('a')), b = await bounds(selector('b'));
    const p = { x: Math.min(a.x, b.x) - 15, y: Math.min(a.y, b.y) - 15 };
    const q = { x: Math.max(a.x + a.width, b.x + b.width) + 15, y: Math.max(a.y + a.height, b.y + b.height) + 15 };
    mouse('mouseMove', p); mouse('mouseDown', p, { button: 'left', clickCount: 1 });
    for (let i = 1; i <= 10; i++) { mouse('mouseMove', { x: p.x + (q.x - p.x) * i / 10, y: p.y + (q.y - p.y) * i / 10 }, { modifiers: ['leftButtonDown'] }); await wait(25); }
    mouse('mouseUp', q, { button: 'left', clickCount: 1 }); await wait(300);
    assert.deepEqual(await js('[...document.querySelectorAll(".react-flow__node.selected")].map(e=>e.dataset.id).sort()'), ['a', 'b']);
    assert.equal(await js('document.querySelectorAll(".edge-route-toolbar").length'), 0, 'box-selection repeated wire tooltips');
  }
  async function openFor(id) {
    await click(await center(selector(id)), 'right');
    await until(() => js('!!document.querySelector(".canvas-context-menu")'), 'context menu missing');
  }
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  await button('打开'); await until(async () => (await model()).name === built.project.name, 'open'); await wait(600);
  const original = await model();
  await boxSelect(); await openFor('b');
  assert.equal(await js('document.querySelector(".canvas-context-title").textContent'), '已选择 2 个模块');
  await menuButton('添加至分区');
  fs.writeFileSync(path.join(directory, 'multi-selection-menu.png'), (await web.capturePage()).toPNG());
  await menuButton('Target region'); await wait(300);
  assert.equal((await model()).graph.nodes.find(n => n.id === 'region').params.members, 'a,b');
  await key('Z', ['control']); assert.deepEqual(await model(), original, 'membership should undo in one step');
  await boxSelect(); await openFor('a'); await menuButton('新建分区并添加'); await wait(300);
  const newRegion = (await model()).graph.nodes.find(n => n.type === 'LayoutRegion' && n.id !== 'region');
  assert.equal(newRegion?.params.members, 'a,b');
  await key('Z', ['control']); assert.deepEqual(await model(), original, 'creation should undo in one step');
  await boxSelect(); await openFor('a'); await menuButton('复制');
  // Stay clear of the minimap and bottom attribution/interaction hints.
  const pane = await bounds('.react-flow'), point = { x: pane.x + pane.width - 220, y: pane.y + pane.height - 60 };
  await click(point, 'right');
  const menu = await bounds('.canvas-context-menu'), viewport = await js('({width:innerWidth,height:innerHeight})');
  assert(menu.x >= 0 && menu.y >= 0 && menu.x + menu.width <= viewport.width && menu.y + menu.height <= viewport.height);
  fs.writeFileSync(path.join(directory, 'clamped-menu.png'), (await web.capturePage()).toPNG());
  await menuButton('粘贴到此处'); await wait(300);
  let p = await model(); assert.equal(p.graph.nodes.length, original.graph.nodes.length + 2);
  const pasted = p.graph.nodes.filter(n => !original.graph.nodes.some(x => x.id === n.id));
  assert(Math.abs(pasted[1].position.x - pasted[0].position.x - 450) < 1e-6);
  await key('Z', ['control']); assert.deepEqual(await model(), original);
  await boxSelect(); await openFor('out');
  assert.equal(await js('document.querySelector(".canvas-context-title").textContent'), '已选择 1 个模块');
  await menuButton('删除'); await wait(200); assert(!(await model()).graph.nodes.some(n => n.id === 'out'));
  await key('Z', ['control']); assert.deepEqual(await model(), original);
  await openFor('out'); await key('Escape'); assert(!await js('!!document.querySelector(".canvas-context-menu")'));
  await fit(); await click(point);
  const wirePoint = await js(`(()=>{const e=document.querySelector('.react-flow__edge-interaction'),p=e.getPointAtLength(e.getTotalLength()/2),s=new DOMPoint(p.x,p.y).matrixTransform(e.getScreenCTM());return {x:s.x,y:s.y};})()`);
  await click(wirePoint); assert.equal(await js('document.querySelectorAll(".edge-route-toolbar").length'), 1, 'single wire still needs its tools');
  await boxSelect(); assert.equal(await js('document.querySelectorAll(".edge-route-toolbar").length'), 0);
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  assert.deepEqual(JSON.parse(fs.readFileSync(fixture)), built.project, 'fixture must not be auto-saved');
  console.log(JSON.stringify({ ok: true, directory, cases: ['box-select-no-tooltips', 'preserve-multiselection', 'add-to-region', 'new-region', 'undo-region-actions', 'copy-paste-at-pointer', 'menu-clamp', 'right-click-unselected', 'delete-undo', 'escape', 'single-wire-toolbar'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error, directory);
  if (web && !web.isDestroyed()) fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
  clearTimeout(timer); app.exit(1);
});
