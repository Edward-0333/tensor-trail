// Build the SDK and desktop first. Uses an isolated profile and test document.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-viewport-');
const fixture = path.join(directory, 'viewport.json');
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile'); delete process.env.TENSOR_TRAIL_DEV_URL;
app.commandLine.appendSwitch('force-device-scale-factor', '1');
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [fixture] });
require('../apps/desktop/dist/main/index.cjs');
const wait = ms => new Promise(r => setTimeout(r, ms)), timer = setTimeout(() => app.exit(1), 120000);
let web;
async function until(fn, label) {
  for (let i = 0; i < 200; i++) { if (await fn()) return; await wait(60); }
  throw Error(label);
}
async function run() {
  const sdk = await import(pathToFileURL(path.join(root, 'packages/model-core/dist/tensor-trail-agent.js')).href);
  const child = id => ({ nodes: [
    { id: `${id}-input`, type: 'Inlet', position: { x: 0, y: 0 } },
    { id: `${id}-output`, type: 'Outlet', position: { x: 800, y: 200 } },
  ], connections: [[`${id}-input.out`, `${id}-output.in`]] });
  const graphA = child('a');
  graphA.nodes.push({ id: 'nested', type: 'Submodule', position: { x: 1400, y: 0 }, subgraph: child('nested') });
  graphA.connections.push(['a-input.out', 'nested.in']);
  const built = sdk.buildModel({ name: 'Viewport memory', nodes: [
    { id: 'source', type: 'Input', params: { shape: '2,256', axes: 'B,C' }, position: { x: 0, y: 0 } },
    { id: 'block-a', type: 'Submodule', position: { x: 1400, y: 0 }, subgraph: graphA },
    { id: 'block-b', type: 'Submodule', position: { x: 1400, y: 1000 }, subgraph: child('b') },
    { id: 'result', type: 'Output', position: { x: 2800, y: 0 } },
  ], connections: [['source.out', 'block-a.in'], ['source.out', 'block-b.in'], ['block-a.out', 'result.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1600, 1000); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = s => web.executeJavaScript(s), model = () => js('window.tensorTrail.getModel().project');
  const viewport = () => js(`(()=>{const m=new DOMMatrix(getComputedStyle(document.querySelector('.react-flow__viewport')).transform);return {x:m.e,y:m.f,zoom:m.a};})()`);
  const click = selector => js(`document.querySelector(${JSON.stringify(selector)}).click()`);
  async function ready(node) {
    await until(() => js(`!!document.querySelector('.react-flow__node[data-id="${node}"]')`), node);
    await wait(300);
  }
  async function enter(id, childId) {
    await js(`document.querySelector('.react-flow__node[data-id="${id}"] button[aria-label="进入 forward()"]').click()`);
    await ready(childId);
  }
  async function up(id) { await click('button[aria-label="上一层级"]'); await ready(id); }
  async function assertView(expected, label) {
    const actual = await viewport();
    for (const key of ['x', 'y', 'zoom'])
      assert(Math.abs(actual[key] - expected[key]) < .001, `${label}: ${key}: ${JSON.stringify({actual,expected})}`);
  }
  async function panAndZoom(dx, dy) {
    await click('button[aria-label="放大画布"]'); await wait(150);
    const p = await js(`(()=>{const r=document.querySelector('.react-flow').getBoundingClientRect();return {x:Math.round(r.x+70),y:Math.round(r.y+r.height/2)};})()`);
    const before = await viewport();
    web.sendInputEvent({type:'mouseMove',...p});
    web.sendInputEvent({type:'mouseDown',button:'middle',clickCount:1,...p});
    for (let i = 1; i <= 6; i++) {
      web.sendInputEvent({type:'mouseMove',x:p.x+Math.round(dx*i/6),y:p.y+Math.round(dy*i/6),modifiers:['middleButtonDown']});
      await wait(25);
    }
    web.sendInputEvent({type:'mouseUp',button:'middle',clickCount:1,x:p.x+dx,y:p.y+dy});
    await wait(150);
    assert.notDeepEqual(await viewport(), before, 'Pan did not run');
    return viewport();
  }
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  const open = () => js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`);
  await open(); await ready('source');
  const original = await model(), initial = await viewport();
  const rootView = await panAndZoom(160, -70);
  await enter('block-a', 'a-input');
  assert.notDeepEqual(await viewport(), rootView, 'Unseen child inherited parent viewport');
  const aView = await panAndZoom(-100, 50);
  await enter('nested', 'nested-input');
  const nestedView = await panAndZoom(70, 40);
  await up('a-input'); await assertView(aView, 'return to parent');
  await enter('nested', 'nested-input'); await assertView(nestedView, 'revisit nested child');
  await click('.breadcrumbs button'); await ready('source'); await assertView(rootView, 'breadcrumb root');
  await enter('block-b', 'b-input');
  const bView = await panAndZoom(50, -30);
  await up('source'); await assertView(rootView, 'return from sibling');
  await enter('block-a', 'a-input'); await assertView(aView, 'revisit first child');
  await up('source');
  await enter('block-b', 'b-input'); await assertView(bView, 'independent sibling viewport');
  await up('source');
  await click('button[aria-label="适配全图"]'); await wait(300);
  const manuallyFitted = await viewport();
  assert.notDeepEqual(manuallyFitted, rootView, 'Explicit fit did not move the viewport');
  await enter('block-a', 'a-input'); await up('source'); await assertView(manuallyFitted, 'fit is remembered');
  await click('.level-inputs-bar [data-input-id="source"]'); await wait(400);
  const focused = await viewport();
  assert.notDeepEqual(focused, manuallyFitted, 'Explicit focus did not override saved viewport');
  await enter('block-a', 'a-input'); await up('source'); await assertView(focused, 'focus is remembered');
  assert.deepEqual(await model(), original, 'Navigation changed model');
  assert.equal(await js(`document.querySelector('.save-state').dataset.status`), 'saved', 'Viewport marked the file dirty');
  // Opening a new document with identical node IDs must not reuse the previous view.
  const other = { ...built.project, name: 'Independent document' };
  fs.writeFileSync(fixture, JSON.stringify(other));
  await open();
  await until(async () => (await model()).name === other.name, 'new document'); await wait(400);
  await assertView(initial, 'new document starts fitted');
  await enter('block-a', 'a-input');
  assert.notDeepEqual(await viewport(), aView, 'Child viewport leaked across documents');
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ok:true,directory,cases:['pan-and-zoom','parent-return','nested-revisit','breadcrumb-jump','sibling-isolation','manual-fit','explicit-focus','document-isolation','unchanged-model-and-save-state']}));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error,directory);
  try {
    if(web&&!web.isDestroyed())fs.writeFileSync(path.join(directory,'failure.png'),(await web.capturePage()).toPNG());
  } finally { clearTimeout(timer);app.exit(1); }
});
