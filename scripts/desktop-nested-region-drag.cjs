// Native mouse checks use an isolated fixture/profile, never the user's canvas.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-nested-drag-');
const fixture = path.join(directory, 'regions.json');
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
  const built = sdk.buildModel({ name: '循环内分区拖动检查', nodes: [
    { id: 'a', type: 'Input', params: { shape: '2,3', axes: 'B,C' }, position: { x: 100, y: 250 } },
    { id: 'b', type: 'ReLU', position: { x: 450, y: 250 } },
    { id: 'out', type: 'Output', position: { x: 1400, y: 600 } },
    { id: 'region', type: 'LayoutRegion', label: '循环内的可拖动分区', position: { x: 40, y: 110 },
      params: { sizing: 'auto', width: 800, height: 440, members: 'a,b' } },
    { id: 'condition', type: 'IfRegion', params: { members: 'region,a,b', condition: 'enabled' } },
    { id: 'loop', type: 'LoopRegion', params: { members: 'condition', statement: 'for i in range(6)' } },
  ], connections: [['a.out', 'b.in'], ['b.out', 'out.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1800, 1100); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = s => web.executeJavaScript(s), model = () => js('window.tensorTrail.getModel().project');
  const selector = id => `.react-flow__node[data-id="${id}"]`;
  const rect = sel => js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)}),r=e.getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height,width:e.offsetWidth};})()`);
  const press = (keyCode, modifiers = ['control']) => {
    web.sendInputEvent({ type: 'keyDown', keyCode, modifiers }); web.sendInputEvent({ type: 'keyUp', keyCode, modifiers });
  };
  async function drag(sel, dx, dy, held) {
    const r = await rect(sel), start = { x: Math.round(r.x + Math.min(r.w / 2, 90)), y: Math.round(r.y + r.h / 2) };
    web.sendInputEvent({ type: 'mouseMove', ...start });
    web.sendInputEvent({ type: 'mouseDown', ...start, button: 'left', clickCount: 1 });
    for (let i = 1; i <= 10; i++) {
      web.sendInputEvent({ type: 'mouseMove', x: Math.round(start.x + dx * i / 10), y: Math.round(start.y + dy * i / 10),
        modifiers: ['leftButtonDown'] }); await wait(35);
    }
    if (held) await held();
    web.sendInputEvent({ type: 'mouseUp', x: start.x + dx, y: start.y + dy, button: 'left', clickCount: 1 });
    await wait(350);
  }
  async function hit(id, suffix) {
    const r = await rect(`${selector(id)} ${suffix}`);
    const x = r.x + Math.min(r.w / 2, 90), y = r.y + r.h / 2;
    return js(`document.elementFromPoint(${x},${y})?.closest('.react-flow__node')?.dataset.id`);
  }
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'api');
  await js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`);
  await until(async () => (await model()).name === built.project.name, 'model'); await wait(900);
  const before = await model();
  const hitRegion = await hit('region', '.layout-region-heading');
  console.log(JSON.stringify({ directory, hitRegion, wrapperStyles: await js(`['loop','condition','region'].map(id=>{const e=document.querySelector('.react-flow__node[data-id="'+id+'"]');return{id,inline:e.style.pointerEvents,computed:getComputedStyle(e).pointerEvents,z:e.style.zIndex}})`) }));
  assert.equal(hitRegion, 'region', 'Code scope intercepts layout region title');
  for (const id of ['loop', 'condition']) assert.equal(await hit(id, '.if-region-heading'), id, 'Scope heading not selectable');
  const frameBefore = await rect(selector('region')), scale = frameBefore.w / frameBefore.width;
  let liveDelta;
  await drag(`${selector('region')} .layout-region-heading`, 90, 50, async () => {
    assert.deepEqual(await model(), before, 'Live drag committed model');
    const a = await rect(selector('a')), frame = await rect(selector('region'));
    assert(frame.x > frameBefore.x + 70 && a.x > frame.x, 'Live group drag did not move members');
    liveDelta = { x: (frame.x - frameBefore.x) / scale, y: (frame.y - frameBefore.y) / scale };
  });
  const after = await model();
  for (const id of ['region', 'a', 'b']) {
    const prev = before.graph.nodes.find(n => n.id === id), next = after.graph.nodes.find(n => n.id === id);
    assert(Math.abs(next.position.x - prev.position.x - liveDelta.x) < 3, `Wrong x delta ${id}`);
    assert(Math.abs(next.position.y - prev.position.y - liveDelta.y) < 3, `Wrong y delta ${id}`);
  }
  for (const id of ['out', 'loop', 'condition'])
    assert.deepEqual(after.graph.nodes.find(n => n.id === id), before.graph.nodes.find(n => n.id === id), `Changed unrelated ${id}`);
  press('Z'); await wait(350); assert.deepEqual(await model(), before, 'Undo must restore whole group');
  press('Z', ['control', 'shift']); await wait(350); assert.deepEqual(await model(), after, 'Redo must restore group');
  await drag(`${selector('region')} .layout-region-heading`, 0, 0);
  assert.equal(await hit('region', '.react-flow__resize-control.handle.bottom.right'), 'region', 'Scope blocks resize handle');
  await drag(`${selector('region')} .react-flow__resize-control.handle.bottom.right`, 60, 45);
  assert.equal((await model()).graph.nodes.find(n => n.id === 'region').params.sizing, 'manual', 'Resize did not commit');
  press('Z'); await wait(350); assert.deepEqual(await model(), after);
  // Selecting an enclosing frame must not make its transparent body eat future drags.
  await drag(`${selector('loop')} .if-region-heading`, 0, 0);
  assert.equal(await hit('region', '.layout-region-heading'), 'region');
  await drag(`${selector('region')} .layout-region-heading`, -40, -20);
  const final = await model(); assert.notDeepEqual(final, after);
  assert(sdk.validateModel(final).ok);
  assert.deepEqual(sdk.validateModel(final).outputs, sdk.validateModel(before).outputs);
  press('S'); await until(() => JSON.stringify(JSON.parse(fs.readFileSync(fixture))) === JSON.stringify(final), 'save');
  await js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`); await wait(600);
  assert.deepEqual(await model(), final);
  assert.equal(await hit('region', '.layout-region-heading'), 'region');
  fs.writeFileSync(path.join(directory, 'nested-drag.png'), (await web.capturePage()).toPNG());
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ ok: true, directory, cases: ['nested-loop-if-title-hit', 'scope-header-hit', 'live-group-drag', 'undo-redo', 'resize', 'selected-scope-pass-through', 'save-reopen'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async e => {
  console.error(e, directory);
  if (web && !web.isDestroyed()) {
    fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
    fs.writeFileSync(path.join(directory, 'failure.json'), JSON.stringify(await web.executeJavaScript('window.tensorTrail.getModel()'), null, 2));
  }
  clearTimeout(timer); app.exit(1);
});
