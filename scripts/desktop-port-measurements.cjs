// Run with Electron after building the SDK and desktop; uses an isolated fixture/profile.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-ports-');
const fixture = path.join(directory, 'ports.json');
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
  const nodes = [], connections = [];
  for (let i = 0; i < 145; i++) {
    const x = Math.floor(i / 15) * 1100, y = (i % 15) * 200;
    nodes.push({ id: `source_${i}`, type: 'Input', params: { shape: '2,3', axes: 'B,C' }, position: { x, y } },
      { id: `out_${i}`, type: 'Output', position: { x: x + 400, y } });
    connections.push([`source_${i}.out`, `out_${i}.in`]);
  }
  nodes.push({ id: 'block', type: 'Submodule', position: { x: 12000, y: 0 }, subgraph: {
    nodes: [{ id: 'input', type: 'Inlet' }, { id: 'output', type: 'Outlet' }],
    connections: [['input.out', 'output.in']],
  } }, { id: 'join', type: 'Concat', position: { x: 12000, y: 400 } },
  { id: 'joined', type: 'Output', position: { x: 12500, y: 400 } });
  connections.push(['source_0.out', 'block.in'], ['source_0.out', 'join.a'], ['source_1.out', 'join.b'], ['join.out', 'joined.in']);
  const built = sdk.buildModel({ name: 'Port measurement regression', nodes, connections });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1800, 1100); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = source => web.executeJavaScript(source), model = () => js('window.tensorTrail.getModel().project');
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  // Count explicit single-node remeasurements, not the batched ResizeObserver initialization.
  await js(`window.__forcedPorts=[];
    const original=Element.prototype.querySelector;
    Element.prototype.querySelector=function(selector){
      const match=typeof selector==='string'&&selector.match(/^\\.react-flow__node\\[data-id="([^"]+)"\\]$/);
      if(match&&this.classList.contains('react-flow'))window.__forcedPorts.push(match[1]);
      return original.call(this,selector);
    };true;`);
  await js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`);
  const counts = () => js(`({nodes:document.querySelectorAll('.react-flow__node').length,
    edges:document.querySelectorAll('.react-flow__edge').length})`);
  const ready = async () => {
    const c = await counts(); return c.nodes === nodes.length && c.edges === connections.length;
  };
  await until(ready, 'initial nodes/edges'); await wait(400);
  assert.deepEqual(await js('window.__forcedPorts'), [], 'Initial mount forced one update per node');
  const original = await model(), measurements = [];
  for (let attempt = 0; attempt < 3; attempt++) {
    await js(`Array.from(document.querySelectorAll('.react-flow__node[data-id="block"] button')).find(b=>b.textContent.includes('进入')).click()`);
    await until(async () => (await counts()).nodes === 2, 'enter child'); await wait(150);
    await js('window.__forcedPorts=[]');
    const start = Date.now();
    await js(`document.querySelector('button[aria-label="上一层级"]').click()`);
    await until(ready, 'return to root'); measurements.push(Date.now() - start); await wait(250);
    assert.deepEqual(await js('window.__forcedPorts'), [], 'Navigation repeated forced measurements');
  }
  assert.deepEqual(await model(), original, 'Navigation changed graph or manual routes');
  await js(`document.querySelector('.react-flow__node[data-id="join"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
  await until(() => js(`!!document.querySelector('select[aria-label="输出端口位置"]')`), 'inspector');
  async function select(label, value) {
    await js(`(()=>{window.__forcedPorts=[];const e=document.querySelector(${JSON.stringify(`select[aria-label="${label}"]`)});
      e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    await until(() => js(`window.__forcedPorts.includes('join')`), `remeasure ${label}`);
    assert((await js('window.__forcedPorts')).every(id => id === 'join'), 'Unrelated nodes were remeasured');
    await wait(150);
  }
  async function assertAttached() {
    // The SVG endpoint must follow the actual DOM handle, not its old position.
    const distance = await js(`(()=>{
      const h=document.querySelector('.react-flow__node[data-id="join"] .react-flow__handle.source').getBoundingClientRect();
      const p=document.querySelector('.react-flow__edge[data-id="${original.graph.edges.find(e => e.source === 'join').id}"] .react-flow__edge-interaction');
      const v=p.getPointAtLength(0),s=new DOMPoint(v.x,v.y).matrixTransform(p.getScreenCTM());
      return Math.hypot(s.x-(h.x+h.width/2),s.y-(h.y+h.height/2));
    })()`);
    assert(distance < 2, `Wire missed the changed handle by ${distance}px`);
  }
  await select('输出端口位置', 'bottom'); await assertAttached();
  await select('输入数量 (input_count)', '3');
  assert.equal(await js(`document.querySelectorAll('.react-flow__node[data-id="join"] .react-flow__handle.target').length`), 3);
  await select('输入数量 (input_count)', '2'); await assertAttached();
  await select('输出端口位置', 'right'); await assertAttached();
  assert.equal(fs.readFileSync(fixture, 'utf8'), JSON.stringify(built.project), 'Fixture was overwritten');
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ ok: true, directory, measurements, nodes: nodes.length,
    cases: ['batched-mount', 'repeated-navigation', 'unchanged-document', 'port-side', 'port-count', 'attached-wires'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error, directory);
  if (web && !web.isDestroyed()) fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
  clearTimeout(timer); app.exit(1);
});
