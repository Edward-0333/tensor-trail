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
  throw Error(typeof label === 'function' ? label() : label);
}
// Runs in the renderer. React Flow connects to the outward face of a handle.
function measureEndpoint(handle, path, source) {
  const side = handle.dataset.handlepos, r = handle.getBoundingClientRect();
  const matrix = path.getScreenCTM();
  const point = path.getPointAtLength(source ? 0 : path.getTotalLength()).matrixTransform(matrix);
  const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
  const x = side === 'left' ? r.left : side === 'right' ? r.right : cx;
  const y = side === 'top' ? r.top : side === 'bottom' ? r.bottom : cy;
  return { side, cx, cy, handle: [x, y], wire: [point.x, point.y], zoom: matrix.a,
    miss: Math.hypot(point.x - x, point.y - y) / matrix.a };
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
  const portIds = ['ref_cam_uv', 'lidar2img', 'bev_pos', 'img_mem'];
  const child = {
    nodes: [...portIds.map(id => ({ id, type: 'Inlet', params: { port: id } })),
      { id: 'features', type: 'Identity' }, { id: 'output', type: 'Outlet' }],
    connections: [['ref_cam_uv.out', 'features.in'], ['features.out', 'output.in']],
  };
  nodes.push({ id: 'block', label: 'SpatialCrossAttention + LN', type: 'Submodule',
    position: { x: 12000, y: 0 }, subgraph: child },
    { id: 'blockOut', type: 'Output', position: { x: 12500, y: 0 } },
    { id: 'branch', type: 'IfRegion', params: { members: 'block' } },
    { id: 'loop', type: 'LoopRegion', params: { members: 'branch' } },
    { id: 'join', type: 'Concat', position: { x: 12000, y: 400 } },
    { id: 'joined', type: 'Output', position: { x: 12500, y: 400 } });
  connections.push(...portIds.map((id, i) => [`source_${i}.out`, `block.${id}`]), ['block.out', 'blockOut.in'],
    ['source_0.out', 'join.a'], ['source_1.out', 'join.b'], ['join.out', 'joined.in']);
  const built = sdk.buildModel({ name: 'Port measurement regression', nodes, connections });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  // Handle measurements need animation frames even when another window covers the test.
  web.setBackgroundThrottling(false);
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
    await js(`document.querySelector('.react-flow__node[data-id="block"] button[aria-label="进入 forward()"]').click()`);
    await until(async () => (await counts()).nodes === child.nodes.length, 'enter child'); await wait(150);
    await js('window.__forcedPorts=[]');
    const start = Date.now();
    await js(`document.querySelector('button[aria-label="上一层级"]').click()`);
    await until(ready, 'return to root'); measurements.push(Date.now() - start); await wait(250);
    assert.deepEqual(await js('window.__forcedPorts'), [], 'Navigation repeated forced measurements');
  }
  assert.deepEqual(await model(), original, 'Navigation changed graph or manual routes');
  await js(`document.querySelector('.react-flow__node[data-id="join"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
  await until(() => js(`!!document.querySelector('select[aria-label="输出端口位置"]')`), 'inspector');
  async function select(nodeId, field, value) {
    const label = { input: '输入端口位置', output: '输出端口位置', input_count: '输入数量 (input_count)' }[field];
    const selector = JSON.stringify(`select[aria-label="${label}"]`);
    await js(`(() => {
      window.__forcedPorts = [];
      const select = document.querySelector(${selector});
      const option = Array.from(select.options).find(o => o.value === ${JSON.stringify(value)});
      if (select.disabled || !option || option.disabled) throw Error('Unavailable option: ' + ${selector});
      option.selected = true;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    })()`);
    const count = field === 'input_count';
    const handles = JSON.stringify(`.react-flow__node[data-id="${nodeId}"] .react-flow__handle.${field === 'output' ? 'source' : 'target'}`);
    let last;
    await until(async () => {
      last = await js(`(() => {
        const node = window.tensorTrail.getModel().project.graph.nodes.find(n => n.id === ${JSON.stringify(nodeId)});
        const handles = [...document.querySelectorAll(${handles})];
        return { selected: document.querySelector(${selector}).value,
          stored: ${count ? 'node.params.input_count' : `node.portLayout?.[${JSON.stringify(field)}]`},
          sides: handles.map(h => h.dataset.handlepos), forced: window.__forcedPorts };
      })()`);
      return last.selected === value && String(last.stored) === value && last.forced.includes(nodeId) &&
        (count ? last.sides.length === Number(value) : last.sides.length > 0 && last.sides.every(side => side === value));
    }, () => `${nodeId} ${field}=${value} did not commit: ${JSON.stringify(last)}`);
    assert(last.forced.every(id => id === nodeId), 'Unrelated nodes were remeasured');
  }
  async function assertAttached(expectedSide) {
    // useUpdateNodeInternals schedules a frame after its selector has already run.
    let endpoint;
    await until(async () => {
      endpoint = await js(`(()=>{
        const h=document.querySelector('.react-flow__node[data-id="join"] .source[data-handleid="out"]');
        const p=document.querySelector('.react-flow__edge[data-id="${original.graph.edges.find(e => e.source === 'join').id}"] .react-flow__edge-interaction');
        return h && p && (${measureEndpoint})(h, p, true);
      })()`);
      return endpoint?.side === expectedSide && endpoint.miss < 1.5;
    }, () => `join.out expected ${expectedSide}, measured in flow pixels: ${JSON.stringify(endpoint)}`);
  }
  await select('join', 'output', 'bottom'); await assertAttached('bottom');
  await select('join', 'input_count', '3'); await assertAttached('bottom');
  await select('join', 'input_count', '2'); await assertAttached('bottom');
  await select('join', 'output', 'right'); await assertAttached('right');
  await js(`document.querySelector('.react-flow__node[data-id="block"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
  await until(() => js(`!!document.querySelector('.submodule-style-panel')`), 'Submodule inspector');
  const blockEdges = original.graph.edges.filter(e => e.source === 'block' || e.target === 'block');
  async function assertBlockSlots() {
    const read = () => js(`(()=>{
      const box=id=>document.querySelector('.react-flow__node[data-id="'+id+'"]').getBoundingClientRect();
      const card=document.querySelector('.submodule-card'), body=card.querySelector('.submodule-content').getBoundingClientRect();
      const zoom=new DOMMatrix(getComputedStyle(document.querySelector('.react-flow__viewport')).transform).a;
      const rect=box('block'), branch=box('branch'), loop=box('loop');
      const enclosed=(outer,inner)=>outer.left<inner.left && outer.top<inner.top && outer.right>inner.right && outer.bottom>inner.bottom;
      const labels=[...card.querySelectorAll('.submodule-port-label')].map(el=>el.getBoundingClientRect());
      const overlaps=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>0.01 && Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>0.01;
      const endpoints=${JSON.stringify(blockEdges)}.map(e=>{
        const source=e.source==='block', port=source?e.sourcePort:e.targetPort;
        const handle=card.querySelector('.react-flow__handle.'+(source?'source':'target')+'[data-handleid="'+port+'"]');
        const slot=card.querySelector('[data-port-id="'+port+'"]');
        const path=document.querySelector('.react-flow__edge[data-id="'+e.id+'"] .react-flow__edge-interaction');
        if (!slot || !handle || !path) return {port, slot:!!slot, handle:!!handle, wire:!!path};
        const label=slot.getBoundingClientRect();
        const {side,cx,cy,miss}=(${measureEndpoint})(handle, path, source);
        const alignment=side==='left'||side==='right'?Math.abs(cy-label.y-label.height/2):Math.abs(cx-label.x-label.width/2);
        return {port, side, miss, alignment:alignment/zoom};
      });
      return {endpoints, enclosed:enclosed(branch,rect)&&enclosed(loop,branch),
        heightError:Math.abs((branch.height-rect.height)/zoom-134),
        collision:labels.some((a,i)=>overlaps(a,body)||labels.slice(i+1).some(b=>overlaps(a,b)))};
    })()`);
    let geometry;
    await until(async () => {
      geometry = await read();
      return geometry.enclosed && geometry.heightError < 1.1 && !geometry.collision &&
        geometry.endpoints.every(endpoint => endpoint.miss < 1.5 && endpoint.alignment < 1);
    }, () => `block slot geometry did not settle: ${JSON.stringify(geometry)}`);
  }
  await assertBlockSlots();
  // Moving the four inputs between sides keeps the card size fixed once the
  // output is on top. ResizeObserver alone cannot refresh these anchors.
  await select('block', 'output', 'top'); await assertBlockSlots();
  await select('block', 'input', 'right'); await assertBlockSlots();
  await select('block', 'output', 'right'); await assertBlockSlots();
  await select('block', 'input', 'bottom'); await assertBlockSlots();
  await select('block', 'output', 'bottom'); await assertBlockSlots();
  assert.equal(fs.readFileSync(fixture, 'utf8'), JSON.stringify(built.project), 'Fixture was overwritten');
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ ok: true, directory, measurements, nodes: nodes.length,
    cases: ['batched-mount', 'repeated-navigation', 'unchanged-document', 'port-side', 'port-count', 'attached-wires', 'submodule-slot-alignment', 'measured-code-regions'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error, directory);
  if (web && !web.isDestroyed()) fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
  clearTimeout(timer); app.exit(1);
});
