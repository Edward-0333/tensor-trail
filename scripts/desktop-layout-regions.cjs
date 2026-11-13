// Run with Electron; fixtures and settings are isolated from the user's project/window.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-regions-');
const fixture = path.join(directory, 'regions.json');
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile'); delete process.env.TENSOR_TRAIL_DEV_URL;
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [fixture] });
dialog.showMessageBox = async () => ({ response: 1 });
require('../apps/desktop/dist/main/index.cjs');
const timer = setTimeout(() => app.exit(1), 120000), wait = ms => new Promise(r => setTimeout(r, ms));
let web;
async function until(fn, label) {
  for (let i = 0; i < 200; i++) { if (await fn()) return; await wait(75); }
  throw Error(label);
}
async function run() {
  const sdk = await import(pathToFileURL(path.join(root, 'packages/model-core/dist/tensor-trail-agent.js')).href);
  const built = sdk.buildModel({ name: '布局分区交互检查', nodes: [
    { id: 'a', type: 'Input', label: '图像特征', params: { shape: '2,256', axes: 'B,C' }, position: { x: 0, y: 0 } },
    { id: 'b', type: 'ReLU', label: '特征激活', position: { x: 450, y: 100 } },
    { id: 'out', type: 'Output', label: '返回特征', position: { x: 900, y: 0 } },
  ], connections: [['a.out', 'b.in'], ['b.out', 'out.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setSize(1800, 1100); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = source => web.executeJavaScript(source);
  const model = () => js('window.tensorTrail.getModel().project');
  const button = label => js(`(()=>{const b=[...document.querySelectorAll('button')].find(x=>x.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Button '+${JSON.stringify(label)});b.click();})()`);
  const selector = id => `.react-flow__node[data-id="${id}"]`;
  const bounds = sel => js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)throw Error('Missing '+${JSON.stringify(sel)});const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,w:e.offsetWidth,h:e.offsetHeight};})()`);
  async function pointer(sel, dx = 0, dy = 0) {
    const r = await bounds(sel), start = { x: Math.round(r.x + Math.min(70, r.width / 2)), y: Math.round(r.y + r.height / 2) };
    web.sendInputEvent({ type: 'mouseMove', ...start });
    web.sendInputEvent({ type: 'mouseDown', button: 'left', clickCount: 1, ...start, modifiers: ['alt'] });
    if (dx || dy) for (let i = 1; i <= 10; i++) {
      web.sendInputEvent({ type: 'mouseMove', x: Math.round(start.x + dx * i / 10), y: Math.round(start.y + dy * i / 10),
        modifiers: ['leftButtonDown', 'alt'] }); await wait(35);
    }
    web.sendInputEvent({ type: 'mouseUp', button: 'left', clickCount: 1, x: start.x + dx, y: start.y + dy, modifiers: ['alt'] });
    await wait(300);
  }
  const press = keyCode => {
    web.sendInputEvent({ type: 'keyDown', keyCode, modifiers: ['control'] });
    web.sendInputEvent({ type: 'keyUp', keyCode, modifiers: ['control'] });
  };
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  await button('打开'); await until(async () => (await model()).name === built.project.name, 'open'); await wait(700);
  await pointer(`${selector('a')} .tensor-node-label`); await button('新建分区');
  let p = await model(), region = p.graph.nodes.find(n => n.type === 'LayoutRegion');
  assert(region && region.params.members === 'a', 'toolbar creation must include selection'); const id = region.id;
  await wait(700);
  const originalA = { ...p.graph.nodes.find(n => n.id === 'a').position };
  await pointer(`${selector(id)} .layout-region-heading`, 80, 40);
  p = await model(); region = p.graph.nodes.find(n => n.id === id);
  const movedA = p.graph.nodes.find(n => n.id === 'a');
  assert(movedA.position.x > originalA.x && movedA.position.y > originalA.y, 'region did not carry member');
  assert(Math.abs(movedA.position.x - region.position.x - 24) < .01);
  await pointer(`${selector(id)} .react-flow__resize-control.handle.bottom.right`, 110, 95);
  p = await model(); region = p.graph.nodes.find(n => n.id === id);
  assert.equal(region.params.sizing, 'manual', 'resize did not commit');
  const beforeDrag = { ...p.graph.nodes.find(n => n.id === 'a').position };
  await pointer(`${selector('a')} .tensor-node-label`, 450, 250);
  p = await model(); const a = p.graph.nodes.find(n => n.id === 'a');
  assert.notDeepEqual(a.position, beforeDrag, 'member drag did not run');
  const size = await bounds(selector('a')), r = await bounds(selector(id));
  assert(a.position.x + size.w <= region.position.x + r.w - 24 + 1, 'member escaped right');
  assert(a.position.y + size.h <= region.position.y + r.h - 24 + 1, 'member escaped bottom');
  assert.equal(p.graph.nodes.find(n => n.id === id).params.members, 'a');
  await pointer(`${selector(id)} .layout-region-heading`);
  await js(`(()=>{const row=[...document.querySelectorAll('.layout-region-panel .if-region-member-list label')].find(e=>e.textContent==='特征激活');row.querySelector('input').click();})()`);
  await wait(400); p = await model(); region = p.graph.nodes.find(n => n.id === id);
  assert.deepEqual(region.params.members.split(','), ['a', 'b']);
  await button('自动适配大小'); await wait(400);
  assert.equal((await model()).graph.nodes.find(n => n.id === id).params.sizing, 'auto');
  // Fit the view through the actual control rather than changing CSS transforms.
  await js(`document.querySelector('button[aria-label="适配全图"]').click()`); await wait(500);
  await pointer(`${selector('a')} .tensor-node-label`);
  await js(`(()=>{const s=document.querySelector('select[aria-label="所属分区"]');s.value='';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await wait(250); p = await model();
  assert.equal(p.graph.nodes.find(n => n.id === id).params.members, 'b', 'membership control did not release');
  const aBeforeReleaseDrag = { ...p.graph.nodes.find(n => n.id === 'a').position };
  await pointer(`${selector('a')} .tensor-node-label`, -350, -170);
  p = await model(); assert(p.graph.nodes.find(n => n.id === 'a').position.x < aBeforeReleaseDrag.x - 100, 'released node remained locked');
  await js(`(()=>{const s=document.querySelector('select[aria-label="所属分区"]');s.value=${JSON.stringify(id)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await wait(400); assert((await model()).graph.nodes.find(n => n.id === id).params.members.includes('a'));
  await js(`document.querySelector('button[aria-label="适配全图"]').click()`); await wait(500);
  await pointer(`${selector(id)} .layout-region-heading`);
  const annotationGeometry = (await model()).graph.nodes.map(n => ({ id: n.id, position: n.position }));
  await js(`(()=>{const e=document.querySelector('textarea[aria-label="分区注释"]');e.focus();e.value='提取视觉特征，输出场景信息。';e.blur();})()`);
  await js(`document.querySelector('input[aria-label="分区输出 特征激活"]').click()`);
  await wait(250);
  assert.equal((await model()).graph.nodes.find(n => n.id === id).params.note, '提取视觉特征，输出场景信息。');
  assert.equal((await model()).graph.nodes.find(n => n.id === id).params.outputs, 'b');
  assert(await js(`!!document.querySelector('${selector('b')} [data-region-output="${id}"]')`));
  assert.deepEqual((await model()).graph.nodes.map(n => ({ id: n.id, position: n.position })), annotationGeometry);
  fs.writeFileSync(path.join(directory, 'regions-dark.png'), (await web.capturePage()).toPNG());
  await js(`document.querySelector('button[aria-label="切换浅色主题"]').click()`); await wait(200);
  fs.writeFileSync(path.join(directory, 'regions-light.png'), (await web.capturePage()).toPNG());
  const expected = await model(); press('S');
  await until(() => fs.existsSync(fixture) && JSON.parse(fs.readFileSync(fixture)).graph.nodes.length === 4, 'save');
  await button('打开'); await wait(600); assert.deepEqual(await model(), expected);
  assert(sdk.validateModel(await model()).ok);
  await js(`(()=>{const api=window.tensorTrail,s=api.getModel();const r=api.patchModel(s.project,
    {operations:[{op:'duplicate_linked',source:'a',id:'alias',position:{x:600,y:600}}]},
    {mode:'draft',layout:'preserve'});if(!r.ok)throw Error(JSON.stringify(r.diagnostics));
    api.applyModel({project:r.project,expectedRevision:s.revision,mode:'draft',allowWarnings:true});})()`);
  await wait(300); await js(`document.querySelector('button[aria-label="适配全图"]').click()`); await wait(500);
  await pointer(`${selector('alias')} .tensor-node-label`);
  const beforeUnlink = await model();
  await js(`void(window.confirm=message=>{window.__unlinkWarning=message;return false;})`);
  await button('解除值关联'); assert.deepEqual(await model(), beforeUnlink);
  assert((await js('window.__unlinkWarning')).includes('输入数量增加 1'));
  await js('void(window.confirm=()=>true)'); await button('解除值关联');
  assert.equal((await model()).graph.nodes.filter(n => n.type === 'Input' && !n.valueSource).length, 2);
  press('Z'); await wait(300); assert.deepEqual(await model(), beforeUnlink);
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ ok: true, directory, cases: ['create-from-selection', 'group-drag', 'manual-resize', 'member-clamp', 'add-member', 'auto-fit', 'release-and-move', 'reassign', 'region-note', 'region-output-badge', 'themes', 'save-reopen', 'input-unlink-confirmation'] }));
  clearTimeout(timer); app.exit(0);
}
run().catch(async error => {
  console.error(error, directory);
  if (web && !web.isDestroyed()) {
    fs.writeFileSync(path.join(directory, 'failure.png'), (await web.capturePage()).toPNG());
    fs.writeFileSync(path.join(directory, 'failure.json'), JSON.stringify(await web.executeJavaScript('window.tensorTrail.getModel()'), null, 2));
  }
  clearTimeout(timer); app.exit(1);
});
