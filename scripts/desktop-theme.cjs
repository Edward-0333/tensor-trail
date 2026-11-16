// Isolated visual/interaction regression. Build the SDK and desktop before running.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'), directory = fs.mkdtempSync('/tmp/tensor-trail-theme-');
const fixture = path.join(directory, 'theme.json');
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile'); delete process.env.TENSOR_TRAIL_DEV_URL;
app.commandLine.appendSwitch('force-device-scale-factor', '1');
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
  const built = sdk.buildModel({ name: 'Workbench appearance', nodes: [
    { id: 'source', type: 'Input', params: { shape: '2,256', axes: 'B,C' }, position: { x: 0, y: 0 } },
    { id: 'layer', type: 'ReLU', position: { x: 400, y: 0 } },
    { id: 'result', type: 'Output', position: { x: 800, y: 0 } },
  ], connections: [['source.out', 'layer.in'], ['layer.out', 'result.in']] });
  assert(built.ok, JSON.stringify(built.diagnostics)); fs.writeFileSync(fixture, JSON.stringify(built.project));
  assert(fs.readFileSync(path.join(root, 'apps/desktop/dist/renderer/assets/Figtree-OFL.txt'), 'utf8').includes('SIL OPEN FONT LICENSE'));
  await app.whenReady(); await until(() => BrowserWindow.getAllWindows().length === 1, 'window');
  const win = BrowserWindow.getAllWindows()[0]; web = win.webContents;
  win.setMinimumSize(360, 580); win.setSize(1600, 1000); win.setIgnoreMouseEvents(true);
  const errors = []; web.on('console-message', e => { if (e.level === 'error') errors.push(e.message); });
  const js = s => web.executeJavaScript(s), model = () => js('window.tensorTrail.getModel().project');
  await until(async () => !web.isLoadingMainFrame() && await js('!!window.tensorTrail').catch(() => false), 'ready');
  const font = await js(`document.fonts.load('400 12px Figtree').then(f=>({count:f.length,loaded:f.every(x=>x.status==='loaded')}))`);
  assert(font.count > 0 && font.loaded, 'Local Figtree font failed to load');
  const fontFile = fs.readdirSync(path.join(root, 'apps/desktop/dist/renderer/assets')).find(f => /^Figtree-variable-.*\.woff2$/.test(f));
  assert(fontFile, 'Missing bundled font');
  assert.equal(await js(`fetch('tensor-trail://app/assets/${fontFile}').then(r=>r.status)`), 200);
  await js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='打开').click()`);
  await until(() => js(`document.querySelectorAll('.react-flow__edge').length===2`), 'project');
  const original = await model();
  const tensorStyle = () => js(`(()=>{const e=document.querySelector('.react-flow__node[data-id="source"] .tensor-node'),s=getComputedStyle(e);
    return {background:s.backgroundColor,font:s.fontFamily,width:e.offsetWidth,height:e.offsetHeight};})()`);
  const originalTensor = await tensorStyle(), layouts = [];
  for (const theme of ['dark', 'light']) {
    if (await js('document.documentElement.dataset.theme') !== theme)
      await js(`document.querySelector('button[aria-label="${theme === 'light' ? '切换浅色主题' : '切换深色主题'}"]').click()`);
    await wait(100);
    assert.deepEqual(await tensorStyle(), originalTensor, 'Theme changed tensor fill/type metrics');
    for (const width of [1600, 1366, 1100, 900, 760, 390]) {
      win.setSize(width, 900); await wait(180);
      const layout = await js(`(()=>{const header=document.querySelector('.topbar'),issues=[];
        if(document.documentElement.scrollWidth>innerWidth)issues.push('page overflow');
        if(innerWidth>=851)for(const e of header.querySelectorAll('button,input')){
          const r=e.getBoundingClientRect();if(!r.width||!r.height)continue;
          if(r.left<0||r.right>innerWidth||r.bottom>header.getBoundingClientRect().bottom)issues.push((e.getAttribute('aria-label')||e.textContent)+' clipped');
          else if(!e.disabled&&!e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)))issues.push('overlapping control');
        }
        return {width:innerWidth,issues};})()`);
      layouts.push({theme,...layout});
      assert.deepEqual(layout.issues, [], `Invalid ${theme} layout at ${width}px`);
      fs.writeFileSync(path.join(directory, `${theme}-${width}.png`), (await web.capturePage()).toPNG());
    }
  }
  win.setSize(1600, 1000); await wait(200);
  // Keyboard-accessible tabs must retain their behavior after the visual change.
  await js(`document.querySelector('#inspector-tab-node').focus()`);
  web.sendInputEvent({type:'keyDown',keyCode:'Right'}); web.sendInputEvent({type:'keyUp',keyCode:'Right'});
  await until(() => js(`document.querySelector('#inspector-tab-hyperparameters').getAttribute('aria-selected')==='true'`), 'tab keyboard navigation');
  await js(`document.querySelector('#inspector-tab-node').click()`);
  await js(`document.querySelector('.level-inputs-bar [data-input-id="source"]').click()`); await wait(300);
  assert(await js(`!!document.querySelector('.react-flow__node[data-id="source"].selected')`));
  assert.deepEqual(await model(), original, 'Theme or tabs modified the graph');
  assert.equal(fs.readFileSync(fixture, 'utf8'), JSON.stringify(built.project));
  assert.deepEqual(errors.filter(e => !e.includes('ResizeObserver')), []);
  console.log(JSON.stringify({ok:true,directory,layouts,cases:['offline-font','bundled-license','responsive-themes','unchanged-tensor-semantics','keyboard-tabs','input-focus','unchanged-project']}));
  clearTimeout(timer);app.exit(0);
}
run().catch(async error => {
  console.error(error,directory);
  try {
    if(web&&!web.isDestroyed())fs.writeFileSync(path.join(directory,'failure.png'),(await web.capturePage()).toPNG());
  } finally { clearTimeout(timer);app.exit(1); }
});
