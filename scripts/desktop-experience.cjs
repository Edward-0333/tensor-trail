const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const os = require('node:os');
const directory = process.env.TENSOR_TRAIL_EXPERIENCE_DIR;
const scale = Number(process.env.TENSOR_TRAIL_EXPERIENCE_SCALE);
if (!directory || ![1, 1.25, 1.5].includes(scale))
  throw Error('An isolated experience test is required');
app.commandLine.appendSwitch('force-device-scale-factor', String(scale));
let mayExit = false;
app.on('before-quit', (event) => {
  if (!mayExit) event.preventDefault();
});
const openAnswers = [];
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [openAnswers.shift()] });
dialog.showMessageBox = async () => ({ response: 1 });
dialog.showErrorBox = (title, content) => console.error(title, content);
const timeout = setTimeout(() => {
  console.error('Experience test timeout');
  app.exit(1);
}, 110000);
let activeWeb;
require('../apps/desktop/dist/main/index.cjs');
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(check, message) {
  const stop = Date.now() + 20000;
  while (Date.now() < stop) {
    if (await check()) return;
    await wait(40);
  }
  throw Error(message);
}
async function run() {
  await app.whenReady();
  await until(() => BrowserWindow.getAllWindows().length === 1, 'No native window');
  const window = BrowserWindow.getAllWindows()[0],
    web = window.webContents;
  activeWeb = web;
  web.on('console-message', (event) => {
    if (event.level === 'error') console.error('[renderer]', event.message);
  });
  const js = async (source) => {
    try {
      return await web.executeJavaScript(
        `try { ${source} } catch(error) { console.error(error.stack); throw error; }`,
      );
    } catch (error) {
      throw Error(error.message + '\nD5 script: ' + source.slice(0, 500));
    }
  };
  await until(
    async () =>
      !web.isLoadingMainFrame() &&
      (await js("Boolean(document.querySelector('.app-shell') && window.tensorTrail)").catch(
        () => false,
      )),
    'No shared editor',
  );
  window.focus();
  assert.ok(
    Math.abs((await js('devicePixelRatio')) - scale) < 0.03,
    'Device scale factor did not apply',
  );
  const model = () => js('window.tensorTrail.getModel().project');
  const click = (label) =>
    js(
      `(()=>{const b=[...document.querySelectorAll('button')].find(x=>x.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Unavailable button '+${JSON.stringify(label)});b.click();return true;})()`,
    );
  const key = (keyCode, modifiers = ['control']) => {
    web.sendInputEvent({ type: 'keyDown', keyCode, modifiers });
    web.sendInputEvent({ type: 'keyUp', keyCode, modifiers });
  };
  const fixture = path.join(directory, 'fixtures/user-project.json');
  const count = JSON.parse(await fs.readFile(fixture, 'utf8')).graph.nodes.length;
  const openUser = async (file = fixture, expectedCount = count) => {
    const previous = await js('window.tensorTrail.getModel().revision');
    openAnswers.push(file);
    const start = performance.now();
    await click('打开');
    await until(
      async () =>
        (await model()).graph.nodes.length === expectedCount &&
        (await js(
          `window.tensorTrail.getModel().revision!==${JSON.stringify(previous)} && document.querySelectorAll('.react-flow__node').length===${expectedCount}`,
        )),
      'Project did not render',
    );
    await js(
      'new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(true))))',
    );
    return Math.round((performance.now() - start) * 10) / 10;
  };
  const load = [];
  load.push(await openUser());
  const longFixture = path.join(
    directory,
    'fixtures',
    '用于验证窗口状态的长项目文件名称'.repeat(4) + '.tensor-trail.json',
  );
  await fs.copyFile(fixture, longFixture);
  await openUser(longFixture);
  const layout = [];
  for (const width of [window.getBounds().width, 1100, 900]) {
    window.setSize(width, Math.min(900, window.getBounds().height));
    await wait(180);
    const probe = await js(
      `(()=>{const issues=[];const header=document.querySelector('.topbar');const r=header.getBoundingClientRect();for(const b of header.querySelectorAll('button,input[aria-label="项目名称"]')){const style=getComputedStyle(b),box=b.getBoundingClientRect();if(style.display==='none'||style.visibility==='hidden'||box.width===0||box.height===0)continue;const label=b.getAttribute('aria-label')||b.textContent.trim();if(box.left<0||box.right>innerWidth+1||box.top<0||box.bottom>r.bottom+1)issues.push(label+' clipped');else if(!b.disabled && !b.contains(document.elementFromPoint(box.x+box.width/2,box.y+box.height/2)))issues.push(label+' overlapped');}return {width:innerWidth,height:innerHeight,dpr:devicePixelRatio,issues,headerOverflow:header.scrollWidth>header.clientWidth+1,bodyOverflow:document.body.scrollWidth>innerWidth+1};})()`,
    );
    layout.push({ ...probe, nativeWidth: width, title: window.getTitle() });
    if (process.env.TENSOR_TRAIL_EXPERIENCE_STAGE !== 'baseline') {
      await js("document.querySelector('.react-flow__controls-fitview').click();true");
      await wait(100);
      const clipped = await js(
        `(()=>{const r=document.querySelector('.react-flow').getBoundingClientRect();return [...document.querySelectorAll('.react-flow__node')].filter(n=>{const b=n.getBoundingClientRect();return b.left<r.left-1||b.right>r.right+1||b.top<r.top-1||b.bottom>r.bottom+1;}).length;})()`,
      );
      assert.equal(clipped, 0, 'Fit-all view clips nodes');
      assert.equal(probe.issues.length, 0, 'Top bar controls clipped or overlapped');
      assert.equal(probe.bodyOverflow, false, 'Body overflows viewport');
    }
    const image = await web.capturePage();
    await fs.writeFile(path.join(directory, `scale-${scale}-width-${width}.png`), image.toPNG());
  }
  window.setSize(
    Math.min(1440, require('electron').screen.getPrimaryDisplay().workAreaSize.width),
    Math.min(940, require('electron').screen.getPrimaryDisplay().workAreaSize.height),
  );
  await wait(150);
  await openUser();
  // Measure real mouse input on a rendered node, not only model API execution time.
  const node = await js(
    `(()=>{const canvas=document.querySelector('.react-flow').getBoundingClientRect();for(const n of document.querySelectorAll('.react-flow__node')){const r=n.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/3;if(r.width>12&&r.height>10&&x>canvas.left+30&&x<canvas.right-80&&y>canvas.top+50&&y<canvas.bottom-60&&document.elementFromPoint(x,y)?.closest('.react-flow__node')===n)return {id:n.getAttribute('data-id'),x,y};}throw Error('No unobscured node drag point');})()`,
  );
  const before = (await model()).graph.nodes.find((n) => n.id === node.id).position;
  await js(
    `window.__d5Frames=[];window.__d5Running=true;window.__d5Long=[];window.__d5Observer=new PerformanceObserver(x=>window.__d5Long.push(...x.getEntries().map(e=>e.duration)));window.__d5Observer.observe({type:'longtask',buffered:false});(()=>{let last=performance.now();function frame(now){window.__d5Frames.push(now-last);last=now;if(window.__d5Running)requestAnimationFrame(frame);}requestAnimationFrame(frame);})();true`,
  );
  await js(
    `window.__d5PointerUpPaint=null;window.addEventListener('pointerup',()=>{const start=performance.now();requestAnimationFrame(()=>requestAnimationFrame(()=>window.__d5PointerUpPaint=performance.now()-start));},{capture:true,once:true});true`,
  );
  const dragStart = performance.now();
  web.sendInputEvent({ type: 'mouseMove', x: Math.round(node.x), y: Math.round(node.y) });
  web.sendInputEvent({
    type: 'mouseDown',
    button: 'left',
    clickCount: 1,
    x: Math.round(node.x),
    y: Math.round(node.y),
  });
  for (let step = 1; step <= 20; step++) {
    web.sendInputEvent({
      type: 'mouseMove',
      button: 'left',
      modifiers: ['leftbuttondown'],
      x: Math.round(node.x + step * 3),
      y: Math.round(node.y + step * 2),
    });
    await wait(16);
  }
  web.sendInputEvent({
    type: 'mouseUp',
    button: 'left',
    clickCount: 1,
    x: Math.round(node.x + 60),
    y: Math.round(node.y + 40),
  });
  await until(async () => {
    const p = (await model()).graph.nodes.find((n) => n.id === node.id).position;
    return p.x !== before.x || p.y !== before.y;
  }, 'Node did not move through native mouse input');
  const dragMs = performance.now() - dragStart;
  await wait(100);
  const frames = await js(
    'window.__d5Running=false;window.__d5Observer.disconnect();({frames:window.__d5Frames,long:window.__d5Long})',
  );
  key('Z');
  await wait(80);
  load.push(await openUser());
  load.push(await openUser());
  const sorted = frames.frames.filter((x) => x > 0).sort((a, b) => a - b);
  const performanceResult = {
    loadMs: load,
    dragMs: Math.round(dragMs),
    pointerUpToTwoFrames: await js('window.__d5PointerUpPaint'),
    frames: sorted.length,
    frameMedian: sorted[Math.floor(sorted.length / 2)],
    frameP95: sorted[Math.floor(sorted.length * 0.95)],
    longTasks: frames.long,
  };
  const cases = ['DPI applied', 'native open user copy', 'layout screenshots', 'native node drag'];
  await fs.writeFile(
    path.join(directory, `performance-${scale}.json`),
    JSON.stringify(performanceResult, null, 2),
  );
  if (process.env.TENSOR_TRAIL_EXPERIENCE_STAGE !== 'baseline')
    cases.push(
      ...(await require('./desktop-experience-interactions.cjs')({
        window,
        web,
        directory,
        scale,
        js,
        model,
        openModel: openUser,
        until,
        wait,
        key,
      })),
    );
  const result = {
    scale,
    layout,
    performance: performanceResult,
    machine: {
      platform: process.platform,
      release: os.release(),
      cpu: os.cpus()[0].model,
      cores: os.cpus().length,
      memoryGiB: Math.round(os.totalmem() / 1024 ** 3),
    },
    cases,
  };
  await fs.writeFile(
    path.join(directory, `processes-${scale}.json`),
    JSON.stringify(app.getAppMetrics().map((x) => x.pid)),
  );
  await fs.writeFile(path.join(directory, `result-${scale}.json`), JSON.stringify(result, null, 2));
  console.log(
    JSON.stringify({
      scale,
      performance: performanceResult,
      layout: layout.map((x) => ({ width: x.width, issues: x.issues, overflow: x.headerOverflow })),
    }),
  );
  clearTimeout(timeout);
  mayExit = true;
  app.exit(0);
}
void run().catch(async (error) => {
  console.error(error.stack);
  try {
    if (activeWeb && !activeWeb.isDestroyed()) {
      await fs.writeFile(
        path.join(directory, `failure-${scale}.png`),
        (await activeWeb.capturePage()).toPNG(),
      );
      await fs.writeFile(
        path.join(directory, `failure-dom-${scale}.json`),
        JSON.stringify(
          await activeWeb.executeJavaScript(
            `({edges:[...document.querySelectorAll('[data-testid^="rf__edge-"]')].map(e=>({attributes:[...e.attributes].map(a=>[a.name,a.value]),inner:e.innerHTML.slice(0,500)})),model:window.tensorTrail?.getModel().project.graph.edges})`,
          ),
          null,
          2,
        ),
      );
    }
  } finally {
    clearTimeout(timeout);
    app.exit(1);
  }
});
