// Run with Electron. All edits and screenshots use an isolated temporary project/profile.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tensor-trail-appearance-'));
process.env.TENSOR_TRAIL_PROFILE_DIR = path.join(directory, 'profile');
delete process.env.TENSOR_TRAIL_DEV_URL;
const fixture = path.join(directory, 'colors.tensor-trail.json');
const input = (id, x, y, grad) => ({
  id,
  label: id,
  type: 'Input',
  definitionVersion: 1,
  position: { x, y },
  params: { shape: '2,1,21', axes: 'batch,ego,feature', dtype: 'float32' },
  ...(grad === undefined ? {} : { tensorAutograd: { requiresGrad: grad, source: 'manual' } }),
});
fs.writeFileSync(
  fixture,
  JSON.stringify({
    schemaVersion: 1,
    name: 'Gradient appearance test',
    variants: [],
    graph: {
      nodes: [
        input('a', 0, 0, true),
        input('b', 300, 0, true),
        input('off', 0, 200, false),
        input('unknown', 300, 200),
      ],
      edges: [],
    },
  }),
);
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [fixture] });
dialog.showSaveDialog = async () => ({ canceled: false, filePath: fixture });
dialog.showMessageBox = async () => ({ response: 1 });
dialog.showErrorBox = (title, body) => console.error(title, body);
const timeout = setTimeout(() => {
  console.error('Appearance test timeout');
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
  const window = BrowserWindow.getAllWindows()[0];
  const web = window.webContents;
  const js = async (code) => {
    try {
      return await web.executeJavaScript(code);
    } catch (error) {
      throw Error(error.message + '\nScript: ' + code);
    }
  };
  await until(
    async () => !web.isLoadingMainFrame() && (await js('!!window.tensorTrail').catch(() => false)),
    'No editor',
  );
  await until(async () => await js('!!document.querySelector(".app-shell")'), 'No mounted editor');
  window.setSize(1440, 900);
  const click = (text) =>
    js(
      `(()=>{const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(text)}); if(!button)throw Error('No button'); button.click();})()`,
    );
  await click('打开');
  await until(
    async () => await js('document.querySelectorAll(".react-flow__node").length===4'),
    'No fixture',
  );
  const style = (id) =>
    js(
      `(()=>{const n=document.querySelector('[data-testid="node-${id}"]');return {bg:getComputedStyle(n).backgroundColor,border:getComputedStyle(n).borderTopColor,text:n.textContent,ink:getComputedStyle(n.querySelector('.tensor-node-shape code')).color};})()`,
    );
  assert.equal((await style('a')).bg, (await style('b')).bg);
  assert.notEqual((await style('a')).bg, (await style('off')).bg);
  assert.notEqual((await style('off')).bg, (await style('unknown')).bg);
  assert.match((await style('a')).text, /grad:on/);
  assert.match((await style('off')).text, /grad:off/);
  assert.match((await style('unknown')).text, /grad:\?/);
  await click('张量配色');
  const change = async (label, value, kind = 'input') => {
    await js(
      `(()=>{const e=document.querySelector(${JSON.stringify(`[aria-label="${label}"]`)});if(!e)throw Error('Missing field');Object.getOwnPropertyDescriptor(${kind === 'select' ? 'HTMLSelectElement' : 'HTMLInputElement'}.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`,
    );
  };
  await change('grad:on 统一背景色', '#123456');
  await until(async () => (await style('a')).bg === 'rgb(18, 52, 86)', 'Palette did not repaint');
  assert.equal((await style('b')).bg, 'rgb(18, 52, 86)');
  assert.equal((await style('a')).ink, 'rgb(255, 255, 255)');
  assert.notEqual((await style('off')).bg, 'rgb(18, 52, 86)');
  const savedPalette = await js('window.tensorTrail.getModel().project.tensorPalette');
  assert.equal(savedPalette.on, '#123456');
  await fs.promises.writeFile(
    path.join(directory, 'palette.png'),
    (await web.capturePage()).toPNG(),
  );
  await js(
    `document.querySelector('[data-testid="node-a"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`,
  );
  await click('节点属性');
  await until(
    async () => await js(`!!document.querySelector('[aria-label="自定义 Tensor 边框色"]')`),
    'No border field',
  );
  await change('自定义 Tensor 边框色', '#ff8800');
  await until(
    async () => (await style('a')).border === 'rgb(255, 136, 0)',
    'Border did not repaint',
  );
  assert.equal((await style('a')).bg, 'rgb(18, 52, 86)');
  assert.notEqual((await style('b')).border, 'rgb(255, 136, 0)');
  const before = (await style('off')).bg;
  await change('自动求导状态标注', 'false', 'select');
  await until(async () => (await style('a')).bg === before, 'Annotation did not change background');
  assert.match((await style('a')).text, /grad:off/);
  assert.equal((await style('a')).border, 'rgb(255, 136, 0)');
  await click('保存');
  await until(
    () => JSON.parse(fs.readFileSync(fixture, 'utf8')).tensorPalette?.on === '#123456',
    'Not saved',
  );
  const saved = JSON.parse(fs.readFileSync(fixture, 'utf8'));
  assert.equal(saved.graph.nodes[0].tensorBorderColor, '#ff8800');
  assert.equal(saved.graph.nodes[0].tensorAutograd.requiresGrad, false);
  await click('打开');
  await until(async () => (await style('a')).bg === before, 'Not restored');
  await fs.promises.writeFile(
    path.join(directory, 'reopened.png'),
    (await web.capturePage()).toPNG(),
  );
  console.log(
    JSON.stringify({
      pass: true,
      directory,
      checks: [
        'palette repaint all matching tensors',
        'unknown distinct from off',
        'contrast',
        'independent border',
        'annotation',
        'save/reopen',
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
