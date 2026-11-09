// Runs inside a real Electron main process, with its own profile and fixture files.
const { app, BrowserWindow, dialog, Menu } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const directory = process.env.TENSOR_TRAIL_TEST_DIR;
if (!directory || !process.env.TENSOR_TRAIL_PROFILE_DIR)
  throw Error('An isolated desktop test profile is required.');
const timeout = setTimeout(() => {
  console.error('Desktop smoke timed out');
  app.exit(1);
}, 75000);
const failures = [];
const phase = process.env.TENSOR_TRAIL_TEST_PHASE || 'full';
let opening = 0,
  choosingSave = 0,
  confirming = 0,
  mayExit = false;
const openAnswers = [],
  saveAnswers = [],
  discardAnswers = [];
const commits = [];
const originalRename = fs.rename;
fs.rename = async (from, to) => {
  const result = await originalRename(from, to);
  commits.push(to);
  return result;
};
dialog.showOpenDialog = async () => {
  opening++;
  return openAnswers.shift() || { canceled: true, filePaths: [] };
};
dialog.showSaveDialog = async () => {
  choosingSave++;
  return saveAnswers.shift() || { canceled: true };
};
dialog.showMessageBox = async (_window, options) => {
  if (options.type === 'question') {
    confirming++;
    return { response: discardAnswers.shift() ?? 2 };
  }
  return { response: 0 };
};
app.on('before-quit', (event) => {
  if (!mayExit) event.preventDefault();
});
const oldErrorBox = dialog.showErrorBox;
dialog.showErrorBox = (title, content) => {
  failures.push(`${title}: ${content}`);
  console.error(title, content);
};
require('../apps/desktop/dist/main/index.cjs');
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(check, message) {
  const stop = Date.now() + 15000;
  while (Date.now() < stop) {
    if (await check()) return;
    await wait(75);
  }
  throw Error(message + (failures.length ? '\n' + failures.join('\n') : ''));
}
async function run() {
  await app.whenReady();
  await until(() => BrowserWindow.getAllWindows().length === 1, 'No desktop window');
  const window = BrowserWindow.getAllWindows()[0];
  const web = window.webContents;
  const js = (expression) => web.executeJavaScript(expression);
  const model = () => js('window.tensorTrail.getModel().project');
  const click = (label) =>
    js(
      `(() => { const b=[...document.querySelectorAll('button')].find(x=>x.textContent.trim()===${JSON.stringify(label)}); if(!b || b.disabled) throw Error('Button unavailable: '+${JSON.stringify(label)});b.click();return true;})()`,
    );
  const rename = (name) =>
    js(
      `(() => {const x=document.querySelector('input[aria-label="项目名称"]');x.focus();x.value=${JSON.stringify(name)};x.blur();return true;})()`,
    );
  const editName = (name) =>
    js(
      `(() => {const x=document.querySelector('input[aria-label="项目名称"]');x.focus();x.value=${JSON.stringify(name)};return true;})()`,
    );
  const press = (key, modifiers = ['control']) => {
    web.sendInputEvent({ type: 'keyDown', keyCode: key, modifiers });
    web.sendInputEvent({ type: 'keyUp', keyCode: key, modifiers });
  };
  const menuClick = (label) => {
    const item = Menu.getApplicationMenu()
      .items.flatMap((x) => x.submenu?.items || [])
      .find((x) => x.label === label);
    assert.ok(item?.enabled, 'Menu item unavailable: ' + label);
    item.click(item, window, {});
  };
  const clean = () => !window.getTitle().startsWith('●');
  const savedName = async (file) =>
    fs
      .readFile(file, 'utf8')
      .then((s) => JSON.parse(s).name)
      .catch(() => null);
  web.on('console-message', (details) => {
    if (details.level === 'error') failures.push(details.message);
  });
  await until(async () => {
    if (web.isLoadingMainFrame()) return false;
    return web
      .executeJavaScript("Boolean(document.querySelector('.app-shell'))")
      .catch(() => false);
  }, 'Shared editor did not render');
  const probe = await web.executeJavaScript(`({url:location.href, title:document.title,
    bridge:Object.keys(window.tensorTrailDesktop || {}), node:typeof window.require,
    content:document.body.innerText.slice(0,1600)})`);
  await fs.writeFile(path.join(directory, 'initial-probe.json'), JSON.stringify(probe, null, 2));
  assert.equal(probe.url, process.env.TENSOR_TRAIL_DEV_URL || 'tensor-trail://app/index.html');
  assert.equal(probe.node, 'undefined');
  assert.ok(probe.bridge.includes('pickOpen'));
  assert.ok(probe.content.includes('打开'), probe.content);
  assert.equal((await model()).graph.nodes.length, 0);
  const preferences = web.getLastWebPreferences();
  assert.equal(preferences.sandbox, true);
  assert.equal(preferences.contextIsolation, true);
  assert.equal(preferences.nodeIntegration, false);
  assert.equal(await js('typeof window.process'), 'undefined');
  await until(() => window.isVisible(), 'Desktop window never became visible');
  await web.executeJavaScript('document.fonts.ready.then(() => true)');
  await wait(300);
  const image = await web.capturePage();
  await fs.writeFile(path.join(directory, `${phase}-blank.png`), image.toPNG());
  await fs.writeFile(path.join(directory, 'initial-probe.json'), JSON.stringify(probe, null, 2));
  const cases = [phase === 'dev' ? 'development-renderer' : 'local-resources', 'blank-start', 'isolated-preload', 'visible-window'];
  let closingFile;
  if (phase !== 'full') {
    assert.equal(await js('document.documentElement.dataset.theme'), 'light');
    assert.equal((await model()).graph.nodes.length, 0);
    assert.ok(window.getTitle().includes('未打开项目'));
    cases.push('persisted-theme', 'no-last-project-restoration');
  } else {
    const invalid = path.join(directory, 'invalid.json');
    await fs.writeFile(invalid, '{"schemaVersion":999}');
    const original = path.join(directory, '中文 项目.json');
    const copy = path.join(directory, '另存 副本.json');
    const recovery = path.join(directory, '恢复当前编辑.json');
    const sdk = await import(
      pathToFileURL(path.join(__dirname, '../packages/model-core/dist/tensor-trail-agent.js')).href
    );
    const built = sdk.buildModel(
      JSON.parse(
        await fs.readFile(path.join(__dirname, '../examples/cnn-transformer.recipe.json'), 'utf8'),
      ),
    );
    assert.equal(built.ok, true);
    const fixture = built.project;
    fixture.name = '原始项目';
    await fs.writeFile(original, '\uFEFF' + JSON.stringify(fixture));
    press('O');
    await until(() => opening === 1, 'Ctrl+O picker not called');
    await wait(100);
    assert.equal((await model()).graph.nodes.length, 0);
    cases.push('cancel-open');
    openAnswers.push({ canceled: false, filePaths: [invalid] });
    await click('打开');
    await until(
      async () => (await js('document.body.innerText')).includes('打开失败'),
      'Invalid file not reported',
    );
    assert.equal((await model()).graph.nodes.length, 0);
    cases.push('invalid-open-preserves-document');
    openAnswers.push({ canceled: false, filePaths: [original] });
    await click('打开');
    await until(
      async () => (await model()).name === '原始项目' && clean(),
      'Native file did not load',
    );
    assert.equal((await model()).graph.nodes.length, fixture.graph.nodes.length);
    cases.push('unicode-bom-open');
    await rename('原路径保存');
    const before = commits.filter((x) => x === original).length;
    press('S');
    await until(
      async () => (await savedName(original)) === '原路径保存' && clean(),
      'Ctrl+S did not save source',
    );
    await wait(100);
    assert.equal(commits.filter((x) => x === original).length, before + 1);
    assert.equal(choosingSave, 0);
    cases.push('ctrl-s-single-source-write');
    await editName('字段未失焦保存');
    press('S');
    await until(
      async () => (await savedName(original)) === '字段未失焦保存' && clean(),
      'Active input was omitted from Ctrl+S',
    );
    cases.push('ctrl-s-includes-focused-field');
    await rename('取消另存为');
    press('S', ['control', 'shift']);
    await until(() => choosingSave === 1, 'Ctrl+Shift+S picker not called');
    await wait(100);
    assert.equal(await savedName(original), '字段未失焦保存');
    assert.ok(!clean());
    press('S');
    await until(
      async () => (await savedName(original)) === '取消另存为' && clean(),
      'Canceled save-as changed source binding',
    );
    assert.equal(choosingSave, 1);
    cases.push('cancel-save-as-keeps-source');
    await rename('另存副本');
    saveAnswers.push({ canceled: false, filePath: copy });
    await click('另存为');
    await until(async () => (await savedName(copy)) === '另存副本' && clean(), 'Save-as failed');
    await rename('继续编辑副本');
    press('S');
    await until(
      async () => (await savedName(copy)) === '继续编辑副本' && clean(),
      'Subsequent save missed new target',
    );
    assert.equal(await savedName(original), '取消另存为');
    cases.push('save-as-switches-only-after-success');
    const external = JSON.parse(await fs.readFile(copy, 'utf8'));
    external.name = '外部更新';
    await fs.writeFile(copy, JSON.stringify(external));
    await rename('保留画布修改');
    press('S');
    await until(
      async () => (await js('document.body.innerText')).includes('其他程序修改'),
      'External conflict was not reported',
    );
    assert.equal(await savedName(copy), '外部更新');
    assert.equal((await model()).name, '保留画布修改');
    assert.ok(!clean());
    saveAnswers.push({ canceled: false, filePath: recovery });
    await click('另存为');
    await until(
      async () => (await savedName(recovery)) === '保留画布修改' && clean(),
      'Conflict recovery save-as failed',
    );
    cases.push('external-conflict-and-recovery');
    await rename('取消切换');
    discardAnswers.push(2);
    const asked = confirming;
    press('N');
    await until(() => confirming === asked + 1, 'Unsaved prompt absent');
    await wait(100);
    assert.equal((await model()).name, '取消切换');
    cases.push('cancel-new-preserves-edit');
    discardAnswers.push(0);
    await click('新建项目');
    await until(async () => (await model()).graph.nodes.length === 0, 'Save-before-new failed');
    assert.equal(await savedName(recovery), '取消切换');
    cases.push('save-before-new');
    discardAnswers.push(2);
    const closeAsk = confirming;
    window.close();
    await until(() => confirming === closeAsk + 1, 'Close prompt absent');
    await wait(100);
    assert.ok(!window.isDestroyed());
    cases.push('cancel-close');
    press('Z');
    await until(
      async () => (await model()).name === '取消切换',
      'Undo did not restore original document',
    );
    await rename('明确放弃当前修改');
    discardAnswers.push(1);
    await click('新建项目');
    await until(async () => (await model()).graph.nodes.length === 0, 'Discard-before-new failed');
    assert.equal(await savedName(recovery), '取消切换');
    press('Z');
    await until(
      async () => (await model()).name === '明确放弃当前修改',
      'Undo after discard did not restore association',
    );
    cases.push('discard-before-new-keeps-disk');
    await rename('撤销恢复后保存');
    press('S');
    await until(
      async () => (await savedName(recovery)) === '撤销恢复后保存' && clean(),
      'Undo restored wrong file association',
    );
    cases.push('undo-restores-file-association');
    await rename('菜单保存');
    menuClick('保存');
    await until(
      async () => (await savedName(recovery)) === '菜单保存' && clean(),
      'Native menu save failed',
    );
    cases.push('native-menu-save');
    const applied = await js(
      `(() => {const current=window.tensorTrail.getModel();current.project.name='AI 修改';return window.tensorTrail.applyModel({project:current.project,expectedRevision:current.revision});})()`,
    );
    assert.equal(applied.ok, true);
    const choicesBeforeAi = choosingSave;
    press('S');
    await until(
      async () => (await savedName(recovery)) === 'AI 修改' && clean(),
      'AI update lost current source association',
    );
    assert.equal(choosingSave, choicesBeforeAi);
    cases.push('ai-modification-keeps-source');
    await js(
      `(() => {const x=document.querySelector('input[aria-label="项目名称"]');x.focus();x.select();return true;})()`,
    );
    await web.insertText('输入框临时文本');
    menuClick('撤销');
    await until(
      async () =>
        (await js(`document.querySelector('input[aria-label="项目名称"]').value`)) === 'AI 修改',
      'Text undo was consumed by graph undo',
    );
    assert.equal((await model()).name, 'AI 修改');
    await js('document.activeElement.blur()');
    cases.push('native-text-undo');
    const invalidReply = await js(
      "window.tensorTrailDesktop.readFile('C:/Windows/not-a-file-token')",
    );
    assert.equal(invalidReply.ok, false);
    assert.equal(invalidReply.error.code, 'FILE_TOKEN');
    cases.push('restricted-file-tokens');
    const loaded = await web.capturePage();
    await fs.writeFile(path.join(directory, 'loaded.png'), loaded.toPNG());
    const themeToggle = await js(
      `(() => { const b=document.querySelector('button[aria-label="切换浅色主题"]');b.click();return true;})()`,
    );
    assert.ok(themeToggle);
    await until(async () => {
      try {
        return (
          JSON.parse(await fs.readFile(path.join(directory, 'profile/settings.json'), 'utf8'))
            .theme === 'light'
        );
      } catch {
        return false;
      }
    }, 'Theme did not persist');
    cases.push('native-preferences');
    const resources = await js("performance.getEntriesByType('resource').map(x=>x.name)");
    assert.ok(resources.every((x) => !/^https?:/.test(x)));
    cases.push('no-vite-server-dependency');
    const second = spawn(process.execPath, [path.join(__dirname, '../apps/desktop')], {
      env: { ...process.env },
      stdio: 'pipe',
      windowsHide: false,
    });
    let secondOutput = '';
    second.stderr.on('data', (chunk) => {
      secondOutput += chunk;
    });
    const secondCode = await new Promise((resolve, reject) => {
      const limit = setTimeout(() => {
        second.kill();
        reject(Error('Second-instance process did not exit'));
      }, 10000);
      second.once('error', (error) => {
        clearTimeout(limit);
        reject(error);
      });
      second.once('exit', (code) => {
        clearTimeout(limit);
        resolve(code);
      });
    });
    assert.equal(secondCode, 0, secondOutput);
    assert.equal(BrowserWindow.getAllWindows().length, 1);
    cases.push('single-instance');
    await editName('关闭前保存');
    discardAnswers.push(0);
    closingFile = recovery;
  }
  // The actual close handshake must destroy the window; the harness delays app exit only to write evidence.
  const beforeCloseProcesses = app.getAppMetrics().map((metric) => metric.pid);
  window.close();
  await until(() => window.isDestroyed(), 'Approved close did not destroy the window');
  if (closingFile) {
    assert.equal(await savedName(closingFile), '关闭前保存');
    cases.push('save-on-close');
  }
  const processIds = [
    ...new Set([...beforeCloseProcesses, ...app.getAppMetrics().map((metric) => metric.pid)]),
  ];
  const result = {
    passed: true,
    phase,
    url: probe.url,
    cases,
    opening,
    choosingSave,
    confirming,
    commits: commits.length,
    processIds,
  };
  await fs.writeFile(path.join(directory, `${phase}-result.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  clearTimeout(timeout);
  dialog.showErrorBox = oldErrorBox;
  mayExit = true;
  app.exit(0);
}
void run().catch((error) => {
  console.error(error.stack);
  clearTimeout(timeout);
  app.exit(1);
});
