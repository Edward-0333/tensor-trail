// Uses the built desktop renderer, including its real file loader and React Flow canvas.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { parseArgs } = require('node:util');
const { setTimeout: delay } = require('node:timers/promises');

const usage = 'node scripts/export-preview.cjs --project <file.tensor-trail.json> --out <out.png> [--path sub1/sub2] [--width 1920] [--height 1080] [--theme light|dark]';

function options(args) {
  const { values } = parseArgs({ args, options: {
    project: { type: 'string' }, out: { type: 'string' }, path: { type: 'string' },
    width: { type: 'string', default: '1920' }, height: { type: 'string', default: '1080' },
    theme: { type: 'string', default: 'light' },
  } });
  if (!values.project || !values.out || !['light', 'dark'].includes(values.theme)) throw Error(usage);
  const project = fs.realpathSync(values.project), out = path.resolve(values.out);
  if (!fs.statSync(project).isFile() || path.extname(out).toLowerCase() !== '.png') throw Error(usage);
  if (fs.existsSync(out)) {
    const input = fs.statSync(project), target = fs.statSync(out);
    if (input.dev === target.dev && input.ino === target.ino) throw Error('PNG destination is the input project.');
  }
  const levels = values.path === undefined ? [] : values.path.split('/');
  if (!levels.every(id => /^[a-zA-Z0-9_-]{1,100}$/.test(id))) throw Error('Expected slash-separated subgraph node IDs.');
  function pixels(value) {
    if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 2147483647)
      throw Error('Viewport dimensions must be positive 32-bit integers.');
    return Number(value);
  }
  return { project, out, levels, width: pixels(values.width), height: pixels(values.height), theme: values.theme };
}

async function until(check, label) {
  const deadline = Date.now() + 30000;
  do {
    if (await check()) return;
    await delay(50);
  } while (Date.now() < deadline);
  throw Error(`Timed out waiting for ${label}.`);
}

async function capture({ project, out, levels, width, height, theme }, profile) {
  const { app, BrowserWindow, dialog } = require('electron');
  process.env.TENSOR_TRAIL_PROFILE_DIR = profile;
  process.env.TENSOR_TRAIL_OFFSCREEN = '1';
  delete process.env.TENSOR_TRAIL_DEV_URL;
  fs.writeFileSync(path.join(profile, 'settings.json'), JSON.stringify({ theme }));
  app.commandLine.appendSwitch('force-device-scale-factor', '1');
  app.disableHardwareAcceleration();
  dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [project] });
  dialog.showErrorBox = (title, message) => console.error(title, message);
  require('../apps/desktop/dist/main/index.cjs');

  await app.whenReady();
  await until(() => BrowserWindow.getAllWindows().length === 1, 'the desktop window');
  const win = BrowserWindow.getAllWindows()[0], web = win.webContents;
  win.setMinimumSize(1, 1);
  win.setContentSize(width, height);
  const js = code => web.executeJavaScript(code);
  await until(() => !web.isLoadingMainFrame() && js('!!window.tensorTrail && !!document.querySelector(".topbar")').catch(() => false), 'the editor');
  await js(`Array.from(document.querySelectorAll('.topbar button')).find(b => b.textContent.trim() === '打开').click()`);
  await until(async () => {
    const state = await js(`({ file: document.querySelector('.save-state[data-status="saved"]')?.title,
      message: document.querySelector('.status-message')?.textContent })`);
    if (state.message?.includes('打开失败')) throw Error(state.message);
    return state.file === path.basename(project);
  }, 'the project to open');
  await until(() => js(`document.documentElement.dataset.theme === ${JSON.stringify(theme)}`), 'the requested theme');

  // Only the preview gets a full-window canvas. Keep the normal controls available for fitting.
  await web.insertCSS(`
    .center-stage { display: flex !important; }
    .canvas-region { position: fixed !important; inset: 0 !important; min-height: 0; z-index: 10000; }
    .canvas-region .canvas-caption, .canvas-region .canvas-instruction,
    .canvas-region .react-flow__panel, .status-message { visibility: hidden !important; }
  `);
  const levelSpecs = await js(`(() => {
    const spec = (graph) => ({
      ids: graph.nodes.map(node => node.id),
      minCount: graph.nodes.filter(node => node.presentation !== 'edge').length,
      maxCount: graph.nodes.length,
    });
    let graph = window.tensorTrail.getModel().project.graph;
    const specs = [spec(graph)];
    for (const id of ${JSON.stringify(levels)}) {
      graph = graph.nodes.find(node => node.id === id)?.subgraph;
      if (!graph) throw Error('No subgraph at node ' + id);
      specs.push(spec(graph));
    }
    return specs;
  })()`);
  for (const [depth, spec] of levelSpecs.entries()) {
    await until(() => js(`(() => {
      if (document.querySelectorAll('.breadcrumbs button').length !== ${depth + 1}) return false;
      const nodes = [...document.querySelectorAll('.react-flow__node')];
      const ids = new Set(${JSON.stringify(spec.ids)});
      return nodes.length >= ${spec.minCount}
        && nodes.length <= ${spec.maxCount}
        && nodes.every(n => ids.has(n.dataset.id) && getComputedStyle(n).visibility === 'visible');
    })()`), 'canvas layout');
    if (depth < levels.length) {
      const selector = `.react-flow__node[data-id="${levels[depth]}"]`;
      await js(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))`);
    }
  }
  await js(`Promise.all([document.fonts.ready, ...Array.from(document.images, img => img.decode())]).then(() => true)`);
  let previousBounds;
  await until(async () => {
    const bounds = await js(`JSON.stringify([...document.querySelectorAll('.react-flow__node')].map(node => {
      const { x, y, width, height } = node.getBoundingClientRect();
      return [node.dataset.id, x, y, width, height];
    }))`);
    const stable = bounds === previousBounds;
    previousBounds = bounds;
    return stable;
  }, 'decoded images and stable node bounds');
  await js(`document.querySelector('button[aria-label="适配全图"]').click()`);
  const painted = () => js('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve(true))))');
  await painted();
  // Interactive fit has a minimum zoom and uses node bounds. Include long manual wires in the image.
  await js(`(() => {
    const canvas = document.querySelector('.react-flow').getBoundingClientRect();
    const viewport = document.querySelector('.react-flow__viewport');
    let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
    for (const el of viewport.querySelectorAll('.react-flow__node, .react-flow__edge path:not(.edge-anchor-hover-wire[data-revealed="false"]), .edge-slide-label, .edge-op-pipeline, .edge-anchor-chip, .loop-feedback-route')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      left = Math.min(left, r.left); right = Math.max(right, r.right);
      top = Math.min(top, r.top); bottom = Math.max(bottom, r.bottom);
    }
    if (left >= canvas.left && right <= canvas.right && top >= canvas.top && bottom <= canvas.bottom) return;
    const factor = 0.94 * Math.min(canvas.width / (right - left), canvas.height / (bottom - top));
    const m = new DOMMatrix(getComputedStyle(viewport).transform);
    const x = canvas.width / 2 + factor * (m.e + canvas.left - (left + right) / 2);
    const y = canvas.height / 2 + factor * (m.f + canvas.top - (top + bottom) / 2);
    viewport.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + m.a * factor + ')';
  })()`);
  await painted();
  const png = await web.capturePage();
  const size = png.getSize();
  if (png.isEmpty() || size.width !== width || size.height !== height)
    throw Error(`Expected ${width}×${height} pixels, captured ${size.width}×${size.height}.`);
  fs.writeFileSync(out, png.toPNG());
  console.log(`${out} (${width}×${height})`);
}

async function main() {
  const scriptIndex = process.argv.findIndex(arg => path.resolve(arg) === __filename);
  const args = process.argv.slice(scriptIndex + 1);
  if (args.includes('--help')) {
    console.log(usage);
    if (process.versions.electron) require('electron').app.exit(0);
    return;
  }
  const config = options(args);
  if (!fs.existsSync(path.join(__dirname, '../apps/desktop/dist/main/index.cjs')))
    throw Error('Run npm run build:desktop first.');
  if (!process.versions.electron) {
    // Release Chromium's file handles before removing its temporary profile (especially on Windows).
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'tensor-trail-preview-'));
    const env = { ...process.env, TENSOR_TRAIL_PREVIEW_PROFILE: profile };
    delete env.ELECTRON_RUN_AS_NODE;
    try {
      const child = require('node:child_process').spawnSync(require('electron'), [__filename, ...args], { env, stdio: 'inherit' });
      if (child.error) throw child.error;
      process.exitCode = child.status ?? 1;
    } finally {
      fs.rmSync(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
    }
    return;
  }
  const { app } = require('electron');
  const profile = process.env.TENSOR_TRAIL_PREVIEW_PROFILE ?? fs.mkdtempSync(path.join(os.tmpdir(), 'tensor-trail-preview-'));
  const timeout = setTimeout(() => { console.error('Preview export timed out.'); app.exit(1); }, 120000);
  app.once('quit', () => {
    clearTimeout(timeout);
    if (!process.env.TENSOR_TRAIL_PREVIEW_PROFILE)
      fs.rmSync(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
  });
  await capture(config, profile);
  app.exit(0);
}

main().catch(error => {
  console.error(error.message);
  if (process.versions.electron) require('electron').app.exit(1);
  else process.exitCode = 1;
});
