const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const stage = process.argv.includes('--baseline') ? 'baseline' : 'verified';
const directory = path.join(root, 'test-results', `experience-${stage}-${Date.now()}`);
fs.mkdirSync(path.join(directory, 'fixtures'), { recursive: true });
const projectArgument = process.argv.indexOf('--project');
if (
  projectArgument >= 0 &&
  (!process.argv[projectArgument + 1] || process.argv[projectArgument + 1].startsWith('--'))
)
  throw Error('Usage: test-desktop-experience.cjs [--project path/to/project.json] [--baseline]');
const original = path.resolve(
  root,
  projectArgument >= 0 ? process.argv[projectArgument + 1] : 'data/pluto.tensor-trail.json',
);
if (!fs.existsSync(original))
  throw Error(
    'Choose a local test project with --project path/to/project.json. The source is only read; all editing uses a copy.',
  );
const bytes = fs.readFileSync(original);
const digest = (data) => crypto.createHash('sha256').update(data).digest('hex');
fs.writeFileSync(path.join(directory, 'fixtures/user-project.json'), bytes);
const results = [];
try {
  for (const scale of [1, 1.25, 1.5]) {
    const env = {
      ...process.env,
      TENSOR_TRAIL_PROFILE_DIR: path.join(directory, `profile-${scale}`),
      TENSOR_TRAIL_EXPERIENCE_DIR: directory,
      TENSOR_TRAIL_EXPERIENCE_SCALE: String(scale),
      TENSOR_TRAIL_EXPERIENCE_STAGE: stage,
    };
    delete env.ELECTRON_RUN_AS_NODE;
    delete env.TENSOR_TRAIL_DEV_URL;
    const result = spawnSync(
      require('electron'),
      [path.join(root, 'scripts/desktop-experience.cjs')],
      {
        cwd: root,
        env,
        stdio: 'inherit',
        windowsHide: false,
        timeout: 120000,
      },
    );
    if (result.error) throw result.error;
    if (result.status !== 0)
      throw Error(`Experience test failed at scale ${scale}: ${result.status}`);
    const processIds = JSON.parse(
      fs.readFileSync(path.join(directory, `processes-${scale}.json`), 'utf8'),
    );
    for (const pid of processIds) {
      let alive = false;
      for (let attempt = 0; attempt < 10; attempt++) {
        try {
          process.kill(pid, 0);
          alive = true;
        } catch {
          alive = false;
          break;
        }
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
      }
      if (alive) throw Error(`Experience test left child process ${pid} running`);
    }
    results.push(JSON.parse(fs.readFileSync(path.join(directory, `result-${scale}.json`), 'utf8')));
  }
} finally {
  if (digest(fs.readFileSync(original)) !== digest(bytes))
    throw Error(
      'Original user project changed during testing; no original file is restored or overwritten.',
    );
}
const summary = {
  stage,
  directory,
  userFileUnchanged: true,
  userFileSha256: digest(bytes),
  results,
};
fs.writeFileSync(path.join(directory, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(
  JSON.stringify({
    stage,
    directory,
    userFileUnchanged: true,
    layoutIssues: results.flatMap((r) => r.layout.flatMap((l) => l.issues)),
    timings: results.map((r) => ({ scale: r.scale, performance: r.performance })),
  }),
);
