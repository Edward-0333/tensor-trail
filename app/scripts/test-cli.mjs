import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensor-trail-cli-'));
const model = path.join(temp, 'model.json');
const next = path.join(temp, 'next.json');
function run(args, expected = 0, input) {
  const result = spawnSync(process.execPath, ['scripts/agent.mjs', ...args], {
    cwd: root,
    input,
    encoding: 'utf8',
    windowsHide: true,
  });
  assert.equal(result.status, expected, result.stdout + result.stderr);
  return JSON.parse(result.stdout);
}
try {
  assert.equal(run(['describe', 'Transformer']).modules[0].type, 'Transformer');
  assert.equal(run(['schema', 'recipe']).title, 'Tensor Trail Recipe v1');
  assert.deepEqual(
    run(['build', 'examples/cnn-transformer.recipe.json', '--out', model]).outputs.output.shape,
    [8, 100, 64],
  );
  assert.equal(run(['validate', model]).ok, true);
  assert.equal(run(['patch', model, 'examples/swiglu.patch.json', '--out', next]).ok, true);
  assert.equal(
    JSON.parse(await fs.readFile(next, 'utf8')).graph.nodes.find((n) => n.id === 'encoder').params
      .ffn,
    'SwiGLU',
  );
  const unchanged = await fs.readFile(model, 'utf8');
  run(['build', 'examples/cnn-transformer.recipe.json', '--out', model], 2);
  assert.equal(await fs.readFile(model, 'utf8'), unchanged);
  const invalid = JSON.parse(
    await fs.readFile(path.join(root, 'examples/cnn-transformer.recipe.json'), 'utf8'),
  );
  invalid.nodes[0].params.shape = '8, 100, 7';
  assert.equal(
    run(['build', '-', '--out', model, '--force'], 1, JSON.stringify(invalid)).ok,
    false,
  );
  assert.equal(await fs.readFile(model, 'utf8'), unchanged);
  assert.equal(run(['validate', '-'], 0, unchanged).ok, true);
  run(['validate', '-'], 2, '{');
  run(['unknown'], 2);
  console.log(
    'CLI smoke passed: catalogue, schema, build, validate, patch, stdin, error exits and output preservation.',
  );
} finally {
  // Delete only the unique directory created by this test, within the OS temp folder.
  const resolved = path.resolve(temp),
    base = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(base) || !path.basename(resolved).startsWith('tensor-trail-cli-'))
    throw new Error('Unexpected test cleanup path');
  await fs.rm(resolved, { recursive: true });
}
