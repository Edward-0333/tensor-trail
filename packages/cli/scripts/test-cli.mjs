import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensor-trail-cli-'));
const model = path.join(temp, 'model.json');
const next = path.join(temp, 'next.json');
function run(args, expected = 0, input) {
  const result = spawnSync(process.execPath, ['packages/cli/scripts/agent.mjs', ...args], {
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
  const structured = run(['build', 'examples/structured-input.recipe.json']);
  assert.equal(structured.outputs.output.kind, 'forward');
  assert.equal(run(['validate', '-'], 0, JSON.stringify(structured.project)).ok, true);
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
  const draftFile = path.join(temp, 'draft.json');
  const draftNext = path.join(temp, 'draft-next.json');
  const partial = {
    name: 'partial',
    nodes: [{ id: 'x', type: 'Input', position: { x: 71.5, y: -20 } }],
    connections: [],
  };
  run(['build', '-', '--out', draftFile], 1, JSON.stringify(partial));
  await assert.rejects(fs.access(draftFile));
  const builtDraft = run(['build', '-', '--draft', '--out', draftFile], 0, JSON.stringify(partial));
  assert.equal(builtDraft.mode, 'draft');
  run(['validate', draftFile], 1);
  assert.equal(run(['validate', draftFile, '--draft']).ok, true);
  const addition = {
    operations: [
      { op: 'add_node', node: { id: 'new', type: 'Identity' } },
      { op: 'connect', from: 'x.out', to: 'new.in' },
    ],
  };
  const nextDraft = run(
    ['patch', draftFile, '-', '--draft', '--out', draftNext],
    0,
    JSON.stringify(addition),
  );
  assert.deepEqual(nextDraft.project.graph.nodes[0].position, { x: 71.5, y: -20 });
  assert.equal(nextDraft.mode, 'draft');
  const auto = run(
    ['patch', draftFile, '-', '--draft', '--layout', 'auto'],
    0,
    JSON.stringify(addition),
  );
  assert.notDeepEqual(auto.project.graph.nodes[0].position, { x: 71.5, y: -20 });
  const protectedBytes = await fs.readFile(draftNext, 'utf8');
  run(
    ['patch', draftNext, '-', '--draft', '--out', draftNext, '--force'],
    1,
    JSON.stringify({ operations: [{ op: 'connect', from: 'missing.out', to: 'new.in' }] }),
  );
  assert.equal(await fs.readFile(draftNext, 'utf8'), protectedBytes);
  const sidecar = structuredClone(builtDraft.project);
  sidecar.runtime = {
    active: 'run',
    snapshots: [{ id: 'run', name: 'run', file: 'runs/run.json' }],
  };
  const unloaded = run(['validate', '-', '--draft'], 0, JSON.stringify(sidecar));
  assert.ok(unloaded.diagnostics.some((d) => d.code === 'RUNTIME_NOT_LOADED'));
  assert.deepEqual(unloaded.project.runtime, sidecar.runtime);
  run(['schema', 'patchOptions']);
  run(['validate', model, '--layout', 'preserve'], 2);
  run(['describe', '--draft'], 2);
  run(['patch', draftFile, '-', '--layout', 'bad'], 2, JSON.stringify(addition));
  console.log(
    'CLI smoke passed: catalogue, schema, build, structured inputs, validate, patch, drafts, preserved/automatic layout, sidecar warnings, stdin, error exits and output preservation.',
  );
} finally {
  // Delete only the unique directory created by this test, within the OS temp folder.
  const resolved = path.resolve(temp),
    base = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(base) || !path.basename(resolved).startsWith('tensor-trail-cli-'))
    throw new Error('Unexpected test cleanup path');
  await fs.rm(resolved, { recursive: true });
}
