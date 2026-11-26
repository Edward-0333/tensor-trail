import assert from 'node:assert/strict';
import { runAgent } from './test-agent-process.cjs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensor-trail-cli-'));
const model = path.join(temp, 'model.json');
const next = path.join(temp, 'next.json');
let invocations = 0;
function run(args, expected = 0, input) {
  invocations++;
  const result = runAgent(args, input);
  assert.equal(result.status, expected, result.stdout);
  return JSON.parse(result.stdout);
}
try {
  assert.equal(run(['describe', 'Transformer']).modules[0].type, 'Transformer');
  const depth = run(['build', '-'], 0, JSON.stringify({
    name: 'Projection depth bounds',
    nodes: [
      { id: 'camera', type: 'Input', params: { shape: '2,6,4', axes: 'batch,camera,z' } },
      { id: 'depth', type: 'Linspace', params: { start: '-2', end: '6', steps: 4, axis: 'z' }, presentation: 'compact' },
      { id: 'ones', type: 'OnesLike', presentation: 'edge' },
      { id: 'floor', type: 'Function', params: { expression: 'x * 1e-5' }, presentation: 'edge' },
      { id: 'max', type: 'Maximum', presentation: 'compact' },
      { id: 'valid', type: 'TensorCompareScalar', params: { operator: 'gt', value: '1e-5' }, presentation: 'edge' },
      { id: 'mask', type: 'Output' },
    ],
    connections: [['depth.out', 'ones.in'], ['ones.out', 'floor.in'], ['floor.out', 'max.b'],
      ['camera.out', 'max.a'], ['max.out', 'valid.in'], ['valid.out', 'mask.in']],
  }));
  assert.deepEqual(depth.nodes.find(n => n.type === 'Maximum').outputs.out, {
    shape: [2, 6, 4], axes: ['batch', 'camera', 'z'], dtype: 'float32',
  });
  for (const type of ['Linspace', 'OnesLike'])
    assert.deepEqual(depth.nodes.find(n => n.type === type).outputs.out.autograd, { requiresGrad: false, source: 'inferred' });
  const mask = run(['validate', '-'], 0, JSON.stringify(depth.project));
  assert.equal(mask.ok, true);
  assert.deepEqual(mask.outputs.mask, { shape: [2, 6, 4], axes: ['batch', 'camera', 'z'], dtype: 'bool' });
  const [gridSample] = run(['describe', 'GridSample']).modules;
  assert.equal(gridSample.type, 'GridSample');
  assert.deepEqual(gridSample.inputs.map(p => p.id), ['in', 'grid']);
  const sampled = run(['build', '-'], 0, JSON.stringify({
    name: 'Plan A',
    nodes: [
      { id: 'x', type: 'Input', params: { shape: '16,32,32,8', axes: 'N,C,H,W' } },
      { id: 'coords', type: 'Input', params: { shape: '16,256,4,2', axes: 'N,Q,P,xy' } },
      { id: 'sample', type: 'GridSample', presentation: 'compact' },
      { id: 'out', type: 'Output' },
    ],
    connections: [['x.out', 'sample.in'], ['coords.out', 'sample.grid'], ['sample.out', 'out.in']],
  }));
  assert.deepEqual(sampled.outputs.out.shape, [16, 32, 256, 4]);
  const sampledReport = run(['validate', '-'], 0, JSON.stringify(sampled.project));
  assert.equal(sampledReport.ok, true);
  assert.deepEqual(sampledReport.outputs.out.axes, ['N', 'C', 'Q', 'P']);
  const [normalize] = run(['describe', 'Normalize']).modules;
  assert.equal(normalize.type, 'Normalize');
  assert.equal(normalize.paramsSchema.properties.dim.default, -3);
  const normalized = run(['build', '-'], 0, JSON.stringify({
    name: 'First normalized image',
    nodes: [
      { id: 'x', type: 'Input', params: { shape: '2,3,8,16', axes: 'B,C,H,W' } },
      { id: 'norm', type: 'Normalize', presentation: 'compact' },
      { id: 'first', type: 'TensorIndex', presentation: 'edge', params: { dim: '0', index: '0' } },
      { id: 'out', type: 'Output' },
    ],
    connections: [['x.out', 'norm.in'], ['norm.out', 'first.in'], ['first.out', 'out.in']],
  }));
  assert.equal(normalized.project.graph.nodes.find(n => n.id === 'first').presentation, 'edge');
  const normalizedReport = run(['validate', '-'], 0, JSON.stringify(normalized.project));
  assert.equal(normalizedReport.ok, true);
  assert.deepEqual(normalizedReport.outputs.out, { shape: [3, 8, 16], axes: ['C', 'H', 'W'], dtype: 'float32' });
  const recipeSchema = run(['schema', 'recipe']);
  assert.equal(recipeSchema.title, 'Tensor Trail Recipe v1');
  assert.deepEqual(recipeSchema.$defs.tensorVisual.properties.mode.enum, ['off', 'block']);
  const owners = recipeSchema.$defs.node.oneOf.filter(n => n.properties.submoduleVisual);
  assert.deepEqual(owners.map(n => n.properties.type.const), ['Submodule']);
  assert.equal(owners[0].properties.submoduleVisual.$ref, '#/$defs/submoduleVisual');
  const style = recipeSchema.$defs.submoduleVisual;
  assert.equal(style.additionalProperties, false);
  assert.deepEqual(style.properties.icon.enum, ['auto', 'backbone', 'attention', 'cross_attention',
    'mlp', 'projection', 'encoder', 'decoder', 'head', 'module']);
  assert.deepEqual(style.properties.tone.enum, ['auto', 'blue', 'violet', 'teal', 'amber', 'rose', 'slate']);
  assert.equal(style.properties.summary.maxLength, 120);
  const patchSchema = run(['schema', 'patch']);
  const operation = patchSchema.properties.operations.items.oneOf.find(s => s.properties.op.const === 'set_node_submodule_visual');
  assert.deepEqual(operation.required, ['op', 'id', 'visual']);
  assert.deepEqual(operation.properties.visual.anyOf, [{ $ref: '#/$defs/submoduleVisual' }, { type: 'null' }]);
  assert.deepEqual(patchSchema.$defs.submoduleVisual, style);
  const styleFile = path.join(temp, 'submodule.json');
  const submodule = run(['build', '-', '--out', styleFile], 0, JSON.stringify({
    name: '封装模块样式', nodes: [
      { id: 'x', type: 'Input' },
      { id: 'unit', type: 'Submodule', submoduleVisual: { icon: 'encoder' }, subgraph: {
        nodes: [{ id: 'data', type: 'Inlet' }, { id: 'ret', type: 'Outlet' }], connections: [['data.out', 'ret.in']],
      } }, { id: 'y', type: 'Output' },
    ], connections: [['x.out', 'unit.in'], ['unit.out', 'y.in']],
  }));
  const styledSubmodule = run(['patch', styleFile, '-', '--layout', 'preserve'], 0, JSON.stringify({
    operations: [{ op: 'set_node_submodule_visual', id: 'unit', visual: { tone: 'rose', summary: '  投影  ' } }],
  }));
  const expectedStyle = structuredClone(submodule.project);
  expectedStyle.graph.nodes[1].submoduleVisual = { tone: 'rose', summary: '投影' };
  assert.deepEqual(styledSubmodule.project, expectedStyle);
  assert.deepEqual(styledSubmodule.outputs, submodule.outputs);
  const standard = run(['build', 'examples/standard-tensor-ops.recipe.json']);
  assert.deepEqual(standard.outputs.values.shape, ['B', 3]);
  assert.equal(standard.outputs.indices.dtype, 'int64');
  const compact = JSON.stringify(standard.project);
  const padded = compact + ' '.repeat(10_000_000 - Buffer.byteLength(compact));
  assert.equal(run(['validate', '-'], 0, padded).ok, true);
  assert.match(run(['validate', '-'], 2, padded + ' ').diagnostics[0].message, /10 MB/);
  // Exercise both CLI file directions with meaningful multi-byte JSON above the former limit.
  const large = path.join(temp, 'large.json'), renamed = path.join(temp, 'renamed.json');
  const largeProject = structuredClone(standard.project);
  for (let i = 0; i < 520; i++) {
    largeProject.graph.nodes.push({
      id: 'arg' + i, type: 'ForwardInput', definitionVersion: 1, label: '输入' + i,
      position: { x: 0, y: i * 100 }, params: { text: '图'.repeat(1990) },
    });
  }
  await fs.writeFile(large, JSON.stringify(largeProject));
  const rename = { operations: [{ op: 'rename_project', name: 'large' }] };
  assert.equal(run(['patch', large, '-', '--out', renamed], 0, JSON.stringify(rename)).ok, true);
  assert.ok((await fs.stat(renamed)).size > 3_000_000);
  assert.equal(run(['validate', renamed]).ok, true);
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
  const slidePatch = {
    operations: [
      { op: 'set_node_media', id: 'x', media: { items: [{ id: 'preview', src: 'data:image/png;base64,YQ==' }] } },
      { op: 'set_node_tensor_visual', id: 'x', visual: { mode: 'block', tone: 'teal' } },
      { op: 'set_edge_presentation', from: 'x.out', to: 'new.in', presentation: 'slide', annotation: '输入特征' },
    ],
  };
  const slides = run(['patch', draftNext, '-', '--draft'], 0, JSON.stringify(slidePatch));
  assert.deepEqual(slides.project.graph.nodes[0].visualMedia, slidePatch.operations[0].media);
  assert.deepEqual(slides.project.graph.nodes[0].tensorVisual, slidePatch.operations[1].visual);
  assert.equal(slides.project.graph.edges[0].annotation, '输入特征');
  assert.equal(slides.project.graph.edges[0].presentation, 'slide');
  const presentationFile = path.join(temp, 'presentation.json');
  const presentationPatch = { operations: [
    { op: 'set_node_presentation', id: 'new', presentation: 'edge' },
    { op: 'set_edge_presentation', from: 'x.out', to: 'new.in', presentation: 'anchor' },
  ] };
  const styled = run(['patch', draftNext, '-', '--draft', '--out', presentationFile], 0, JSON.stringify(presentationPatch));
  const loaded = run(['validate', presentationFile, '--draft']);
  assert.equal(loaded.project.graph.nodes.find(n => n.id === 'new').presentation, 'edge');
  assert.equal(loaded.project.graph.edges[0].presentation, 'anchor');
  assert.deepEqual(styled.nodes, nextDraft.nodes);
  const restored = run(['patch', presentationFile, '-', '--draft'], 0, JSON.stringify({
    operations: presentationPatch.operations.map(op => ({ ...op, presentation: null })),
  }));
  assert.deepEqual(restored.project, nextDraft.project);
  const invalidStyle = { operations: [{ op: 'set_node_presentation', id: 'new', presentation: 'compact' }] };
  run(['patch', presentationFile, '-', '--draft', '--out', presentationFile, '--force'], 1, JSON.stringify(invalidStyle));
  assert.deepEqual(JSON.parse(await fs.readFile(presentationFile, 'utf8')), loaded.project);
  const operatorPreview = { op: 'set_node_tensor_visual', id: 'new', visual: {
    mode: 'block', color: '#3b82f6', depthStyle: 'solid', width: 420,
    boxWidth: 64, boxHeight: 28, boxDepth: 40, axisLabels: ['queue', 'query', 'head', 'dim'],
  } };
  const preview = run(['patch', draftNext, '-', '--draft'], 0, JSON.stringify({ operations: [operatorPreview] }));
  assert.deepEqual(preview.project.graph.nodes.find(n => n.id === 'new').tensorVisual, operatorPreview.visual);
  assert.deepEqual(run(['validate', '-', '--draft'], 0, JSON.stringify(preview.project)).project, preview.project);
  const obsolete = { ...operatorPreview, visual: { mode: 'operation' } };
  assert.equal(run(['patch', draftNext, '-', '--draft'], 1, JSON.stringify({ operations: [obsolete] })).ok, false);
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
    `CLI smoke passed (${invocations} invocations): catalogue, schema, projection depth bounds, GridSample, Normalize, TensorIndex edges, six standard tensor ops, 10 MB stdin boundary, build, structured inputs, validate, patch, node media, submodule visuals, tensor visuals, node presentation, slide arrows, anchors, drafts, preserved/automatic layout, sidecar warnings, stdin, error exits and output preservation.`,
  );
} finally {
  // Delete only the unique directory created by this test, within the OS temp folder.
  const resolved = path.resolve(temp),
    base = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(base) || !path.basename(resolved).startsWith('tensor-trail-cli-'))
    throw new Error('Unexpected test cleanup path');
  await fs.rm(resolved, { recursive: true });
}
