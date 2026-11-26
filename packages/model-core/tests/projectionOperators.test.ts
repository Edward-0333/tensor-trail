import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { buildModel, describeModules, getSchemas, patchModel, validateModel } from '../src/agent';
import type { Recipe } from '../src/agent/types';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Params, type Project, type RuntimeSnapshot } from '../src/core/types';
import { registry } from '../src/modules';
import { linspaceCode } from '../src/modules/linspace';
import { elementwiseCode } from '../src/modules/elementwise';

const floating = ['float16', 'bfloat16', 'float32', 'float64'];
const linspaceDtypes = [...floating, 'int32', 'int64'];
const off = { requiresGrad: false, source: 'inferred' };
const manual = (requiresGrad: boolean) => ({ requiresGrad, source: 'manual' as const });
const linspace = (params: Params) =>
  registry.get('Linspace')!.infer!({}, registry.create('Linspace', params).params).outputs.out;

function sequenceRecipe(params: Params = {}): Recipe {
  return {
    name: 'Sequence and lower bound',
    nodes: [
      { id: 'lower', type: 'Input', params: { shape: '1', axes: 'step' } },
      { id: 'z', type: 'Linspace', params }, { id: 'max', type: 'Maximum' }, { id: 'out', type: 'Output' },
    ],
    connections: [['lower.out', 'max.a'], ['z.out', 'max.b'], ['max.out', 'out.in']],
  };
}

function expectInvalidSequence(params: Params, parameter: string) {
  expect(() => linspace(params)).toThrow(parameter);
  expect(linspaceCode(registry.create('Linspace', params))).not.toContain('out =');
  const built = buildModel(sequenceRecipe(params));
  expect(built.ok).toBe(false);
  expect(built.diagnostics.filter(d => d.code !== 'UPSTREAM_BLOCKED')).toEqual([
    expect.objectContaining({ code: 'INVALID_PARAMETER', severity: 'error', path: ['z'], parameter }),
  ]);
}

function coordinates(dtype = 'float32'): Recipe {
  const params = { shape: '4', axes: 'depth', dtype };
  return {
    name: 'Projection coordinates',
    nodes: [
      { id: 'a', type: 'Input', params }, { id: 'b', type: 'Input', params },
      { id: 'samples', type: 'Linspace', params: { axis: 'depth', dtype }, presentation: 'compact' },
      { id: 'ones', type: 'OnesLike', presentation: 'edge' }, { id: 'zeros', type: 'ZerosLike', presentation: 'compact' },
      { id: 'max', type: 'Maximum', presentation: 'compact' }, { id: 'out', type: 'Output' },
    ],
    connections: [['a.out', 'max.a'], ['b.out', 'max.b'], ['a.out', 'ones.in'],
      ['ones.out', 'zeros.in'], ['max.out', 'out.in']],
  };
}

function snapshot(project: Project, gradMode: string, dtype: string): RuntimeSnapshot {
  project.runtime = { active: 'run', snapshots: [{ id: 'run', name: 'projection', file: 'runs/run.json' }] };
  return {
    format: 'tensor-trail.runtime-snapshot', version: 1, id: 'run', name: 'projection',
    capturedAt: '2026-11-26T00:00:00Z', context: { gradMode },
    bindings: [{ nodeType: 'Input', nodePath: ['a'], value: {
      shape: [4], axes: ['depth'], dtype, autograd: { requiresGrad: dtype === 'float32', source: 'runtime' },
    } }],
  };
}

describe('projection coordinate operators', () => {
  it.each(linspaceDtypes)('creates an independent 1D %s sequence for all valid step counts', dtype => {
    for (const steps of [0, 1, 4, Number.MAX_SAFE_INTEGER]) {
      expect(linspace({ start: ' +2.e1 ', end: '-.5E-2', steps, axis: ' z ', dtype }))
        .toEqual({ shape: [steps], axes: ['z'], dtype, autograd: off });
    }
    expect(linspace({ axis: '  ' }).axes).toEqual(['step']);
  });

  it.each(['start', 'end'])('rejects invalid %s without generating code or a tensor', field => {
    for (const value of ['', ' ', 'NaN', '+Infinity', '-Infinity', '1e309', '0x10', '1+2', '3e', '1_000']) {
      expectInvalidSequence({ [field]: value }, field);
    }
  });

  it.each([
    { steps: -1 }, { steps: 1.2 }, { steps: '4' }, { steps: 2 ** 53 }, { dtype: 'bool' },
    { dtype: 'uint8' }, { dtype: 'float8' }, { axis: 'z'.repeat(65) },
  ] as Params[])('rejects invalid fields %o directly and through the agent', params => {
    expectInvalidSequence(params, Object.keys(params)[0]);
  });

  it.each(['start', 'end'])('binds %s before checking whether its value will be used', field => {
    for (const value of ['100000000000000000000', '18446744073709551616', '-9223372036854775809', '1e309']) {
      for (const steps of [0, 1, 2])
        expectInvalidSequence({ [field]: value, dtype: 'float64', steps }, field);
    }
    expect(buildModel(sequenceRecipe({ [field]: '100000000000000000000.0', steps: 0 })).ok).toBe(true);
    expectInvalidSequence({ start: '1e5', end: '100000000000000000000', steps: 2, dtype: 'float16' }, 'end');
  });

  it.each(['9223372036854775808', '18446744073709551615'])('binds unsigned integer literal %s before target dtype conversion', value => {
    for (const params of [
      { start: value, end: '0', steps: 2, dtype: 'float64' },
      { start: '0', end: value, steps: 2, dtype: 'float64' },
      { start: '0', end: value, steps: 1, dtype: 'int32' },
      { start: value, end: value, steps: 0, dtype: 'int64' },
    ]) {
      expect(buildModel(sequenceRecipe(params))).toMatchObject({ ok: true, diagnostics: [] });
      expect(linspaceCode(registry.create('Linspace', params))).toContain(
        `out = torch.linspace(${params.start}, ${params.end}, ${params.steps}, dtype=torch.${params.dtype})`,
      );
    }
    expectInvalidSequence({ start: value, steps: 1, dtype: 'int64' }, 'start');
    expectInvalidSequence({ start: value, steps: 2, dtype: 'int64' }, 'start');
    expectInvalidSequence({ end: value, steps: 2, dtype: 'int64' }, 'end');
  });

  it.each([
    ['float16', '-65504', '65504', '-65504.1', '1e5'],
    ['bfloat16', '-3.3895313892515355e38', '3.3895313892515355e38', '-3.38954e38', '3.38954e38'],
    ['float32', '-3.4028234663852886e38', '3.4028234663852886e38', '-3.4028235e38', '3.4028235e38'],
    ['int32', '-2147483648', '2147483647', '-2147483648.5', '2147483647.5'],
    ['int64', '-9223372036854775808', '9223372036854775807', '-9.223372036854778e18', '9223372036854775807.0'],
  ])('checks exactly the endpoints consumed by %s at steps 0, 1 and 2', (dtype, min, max, below, above) => {
    expect(buildModel(sequenceRecipe({ start: min, end: max, steps: 2, dtype }))).toMatchObject({ ok: true, diagnostics: [] });
    for (const value of [below, above]) {
      expect(linspace({ start: value, end: value, steps: 0, dtype }).shape).toEqual([0]);
      expect(linspace({ start: min, end: value, steps: 1, dtype }).shape).toEqual([1]);
      expectInvalidSequence({ start: value, steps: 1, dtype }, 'start');
      expectInvalidSequence({ start: value, steps: 2, dtype }, 'start');
      expectInvalidSequence({ end: value, steps: 2, dtype }, 'end');
    }
  });

  it('preserves binary64 and in-range fractional literals in generated code', () => {
    for (const dtype of ['float64', 'int32', 'int64']) {
      const start = dtype === 'float64' ? '-1.7976931348623157e308' : '-3.9';
      const end = dtype === 'float64' ? '1.7976931348623157e308' : '3.9';
      const params = { start, end, dtype, steps: 2 };
      expect(linspaceCode(registry.create('Linspace', params))).toContain(`torch.linspace(${start}, ${end}, 2, dtype=torch.${dtype})`);
      expect(buildModel(sequenceRecipe(params)).ok).toBe(true);
    }
  });

  it('uses merged Linspace parameters for patch validation and reports the failing endpoint', () => {
    const project = buildModel(sequenceRecipe({ dtype: 'float16', steps: 1, end: '1e5' })).project!;
    const result = patchModel(project, { operations: [{ op: 'set_params', id: 'z', params: { steps: 2 } }] });
    expect(result.ok).toBe(false);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ code: 'INVALID_PARAMETER', path: ['z'], parameter: 'end' }));
    expect(validateModel(project).ok).toBe(true);
  });

  it('counts axis code points consistently in JSON Schema, inference, agent build, patch and saved projects', () => {
    const schema = describeModules().modules.find(m => m.type === 'Linspace')!.paramsSchema;
    const validateParams = new Ajv2020().compile(schema);
    for (const count of [33, 64, 65]) {
      const axis = '𠮷'.repeat(count), valid = count <= 64;
      expect(validateParams({ axis })).toBe(valid);
      const project = buildModel(sequenceRecipe()).project!;
      const node = project.graph.nodes.find(n => n.id === 'z')!;
      const patched = patchModel(project, { operations: [{ op: 'set_params', id: 'z', params: { axis } }] });
      node.params.axis = axis;
      const checked = validateModel(project);
      if (valid) {
        expect(linspace({ axis }).axes).toEqual([axis]);
        expect(buildModel(sequenceRecipe({ axis }))).toMatchObject({ ok: true, diagnostics: [] });
        expect(patched).toMatchObject({ ok: true, diagnostics: [] });
        expect(patched.project!.graph.nodes.find(n => n.id === 'z')!.params.axis).toBe(axis);
        expect(parseProject(serializeProject(project), registry)).toEqual(project);
        expect(checked.ok).toBe(true);
      } else {
        expectInvalidSequence({ axis }, 'axis');
        for (const response of [checked, patched]) {
          expect(response.ok).toBe(false);
          expect(response.diagnostics).toEqual([expect.objectContaining({ code: 'INVALID_PARAMETER', path: ['z'], parameter: 'axis' })]);
        }
        expect(() => parseProject(serializeProject(project), registry)).toThrow(/axis/);
      }
    }
  });

  it.each([false, true])('rejects Linear(3,2) after broadcasting [D] and [4] with reversed inputs %s', reversed => {
    const built = buildModel({
      name: 'Known broadcast width',
      nodes: [
        { id: 'symbol', type: 'Input', params: { shape: 'D', axes: 'features' } },
        { id: 'four', type: 'Input', params: { shape: '4', axes: 'width' } },
        { id: 'max', type: 'Maximum' }, { id: 'linear', type: 'Linear', params: { in_features: 3, out_features: 2 } },
        { id: 'out', type: 'Output' },
      ],
      connections: [['symbol.out', reversed ? 'max.b' : 'max.a'], ['four.out', reversed ? 'max.a' : 'max.b'],
        ['max.out', 'linear.in'], ['linear.out', 'out.in']],
    });
    expect(built.ok).toBe(false);
    expect(built.nodes.find(n => n.type === 'Maximum')).toMatchObject({ status: 'warning', outputs: { out: { shape: [4], axes: ['width'] } } });
    expect(built.diagnostics).toContainEqual(expect.objectContaining({ path: ['max'], code: 'SHAPE_WARNING', message: expect.stringContaining('D 为 1 或 4') }));
    expect(built.diagnostics).toContainEqual(expect.objectContaining({ path: ['linear'], code: 'SHAPE_ERROR' }));
  });

  it('generates Python using the selected literals and dtype, including integer leading zeros', () => {
    expect(linspaceCode(registry.create('Linspace', { start: ' -0002 ', end: '+6e2', steps: 1, dtype: 'int32' })))
      .toMatch(/^out = torch\.linspace\(-2, \+6e2, 1, dtype=torch\.int32\)\n#/);
    expect(elementwiseCode(registry.create('OnesLike'))).toBe('out = torch.ones_like(x)');
    expect(elementwiseCode(registry.create('Maximum'))).toMatch(/^out = torch\.maximum\(a, b\)\n#/);
  });

  it('publishes native ports, fields and presentation in both agent schemas', () => {
    const modules = describeModules().modules;
    const schemas = getSchemas();
    for (const [type, category, inputs, modes] of [
      ['Linspace', '维度操作', [], ['default', 'compact']],
      ['OnesLike', '数学运算', ['in'], ['default', 'compact', 'edge']],
      ['Maximum', '数学运算', ['a', 'b'], ['default', 'compact']],
    ] as const) {
      const def = modules.find(m => m.type === type)!;
      expect(def.category).toBe(category);
      expect(def.inputs.map(p => p.id)).toEqual(inputs);
      expect(def.outputs).toEqual([{ id: 'out', label: '输出' }]);
      for (const schema of [schemas.recipe, schemas.patch]) {
        expect(schema.$defs.node.oneOf).toContainEqual(expect.objectContaining({
          properties: expect.objectContaining({ type: { const: type }, presentation: { enum: modes }, params: def.paramsSchema }),
        }));
      }
    }
    expect(registry.create('Linspace').params).toEqual({ start: '0.0', end: '1.0', steps: 4, dtype: 'float32', axis: 'step' });
    expect(modules.find(m => m.type === 'Linspace')!.paramsSchema.properties).toMatchObject({
      dtype: { enum: linspaceDtypes }, axis: { maxLength: 64 }, steps: { minimum: 0, maximum: Number.MAX_SAFE_INTEGER },
    });
  });

  it('projects a homogeneous reference pillar through batched lidar2img matrices', () => {
    const built = buildModel({
      name: 'Reference pillar camera projection',
      nodes: [
        { id: 'z', type: 'Linspace', params: { start: '-2', end: '6', steps: 5, axis: 'sample' } },
        { id: 'xy', type: 'ZerosLike' }, { id: 'w', type: 'OnesLike' },
        { id: 'pillar', type: 'Stack', params: { dim: 0, input_count: 4 } },
        { id: 'lidar2img', type: 'Input', params: { shape: '2,6,4,4', axes: 'batch,camera,component,homogeneous' } },
        { id: 'project', type: 'MatMul' },
        { id: 'uv', type: 'TensorIndex', params: { mode: 'slice', dim: '-2', start: '0', stop: '2' } },
        { id: 'depth', type: 'TensorIndex', params: { mode: 'slice', dim: '-2', start: '2', stop: '3' } },
        { id: 'unit', type: 'OnesLike', presentation: 'edge' },
        { id: 'epsilon', type: 'Function', params: { expression: 'x * 1e-5' }, presentation: 'edge' },
        { id: 'denom', type: 'Maximum', presentation: 'compact' }, { id: 'divide', type: 'Divide' },
        { id: 'positive', type: 'TensorCompareScalar', params: { operator: 'gt', value: '1e-5' }, presentation: 'edge' },
        { id: 'image', type: 'Output' }, { id: 'mask', type: 'Output' },
      ],
      connections: [
        ['z.out', 'xy.in'], ['z.out', 'w.in'], ['xy.out', 'pillar.a'], ['xy.out', 'pillar.b'],
        ['z.out', 'pillar.c'], ['w.out', 'pillar.d'], ['pillar.out', 'project.b'], ['lidar2img.out', 'project.a'],
        ['project.out', 'uv.in'], ['project.out', 'depth.in'], ['depth.out', 'unit.in'],
        ['unit.out', 'epsilon.in'], ['depth.out', 'denom.a'], ['epsilon.out', 'denom.b'],
        ['uv.out', 'divide.a'], ['denom.out', 'divide.b'], ['divide.out', 'image.in'],
        ['depth.out', 'positive.in'], ['positive.out', 'mask.in'],
      ],
    });
    expect(built).toMatchObject({ ok: true, diagnostics: [] });
    expect(built.nodes.find(n => n.type === 'Stack')!.outputs.out).toMatchObject({ shape: [4, 5] });
    expect(built.nodes.find(n => n.type === 'MatMul')!.outputs.out).toMatchObject({ shape: [2, 6, 4, 5] });
    const restored = parseProject(serializeProject(built.project!), registry);
    expect(validateModel(restored).outputs).toMatchObject({
      image: { shape: [2, 6, 2, 5], axes: ['batch', 'camera', 'component', 'sample'], dtype: 'float32' },
      mask: { shape: [2, 6, 1, 5], dtype: 'bool' },
    });
  });

  it('preserves two value sources only for maximum across serialization and validation', () => {
    const built = buildModel(coordinates());
    expect(built.ok, JSON.stringify(built.diagnostics)).toBe(true);
    const project = built.project!;
    project.lineageEnabled = true;
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    const checked = validateModel(restored);
    expect(checked.ok).toBe(true);
    expect(checked.outputs.out.lineage?.sources.map(s => s.id)).toEqual(['a', 'b']);
    for (const [id, initializations] of [['ones', ['ones']], ['zeros', ['zeros']], ['samples', []]] as const) {
      expect(checked.nodes.find(n => n.path[0] === id)!.outputs.out.lineage)
        .toMatchObject({ origins: ['constant'], sources: [], initializations, unknown: false });
    }
    const parameter = { dtype: 'float64', axes: ['z'], shape: [4], parameterRef: { module: 'proj', name: 'weight' },
      autograd: manual(true) };
    expect(registry.get('OnesLike')!.infer!({ in: parameter }, {}).outputs.out)
      .toEqual({ dtype: 'float64', axes: ['z'], shape: [4], autograd: off });
    expect(registry.get('Maximum')!.infer!({ a: parameter, b: parameter }, {}).outputs.out).not.toHaveProperty('parameterRef');
  });

  it.each([
    [undefined, undefined, 'grad'], ['grad', undefined, 'grad'],
    ['no_grad', undefined, 'no_grad'], ['inference_mode', undefined, 'inference_mode'],
    ['grad', 'no_grad', 'no_grad'], ['grad', 'inference_mode', 'inference_mode'],
    ['inference_mode', 'grad', 'grad'], ['no_grad', 'unknown', 'no_grad'], ['unknown', undefined, 'unknown'],
  ] as const)('combines project %s and snapshot %s with manual grad annotations', (projectMode, snapshotMode, effective) => {
    for (const dtype of ['float32', 'int64']) {
      const project = buildModel(coordinates(dtype)).project!;
      if (projectMode) project.hyperparameters = { gradMode: { type: 'string', value: projectMode } };
      const bound = snapshotMode && snapshot(project, snapshotMode, dtype);
      project.graph.nodes[0].tensorAutograd = manual(dtype === 'float32');
      project.graph.nodes[1].tensorAutograd = manual(false);
      for (const override of [undefined, true, false]) {
        const ops = project.graph.nodes.filter(n => ['Maximum', 'Linspace', 'OnesLike', 'ZerosLike'].includes(n.type));
        for (const n of ops) n.tensorAutograd = override === undefined ? undefined : manual(override);
        const checked = validateModel(project, bound ? { runtimeSnapshot: bound } : {});
        expect(checked.ok, JSON.stringify(checked.diagnostics)).toBe(true);
        for (const n of ops) {
          const canEnable = dtype === 'float32' && effective === 'grad';
          const inferred = n.type !== 'Maximum' || dtype !== 'float32' ? off
            : effective === 'unknown' ? undefined : { requiresGrad: canEnable, source: 'inferred' };
          const expected = override === false || (override && canEnable) ? manual(override) : inferred;
          expect(tensorValue(checked.nodes.find(step => step.path[0] === n.id)!.outputs.out).autograd).toEqual(expected);
        }
      }
    }
  });

  it('propagates either Maximum input and retains unknown when neither input is known on', () => {
    const project = buildModel(coordinates()).project!;
    for (const a of [undefined, true, false]) {
      for (const b of [undefined, true, false]) {
        project.graph.nodes[0].tensorAutograd = a === undefined ? undefined : manual(a);
        project.graph.nodes[1].tensorAutograd = b === undefined ? undefined : manual(b);
        const grad = tensorValue(inferProject(project, registry).nodes.max.outputs.out).autograd;
        expect(grad).toEqual(a || b ? { requiresGrad: true, source: 'inferred' }
          : a === false && b === false ? off : undefined);
      }
    }
  });
});
