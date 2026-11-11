import { describe, expect, it } from 'vitest';
import { buildModel, validateModel } from '../src/agent';
import { parseProject, serializeProject } from '../src/core/serialization';
import { tensorValue, type Params } from '../src/core/types';
import { registry } from '../src/modules';
import { agentCompletionCode } from '../src/modules/agentCompletion';

describe('LearnedParameter source', () => {
  it('keeps the legacy normal initialization when no source is specified', () => {
    const node = registry.create('LearnedParameter');
    delete node.params.initialization;
    delete node.params.source_note;
    const result = registry.get(node.type)!.infer!({}, node.params);
    expect(result.outputs.out.shape).toEqual([1, 1, 128]);
    expect(result.notes!.join()).toContain('std=0.02');
    expect(agentCompletionCode(node)).toContain('nn.init.normal_');
  });

  it('describes an existing weight without inventing initialization or grad mode', () => {
    const node = registry.create('LearnedParameter', {
      shape: '1040,3',
      initialization: 'existing',
      source_note: 'OPUSHead._init_layers: nn.Embedding(1040,3); x~U(0,1.1), y/z~U(0,1).',
    });
    const result = registry.get(node.type)!.infer!({}, node.params);
    expect(result.outputs.out.shape).toEqual([1040, 3]);
    expect(result.outputs.out).not.toHaveProperty('autograd');
    expect(result.notes!.join()).not.toContain('std=0.02');
    const code = agentCompletionCode(node)!;
    expect(code).toContain('out = existing_parameter');
    expect(code).toContain('OPUSHead._init_layers');
    expect(code).not.toContain('nn.init.normal_');
    expect(code).not.toContain('torch.empty');
  });

  it.each([1, 2, 3])('matches SparseWorld indexing and repeat with B=%i', (batch) => {
    const result = buildModel({
      name: 'initial query points',
      nodes: [
        { id: 'image', type: 'Input', params: { shape: `${batch},30,256,64,176`, axes: '' } },
        { id: 'size', type: 'TensorSize', params: { axes: '0', names: 'B' } },
        {
          id: 'weight',
          type: 'LearnedParameter',
          params: { shape: '1040,3', initialization: 'existing' },
        },
        { id: 'batch_axis', type: 'Unsqueeze', params: { dim: 0 } },
        { id: 'point_axis', type: 'Unsqueeze', params: { dim: 2 } },
        { id: 'repeat', type: 'Repeat', params: { repeats: 'd0,1,1,1' } },
        { id: 'out', type: 'Output' },
      ],
      connections: [
        ['image.out', 'size.in'],
        ['weight.out', 'batch_axis.in'],
        ['batch_axis.out', 'point_axis.in'],
        ['point_axis.out', 'repeat.in'],
        ['size.dim0', 'repeat.d0'],
        ['repeat.out', 'out.in'],
      ],
    });
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(tensorValue(result.nodes.find((n) => n.path[0] === 'point_axis')!.outputs.out).shape).toEqual([
      1, 1040, 1, 3,
    ]);
    expect(tensorValue(result.outputs.out).shape).toEqual([batch, 1040, 1, 3]);
    expect(result.outputs.out).not.toHaveProperty('autograd');
    expect(validateModel(parseProject(serializeProject(result.project!), registry)).ok).toBe(true);
  });

  it.each<Params>([{ initialization: 'invalid' }, { source_note: 1 }, { source_note: 'x'.repeat(2000) }])(
    'rejects invalid source settings %j',
    (params) => {
      const node = registry.create('LearnedParameter', params);
      expect(() => registry.get(node.type)!.infer!({}, node.params)).toThrow();
    },
  );
});
