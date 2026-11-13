import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { inferGraph } from '@tensor-trail/model-core/core/engine';
import { ports } from '@tensor-trail/model-core/core/graph';
import { registry } from '@tensor-trail/model-core/modules';
import type { Graph, ModelNode, Params, TensorSpec, ValueSpec } from '@tensor-trail/model-core/core/types';
import { DEFAULT_TENSOR_PALETTE } from '@tensor-trail/model-core/core/tensorAppearance';
import { finalOutputValues, FINAL_OUTPUT_TITLE } from '../src/views/finalOutputs';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { useEditor } from '../src/editor/store';

function fixture() {
  const graph: Graph = { nodes: [], edges: [] };
  const add = (type: string, id: string, params: Params = {}) => {
    const node = registry.create(type, params, id);
    graph.nodes.push(node);
    return node;
  };
  const input = (id: string) => add('Input', id, { shape: '2,3', axes: 'B,C' });
  const link = (source: string, target: string, targetPort = 'in', sourcePort = 'out') =>
    graph.edges.push({ id: `e${graph.edges.length}`, source, target, targetPort, sourcePort });
  const alias = (source: ModelNode, id: string) => {
    graph.nodes.push({ ...structuredClone(source), id, valueSource: source.id });
  };
  const infer = () => {
    const result = inferGraph(graph, registry);
    expect(result.issues).toEqual([]);
    expect(Object.values(result.nodes).filter(n => ['error', 'blocked'].includes(n.status))).toEqual([]);
    return result;
  };
  return { graph, add, input, link, alias, infer };
}

describe('Unchanged final return values', () => {
  it('traces nested packing and aliases with definition-only owners without changing the graph or specs', () => {
    const f = fixture();
    const a = f.input('a'); f.input('b');
    f.alias(a, 'a_at_return'); f.alias(a, 'a_elsewhere');
    f.add('TensorSequencePack', 'list', { kind: 'list', input_count: 2 });
    const list = f.add('ValueVariable', 'list_value'); f.alias(list, 'list_at_return');
    f.add('DictionaryPack', 'pack', { keys: 'predictions' }); f.add('Output', 'return');
    f.link('a_at_return', 'list', 'item0'); f.link('b', 'list', 'item1');
    f.link('list', 'list_value'); f.link('list_at_return', 'pack', 'predictions'); f.link('pack', 'return');
    const result = f.infer(), before = structuredClone({ graph: f.graph, result });
    expect(finalOutputValues(f.graph, result)).toEqual(new Set(['return', 'list_value', 'list_at_return', 'a', 'a_at_return', 'a_elsewhere', 'b']));
    expect(f.graph.edges.some(e => e.source === 'a' || e.source === 'list_value')).toBe(false);
    expect({ graph: f.graph, result }).toEqual(before);
  });

  it('excludes overwritten dictionary fields and partially retained dictionary variables', () => {
    const f = fixture();
    for (const id of ['old', 'kept', 'replacement']) f.input(id);
    f.add('DictionaryPack', 'base_pack', { keys: 'x,y' }); f.add('ValueVariable', 'base');
    f.add('DictionaryPack', 'new_pack', { keys: 'x' }); f.add('DictionaryUpdate', 'update'); f.add('Output', 'return');
    f.link('old', 'base_pack', 'x'); f.link('kept', 'base_pack', 'y'); f.link('base_pack', 'base');
    f.link('base', 'update', 'base'); f.link('replacement', 'new_pack', 'x');
    f.link('new_pack', 'update', 'updates'); f.link('update', 'return');
    expect(finalOutputValues(f.graph, f.infer())).toEqual(new Set(['return', 'replacement', 'kept']));
  });

  it('tracks field selection through chained shallow updates', () => {
    const f = fixture();
    for (const id of ['x0', 'x1', 'x2', 'y', 'z']) f.input(id);
    for (const [id, keys] of [['base', 'x,y'], ['updates1', 'x,z'], ['updates2', 'x']]) {
      f.add('DictionaryPack', id, { keys });
    }
    f.add('DictionaryUpdate', 'update1'); f.add('ValueVariable', 'intermediate');
    f.add('DictionaryUpdate', 'update2'); f.add('Output', 'return');
    f.link('x0', 'base', 'x'); f.link('y', 'base', 'y');
    f.link('x1', 'updates1', 'x'); f.link('z', 'updates1', 'z'); f.link('x2', 'updates2', 'x');
    f.link('base', 'update1', 'base'); f.link('updates1', 'update1', 'updates');
    f.link('update1', 'intermediate'); f.link('intermediate', 'update2', 'base');
    f.link('updates2', 'update2', 'updates'); f.link('update2', 'return');
    expect(finalOutputValues(f.graph, f.infer())).toEqual(new Set(['return', 'x2', 'y', 'z']));
  });

  it.each(['Detach', 'ReLU', 'TensorSequenceGet', 'SelectValue', 'TensorListAppend', 'Submodule'])(
    'stops at %s instead of treating all ancestors as final outputs', type => {
      const f = fixture();
      f.input('before'); f.add('ReLU', 'operation'); f.add('TensorVariable', 'after'); f.add('Output', 'return');
      f.link('before', 'operation'); f.link('operation', 'after'); f.link('after', 'return');
      const result = f.infer();
      // Boundary behavior depends on operation identity, not its particular shape contract.
      f.graph.nodes.find(n => n.id === 'operation')!.type = type;
      expect(finalOutputValues(f.graph, result)).toEqual(new Set(['return', 'after']));
    },
  );

  it('still marks a returned value that is also used by a computational branch', () => {
    const f = fixture(); f.input('x'); f.add('ReLU', 'other_use'); f.add('Output', 'return');
    f.link('x', 'other_use'); f.link('x', 'return');
    expect(finalOutputValues(f.graph, f.infer())).toEqual(new Set(['return', 'x']));
  });

  it('recomputes after a return connection changes without storing annotations', () => {
    const f = fixture(); f.input('a'); f.input('b'); f.add('Output', 'return'); f.link('a', 'return');
    expect(finalOutputValues(f.graph, f.infer()).has('a')).toBe(true);
    f.graph.edges[0].source = 'b';
    expect(finalOutputValues(f.graph, f.infer())).toEqual(new Set(['return', 'b']));
    f.graph.edges = [];
    expect(finalOutputValues(f.graph, inferGraph(f.graph, registry))).toEqual(new Set());
    expect(finalOutputValues(f.graph)).toEqual(new Set());
  });

  it('marks function outlets but not intermediate loop-state outlets', () => {
    const f = fixture(); f.input('x'); const out = f.add('Outlet', 'return', { port: 'result' }); f.link('x', 'return');
    const result = f.infer();
    expect(finalOutputValues(f.graph, result, registry.create('Submodule'))).toEqual(new Set(['return', 'x']));
    for (const type of ['ForLoop', 'WhileLoop', 'ForEachTensor']) {
      expect(finalOutputValues(f.graph, result, { ...out, type })).toEqual(new Set());
    }
    f.alias(out, 'out_copy');
    f.graph.nodes.splice(f.graph.nodes.indexOf(out), 1);
    expect(finalOutputValues(f.graph, result)).toEqual(new Set());
  });

  it('does not mark integers, device metadata, errors or incomplete return values', () => {
    const f = fixture(); f.input('x'); f.add('Output', 'return'); f.link('x', 'return');
    const result = f.infer();
    const metadata: ValueSpec[] = [{ kind: 'int', value: 2 }, { kind: 'device', reference: 'cuda:0' }];
    for (const spec of metadata) {
      const changed = structuredClone(result);
      changed.nodes.return.outputs.out = spec;
      expect(finalOutputValues(f.graph, changed)).toEqual(new Set());
    }
    result.nodes.return.status = 'error';
    expect(finalOutputValues(f.graph, result)).toEqual(new Set());
  });

  it('terminates on stale results for cyclic or broken graphs', () => {
    const f = fixture(); f.input('x'); f.add('ValueVariable', 'v'); f.add('Output', 'return');
    f.link('x', 'v'); f.link('v', 'return'); const result = f.infer();
    f.graph.edges[0].source = 'v';
    expect(() => finalOutputValues(f.graph, result)).not.toThrow();
    f.graph.nodes.find(n => n.id === 'v')!.valueSource = 'missing';
    expect(finalOutputValues(f.graph, result)).toEqual(new Set(['return']));
  });
});

describe('Final-output presentation', () => {
  it.each(['TensorVariable', 'ValueVariable', 'Inlet', 'Output'])(
    'adds a compact semantic border without replacing %s gradients, source badges or custom color data', type => {
      useEditor.getState().load({ schemaVersion: 1, name: 'final values', variants: [], graph: { nodes: [], edges: [] } });
      const model = registry.create(type, {}, 'value'); model.tensorBorderColor = '#123456';
      const spec: TensorSpec = {
        shape: [2, 3], axes: ['B', 'C'], dtype: 'float32',
        autograd: { requiresGrad: true, source: 'manual' },
        lineage: { sources: [{ id: 'image', index: 1, label: 'img' }], origins: [], initializations: [], unknown: false, evidence: 'inferred' },
      };
      const props: NodeProps<TrailNode> = {
        id: model.id, type: 'module', selected: true, dragging: false, zIndex: 0,
        selectable: true, deletable: true, draggable: true, isConnectable: true,
        positionAbsoluteX: 0, positionAbsoluteY: 0,
        data: {
          model, ...ports(model, registry), finalOutput: true,
          result: { status: 'ok', inputs: { in: spec }, outputs: { out: spec }, notes: [], messages: [] },
          category: '输入输出', reverse: false, active: false, dimmed: false, open: () => {},
        },
      };
      const render = () => renderToStaticMarkup(createElement(ReactFlowProvider, null, createElement(ModuleNode, props)));
      const html = render();
      expect(html).toContain('final-output-value'); expect(html).toContain('data-final-output="true"');
      expect(html).toContain(FINAL_OUTPUT_TITLE); expect(html).toContain('--tensor-border:#123456');
      expect(html).toContain(`--tensor-bg:${DEFAULT_TENSOR_PALETTE.on}`);
      expect(html).toContain('data-grad-state="on"');
      expect(html).toContain('data-source-index="1"');
      expect(html).toContain('selected');
      props.data.finalOutput = false;
      expect(render()).not.toContain('data-final-output');
      expect(model.tensorBorderColor).toBe('#123456');
    },
  );
});
