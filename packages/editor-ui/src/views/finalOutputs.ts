import {
  isDictionarySpec, isTensorSequenceSpec, isTensorSpec,
  type Graph, type GraphResult, type ModelNode, type ValueSpec,
} from '@tensor-trail/model-core/core/types';
import { valueOwner } from '@tensor-trail/model-core/core/valueLinks';

export const FINAL_OUTPUT_TITLE = '本层最终返回值：到 return 仅经引用或容器组装；绿色粗边框不表示停止梯度，其他分支仍可使用该值。';

const valueTypes = new Set([
  'Input', 'ForwardInput', 'DictionaryInput', 'Inlet', 'Output', 'Outlet',
  'TensorVariable', 'ValueVariable', 'BufferSource',
]);
const returnValue = (value?: ValueSpec) => value && (
  isTensorSpec(value) || isDictionarySpec(value) || isTensorSequenceSpec(value)
);

/** Trace unchanged return values, not general ancestors or numerical dependencies. */
export function finalOutputValues(graph: Graph, result?: GraphResult, owner?: ModelNode): Set<string> {
  const final = new Set<string>();
  // Loop outlets describe one iteration's state, not the enclosing forward's return.
  if (!result || (owner && ['ForLoop', 'WhileLoop', 'ForEachTensor'].includes(owner.type))) return final;
  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  const incoming = new Map<string, typeof graph.edges>();
  for (const edge of graph.edges) {
    const key = JSON.stringify([edge.target, edge.targetPort]);
    incoming.set(key, [...(incoming.get(key) ?? []), edge]);
  }
  const visited = new Set<string>();
  // Undefined means the complete value; a key set means only those dictionary fields survive.
  const visitInput = (id: string, port: string, keys?: Set<string>) => {
    const edges = incoming.get(JSON.stringify([id, port])) ?? [];
    if (edges.length === 1) visit(edges[0].source, edges[0].sourcePort, keys);
  };
  const visit = (id: string, port: string, keys?: Set<string>) => {
    const node = nodes.get(id), info = result.nodes[id];
    if (!node || !info || info.status === 'error' || info.status === 'blocked') return;
    const spec = info.outputs[port] ?? (['Output', 'Outlet'].includes(node.type) ? info.inputs.in : undefined);
    if (!returnValue(spec)) return;
    if (keys) {
      if (!keys.size || !spec || !isDictionarySpec(spec)) return;
      const ownKeys = Object.keys(spec.fields);
      const selected = keys;
      if (ownKeys.every(key => selected.has(key))) keys = undefined;
    }
    const key = JSON.stringify([id, port, keys ? [...keys].sort() : null]);
    if (visited.has(key)) return;
    visited.add(key);

    if (valueTypes.has(node.type)) {
      let source: ModelNode;
      try { source = valueOwner(graph, node); } catch { return; }
      if (!keys) {
        for (const member of graph.nodes) {
          if (member.id === source.id || (member.valueSource === source.id && member.type === source.type)) {
            final.add(member.id);
          }
        }
      }
      if (node.valueSource) visit(source.id, port, keys);
      else visitInput(id, 'in', keys);
      return;
    }
    if (node.type === 'DictionaryPack' && port === 'out' && spec && isDictionarySpec(spec)) {
      for (const field of Object.keys(spec.fields)) {
        if (!keys || keys.has(field)) visitInput(id, field);
      }
    } else if (node.type === 'TensorSequencePack' && port === 'out' && !keys) {
      for (const input of Object.keys(info.inputs)) visitInput(id, input);
    } else if (node.type === 'DictionaryUpdate' && port === 'out') {
      const { base, updates } = info.inputs;
      if (!base || !updates || !isDictionarySpec(base) || !isDictionarySpec(updates)) return;
      const wanted = keys ?? new Set([...Object.keys(base.fields), ...Object.keys(updates.fields)]);
      // update is shallow: overwritten fields must never trace back to the old base value.
      visitInput(id, 'updates', new Set(Object.keys(updates.fields).filter(field => wanted.has(field))));
      visitInput(id, 'base', new Set(Object.keys(base.fields).filter(field =>
        wanted.has(field) && !Object.hasOwn(updates.fields, field))));
    }
  };
  for (const node of graph.nodes) {
    if (!node.valueSource && ['Output', 'Outlet'].includes(node.type)) visit(node.id, 'out');
  }
  return final;
}
