import type { Graph, ModelNode, ModuleDefinition, Params } from '../core/types';
import { clone, uid } from '../core/types';

export class ModuleRegistry {
  private definitions = new Map<string, ModuleDefinition>();
  register(def: ModuleDefinition) {
    if (this.definitions.has(def.type)) throw new Error(`模块 ${def.type} 已注册。`);
    this.definitions.set(def.type, def);
  }
  get(type: string) {
    return this.definitions.get(type);
  }
  list() {
    return [...this.definitions.values()].filter((d) => d.category !== '内部');
  }
  create(type: string, params: Params = {}, id: string = uid()): ModelNode {
    const def = this.get(type);
    if (!def) throw new Error(`未知模块 ${type}`);
    const values = { ...Object.fromEntries(def.fields.map((f) => [f.key, f.default])), ...params };
    return {
      id,
      type,
      definitionVersion: 1,
      label: def.label,
      params: values,
      position: { x: 0, y: 0 },
      ...(def.build ? { subgraph: def.build(values) } : {}),
    };
  }
  fromVariant(name: string, graph: Graph): ModelNode {
    return { ...this.create('Block'), label: name, subgraph: clone(graph), customized: true };
  }
}
