import type { Graph } from '@tensor-trail/model-core/core/types';
import { graphAt } from '@tensor-trail/model-core/core/graph';
import { isForwardInput } from '@tensor-trail/model-core/core/inputTypes';
import { linkedValueNodes, valueOwner } from '@tensor-trail/model-core/core/valueLinks';
import { useEditor } from './store';

export function valueUnlinkWarning(graph: Graph, root: boolean, id: string): string | undefined {
  if (linkedValueNodes(graph, id).length < 2) return;
  const owner = valueOwner(graph, id);
  if ((root && (owner.type === 'Input' || isForwardInput(owner.type))) || owner.type === 'Inlet')
    return `解除“${owner.label}”的值关联将新增一个独立 forward 输入，输入数量增加 1。此操作不会解除分区归属；移出分区请使用“所属分区”。是否继续解除值关联？`;
  if (owner.type === 'Outlet')
    return '解除值关联将新增一个独立输出接口，不会解除分区归属。是否继续？';
}

export function unlinkValueWithConfirmation(id: string, confirm = (message: string) => window.confirm(message)): void {
  const state = useEditor.getState();
  const warning = valueUnlinkWarning(graphAt(state.project.graph, state.path), !state.path.length, id);
  if (warning && !confirm(warning)) return;
  state.unlinkValue(id);
}
