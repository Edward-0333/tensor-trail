import { isModuleSpec, isControlSpec, type ValueSpec } from '@tensor-trail/model-core/core/types';
import { isScalarTensor, valueDisplayText } from './valueDisplayText';

/** Module types remain available in the inspector, not on canvas nodes or wires. */
export function canvasValueLabel(
  value?: ValueSpec,
  context: 'node' | 'edge' = 'edge',
): string | undefined {
  if (value && isControlSpec(value))
    return value.branch === undefined ? '条件（未执行）' : value.branch ? 'True' : 'False';
  if (context === 'edge' && isScalarTensor(value)) return '标量';
  return value && isModuleSpec(value) ? undefined : value ? valueDisplayText(value) : '待推导';
}
