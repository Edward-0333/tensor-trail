import type { CSSProperties } from 'react';
import { isDeviceSpec, isDtypeSpec, isControlSpec, isIntegerSpec, type ModelNode, type ValueSpec } from '@tensor-trail/model-core/core/types';
import { canvasValueLabel } from './canvasValueLabel';

export function isCountContract(target: ModelNode | undefined, port: string): boolean {
  return target?.type === 'BooleanIndexAxis' && target.params.declared_count === true && port === 'count';
}

export function valueEdgeLabel(value: ValueSpec | undefined, countContract = false): string | undefined {
  return countContract
    ? `数量契约 · ${value && isIntegerSpec(value) ? value.value : '?'}（非实测）`
    : canvasValueLabel(value);
}

/** Metadata/contract dependencies differ from data; ordinary integer wires stay solid. */
export function valueEdgeStyle(value: ValueSpec | undefined, error = false, countContract = false): CSSProperties {
  const device = value && (isDeviceSpec(value) || isDtypeSpec(value));
  const control = value && isControlSpec(value);
  return {
    stroke: error ? 'var(--danger)' : countContract ? 'var(--text-secondary)' : control ? 'var(--category-pink)' : device ? 'var(--text-secondary)' : 'var(--edge)',
    strokeWidth: 1.5,
    ...(device || countContract ? { strokeDasharray: '5 4' } : {}),
    ...(control && !countContract ? { strokeDasharray: '7 4' } : {}),
  };
}
