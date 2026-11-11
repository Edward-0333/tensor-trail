import type { CSSProperties } from 'react';
import { isDeviceSpec, isControlSpec, type ValueSpec } from '@tensor-trail/model-core/core/types';

/** Device metadata is visually distinct; ordinary Tensor/size wires keep their style. */
export function valueEdgeStyle(value: ValueSpec | undefined, error = false): CSSProperties {
  const device = value && isDeviceSpec(value);
  const control = value && isControlSpec(value);
  return {
    stroke: error ? 'var(--danger)' : control ? 'var(--category-pink)' : device ? 'var(--text-secondary)' : 'var(--edge)',
    strokeWidth: 1.5,
    ...(device ? { strokeDasharray: '5 4' } : {}),
    ...(control ? { strokeDasharray: '7 4' } : {}),
  };
}
