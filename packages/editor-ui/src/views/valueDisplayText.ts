import { isTensorSpec, shapeText, type ValueSpec } from '@tensor-trail/model-core/core/types';

export const isScalarTensor = (value?: ValueSpec): boolean =>
  !!value && isTensorSpec(value) && value.shape.length === 0;

/** Presentation only: keep shape=[] in inference, snapshots and serialized projects. */
export function valueDisplayText(value?: ValueSpec): string {
  return isScalarTensor(value) ? '标量（0D）' : shapeText(value);
}
