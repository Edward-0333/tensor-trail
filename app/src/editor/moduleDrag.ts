import type { DragEvent } from 'react';

export const MODULE_DRAG_TYPE = 'application/x-tensor-trail-module';

export interface ModuleDrag {
  type: string;
  variantId?: string;
}

export function startModuleDrag(event: DragEvent<HTMLElement>, module: ModuleDrag) {
  event.dataTransfer.setData(MODULE_DRAG_TYPE, JSON.stringify(module));
  event.dataTransfer.effectAllowed = 'copy';
}

export function readModuleDrag(transfer: DataTransfer): ModuleDrag | undefined {
  try {
    const value: unknown = JSON.parse(transfer.getData(MODULE_DRAG_TYPE));
    if (
      value &&
      typeof value === 'object' &&
      'type' in value &&
      typeof value.type === 'string' &&
      (!('variantId' in value) || typeof value.variantId === 'string')
    )
      return value as ModuleDrag;
  } catch {
    // Ignore external drags that do not contain a library module.
  }
}
