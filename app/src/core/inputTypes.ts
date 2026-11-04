export const isLegacyForwardInput = (type: string) => ['DictInput', 'ListInput'].includes(type);
export const isForwardInput = (type: string) =>
  type === 'ForwardInput' || isLegacyForwardInput(type);
export const canonicalModuleType = (type: string) =>
  isLegacyForwardInput(type) ? 'ForwardInput' : type;
