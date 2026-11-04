export const isLegacyForwardInput = (type: string) => ['DictInput', 'ListInput'].includes(type);
export const isForwardInput = (type: string) =>
  ['ForwardInput', 'DictionaryInput'].includes(type) || isLegacyForwardInput(type);
export const canonicalModuleType = (type: string) =>
  isLegacyForwardInput(type) ? 'ForwardInput' : type;
