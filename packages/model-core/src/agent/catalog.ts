import { MAX_EDGES_PER_GRAPH, MAX_EXPANDED_NODES, MAX_NODE_VISITS, MAX_PROJECT_BYTES } from '../core/limits';
import { registry } from '../modules/index';
import { type Field, identifierPattern } from '../core/types';
import { maxSubmoduleSummaryLength, submoduleIcons } from '../core/submoduleVisual';
import { imageSourcePattern, maxImageSourceLength, maxMediaItems, mediaLayouts, imageEmphases, supportsVisualMedia } from '../core/visualMedia';
import { edgeOperatorTypes, nodePresentations, supportsCompactPresentation } from '../core/nodePresentation';
import {
  supportsTensorVisual, tensorVisualModes, tensorVisualTones, tensorDepthStyles, hexColorPattern,
  minTensorVisualWidth, maxTensorVisualWidth, minTensorBoxWidth, maxTensorBoxWidth,
  minTensorBoxHeight, maxTensorBoxHeight, minTensorBoxDepth, maxTensorBoxDepth, maxAxisLabels, maxAxisLabelLength,
} from '../core/tensorVisual';
export { identifierPattern };
const identifier = { type: 'string', pattern: identifierPattern };
const name = { type: 'string', minLength: 1, maxLength: 200, pattern: '\\S' };
const endpoint = { type: 'string', pattern: '^[A-Za-z0-9_-]{1,100}\\.[A-Za-z0-9_-]{1,100}$' };
const object = (properties: Record<string, unknown>, required: string[]) => ({
  type: 'object',
  properties,
  required,
  additionalProperties: false,
});
function fieldSchema(f: Field) {
  if (f.key === 'schema')
    return {
      anyOf: [{ type: 'object' }, { type: 'string', maxLength: f.maxLength ?? 64000 }],
      description: 'DictionarySpec object (preferred); legacy JSON strings remain readable.',
      default: f.default,
    };
  return {
    ...(f.kind === 'number'
      ? { type: f.integer === false ? 'number' : 'integer', minimum: f.min ?? 1, maximum: f.max ?? Number.MAX_SAFE_INTEGER }
      : f.kind === 'boolean'
        ? { type: 'boolean' }
        : f.kind === 'select'
          ? { type: 'string', enum: f.options }
          : { type: 'string', maxLength: f.maxLength ?? 1999 }),
    default: f.default,
    description: f.help || f.label,
  };
}

const joiningPortDescription = 'input_count selects 1–16 ordered Tensor inputs a, b, c, ...; omitted count defaults to 2 for legacy Concat.';
const dynamicPortDescriptions: Record<string, string> = {
  SemanticOp: 'inputs declares 1–12 ordered value ports and outputs declares 1–8 Tensor ports. Both use comma-separated identifiers.',
  TensorSize: 'Output ports depend on axes: dim0, dim1, ...; names changes labels only.',
  Chunk: 'chunks selects 2–9 output ports out0, out1, ... . torch.chunk may return fewer values for a concrete axis length.',
  Concat: joiningPortDescription,
  Stack: joiningPortDescription,
  View: 'Tensor input in plus integer inputs d0–d7 referenced in shape, each appearing once as a port; example shape: d0, d1, -1.',
  TensorIndex: 'Slice mode with stopFromInput=true adds integer input stop alongside Tensor input in.',
};

export function describeModules(includeInternal = false) {
  return structuredClone({
    apiVersion: 1 as const,
    editing: {
      validationModes: ['strict', 'draft'],
      defaultValidationMode: 'strict',
      patchLayouts: ['preserve', 'auto'],
      defaultPatchLayout: 'preserve',
      position:
        'RecipeNode.position and set_position accept finite x/y coordinates. Explicit layout operations always reflow their target graph.',
      linkedCopies:
        'duplicate_linked creates a same-graph value alias with no independent inputs. source may be an existing alias; the new copy refers directly to its canonical owner.',
      draft:
        'Missing input/output and unconnected inputs are warnings; structural and known shape errors remain errors. Unloaded runtime sidecars use static inference with an explicit warning; references are retained.',
    },
    modules: registry.list(includeInternal).map((d) => ({
      type: d.type,
      label: d.label,
      category: d.category,
      description: d.description,
      composite: !!d.composite,
      internal: d.category === '内部',
      inputs: d.inputs,
      outputs: d.outputs,
      ...(d.getPorts ? { dynamicPorts: dynamicPortDescriptions[d.type] ?? 'Ports depend on module parameters.' } : {}),
      fields: d.fields,
      paramsSchema: object(Object.fromEntries(d.fields.map((f) => [f.key, fieldSchema(f)])), []),
    })),
    limits: {
      maxBytes: MAX_PROJECT_BYTES,
      maxExpandedNodes: MAX_EXPANDED_NODES,
      maxEdgesPerGraph: MAX_EDGES_PER_GRAPH,
      maxNodeVisits: MAX_NODE_VISITS,
      maxDepth: 12,
      maxOperations: 500,
    },
    semantics:
      'Shape inference only. Omitted parameters use registry defaults. No tensor execution or training.',
  });
}

export function getSchemas() {
  const position = object({ x: { type: 'number' }, y: { type: 'number' } }, ['x', 'y']);
  const mode = { type: 'string', enum: ['strict', 'draft'], default: 'strict' };
  const submoduleVisual = object({
    icon: { enum: submoduleIcons },
    tone: { enum: tensorVisualTones },
    summary: { type: 'string', maxLength: maxSubmoduleSummaryLength },
  }, []);
  const tensorVisual = object({
    mode: { enum: tensorVisualModes, description: 'Show a single inferred tensor block, or turn the preview off.' },
    tone: { enum: tensorVisualTones },
    color: { type: 'string', pattern: hexColorPattern },
    depthStyle: { enum: tensorDepthStyles },
    caption: { type: 'string', maxLength: 120 },
    width: { type: 'number', minimum: minTensorVisualWidth, maximum: maxTensorVisualWidth },
    boxWidth: { type: 'number', minimum: minTensorBoxWidth, maximum: maxTensorBoxWidth },
    boxHeight: { type: 'number', minimum: minTensorBoxHeight, maximum: maxTensorBoxHeight },
    boxDepth: { type: 'number', minimum: minTensorBoxDepth, maximum: maxTensorBoxDepth },
    axisLabels: {
      type: 'array', maxItems: maxAxisLabels,
      items: { type: 'string', minLength: 1, maxLength: 64, pattern: `^(?=[\\s\\S]*\\S)[\\s\\S]{1,${maxAxisLabelLength}}$` },
    },
  }, ['mode']);
  const visualMedia = object({
    layout: { enum: mediaLayouts },
    tileWidth: { type: 'number', minimum: 80, maximum: 280 },
    items: {
      type: 'array', minItems: 1, maxItems: maxMediaItems, uniqueItems: true,
      description: 'Image IDs must be unique within this node.',
      items: object({
        id: identifier, src: { type: 'string', pattern: imageSourcePattern.source, maxLength: maxImageSourceLength },
        caption: { type: 'string', maxLength: 120 }, badge: { type: 'string', maxLength: 32 },
        emphasis: { enum: imageEmphases },
      }, ['id', 'src']),
    },
  }, ['items']);
  const node = {
    oneOf: registry.list(true).map((d) =>
      object(
        {
          id: identifier,
          type: { const: d.type },
          label: name,
          position,
          ...(supportsVisualMedia(d) ? { presentation: { enum: ['default',
            ...(supportsCompactPresentation(d) ? ['compact'] : []),
            ...(edgeOperatorTypes.has(d.type) ? ['edge'] : [])] } } : {}),
          ...(d.type === 'Submodule' ? { submoduleVisual: { $ref: '#/$defs/submoduleVisual' } } : {}),
          ...(supportsTensorVisual(d) ? { tensorVisual: { $ref: '#/$defs/tensorVisual' } } : {}),
          ...(supportsVisualMedia(d) ? { visualMedia: { $ref: '#/$defs/visualMedia' } } : {}),
          params: object(Object.fromEntries(d.fields.map((f) => [f.key, fieldSchema(f)])), []),
          ...(d.composite ? { subgraph: { $ref: '#/$defs/graph' } } : {}),
        },
        ['id', 'type', ...(d.composite && !d.build ? ['subgraph'] : [])],
      ),
    ),
  };
  const graph = object(
    {
      nodes: { type: 'array', maxItems: MAX_EXPANDED_NODES, items: { $ref: '#/$defs/node' } },
      connections: {
        type: 'array',
        maxItems: MAX_EDGES_PER_GRAPH,
        items: { type: 'array', prefixItems: [endpoint, endpoint], minItems: 2, maxItems: 2 },
      },
    },
    ['nodes', 'connections'],
  );
  const path = { type: 'array', maxItems: 12, items: identifier };
  const shortText = { type: 'string', maxLength: 240 };
  const ops = [
    object({ op: { const: 'collapse_calculations' }, path, ids: { type: 'array', minItems: 2, uniqueItems: true, items: identifier },
      id: identifier, label: name, position, summary: shortText, formula: shortText }, ['op', 'ids']),
    object({ op: { const: 'expand_calculation' }, path, id: identifier }, ['op', 'id']),
    object({ op: { const: 'auto_simplify' }, path, recursive: { type: 'boolean' }, min_size: { type: 'integer', minimum: 2 } }, ['op']),
    object({ op: { const: 'assign_region' }, path, ids: { type: 'array', minItems: 1, maxItems: 500, uniqueItems: true, items: identifier },
      region: { anyOf: [identifier, { type: 'null' }] } }, ['op', 'ids', 'region']),
    object({ op: { const: 'fit_region' }, path, id: identifier }, ['op', 'id']),
    object({ op: { const: 'resize_region' }, path, id: identifier, position,
      width: { type: 'number', minimum: 360, maximum: 1000000 }, height: { type: 'number', minimum: 200, maximum: 1000000 } }, ['op', 'id', 'width', 'height']),
    object({ op: { const: 'add_node' }, path, node: { $ref: '#/$defs/node' } }, ['op', 'node']),
    object({ op: { const: 'replace_node' }, path, id: identifier, node: { $ref: '#/$defs/node' } }, ['op', 'id', 'node']),
    object({ op: { const: 'duplicate_linked' }, path, source: identifier, id: identifier, position }, ['op', 'source', 'id']),
    object({ op: { const: 'link_value' }, path, source: identifier, id: identifier }, ['op', 'source', 'id']),
    object({ op: { const: 'remove_node' }, path, id: identifier }, ['op', 'id']),
    object({ op: { const: 'set_lineage_enabled' }, enabled: { type: 'boolean' } }, ['op', 'enabled']),
    object({ op: { const: 'set_information_source' }, path, id: identifier, source: { type: 'null', description: '仅用于清除旧声明；输入来源由根层接口自动编号。' } }, ['op', 'id', 'source']),
    object(
      {
        op: { const: 'set_params' },
        path,
        id: identifier,
        params: { type: 'object', additionalProperties: { anyOf: [{ type: 'string' }, { type: 'number' }, { type: 'boolean' }] } },
      },
      ['op', 'id', 'params'],
    ),
    object({ op: { const: 'rename_node' }, path, id: identifier, label: name }, [
      'op',
      'id',
      'label',
    ]),
    object({ op: { const: 'set_position' }, path, id: identifier, position }, [
      'op',
      'id',
      'position',
    ]),
    object({ op: { const: 'set_node_presentation' }, path, id: identifier,
      presentation: { enum: [...nodePresentations, null] } }, ['op', 'id', 'presentation']),
    object({ op: { const: 'set_node_submodule_visual' }, path, id: identifier,
      visual: { anyOf: [{ $ref: '#/$defs/submoduleVisual' }, { type: 'null' }] } }, ['op', 'id', 'visual']),
    object({ op: { const: 'set_node_tensor_visual' }, path, id: identifier,
      visual: { anyOf: [{ $ref: '#/$defs/tensorVisual' }, { type: 'null' }] } }, ['op', 'id', 'visual']),
    object({ op: { const: 'set_node_media' }, path, id: identifier,
      media: { anyOf: [{ $ref: '#/$defs/visualMedia' }, { type: 'null' }] } }, ['op', 'id', 'media']),
    object({ op: { const: 'set_edge_presentation' }, path, from: endpoint, to: endpoint,
      annotation: { type: ['string', 'null'], maxLength: 160 },
      presentation: { enum: ['default', 'slide', 'anchor', null] } }, ['op', 'from', 'to']),
    object({ op: { const: 'set_tensor_border' }, path, id: identifier,
      color: { type: ['string', 'null'], pattern: '^#[0-9a-fA-F]{6}$' } }, ['op', 'id', 'color']),
    ...['connect', 'disconnect'].map((op) =>
      object({ op: { const: op }, path, from: endpoint, to: endpoint }, ['op', 'from', 'to']),
    ),
    object({ op: { const: 'layout' }, path }, ['op']),
    object({ op: { const: 'rename_project' }, name }, ['op', 'name']),
  ];
  const base = { $schema: 'https://json-schema.org/draft/2020-12/schema', $defs: { node, graph, visualMedia, tensorVisual, submoduleVisual } };
  return structuredClone({
    validationOptions: {
      $schema: base.$schema,
      ...object(
        {
          mode,
          runtimeSnapshot: {
            type: 'object',
            description:
              'Optional validated runtime-snapshot v1, supplied by the host; not persisted.',
          },
        },
        [],
      ),
    },
    patchOptions: {
      $schema: base.$schema,
      ...object(
        {
          mode,
          layout: { type: 'string', enum: ['preserve', 'auto'], default: 'preserve' },
          runtimeSnapshot: {
            type: 'object',
            description: 'Optional loaded runtime-snapshot v1; not persisted.',
          },
        },
        [],
      ),
    },
    recipe: {
      ...base,
      title: 'Tensor Trail Recipe v1',
      ...object(
        {
          recipeVersion: { const: 1 },
          name,
          hyperparameters: {
            type: 'object',
            maxProperties: 128,
            propertyNames: {
              type: 'string',
              pattern: '^(?!__proto__$|prototype$|constructor$|self$)[A-Za-z_][A-Za-z0-9_]{0,63}$',
            },
            additionalProperties: {
              oneOf: [
                [
                  'int',
                  {
                    type: 'integer',
                    minimum: -Number.MAX_SAFE_INTEGER,
                    maximum: Number.MAX_SAFE_INTEGER,
                  },
                ],
                ['float', { type: 'number' }],
                ['bool', { type: 'boolean' }],
                ['string', { type: 'string', maxLength: 1999 }],
              ].map(([type, value]) =>
                object(
                  { type: { const: type }, value, source: { type: 'string', maxLength: 256 } },
                  ['type', 'value'],
                ),
              ),
            },
          },
          ...graph.properties,
        },
        ['name', 'nodes', 'connections'],
      ),
    },
    patch: {
      ...base,
      title: 'Tensor Trail Patch v1',
      ...object(
        {
          patchVersion: { const: 1 },
          operations: { type: 'array', minItems: 1, maxItems: 500, items: { oneOf: ops } },
        },
        ['operations'],
      ),
    },
  });
}
