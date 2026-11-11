import { registry } from '../modules/index';
import type { Field } from '../core/types';

export const identifierPattern = '^(?!__proto__$|constructor$|prototype$)[A-Za-z0-9_-]{1,100}$';
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
      type: ['string', 'object'],
      maxLength: f.maxLength ?? 64000,
      description: 'DictionarySpec object (preferred); legacy JSON strings remain readable.',
      default: f.default,
    };
  return {
    ...(f.kind === 'number'
      ? { type: 'integer', minimum: f.min ?? 1, maximum: Number.MAX_SAFE_INTEGER }
      : f.kind === 'boolean'
        ? { type: 'boolean' }
        : f.kind === 'select'
          ? { type: 'string', enum: f.options }
          : { type: 'string', maxLength: f.maxLength ?? 1999 }),
    default: f.default,
    description: f.help || f.label,
  };
}

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
      ...(d.getPorts
        ? {
            dynamicPorts:
              d.type === 'TensorSize'
                ? 'Output ports depend on axes: dim0, dim1, ...; names changes labels only.'
                : ['Concat', 'Stack'].includes(d.type)
                  ? 'input_count selects 1–16 ordered Tensor inputs a, b, c, ...; omitted count defaults to 2 for legacy Concat.'
                  : d.type === 'View'
                    ? 'Tensor input in plus integer inputs d0–d7 referenced in shape, each appearing once as a port; example shape: d0, d1, -1.'
                    : d.type === 'TensorIndex'
                      ? 'Slice mode with stopFromInput=true adds integer input stop alongside Tensor input in.'
                      : 'Ports depend on module parameters.',
          }
        : {}),
      fields: d.fields,
      paramsSchema: object(Object.fromEntries(d.fields.map((f) => [f.key, fieldSchema(f)])), []),
    })),
    limits: {
      maxBytes: 3_000_000,
      maxExpandedNodes: 2000,
      maxEdgesPerGraph: 8000,
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
  const node = {
    oneOf: registry.list(true).map((d) =>
      object(
        {
          id: identifier,
          type: { const: d.type },
          label: name,
          position,
          params: object(Object.fromEntries(d.fields.map((f) => [f.key, fieldSchema(f)])), []),
          ...(d.composite ? { subgraph: { $ref: '#/$defs/graph' } } : {}),
        },
        ['id', 'type', ...(d.composite && !d.build ? ['subgraph'] : [])],
      ),
    ),
  };
  const graph = object(
    {
      nodes: { type: 'array', maxItems: 2000, items: { $ref: '#/$defs/node' } },
      connections: {
        type: 'array',
        maxItems: 8000,
        items: { type: 'array', prefixItems: [endpoint, endpoint], minItems: 2, maxItems: 2 },
      },
    },
    ['nodes', 'connections'],
  );
  const path = { type: 'array', maxItems: 12, items: identifier };
  const ops = [
    object({ op: { const: 'add_node' }, path, node: { $ref: '#/$defs/node' } }, ['op', 'node']),
    object({ op: { const: 'replace_node' }, path, id: identifier, node: { $ref: '#/$defs/node' } }, ['op', 'id', 'node']),
    object({ op: { const: 'duplicate_linked' }, path, source: identifier, id: identifier, position }, ['op', 'source', 'id']),
    object({ op: { const: 'remove_node' }, path, id: identifier }, ['op', 'id']),
    object(
      {
        op: { const: 'set_params' },
        path,
        id: identifier,
        params: { type: 'object', additionalProperties: { type: ['string', 'number', 'boolean'] } },
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
    object({ op: { const: 'set_tensor_border' }, path, id: identifier,
      color: { type: ['string', 'null'], pattern: '^#[0-9a-fA-F]{6}$' } }, ['op', 'id', 'color']),
    ...['connect', 'disconnect'].map((op) =>
      object({ op: { const: op }, path, from: endpoint, to: endpoint }, ['op', 'from', 'to']),
    ),
    object({ op: { const: 'layout' }, path }, ['op']),
    object({ op: { const: 'rename_project' }, name }, ['op', 'name']),
  ];
  const base = { $schema: 'https://json-schema.org/draft/2020-12/schema', $defs: { node, graph } };
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
