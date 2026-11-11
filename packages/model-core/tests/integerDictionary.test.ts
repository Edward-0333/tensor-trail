import { describe, expect, it } from 'vitest';
import { parseDictionarySchema } from '../src/core/dictionary';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject } from '../src/core/serialization';
import { shapeText, type DictionarySpec, type Project } from '../src/core/types';
import { registry } from '../src/modules/index';
import { parseFieldPath } from '../src/modules/indexing';
import { validateModel } from '../src/agent/index';

const ego = { shape: [1, 1, 21], axes: ['batch', 'ego', 'state_feature'], dtype: 'float32' };
const schema = {
  kind: 'dict',
  fields: { temporal_ego_states: { kind: 'dict', keyType: 'int', fields: { 0: ego, 1: ego } } },
};
const rule = registry.get('DictionaryGet')!.inferValue!;
const source = () => parseDictionarySchema(JSON.stringify(schema));

describe('Integer-key dictionaries', () => {
  it('looks up an integer time key without indexing the tensor batch axis', () => {
    const before = source();
    const result = rule({ in: before }, { path: '["temporal_ego_states", 0]' });
    expect(result.outputs.out).toEqual(ego);
    expect(result.notes?.join()).toContain('[0]');
    expect(before).toEqual(source());
    expect(result.outputs.out).not.toBe(before.fields.temporal_ego_states);
  });

  it('keeps legacy string keys distinct from integer keys', () => {
    const temporal = rule({ in: source() }, { path: 'temporal_ego_states' }).outputs.out;
    expect(shapeText(temporal)).toBe('dict[int] · 2 个字段');
    expect(rule({ in: temporal }, { path: '[0]' }).outputs.out).toEqual(ego);
    expect(() => rule({ in: temporal }, { path: '["0"]' })).toThrow('整数键');
    const strings = parseDictionarySchema(JSON.stringify({ kind: 'dict', fields: { '0': ego } }));
    expect(rule({ in: strings }, { path: '["0"]' }).outputs.out).toEqual(ego);
    expect(() => rule({ in: strings }, { path: '[0]' })).toThrow('字符串键');
    expect(() => rule({ in: temporal }, { path: '[6]' })).toThrow('不存在字段');
  });

  it('supports canonical negative integer keys and empty integer dictionaries', () => {
    const negative = parseDictionarySchema(
      JSON.stringify({
        kind: 'dict',
        keyType: 'int',
        fields: { '-1': ego },
      }),
    );
    expect(rule({ in: negative }, { path: '[-1]' }).outputs.out).toEqual(ego);
    expect(parseDictionarySchema('{"kind":"dict","keyType":"int","fields":{}}')).toEqual({
      kind: 'dict',
      keyType: 'int',
      fields: {},
    });
  });

  it.each(['01', '-0', '1.5', '9007199254740992', 'time', '__proto__'])(
    'rejects invalid integer schema key %s',
    (key) => {
      expect(() =>
        parseDictionarySchema(
          JSON.stringify({
            kind: 'dict',
            keyType: 'int',
            fields: { [key]: ego },
          }),
        ),
      ).toThrow();
    },
  );

  it.each(['[true]', '[null]', '[1.5]', '[9007199254740992]', '[]'])(
    'rejects invalid lookup path %s',
    (path) => expect(() => parseFieldPath(path)).toThrow(),
  );

  it('rejects unsupported key types rather than silently coercing them', () => {
    expect(() => parseDictionarySchema('{"kind":"dict","keyType":"float","fields":{}}')).toThrow(
      'keyType',
    );
  });

  it('preserves lookup and dimension values through editing and save/reopen', () => {
    const project: Project = {
      schemaVersion: 1,
      name: 'SparseWorld first lines',
      variants: [],
      graph: {
        nodes: [
          registry.create('Input', { shape: '1,30,3,256,704', axes: '', dtype: 'uint8' }, 'img'),
          registry.create('TensorSize', { axes: '0', names: 'B' }, 'img_size'),
          registry.create('LocalParameter', {}, 'B'),
          registry.create('ForwardInput', { schema: JSON.stringify(schema) }, 'kwargs'),
          registry.create('DictionaryGet', { path: '["temporal_ego_states",0]' }, 'get_ego'),
          registry.create('TensorVariable', {}, 'ego_states'),
          registry.create('TensorSize', { axes: '0,1,2', names: 'bs,_,dim_' }, 'ego_size'),
          ...['bs', 'unused', 'dim_'].map((id) => registry.create('LocalParameter', {}, id)),
        ],
        edges: [
          ['img', 'out', 'img_size'],
          ['img_size', 'dim0', 'B'],
          ['kwargs', 'out', 'get_ego'],
          ['get_ego', 'out', 'ego_states'],
          ['ego_states', 'out', 'ego_size'],
          ['ego_size', 'dim0', 'bs'],
          ['ego_size', 'dim1', 'unused'],
          ['ego_size', 'dim2', 'dim_'],
        ].map(([source, sourcePort, target], i) => ({
          id: `e${i}`,
          source,
          sourcePort,
          target,
          targetPort: 'in',
        })),
      },
    };
    const restored = parseProject(serializeProject(project), registry);
    expect(restored).toEqual(project);
    const r = inferProject(restored, registry);
    expect(Object.values(r.nodes).every((node) => node.status === 'ok')).toBe(true);
    expect(r.nodes.img.outputs.out).toMatchObject({ dtype: 'uint8' });
    expect(r.nodes.ego_states.outputs.out).toEqual(ego);
    for (const [id, value] of Object.entries({ B: 1, bs: 1, unused: 1, dim_: 21 }))
      expect(r.nodes[id].outputs.out).toEqual({ kind: 'int', value });
    // This is a partial body, not the final return of forward_backbone.
    expect(validateModel(restored).diagnostics.map((d) => d.code)).toEqual(['MISSING_OUTPUT']);
    const updated = source();
    (updated.fields.temporal_ego_states as DictionarySpec).fields['0'] = {
      ...ego,
      shape: [2, 1, 21],
    };
    restored.graph.nodes.find((node) => node.id === 'kwargs')!.params.schema =
      JSON.stringify(updated);
    expect(inferProject(restored, registry).nodes.bs.outputs.out).toEqual({
      kind: 'int',
      value: 2,
    });
    expect(inferProject(restored, registry).nodes.B.outputs.out).toEqual({ kind: 'int', value: 1 });
  });
});
