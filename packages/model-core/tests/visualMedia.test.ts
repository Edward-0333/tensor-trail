import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import { buildModel, getSchemas, patchModel } from '../src/agent';
import { inferProject } from '../src/core/engine';
import { parseProject, serializeProject, validateProjectObject } from '../src/core/serialization';
import { effectiveValueNode } from '../src/core/valueLinks';
import { maxImageSourceLength } from '../src/core/visualMedia';
import { registry } from '../src/modules';
import { example } from '../src/modules/examples';

const image = { id: 'mask', src: 'data:image/png;base64,YQ==' };
const media = { items: [image], tileWidth: 80.5 };
const recipe = {
  name: '演示',
  nodes: [
    { id: 'x', type: 'Input', visualMedia: media },
    { id: 'y', type: 'Output' },
  ],
  connections: [['x.out', 'y.in']],
};
const edgeOp = { op: 'set_edge_presentation', from: 'x.out', to: 'y.in' };

describe('节点插图与箭头展示协议', () => {
  it.each(['png', 'jpeg', 'webp'])('往返保存内嵌 %s 图片', (format) => {
    const project = buildModel(recipe).project!;
    project.graph.nodes[0].visualMedia!.items[0].src = image.src.replace('png', format);
    expect(parseProject(serializeProject(project), registry)).toEqual(project);
  });

  it('拒绝不安全来源、非法字段、数量或尺寸，项目与 Patch 复用同一校验', () => {
    const project = buildModel(recipe).project!;
    const invalid = [
      ...[
        'https://example.org/x.png',
        'javascript:alert(1)',
        'data:image/svg+xml;base64,YQ==',
        'data:image/png;base64,YQ==\n',
        'data:image/png;base64,',
        'data:image/png;base64,AA*A',
        'data:image/jpeg;base64,' + 'a'.repeat(maxImageSourceLength),
      ].map((src) => ({ items: [{ ...image, src }] })),
      { items: [] },
      { items: Array.from({ length: 13 }, (_, i) => ({ ...image, id: String(i) })) },
      { items: [image, { ...image, caption: '另一个条目' }] },
      { items: [{ ...image, id: 'constructor' }] },
      ...[
        { id: 'a.b' },
        { caption: '字'.repeat(121) },
        { badge: 'a'.repeat(33) },
        { caption: 1 },
        { emphasis: 'hidden' },
        { url: 'remote' },
      ].map((fields) => ({ items: [{ ...image, ...fields }] })),
      { ...media, layout: 'free' },
      { ...media, tileWidth: 79 },
      { ...media, tileWidth: 281 },
      { ...media, tileWidth: Infinity },
      { ...media, extra: true },
    ];
    for (const value of invalid) {
      const candidate = structuredClone(project);
      Object.assign(candidate.graph.nodes[0], { visualMedia: value });
      expect(() => validateProjectObject(candidate, registry)).toThrow();
      expect(
        patchModel(project, { operations: [{ op: 'set_node_media', id: 'x', media: value }] }).ok,
      ).toBe(false);
    }
    for (const type of ['LayoutRegion', 'LoopRegion', 'IfRegion']) {
      expect(
        buildModel({ ...recipe, nodes: [...recipe.nodes, { type, id: 'r', visualMedia: media }] })
          .ok,
      ).toBe(false);
      const candidate = structuredClone(project);
      candidate.graph.nodes.push({ ...registry.create(type), visualMedia: media });
      expect(() => parseProject(serializeProject(candidate), registry)).toThrow(/范围框/);
    }
  });

  it('逐项合法的图片仍受整个项目的 10 MB 上限约束', () => {
    const project = buildModel(recipe).project!;
    const src = 'data:image/png;base64,'.padEnd(maxImageSourceLength, 'A');
    const large = { items: Array.from({ length: 10 }, (_, i) => ({ id: String(i), src })) };
    project.graph.nodes[0].visualMedia = large;
    expect(validateProjectObject(project, registry)).toBe(project);
    expect(
      patchModel(project, { operations: [{ op: 'set_node_media', id: 'y', media: large }] }).ok,
    ).toBe(false);
    project.graph.nodes[1].visualMedia = large;
    expect(() => validateProjectObject(project, registry)).toThrow('10 MB');
    expect(() => parseProject(JSON.stringify(project), registry)).toThrow('10 MB');
  });

  it('仅修改指定子图的节点与边，保留手动布局与推导结果', () => {
    const project = example();
    project.lineageEnabled = true;
    const parent = project.graph.nodes.find((n) => n.id === 'transformer')!;
    const child = parent.subgraph!;
    const edge = child.edges[0];
    edge.labelPosition = 0.29;
    edge.waypoints = [{ x: 247, y: 30 }];
    const path = [parent.id];
    const nodeOp = { op: 'set_node_media', path, id: edge.source, media };
    const arrow = {
      op: 'set_edge_presentation',
      path,
      from: `${edge.source}.${edge.sourcePort}`,
      to: `${edge.target}.${edge.targetPort}`,
    };
    const edited = patchModel(
      project,
      { operations: [nodeOp, { ...arrow, presentation: 'slide', annotation: '特征' }] },
      { layout: 'preserve' },
    );
    expect(edited.ok).toBe(true);
    const expected = structuredClone(project);
    const decorated = expected.graph.nodes.find((n) => n.id === parent.id)!.subgraph!;
    decorated.nodes.find((n) => n.id === edge.source)!.visualMedia = media;
    Object.assign(decorated.edges[0], { annotation: '特征', presentation: 'slide' });
    expect(edited.project).toEqual(expected);
    expect(inferProject(expected, registry)).toEqual(inferProject(project, registry));
    const regenerated = patchModel(expected, {
      operations: [{ op: 'set_params', id: parent.id, params: { heads: 4 } }],
    });
    expect(regenerated.ok).toBe(true);
    const rebuilt = regenerated.project!.graph.nodes.find((n) => n.id === parent.id)!.subgraph!;
    expect(rebuilt.nodes.find((n) => n.id === edge.source)!.visualMedia).toEqual(media);
    expect(
      rebuilt.edges.find(
        (e) =>
          e.source === edge.source && e.target === edge.target && e.targetPort === edge.targetPort,
      ),
    ).toMatchObject({ presentation: 'slide', annotation: '特征', labelPosition: 0.29 });
    for (const annotation of [null, '']) {
      expect(
        patchModel(expected, {
          operations: [
            { ...nodeOp, media: null },
            { ...arrow, annotation, presentation: null },
          ],
        }).project,
      ).toEqual(project);
    }
  });

  it('按端点部分更新，拒绝未知连线与非法样式', () => {
    const original = buildModel(recipe).project!;
    const edited = patchModel(original, {
      operations: [
        { ...edgeOp, annotation: '第一步' },
        { ...edgeOp, presentation: 'slide' },
      ],
    });
    expect(edited.project!.graph.edges[0]).toMatchObject({
      annotation: '第一步',
      presentation: 'slide',
    });
    for (const invalid of [
      { from: 'bad' },
      { to: 'y.missing' },
      { annotation: 10 },
      { annotation: '多'.repeat(161) },
      { presentation: 'large' },
    ]) {
      expect(patchModel(original, { operations: [{ ...edgeOp, ...invalid }] }).ok).toBe(false);
    }
    for (const invalid of [
      { annotation: null },
      { annotation: '字'.repeat(161) },
      { presentation: null },
      { presentation: 'large' },
    ]) {
      const project = structuredClone(original);
      Object.assign(project.graph.edges[0], invalid);
      expect(() => parseProject(JSON.stringify(project), registry)).toThrow();
    }
  });

  it('关联副本独立保存自己的插图', () => {
    const result = patchModel(buildModel(recipe).project!, {
      operations: [{ op: 'duplicate_linked', source: 'x', id: 'copy' }],
    });
    const graph = result.project!.graph;
    expect(effectiveValueNode(graph, graph.nodes.at(-1)!).visualMedia).toBeUndefined();
    const changed = patchModel(result.project, {
      operations: [
        { op: 'set_node_media', id: 'copy', media: { items: [{ ...image, caption: '局部' }] } },
      ],
    });
    const updated = changed.project!.graph;
    expect(effectiveValueNode(updated, updated.nodes.at(-1)!).visualMedia!.items[0].caption).toBe(
      '局部',
    );
    expect(updated.nodes[0].visualMedia).toEqual(media);
  });

  it('发布可执行的 Recipe 与 Patch 2020-12 Schema', () => {
    const schemas = getSchemas();
    const ajv = new Ajv2020({ strict: false, inlineRefs: false });
    const validRecipe = ajv.compile(schemas.recipe);
    const validPatch = ajv.compile(schemas.patch);
    expect(validRecipe(recipe)).toBe(true);
    expect(
      validRecipe({
        ...recipe,
        nodes: [{ type: 'LayoutRegion', id: 'region', visualMedia: media }],
      }),
    ).toBe(false);
    const op = { op: 'set_node_media', id: 'x', media };
    for (const operations of [
      [op],
      [{ ...op, media: null }],
      [{ ...edgeOp, annotation: null, presentation: 'slide' }],
    ])
      expect(validPatch({ operations })).toBe(true);
    for (const invalid of [
      { ...op, media: { items: [] } },
      { ...op, media: { items: [{ ...image, src: 'https://example.org/i' }] } },
      { ...edgeOp, presentation: 'invalid' },
      { ...edgeOp, annotation: '字'.repeat(161) },
    ])
      expect(validPatch({ operations: [invalid] })).toBe(false);
  });
});
