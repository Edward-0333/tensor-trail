import type { Graph, Project, Registry } from './types';
import { graphIssues } from './graph';
function requireThat(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const identifier = (value: unknown): value is string =>
  typeof value === 'string' &&
  /^[A-Za-z0-9_-]{1,100}$/.test(value) &&
  !['__proto__', 'constructor', 'prototype'].includes(value);
const label = (value: unknown): value is string =>
  typeof value === 'string' && value.length > 0 && value.length <= 200;

export function parseProject(text: string, registry: Registry): Project {
  requireThat(text.length <= 3_000_000, '项目文件超过 3 MB，第一版暂不支持如此大的项目。');
  const value: unknown = JSON.parse(text);
  requireThat(
    record(value) && value.schemaVersion === 1,
    '项目格式版本不受支持；需要 schemaVersion=1。',
  );
  requireThat(
    label(value.name) && Array.isArray(value.variants) && value.variants.length <= 100,
    '项目名称或变体列表无效。',
  );
  let total = 0;
  const inspect = (candidate: unknown, depth = 0): void => {
    requireThat(depth <= 12, '子图嵌套超过 12 层。');
    requireThat(
      record(candidate) && Array.isArray(candidate.nodes) && Array.isArray(candidate.edges),
      '图必须包含 nodes 和 edges 数组。',
    );
    total += candidate.nodes.length;
    requireThat(
      total <= 2000 && candidate.edges.length <= 8000,
      '项目节点或连接数量超过第一版上限。',
    );
    const inlet = new Set<string>(),
      outlet = new Set<string>();
    for (const node of candidate.nodes) {
      requireThat(
        record(node) &&
          identifier(node.id) &&
          label(node.type) &&
          label(node.label) &&
          node.definitionVersion === 1,
        '节点 ID、名称或定义版本无效。',
      );
      const def = registry.get(node.type);
      requireThat(def, `尚未注册模块 ${node.type}。`);
      requireThat(
        record(node.params) &&
          Object.values(node.params).every(
            (p) =>
              (typeof p === 'number' && Number.isFinite(p)) ||
              typeof p === 'boolean' ||
              (typeof p === 'string' && p.length < 2000),
          ),
        '节点参数必须是有限数值、开关或短文本。',
      );
      requireThat(
        record(node.position) &&
          Number.isFinite(node.position.x) &&
          Number.isFinite(node.position.y),
        '节点位置无效。',
      );
      requireThat(
        node.customized === undefined || typeof node.customized === 'boolean',
        'customized 必须为布尔值。',
      );
      if (def.composite) {
        requireThat(node.subgraph, '组合模块缺少子图。');
        inspect(node.subgraph, depth + 1);
      } else requireThat(node.subgraph === undefined, '普通算子不能携带子图。');
      if (node.type === 'Inlet' || node.type === 'Outlet') {
        requireThat(depth > 0, '根图不能使用子图边界节点。');
        const set = node.type === 'Inlet' ? inlet : outlet,
          port = node.params.port;
        requireThat(identifier(port) && !set.has(port), '子图边界端口重复或无效。');
        set.add(port);
      }
    }
    if (depth > 0)
      requireThat(inlet.size > 0 && outlet.size > 0, '子图至少需要一个 Inlet 和一个 Outlet。');
    for (const edge of candidate.edges) {
      requireThat(
        record(edge) &&
          identifier(edge.id) &&
          identifier(edge.source) &&
          identifier(edge.target) &&
          identifier(edge.sourcePort) &&
          identifier(edge.targetPort),
        '连接数据无效。',
      );
    }
    const issues = graphIssues(candidate as unknown as Graph, registry);
    requireThat(!issues.length, issues.join('\n'));
  };
  inspect(value.graph);
  const ids = new Set<string>();
  for (const variant of value.variants) {
    requireThat(
      record(variant) && identifier(variant.id) && label(variant.name) && !ids.has(variant.id),
      '变体名称或 ID 无效/重复。',
    );
    ids.add(variant.id);
    inspect(variant.graph, 1);
  }
  return value as unknown as Project;
}
export const serializeProject = (project: Project) => JSON.stringify(project, null, 2);
