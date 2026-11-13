import type { Graph, ModelNode, Project, Registry } from './types';
import { graphIssues, orderGraph } from './graph';
import { planAttributes } from './attributes';
import { canonicalModuleType } from './inputTypes';
import { validateHyperparameters } from './hyperparameters';
import { migrateTensorSelection } from './indexingTypes';
import { syncValueLinks } from './valueLinks';
import { isHexColor, validateAutograd } from './tensorAppearance';
import { parseDictionarySchema } from './dictionary';
import { validateRuntimeReferences } from './runtimeSnapshot';
import { checkForEachBody } from './forEach';
import { checkWhileBody } from './whileLoop';
import { checkForLoopBody } from './forLoop';
import { validateInformationSource } from './informationSources';
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
  return validateProjectObject(JSON.parse(text), registry);
}

/** Validate a private draft directly, avoiding a stringify/parse clone on every edit. */
export function validateProjectObject(value: unknown, registry: Registry): Project {
  // Count pretty-printed JSON without allocating a project-sized string or parsing it again.
  let size = 0;
  const ancestors = new Set<object>();
  const count = (item: unknown, depth = 0): void => {
    requireThat(depth <= 80, 'JSON 嵌套过深。');
    if (item && typeof item === 'object') {
      requireThat(!ancestors.has(item), '项目不能包含循环引用。'); ancestors.add(item);
      const entries = Array.isArray(item) ? item.map(child=>[undefined,child] as const)
        : Object.entries(item).filter(([,child])=>child!==undefined);
      size += 2;
      for (const [key,child] of entries) {
        size += 2 + (depth+1)*2 + (key === undefined ? 0 : JSON.stringify(key).length+2);
        count(child,depth+1);
      }
      if(entries.length) size += depth*2;
      ancestors.delete(item);
    } else size += (JSON.stringify(item) ?? 'null').length;
    requireThat(size <= 3_000_000, '项目文件超过 3 MB。');
  };
  count(value);
  requireThat(
    record(value) && value.schemaVersion === 1,
    '项目格式版本不受支持；需要 schemaVersion=1。',
  );
  requireThat(
    label(value.name) && Array.isArray(value.variants) && value.variants.length <= 100,
    '项目名称或变体列表无效。',
  );
  let total = 0;
  if (value.hyperparameters !== undefined) validateHyperparameters(value.hyperparameters);
  if (value.runtime !== undefined) validateRuntimeReferences(value.runtime);
  requireThat(value.lineageEnabled === undefined || typeof value.lineageEnabled === 'boolean', '信息来源分析开关必须为布尔值。');
  if (value.tensorPalette !== undefined)
    requireThat(record(value.tensorPalette) &&
      Object.keys(value.tensorPalette).length === 3 &&
      ['on', 'off', 'unknown'].every((key) => isHexColor((value.tensorPalette as Record<string, unknown>)[key])),
      'Tensor 统一背景色需要 on、off、unknown 三个六位十六进制颜色。');
  const inspect = (candidate: unknown, depth = 0, ownerType?: string): void => {
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
      const type = canonicalModuleType(node.type);
      if (node.type !== type) {
        node.type = type;
        if (['字典输入 Dict', '列表输入 List'].includes(node.label)) node.label = '网络输入';
      }
      const def = registry.get(type);
      requireThat(def, `尚未注册模块 ${node.type}。`);
      if (record(node.params) && ['ForwardInput', 'DictionaryInput'].includes(type) &&
          node.params.schema !== undefined && node.params.schema !== '') {
        // Keep malformed legacy text editable; a valid schema is always stored as an object.
        try { node.params.schema = parseDictionarySchema(node.params.schema); }
        catch (error) { if (typeof node.params.schema !== 'string') throw error; }
      }
      requireThat(
        record(node.params) &&
          Object.entries(node.params).every(
            ([key, p]) =>
              (key === 'schema' && ['ForwardInput', 'DictionaryInput'].includes(type) && record(p)) ||
              (typeof p === 'number' && Number.isFinite(p)) ||
              typeof p === 'boolean' ||
              (typeof p === 'string' &&
                p.length <= (def.fields.find((f) => f.key === key)?.maxLength ?? 1999)),
          ),
        '节点参数必须是有限数值、开关或短文本。',
      );
      migrateTensorSelection(node as unknown as ModelNode);
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
      requireThat(
        node.valueSource === undefined || identifier(node.valueSource),
        '共享值来源必须是有效节点 ID。',
      );
      const layout = node.portLayout;
      if (node.loopFeedbackRoutes !== undefined) {
        requireThat(type === 'ForLoop' && record(node.loopFeedbackRoutes) &&
          Object.keys(node.loopFeedbackRoutes).length <= 8,
          '循环回传布局仅支持 ForLoop，最多 8 条状态路径。');
        for (const [state, route] of Object.entries(node.loopFeedbackRoutes)) {
          requireThat(identifier(state) && record(route) &&
            ['left', 'right'].includes(String(route.targetSide)) &&
            Array.isArray(route.waypoints) && route.waypoints.length <= 100 &&
            route.waypoints.every(point => record(point) && Number.isFinite(point.x) && Number.isFinite(point.y)) &&
            (route.labelPosition === undefined || (typeof route.labelPosition === 'number' &&
              Number.isFinite(route.labelPosition) && route.labelPosition >= 0 && route.labelPosition <= 1)),
            '循环回传布局无效：需要有效状态名、左右端点、最多 100 个有限坐标点和 0-1 的文字位置。');
        }
      }
      requireThat(
        (node.tensorColor === undefined || isHexColor(node.tensorColor)) &&
          (node.tensorBorderColor === undefined || isHexColor(node.tensorBorderColor)),
        'Tensor 边框色必须为六位十六进制颜色。',
      );
      if (node.tensorColor !== undefined) {
        node.tensorBorderColor ??= node.tensorColor;
        delete node.tensorColor;
      }
      if (node.tensorAutograd !== undefined)
        validateAutograd(node.tensorAutograd, node.type === 'Input' ? String(node.params.dtype) : undefined);
      if (node.informationSource !== undefined) validateInformationSource(node.informationSource);
      requireThat(
        layout === undefined ||
          (record(layout) &&
            ['input', 'output'].every(
              (key) =>
                layout[key] === undefined ||
                ['left', 'right', 'top', 'bottom'].includes(String(layout[key])),
            )),
        '端口位置无效，请选择左、右、上或下。',
      );
      if (def.composite) {
        requireThat(node.subgraph, '组合模块缺少子图。');
        inspect(node.subgraph, depth + 1, type);
        if (type === 'ForEachTensor') checkForEachBody(node.subgraph as Graph);
        if (type === 'WhileLoop') checkWhileBody(node as unknown as ModelNode);
        if (type === 'ForLoop') checkForLoopBody(node as unknown as ModelNode);
      } else requireThat(node.subgraph === undefined, '普通算子不能携带子图。');
      if (node.type === 'Inlet' || node.type === 'Outlet') {
        requireThat(depth > 0, '根图不能使用子图边界节点。');
        if (node.valueSource === undefined) {
          const set = node.type === 'Inlet' ? inlet : outlet,
            port = node.params.port;
          requireThat(identifier(port) && !set.has(port), '子图边界端口重复或无效。');
          set.add(port);
        }
      }
    }
    // Constructor provenance has no forward inputs, but must expose one result.
    if (depth > 0 && ['EmbeddingXYZInit', 'EmbeddingXYZModuleInit'].includes(ownerType ?? ''))
      requireThat(inlet.size === 0 && outlet.size === 1 && outlet.has('out'),
        'Embedding 初始化子图不能接收 forward 输入，且必须有唯一 out 出口。');
    else if (depth > 0)
      requireThat(
        inlet.size > 0 && (outlet.size > 0 || ownerType === 'Submodule'),
        ownerType === 'Submodule'
          ? '子模块至少需要一个 Inlet 输入。'
          : '子图至少需要一个 Inlet 和一个 Outlet。',
      );
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
      requireThat(
        edge.waypoints === undefined ||
          (Array.isArray(edge.waypoints) &&
            edge.waypoints.length <= 100 &&
            edge.waypoints.every(
              (point) => record(point) && Number.isFinite(point.x) && Number.isFinite(point.y),
            )),
        '连线控制点无效；每条连线最多支持 100 个有限坐标点。',
      );
      requireThat(
        edge.labelPosition === undefined ||
          (typeof edge.labelPosition === 'number' &&
            Number.isFinite(edge.labelPosition) &&
            edge.labelPosition >= 0 &&
            edge.labelPosition <= 1),
        '连线文字位置必须是 0 到 1 之间的有限数值。',
      );
    }
    const issues = graphIssues(candidate as unknown as Graph, registry);
    requireThat(!issues.length, issues.join('\n'));
    syncValueLinks(candidate as unknown as Graph);
  };
  const checkAttributeCycles = (graph: Graph) => {
    const plan = planAttributes(graph);
    requireThat(!plan.issues.length, plan.issues.join('\n'));
    const visit = (g: Graph, path: string[] = []) => {
      orderGraph(g, plan.dependencies.get(JSON.stringify(path)));
      for (const node of g.nodes) if (node.subgraph) visit(node.subgraph, [...path, node.id]);
    };
    visit(graph);
  };
  inspect(value.graph);
  checkAttributeCycles(value.graph as Graph);
  const ids = new Set<string>();
  for (const variant of value.variants) {
    requireThat(
      record(variant) && identifier(variant.id) && label(variant.name) && !ids.has(variant.id),
      '变体名称或 ID 无效/重复。',
    );
    ids.add(variant.id);
    inspect(variant.graph, 1);
    checkAttributeCycles(variant.graph as Graph);
  }
  return value as unknown as Project;
}
export const serializeProject = (project: Project) => JSON.stringify(project, null, 2);
