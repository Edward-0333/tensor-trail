import type { Project, RuntimeSnapshot, ValueSpec } from './types';
import { clone, isDictionarySpec, isTensorSequenceSpec, isTensorSpec } from './types';
import { parseDictionarySchema } from './dictionary';

const record = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === 'object' && !Array.isArray(v);
const id = (v: unknown): v is string =>
  typeof v === 'string' &&
  /^[A-Za-z0-9_-]{1,100}$/.test(v) &&
  !['__proto__', 'constructor', 'prototype'].includes(v);
const name = (v: unknown): v is string => typeof v === 'string' && v.length > 0 && v.length <= 200;
function check(v: unknown, message: string): asserts v {
  if (!v) throw Error(message);
}
export function safeSnapshotPath(file: unknown): file is string {
  return (
    typeof file === 'string' &&
    file.length <= 500 &&
    file.endsWith('.json') &&
    !/[\\:\x00-\x1f]/.test(file) &&
    file.split('/').every((part) => part && part !== '.' && part !== '..')
  );
}
function markRuntime(value: ValueSpec): void {
  if (isDictionarySpec(value)) {
    Object.values(value.fields).forEach(markRuntime);
  } else if (isTensorSequenceSpec(value)) {
    value.items.forEach(markRuntime);
  } else if (isTensorSpec(value) && value.autograd) value.autograd.source = 'runtime';
}
export function parseRuntimeSnapshot(value: unknown): RuntimeSnapshot {
  if (typeof value === 'string') {
    check(value.length <= 3_000_000, '运行快照超过 3 MB。');
    value = JSON.parse(value.replace(/^\uFEFF/, ''));
  }
  check(
    record(value) && value.format === 'tensor-trail.runtime-snapshot' && value.version === 1,
    '运行快照格式无效。',
  );
  check(
    id(value.id) &&
      name(value.name) &&
      typeof value.capturedAt === 'string' &&
      value.capturedAt.length <= 100 &&
      Number.isFinite(Date.parse(value.capturedAt)),
    '运行快照标识或时间无效。',
  );
  check(
    record(value.context) && JSON.stringify(value.context).length <= 64_000,
    '运行上下文必须是最多 64000 字符的对象。',
  );
  check(
    Array.isArray(value.bindings) && value.bindings.length > 0 && value.bindings.length <= 256,
    '运行快照需要 1–256 个输入绑定。',
  );
  const paths = new Set<string>();
  const bindings = value.bindings.map((binding) => {
    check(
      record(binding) &&
        Array.isArray(binding.nodePath) &&
        binding.nodePath.length > 0 &&
        binding.nodePath.length <= 13 &&
        binding.nodePath.every(id),
      '输入绑定的节点路径无效。',
    );
    const key = JSON.stringify(binding.nodePath);
    check(!paths.has(key), '运行快照有重复输入绑定。');
    paths.add(key);
    check(
      ['Input', 'ForwardInput', 'DictionaryInput'].includes(String(binding.nodeType)),
      '第一阶段仅支持运行时输入快照。',
    );
    const spec =
      binding.nodeType === 'Input'
        ? parseDictionarySchema({ kind: 'dict', fields: { value: binding.value } }).fields.value
        : parseDictionarySchema(binding.value);
    check(isTensorSpec(spec) || isDictionarySpec(spec), '输入绑定必须是 Tensor 或字典。');
    check(binding.nodeType !== 'Input' || isTensorSpec(spec), 'Input 绑定必须是 Tensor。');
    check(binding.nodeType === 'Input' || isDictionarySpec(spec), '网络输入绑定必须是字典。');
    markRuntime(spec);
    return {
      nodePath: [...binding.nodePath] as string[],
      nodeType: binding.nodeType as RuntimeSnapshot['bindings'][number]['nodeType'],
      value: spec,
    };
  });
  return {
    format: 'tensor-trail.runtime-snapshot',
    version: 1,
    id: value.id,
    name: value.name,
    capturedAt: value.capturedAt,
    context: clone(value.context),
    bindings,
  };
}
export function validateRuntimeReferences(
  value: unknown,
): asserts value is NonNullable<Project['runtime']> {
  check(
    record(value) && Array.isArray(value.snapshots) && value.snapshots.length <= 100,
    '运行快照引用列表无效。',
  );
  const ids = new Set<string>();
  for (const ref of value.snapshots) {
    check(
      record(ref) && id(ref.id) && name(ref.name) && safeSnapshotPath(ref.file) && !ids.has(ref.id),
      '快照引用需要唯一 ID、名称及安全的相对 JSON 路径。',
    );
    ids.add(ref.id);
  }
  check(
    value.active === undefined || (typeof value.active === 'string' && ids.has(value.active)),
    '选中的运行快照不存在。',
  );
  if (value.embedded !== undefined) {
    const snapshot = parseRuntimeSnapshot(value.embedded);
    check(ids.has(snapshot.id), '内嵌快照必须属于已声明的运行。');
    value.embedded = snapshot;
  }
}
export function checkSnapshotBindings(project: Project, snapshot: RuntimeSnapshot): void {
  for (const binding of snapshot.bindings) {
    let graph = project.graph;
    for (const part of binding.nodePath.slice(0, -1)) {
      const child = graph.nodes.find((n) => n.id === part)?.subgraph;
      check(child, `快照子图不存在：${binding.nodePath.join('/')}`);
      graph = child;
    }
    const node = graph.nodes.find((n) => n.id === binding.nodePath.at(-1));
    check(
      node && node.type === binding.nodeType && !node.valueSource,
      `快照输入不存在、类型改变或是共享副本：${binding.nodePath.join('/')}`,
    );
  }
}
export function activeRuntimeSnapshot(
  project: Project,
  loaded?: RuntimeSnapshot,
): RuntimeSnapshot | undefined {
  const selected = project.runtime?.active;
  const snapshot = loaded?.id === selected ? loaded : project.runtime?.embedded;
  return snapshot?.id === selected ? snapshot : undefined;
}
export function portableProject(project: Project, loaded?: RuntimeSnapshot): Project {
  const copy = clone(project);
  const snapshot = activeRuntimeSnapshot(project, loaded);
  if (project.runtime?.active) {
    check(snapshot, '当前运行快照尚未加载，不能导出完整单文件。');
    checkSnapshotBindings(project, snapshot);
    copy.runtime!.embedded = clone(snapshot);
    copy.runtime!.snapshots = copy.runtime!.snapshots.filter((ref) => ref.id === snapshot.id);
  } else delete copy.runtime;
  return copy;
}
