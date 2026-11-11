import { useEffect, useState } from 'react';
import type { Field, ModelNode, ValueSpec } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, shapeText } from '@tensor-trail/model-core/core/types';
import { resolveIntegerExpression } from '@tensor-trail/model-core/core/hyperparameters';
import { parseFieldPath } from '@tensor-trail/model-core/modules/indexing';
import { dictionaryFieldKey } from '@tensor-trail/model-core/core/dictionary';
import { useEditor } from './store';

export function IndexExpressionField({ node, field }: { node: ModelNode; field: Field }) {
  const parameters = useEditor((state) => state.project.hyperparameters);
  const value = String(node.params[field.key] ?? '');
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [node.id, value]);
  let preview: string;
  let invalid = false;
  try {
    preview =
      draft === '' && ['start', 'stop'].includes(field.key)
        ? field.key === 'start'
          ? '从该轴起点开始'
          : '取到该轴末尾'
        : `解析结果：${resolveIntegerExpression(draft, parameters)}`;
  } catch (error) {
    preview = (error as Error).message;
    invalid = true;
  }
  const list = `hyperparameters-${node.id}-${field.key}`;
  return (
    <div className="index-expression-field">
      <label>
        {field.label}
        <input
          aria-label={`${field.label} (${field.key})`}
          value={draft}
          maxLength={256}
          list={list}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') event.currentTarget.blur();
          }}
          onBlur={() => {
            if (draft !== value) useEditor.getState().setParam(node.id, field.key, draft);
          }}
        />
      </label>
      <datalist id={list}>
        {Object.entries(parameters ?? {})
          .filter(([, p]) => ['int', 'float'].includes(p.type))
          .map(([name]) => (
            <option key={name} value={name} />
          ))}
      </datalist>
      <small className={invalid ? 'expression-error' : 'muted'}>{preview}</small>
      <p className="microcopy">
        支持整数、负数及表达式，如 history_steps - 1；也可写 self.history_steps。
      </p>
    </div>
  );
}

export function DictionaryPathField({ node, input }: { node: ModelNode; input?: ValueSpec }) {
  const options: { path: (string | number)[]; label: string }[] = [];
  function walk(value: ValueSpec, path: (string | number)[]) {
    if (path.length)
      options.push({
        path,
        label: `${path.map((key) => `[${JSON.stringify(key)}]`).join('')} · ${shapeText(value)}`,
      });
    if (isDictionarySpec(value))
      Object.entries(value.fields).forEach(([key, child]) =>
        walk(child, [...path, dictionaryFieldKey(value, key)]),
      );
  }
  if (input && isDictionarySpec(input)) walk(input, []);
  let chosen = '';
  try {
    chosen = JSON.stringify(parseFieldPath(node.params.path));
  } catch {
    /* Incomplete path. */
  }
  if (!options.some((option) => JSON.stringify(option.path) === chosen)) chosen = '';
  return (
    <div className="dictionary-path-field">
      <label>
        字段路径
        <input
          aria-label="字段路径 (path)"
          value={String(node.params.path ?? '')}
          placeholder="agent.position"
          onChange={(event) => useEditor.getState().setParam(node.id, 'path', event.target.value)}
        />
      </label>
      <label>
        从输入字典选择
        <select
          aria-label="选择字典字段"
          value={chosen}
          disabled={!options.length}
          onChange={(event) => {
            if (event.target.value)
              useEditor.getState().setParam(node.id, 'path', event.target.value);
          }}
        >
          <option value="">{options.length ? '选择一个字段' : '先连接字典输入'}</option>
          {options.map((option) => (
            <option key={JSON.stringify(option.path)} value={JSON.stringify(option.path)}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <p className="microcopy">
        支持 agent.position 或 ["agent", "position"]。整数键使用数组格式，如 ["temporal_ego_states",
        0]；0 与 "0" 不等价。
      </p>
    </div>
  );
}
