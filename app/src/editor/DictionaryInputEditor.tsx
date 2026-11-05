import { useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import type { DictionarySpec, ModelNode, TensorSpec } from '../core/types';
import { clone, isDictionarySpec } from '../core/types';
import {
  checkDictionaryKey,
  DICTIONARY_DTYPES,
  dictionaryShape,
  emptyDictionary,
  parseDictionarySchema,
} from '../core/dictionary';
import { useEditor } from './store';
import { valueOwner } from '../core/valueLinks';

export function DictionaryInputEditor({ node }: { node: ModelNode }) {
  const [error, setError] = useState('');
  let schema: DictionarySpec;
  try {
    schema = parseDictionarySchema(node.params.schema);
  } catch (e) {
    return (
      <div className="dictionary-editor">
        <p className="diagnostic error">{(e as Error).message}</p>
        <p className="muted">修正结构描述后可继续编辑字段。</p>
        <textarea
          aria-label="修正字典结构描述"
          defaultValue={String(node.params.schema ?? '')}
          onBlur={(event) => useEditor.getState().setParam(node.id, 'schema', event.target.value)}
        />
      </div>
    );
  }

  function change(path: string[], update: (dict: DictionarySpec) => void) {
    try {
      const draft = clone(schema);
      let parent = draft;
      for (const key of path) {
        const next = parent.fields[key];
        if (!next || !isDictionarySpec(next)) throw new Error('字段路径已失效，请重新选择。');
        parent = next;
      }
      update(parent);
      const text = JSON.stringify(parseDictionarySchema(JSON.stringify(draft)));
      if (text === JSON.stringify(schema)) {
        setError('');
        return;
      }
      const committed = useEditor.getState().edit((graph) => {
        const current = valueOwner(graph, node.id);
        current.params.schema = text;
      });
      setError(committed ? '' : useEditor.getState().notice);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function fields(dict: DictionarySpec, path: string[]) {
    return (
      <div className="dictionary-fields">
        {Object.entries(dict.fields).map(([key, value]) => {
          const fieldPath = [...path, key];
          const label = fieldPath.map((part) => `[${JSON.stringify(part)}]`).join('');
          const nested = isDictionarySpec(value);
          const updateTensor = (update: (tensor: TensorSpec) => void) =>
            change(path, (parent) => {
              const tensor = parent.fields[key];
              if (isDictionarySpec(tensor)) throw new Error('当前字段不是 Tensor。');
              update(tensor);
            });
          return (
            <section className="dictionary-field" key={key}>
              <div className="dictionary-field-heading">
                <input
                  aria-label={`字段名 ${label}`}
                  defaultValue={key}
                  onBlur={(event) => {
                    const name = event.target.value;
                    if (name === key) {
                      setError('');
                      return;
                    }
                    change(path, (parent) => {
                      checkDictionaryKey(name);
                      if (Object.hasOwn(parent.fields, name))
                        throw new Error(`字段 ${name} 已存在。`);
                      parent.fields = Object.fromEntries(
                        Object.entries(parent.fields).map(([old, child]) => [
                          old === key ? name : old,
                          child,
                        ]),
                      );
                    });
                  }}
                />
                <span>{nested ? 'dict' : 'Tensor'}</span>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`删除字段 ${label}`}
                  onClick={() =>
                    change(path, (parent) => {
                      delete parent.fields[key];
                    })
                  }
                >
                  <Trash2 size={13} />
                </button>
              </div>
              {nested ? (
                <details>
                  <summary>展开 {Object.keys(value.fields).length} 个字段</summary>
                  {fields(value, fieldPath)}
                </details>
              ) : (
                <div className="dictionary-tensor-fields">
                  <label>
                    形状
                    <input
                      aria-label={`形状 ${label}`}
                      key={JSON.stringify(value.shape)}
                      defaultValue={value.shape.length ? value.shape.join(', ') : '[]'}
                      onBlur={(event) =>
                        updateTensor((tensor) => {
                          const shape = dictionaryShape(event.target.value);
                          tensor.axes = shape.map((_, i) => tensor.axes[i] ?? `axis${i}`);
                          tensor.shape = shape;
                        })
                      }
                    />
                  </label>
                  <label>
                    类型
                    <select
                      aria-label={`类型 ${label}`}
                      value={value.dtype}
                      onChange={(event) =>
                        updateTensor((tensor) => {
                          tensor.dtype = event.target.value;
                        })
                      }
                    >
                      {DICTIONARY_DTYPES.map((dtype) => (
                        <option key={dtype}>{dtype}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    轴含义
                    <input
                      aria-label={`轴含义 ${label}`}
                      key={JSON.stringify(value.axes)}
                      defaultValue={value.axes.join(', ')}
                      onBlur={(event) =>
                        updateTensor((tensor) => {
                          tensor.axes = event.target.value.trim()
                            ? event.target.value.split(',').map((axis) => axis.trim())
                            : tensor.shape.map((_, i) => `axis${i}`);
                        })
                      }
                    />
                  </label>
                </div>
              )}
            </section>
          );
        })}
        <div className="dictionary-add-actions">
          {(['Tensor', 'dict'] as const).map((kind) => (
            <button
              type="button"
              className="soft-button"
              key={kind}
              aria-label={`添加 ${kind} 到 ${path.length ? path.join('.') : '根字典'}`}
              onClick={() =>
                change(path, (parent) => {
                  let key = kind === 'dict' ? 'group' : 'tensor';
                  const base = key;
                  for (let i = 2; Object.hasOwn(parent.fields, key); i++) key = `${base}${i}`;
                  parent.fields[key] =
                    kind === 'dict'
                      ? emptyDictionary()
                      : {
                          shape: ['B'],
                          axes: ['batch'],
                          dtype: 'float32',
                        };
                })
              }
            >
              <Plus size={12} />
              {kind === 'dict' ? '子字典' : 'Tensor'}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="dictionary-editor">
      <p className="muted">
        定义字段结构；整份字典从一个端口输出。形状支持 B、T 等符号，[] 表示标量，0 表示空维度。
      </p>
      {error && (
        <div role="alert" className="diagnostic error">
          {error}
        </div>
      )}
      {fields(schema, [])}
    </div>
  );
}
