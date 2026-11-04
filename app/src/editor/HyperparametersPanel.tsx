import { useEffect, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import type { Hyperparameter } from '../core/types';
import { checkHyperparameterName, parseHyperparameterValue } from '../core/hyperparameters';
import { useEditor } from './store';

const types: Hyperparameter['type'][] = ['int', 'float', 'bool', 'string'];
const defaultValue = (type: Hyperparameter['type']) =>
  type === 'bool' ? false : type === 'string' ? '' : 0;

export function HyperparametersPanel() {
  const state = useEditor();
  const parameters = state.project.hyperparameters ?? {};
  const [name, setName] = useState('');
  const [type, setType] = useState<Hyperparameter['type']>('int');
  const [value, setValue] = useState('0');
  const [error, setError] = useState('');
  useEffect(() => {
    setName('');
    setError('');
  }, [state.documentId]);
  const commit = (key: string, parameter?: Hyperparameter) => {
    const ok = state.setHyperparameter(key, parameter);
    setError(ok ? '' : useEditor.getState().notice);
    return ok;
  };
  if (!state.hasProject) return <p className="muted">打开或新建项目后，可设置模型超参数。</p>;
  return (
    <section className="hyperparameters-panel">
      <p className="muted">
        项目级公共配置，可在整数索引中引用，例如 history_steps - 1。当前层及子图共享这些值。
      </p>
      <p className="microcopy">修改后重新推导图中形状；训练配置和已采集的 data 快照保持独立。</p>
      {error && (
        <div role="alert" className="diagnostic error">
          {error}
        </div>
      )}
      <div className="hyperparameter-list">
        {Object.entries(parameters).map(([key, parameter]) => (
          <div className="hyperparameter-row" key={key}>
            <div className="hyperparameter-heading">
              <code>{key}</code>
              <button
                className="icon-button danger"
                aria-label={`删除超参数 ${key}`}
                onClick={() => commit(key)}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div className="hyperparameter-value">
              <select
                aria-label={`超参数类型 ${key}`}
                value={parameter.type}
                onChange={(event) => {
                  const type = event.target.value as Hyperparameter['type'];
                  commit(key, { type, value: defaultValue(type), source: '手动编辑' });
                }}
              >
                {types.map((type) => (
                  <option key={type}>{type}</option>
                ))}
              </select>
              {parameter.type === 'bool' ? (
                <select
                  aria-label={`超参数值 ${key}`}
                  value={String(parameter.value)}
                  onChange={(event) =>
                    commit(key, {
                      ...parameter,
                      value: event.target.value === 'true',
                      source: '手动编辑',
                    })
                  }
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : (
                <input
                  aria-label={`超参数值 ${key}`}
                  key={`${parameter.type}:${parameter.value}`}
                  defaultValue={String(parameter.value)}
                  maxLength={1999}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') event.currentTarget.blur();
                  }}
                  onBlur={(event) => {
                    try {
                      const value = parseHyperparameterValue(parameter.type, event.target.value);
                      if (value !== parameter.value)
                        commit(key, { ...parameter, value, source: '手动编辑' });
                      else setError('');
                    } catch (error) {
                      setError((error as Error).message);
                    }
                  }}
                />
              )}
            </div>
            {parameter.source && <small className="muted">来源：{parameter.source}</small>}
          </div>
        ))}
      </div>
      {!Object.keys(parameters).length && <p className="muted">尚未定义超参数。</p>}
      <form
        className="hyperparameter-add"
        onSubmit={(event) => {
          event.preventDefault();
          try {
            const key = name.trim();
            checkHyperparameterName(key);
            if (Object.hasOwn(parameters, key)) throw new Error(`超参数 ${key} 已存在。`);
            if (
              commit(key, {
                type,
                value: parseHyperparameterValue(type, value),
                source: '手动添加',
              })
            )
              setName('');
          } catch (error) {
            setError((error as Error).message);
          }
        }}
      >
        <label>
          名称
          <input
            aria-label="新超参数名称"
            value={name}
            maxLength={64}
            placeholder="history_steps"
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <div className="hyperparameter-value">
          <select
            aria-label="新超参数类型"
            value={type}
            onChange={(event) => {
              const type = event.target.value as Hyperparameter['type'];
              setType(type);
              setValue(String(defaultValue(type)));
            }}
          >
            {types.map((type) => (
              <option key={type}>{type}</option>
            ))}
          </select>
          {type === 'bool' ? (
            <select
              aria-label="新超参数值"
              value={value}
              onChange={(event) => setValue(event.target.value)}
            >
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          ) : (
            <input
              aria-label="新超参数值"
              value={value}
              maxLength={1999}
              onChange={(event) => setValue(event.target.value)}
            />
          )}
        </div>
        <button className="soft-button" type="submit">
          <Plus size={14} />
          添加超参数
        </button>
      </form>
    </section>
  );
}
