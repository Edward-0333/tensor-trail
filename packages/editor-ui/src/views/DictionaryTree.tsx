import type { DictionarySpec, TensorSequenceSpec } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSequenceSpec, isTensorSpec } from '@tensor-trail/model-core/core/types';
import { valueDisplayText } from './valueDisplayText';
import { gradLabel, tensorGradState } from '@tensor-trail/model-core/core/tensorAppearance';

export function TensorSequenceTree({ value }: { value: TensorSequenceSpec }) {
  return (
    <ul className="dictionary-value-tree">
      {value.items.map((item, i) => (
        <li key={i}>
          <code>[{i}]</code>
          <span>{valueDisplayText(item)} · {item.dtype} · {gradLabel(tensorGradState(item))}</span>
        </li>
      ))}
      {!value.items.length && <li className="muted">{value.kind === 'list' ? '空列表 []' : '空元组 ()'}</li>}
    </ul>
  );
}

export function DictionaryTree({ value }: { value: DictionarySpec }) {
  return (
    <ul className="dictionary-value-tree">
      {Object.entries(value.fields).map(([key, child]) => (
        <li key={key}>
          {isDictionarySpec(child) || isTensorSequenceSpec(child) ? (
            <details>
              <summary>
                <code>{value.keyType === 'int' ? `${key} (int)` : key}</code> · {valueDisplayText(child)}
              </summary>
              {isDictionarySpec(child) ? <DictionaryTree value={child} /> : <TensorSequenceTree value={child} />}
            </details>
          ) : (
            <>
              <code>{value.keyType === 'int' ? `${key} (int)` : key}</code>
              <span>
                {valueDisplayText(child)}
                {isTensorSpec(child) ? ` · ${child.dtype} · ${gradLabel(tensorGradState(child))}` : ''}
              </span>
            </>
          )}
        </li>
      ))}
      {!Object.keys(value.fields).length && <li className="muted">空字典</li>}
    </ul>
  );
}
