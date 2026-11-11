import type { DictionarySpec } from '@tensor-trail/model-core/core/types';
import { isDictionarySpec, isTensorSpec } from '@tensor-trail/model-core/core/types';
import { valueDisplayText } from './valueDisplayText';
import { gradLabel, tensorGradState } from '@tensor-trail/model-core/core/tensorAppearance';

export function DictionaryTree({ value }: { value: DictionarySpec }) {
  return (
    <ul className="dictionary-value-tree">
      {Object.entries(value.fields).map(([key, child]) => (
        <li key={key}>
          {isDictionarySpec(child) ? (
            <details>
              <summary>
                <code>{value.keyType === 'int' ? `${key} (int)` : key}</code> · {valueDisplayText(child)}
              </summary>
              <DictionaryTree value={child} />
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
