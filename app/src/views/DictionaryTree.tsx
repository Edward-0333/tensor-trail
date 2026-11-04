import type { DictionarySpec } from '../core/types';
import { isDictionarySpec, shapeText } from '../core/types';

export function DictionaryTree({ value }: { value: DictionarySpec }) {
  return (
    <ul className="dictionary-value-tree">
      {Object.entries(value.fields).map(([key, child]) => (
        <li key={key}>
          {isDictionarySpec(child) ? (
            <details>
              <summary>
                <code>{key}</code> · {shapeText(child)}
              </summary>
              <DictionaryTree value={child} />
            </details>
          ) : (
            <>
              <code>{key}</code>
              <span>
                {shapeText(child)} · {child.dtype}
              </span>
            </>
          )}
        </li>
      ))}
      {!Object.keys(value.fields).length && <li className="muted">空字典</li>}
    </ul>
  );
}
