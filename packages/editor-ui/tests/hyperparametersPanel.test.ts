import { parseHTML } from 'linkedom';
import { act, createElement } from 'react';
import type { Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HyperparametersPanel } from '../src/editor/HyperparametersPanel';
import { useEditor } from '../src/editor/store';

describe('string hyperparameter editing', () => {
  let root: Root;
  const state = useEditor.getState;
  const input = (label: string) => document.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;

  beforeEach(async () => {
    const { window } = parseHTML('<html><body><main></main></body></html>');
    vi.stubGlobal('window', window);
    vi.stubGlobal('document', window.document);
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    document.oninput = null; // Expose input-event support before React initializes its event system.
    state().newProject();
    const { createRoot } = await import('react-dom/client');
    root = createRoot(document.querySelector('main')!);
    await act(() => root.render(createElement(HyperparametersPanel)));
  });
  afterEach(async () => {
    await act(() => root.unmount());
    vi.unstubAllGlobals();
  });

  // Update the DOM value before dispatching the browser's input event to React.
  async function typeText(label: string, text: string) {
    const field = input(label);
    field.type = 'text'; // linkedom omits the browser's default input type.
    field.setAttribute('value', text);
    await act(() => { field.dispatchEvent(new window.Event('input', { bubbles: true })); });
  }
  const submit = () => act(() => {
    document.querySelector('form')!.dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  });

  it('adds and edits non-BMP values up to 1999 code points and rejects 2000', async () => {
    const select = document.querySelector<HTMLSelectElement>('[aria-label="新超参数类型"]')!;
    const option = Array.from(select.options).find(o => o.textContent === 'string')!;
    option.value = 'string'; // linkedom does not derive the option value from its text.
    option.selected = true;
    await act(() => { select.dispatchEvent(new window.Event('change', { bubbles: true })); });
    expect(input('新超参数值').hasAttribute('maxlength')).toBe(false);
    await typeText('新超参数名称', 'label');
    const full = '𠮷'.repeat(1999);
    await typeText('新超参数值', full + '𠮷');
    await submit();
    expect(state().project.hyperparameters?.label).toBeUndefined();
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('超出范围');
    const thousand = '𠮷'.repeat(1000);
    await typeText('新超参数值', thousand);
    await submit();
    expect(state().project.hyperparameters?.label?.value).toBe(thousand);

    const edit = async (text: string) => {
      const field = input('超参数值 label');
      expect(field.hasAttribute('maxlength')).toBe(false);
      field.value = text;
      await act(() => { field.dispatchEvent(new window.Event('focusout', { bubbles: true })); });
    };
    await edit(full);
    expect(state().project.hyperparameters?.label?.value).toBe(full);
    const before = state().project;
    await edit(full + '𠮷');
    expect(state().project).toBe(before);
    expect(input('超参数值 label').value).toBe(full);
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('超出范围');
  });
});
