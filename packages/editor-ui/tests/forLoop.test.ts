import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { inferProject, resultAt } from '@tensor-trail/model-core/core/engine';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { registry } from '@tensor-trail/model-core/modules';
import { decoderLoopRecipe } from '../../model-core/tests/fixtures/decoderLoop';
import { ForLoopControls } from '../src/editor/ForLoopControls';
import { useEditor } from '../src/editor/store';

describe('ForLoop iteration UI', () => {
  it('shows layer-specific shapes and append progress, not shared-weight semantics', () => {
    const p = buildModel(decoderLoopRecipe()).project!, result = inferProject(p, registry).nodes.loop;
    const html = renderToStaticMarkup(createElement(ForLoopControls, { result, count: 6, selected: 5, onSelect: () => {} }));
    expect(html).toContain('value="5" selected');
    expect(html).toContain('[2, 1040, 32, 3] → [2, 1040, 48, 3]');
    expect(html).toContain('list 5 → 6');
    expect(html).toContain('不表示网络权重共享');
    expect(html).toContain('不执行 Python');
  });

  it('does not mutate project/history when selecting a layer, and prevents exporting a body as a normal variant', () => {
    const p = buildModel(decoderLoopRecipe()).project!;
    useEditor.getState().load(p);
    useEditor.getState().navigate(['loop']);
    const before = useEditor.getState(), r = inferProject(p, registry);
    useEditor.getState().selectLoopIteration(['loop'], 4);
    const after = useEditor.getState();
    expect(after.project).toBe(before.project);
    expect(after.past).toBe(before.past);
    expect(resultAt(r, after.path, after.loopSelections)!.outputs.query_points).toMatchObject({ shape: [2, 1040, 32, 3] });
    expect(resolveForwardView(p.graph, ['loop']).current.title).toContain('循环体');
    useEditor.getState().saveVariant('not a normal submodule');
    expect(useEditor.getState().notice).toContain('ForLoop 具有状态反馈语义');
    expect(useEditor.getState().project).toBe(before.project);
  });

  it('shows zero iterations and failures without claiming valid state updates', () => {
    const result = inferProject(buildModel(decoderLoopRecipe()).project!, registry).nodes.loop;
    const render = (r: typeof result) => renderToStaticMarkup(createElement(ForLoopControls, { result: r, count: 6, onSelect: () => {} }));
    expect(render({ ...result, iterations: [] })).toContain('保留初始状态');
    const failure = render({ ...result, status: 'error', outputs: {} });
    expect(failure).toContain('尚无有效的最终状态');
    expect(failure).not.toContain('list 0');
  });
});
