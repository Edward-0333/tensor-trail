import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { buildModel } from '@tensor-trail/model-core/agent';
import { inferProject, resultAt } from '@tensor-trail/model-core/core/engine';
import { resolveForwardView } from '@tensor-trail/model-core/core/hierarchy';
import { serializeProject } from '@tensor-trail/model-core/core/serialization';
import { registry } from '@tensor-trail/model-core/modules';
import { loopRecipe, reshapedFeatures } from '../../model-core/tests/fixtures/forEach';
import { ForEachControls } from '../src/editor/ForEachControls';
import { useEditor } from '../src/editor/store';

describe('ForEach iteration viewing', () => {
  it('selects each body result without changing the project, history, runtime or final output', () => {
    useEditor.getState().load(buildModel(loopRecipe()).project!);
    useEditor.getState().navigate(['loop']);
    const before = useEditor.getState(),
      serialized = serializeProject(before.project);
    const inference = inferProject(before.project, registry);
    for (let i = 0; i < 4; i++) {
      useEditor.getState().selectLoopIteration(['loop'], i);
      const state = useEditor.getState();
      expect(resultAt(inference, state.path, state.loopSelections)!.outputs.out).toMatchObject({
        shape: reshapedFeatures[i],
      });
      expect(state.project).toBe(before.project);
      expect(state.past).toBe(before.past);
      expect(state.future).toBe(before.future);
      expect(state.runtimeSnapshot).toBe(before.runtimeSnapshot);
    }
    expect(serializeProject(useEditor.getState().project)).toBe(serialized);
    expect(resolveForwardView(before.project.graph, ['loop']).current.title).toContain('循环体');
    useEditor.getState().load(before.project);
    expect(useEditor.getState().loopSelections).toEqual({});
  });
  it('renders the selected iteration and collection progress, with a static-only disclaimer', () => {
    const result = inferProject(buildModel(loopRecipe()).project!, registry).nodes.loop;
    const html = renderToStaticMarkup(
      createElement(ForEachControls, { result, selected: 3, onSelect: () => {} }),
    );
    expect(html).toContain('循环次数：4');
    expect(html).toContain('[60, 256, 8, 22]');
    expect(html).toContain('[2, 30, 256, 8, 22]');
    expect(html).toContain('列表元素数：3 → 4');
    expect(html).toContain('不执行 Python');
    expect(html).toContain('value="3" selected');
    const empty = renderToStaticMarkup(
      createElement(ForEachControls, { result: { ...result, iterations: [] }, onSelect: () => {} }),
    );
    expect(empty).toContain('跳过循环体');
    expect(empty).not.toContain('<select');
    const failed = renderToStaticMarkup(
      createElement(ForEachControls, {
        result: { ...result, status: 'error' },
        onSelect: () => {},
      }),
    );
    expect(failed).toContain('尚无有效的完整收集结果');
    expect(failed).not.toContain('列表元素数');
  });
});
