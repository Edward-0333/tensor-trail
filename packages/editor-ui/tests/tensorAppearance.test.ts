import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it } from 'vitest';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { DEFAULT_TENSOR_PALETTE } from '@tensor-trail/model-core/core/tensorAppearance';
import { registry } from '@tensor-trail/model-core/modules/index';
import { useEditor } from '../src/editor/store';
import { TensorAppearancePanel } from '../src/editor/TensorAppearancePanel';
import { TensorView } from '../src/views/TensorView';

const state = () => useEditor.getState();
describe('Project-wide backgrounds and per-node borders', () => {
  beforeEach(() => {
    state().load({
      schemaVersion: 1,
      name: 'colors',
      variants: [],
      graph: { nodes: [registry.create('Input', {}, 'x')], edges: [] },
    });
    useEditor.setState({ past: [], future: [], path: [] });
  });
  it('changes and resets the shared palette with undo, redo and serialization', () => {
    state().setTensorPaletteColor('on', '#102030');
    expect(state().project.tensorPalette).toEqual({ ...DEFAULT_TENSOR_PALETTE, on: '#102030' });
    state().undo();
    expect(state().project.tensorPalette).toBeUndefined();
    state().redo();
    expect(state().project.tensorPalette?.on).toBe('#102030');
    expect(parseProject(serializeProject(state().project), registry).tensorPalette).toEqual(
      state().project.tensorPalette,
    );
    const before = state().project;
    state().setTensorPaletteColor('off', 'url(bad)');
    expect(state().project).toBe(before);
    state().resetTensorPalette();
    expect(state().project.tensorPalette).toBeUndefined();
  });
  it('stores borders separately from backgrounds, with a legacy setter alias', () => {
    state().setTensorBorderColor('x', '#123456');
    expect(state().project.graph.nodes[0].tensorBorderColor).toBe('#123456');
    expect(state().project.tensorPalette).toBeUndefined();
    expect(state().project.graph.nodes[0].tensorColor).toBeUndefined();
    state().setTensorColor('x', '#abcdef');
    expect(state().project.graph.nodes[0].tensorBorderColor).toBe('#abcdef');
    state().setTensorBorderColor('x');
    expect(state().project.graph.nodes[0].tensorBorderColor).toBeUndefined();
  });
  it('shares autograd annotations across linked values but not their borders', () => {
    state().duplicateLinked('x');
    const alias = state().selected!;
    state().setTensorAutograd(alias, true);
    const r = inferProject(state().project, registry);
    expect(r.nodes[alias].outputs).toEqual(r.nodes.x.outputs);
    expect(r.nodes.x.outputs.out).toHaveProperty('autograd', {
      requiresGrad: true,
      source: 'manual',
    });
    state().setTensorBorderColor(alias, '#123456');
    expect(
      state().project.graph.nodes.find((n) => n.id === 'x')?.tensorBorderColor,
    ).toBeUndefined();
    state().unlinkValue(alias);
    state().setTensorAutograd('x', false);
    expect(inferProject(state().project, registry).nodes[alias].outputs.out).toHaveProperty(
      'autograd.requiresGrad',
      true,
    );
    state().setTensorAutograd(alias);
    expect(inferProject(state().project, registry).nodes[alias].outputs.out).not.toHaveProperty(
      'autograd',
    );
  });
  it('rejects grad:on for integer tensors', () => {
    state().setParam('x', 'dtype', 'uint8');
    const before = state().project;
    state().setTensorAutograd('x', true);
    expect(state().project).toBe(before);
    expect(state().notice).toContain('不能');
  });
  it('renders accessible palette settings and distinguishes unknown from false in the inspector', () => {
    const panel = renderToStaticMarkup(createElement(TensorAppearancePanel));
    expect(panel).toContain('grad:on 统一背景色');
    expect(panel).toContain('grad:off 统一背景色');
    expect(panel).toContain('grad:? 统一背景色');
    const render = () =>
      renderToStaticMarkup(
        createElement(TensorView, { result: inferProject(state().project, registry).nodes.x }),
      );
    expect(render()).toContain('grad:?');
    state().setTensorAutograd('x', false);
    expect(render()).toContain('grad:off');
    expect(render()).toContain('手工标注');
  });
});
