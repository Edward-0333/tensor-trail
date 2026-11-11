import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { expect, it } from 'vitest';
import { ports } from '@tensor-trail/model-core/core/graph';
import type { Params } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { ModuleNode, type TrailNode } from '../src/views/ModuleNode';
import { valueEdgeStyle } from '../src/views/valueEdgeStyle';
import { canvasValueLabel } from '../src/views/canvasValueLabel';
import { TensorView } from '../src/views/TensorView';
import { useEditor } from '../src/editor/store';
import { buildModel } from '@tensor-trail/model-core/agent';

function render(type: string, params: Params = {}) {
  const model = registry.create(type,params,'node');
  const props: NodeProps<TrailNode> = {id:model.id,type:'module',selected:false,dragging:false,zIndex:0,
    selectable:true,deletable:true,draggable:true,isConnectable:true,positionAbsoluteX:0,positionAbsoluteY:0,
    data:{model,...ports(model,registry),category:registry.get(type)!.category,reverse:false,active:false,dimmed:false,open:()=>{}}};
  return renderToStaticMarkup(createElement(ReactFlowProvider,null,createElement(ModuleNode,props)));
}
it('shows While as a bounded state template, never a network forward or iteration trace',()=>{
  const html=render('WhileLoop'); expect(html).toContain('while-loop-node');
  expect(html).toContain('i &lt; num_points'); expect(html).toContain('一轮符号模板 · 未执行');
  expect(html).not.toContain('forward()'); expect(html).not.toContain('tensor-grad-badge');
});
it('shows a compact comparison with separate branch scope presentation',()=>{
  expect(render('IfCondition')).toContain('condition-test-node'); expect(render('IfCondition')).toContain('条件比较 · 未执行');
  expect(render('IfCondition')).not.toContain('condition-node');
  expect(render('BranchValue')).toContain('True 分支');
  expect(render('BranchValue',{branch:false})).toContain('False 分支');
  expect(render('BranchValue',{branch:false})).toContain('maps 不修改');
});
it('uses dashed control wires and keeps regular Tensor wires solid',()=>{
  const control={kind:'control' as const,expression:'count == quota',branch:true};
  expect(valueEdgeStyle(control).strokeDasharray).toBe('7 4'); expect(canvasValueLabel(control)).toBe('True');
  expect(valueEdgeStyle({shape:[1040],axes:['query'],dtype:'float32'}).strokeDasharray).toBeUndefined();
  const html=renderToStaticMarkup(createElement(TensorView,{result:{status:'ok',inputs:{},outputs:{out:control},notes:[],messages:[]}}));
  expect(html).toContain('未执行'); expect(html).not.toContain('tensor-autograd-summary');
});
it('shows quotas as a Python integer list, not as a Tensor',()=>{
  const html=renderToStaticMarkup(createElement(TensorView,{result:{status:'ok',inputs:{},outputs:{out:{kind:'int-list',values:[720,60,60,60,60,40,40]}},notes:[],messages:[]}}));
  expect(html).toContain('list[int]'); expect(html).toContain('不是 Tensor'); expect(html).not.toContain('tensor-axes');
});
it('does not save a state loop body as an ordinary stateless variant',()=>{
  const built=buildModel({name:'loop',nodes:[{id:'loop',type:'WhileLoop'}],connections:[]},{mode:'draft'});
  expect(built.ok).toBe(true); useEditor.getState().load(built.project!); useEditor.getState().navigate(['loop']);
  const variants=useEditor.getState().project.variants;
  useEditor.getState().saveVariant('invalid');
  expect(useEditor.getState().project.variants).toBe(variants);
  expect(useEditor.getState().notice).toContain('状态反馈语义');
});
