import type { Project } from '../core/types';
import { registry } from './index';

export function example(kind: 'transformer' | 'mixed' | 'blank' = 'transformer'): Project {
  const project: Project = {
    schemaVersion: 1,
    name:
      kind === 'mixed'
        ? 'Transformer × CNN × LSTM'
        : kind === 'blank'
          ? '我的新网络'
          : '走进 Transformer',
    graph: { nodes: [], edges: [] },
    variants: [],
  };
  const input = registry.create('Input', {}, 'input');
  input.position = { x: 0, y: 120 };
  project.graph.nodes.push(input);
  if (kind === 'blank') return project;
  const block = registry.create('Transformer', {}, 'transformer');
  block.position = { x: 290, y: 120 };
  project.graph.nodes.push(block);
  if (kind === 'mixed') {
    project.graph.nodes.push(
      registry.create('Permute', {}, 'to_channels'),
      registry.create('Conv1d', {}, 'cnn'),
      registry.create('Permute', {}, 'to_sequence'),
      registry.create('LSTM', { input_size: 128, hidden_size: 32 }, 'lstm'),
    );
  }
  project.graph.nodes.push(registry.create('Output', {}, 'output'));
  project.graph.nodes.forEach((node, i) => {
    node.position = { x: i < 4 ? i * 290 : (7 - i) * 290, y: i < 4 ? 100 : 390 };
    if (i)
      project.graph.edges.push({
        id: `edge_${i}`,
        source: project.graph.nodes[i - 1].id,
        sourcePort: 'out',
        target: node.id,
        targetPort: 'in',
      });
  });
  return project;
}
