import type { Graph, ModelNode } from '../core/types';

export interface LocalParameterColor {
  ownerId: string;
  index: number;
  count: number;
  color: string;
}

const palette = ['orange', 'blue', 'purple', 'teal', 'pink', 'green', 'cyan', 'sky'];
const definitionColor = (index: number) =>
  index < palette.length
    ? `var(--category-${palette[index]})`
    : `hsl(${((index * 137.508) % 360).toFixed(3)} 62% 48%)`;

/** Names may be rebound. Color the value definition, never its current numeric result.
 * Aliases use their canonical owner, so moving/copying an alias or changing input sizes
 * cannot change the color. This is derived view state, scoped to one forward graph.
 */
export function localParameterColors(graph: Graph): Map<string, LocalParameterColor> {
  const names = new Map<string, ModelNode[]>();
  for (const node of graph.nodes) {
    if (node.type !== 'LocalParameter' || node.valueSource !== undefined) continue;
    const name = node.label.trim();
    const definitions = names.get(name) ?? [];
    definitions.push(node);
    names.set(name, definitions);
  }
  const owners = new Map<string, LocalParameterColor>();
  for (const definitions of names.values())
    definitions.forEach((node, index) =>
      owners.set(node.id, {
        ownerId: node.id,
        index,
        count: definitions.length,
        color: definitionColor(index),
      }),
    );
  const colors = new Map<string, LocalParameterColor>();
  for (const node of graph.nodes) {
    if (node.type !== 'LocalParameter') continue;
    const color = owners.get(node.valueSource ?? node.id);
    if (color) colors.set(node.id, color);
  }
  return colors;
}
