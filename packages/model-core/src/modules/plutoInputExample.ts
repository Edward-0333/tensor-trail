import type { DictionarySpec, Dim, Project, TensorSpec } from '../core/types';
import { registry } from './index';

// Input-contract example from PLUTO's forward/encoders and feature builder.
// Includes fields read by the model, not every training/cache auxiliary field.
// T is the input sequence length, not a fixed history_steps=21.
export function plutoInputSchema(): DictionarySpec {
  const t = (shape: Dim[], dtype = 'float32'): TensorSpec => ({
    shape,
    dtype,
    axes: shape.map((dim, i) => (typeof dim === 'string' ? dim : `axis${i}`)),
  });
  const dict = (fields: DictionarySpec['fields']): DictionarySpec => ({ kind: 'dict', fields });
  return dict({
    agent: dict({
      position: t(['B', 'A', 'T', 2]),
      heading: t(['B', 'A', 'T']),
      velocity: t(['B', 'A', 'T', 2]),
      shape: t(['B', 'A', 'T', 2]),
      category: t(['B', 'A'], 'int8'),
      valid_mask: t(['B', 'A', 'T'], 'bool'),
    }),
    map: dict({
      polygon_center: t(['B', 'L', 3]),
      valid_mask: t(['B', 'L', 'P'], 'bool'),
      point_position: t(['B', 'L', 3, 'P', 2]),
      point_vector: t(['B', 'L', 3, 'P', 2]),
      point_orientation: t(['B', 'L', 3, 'P']),
      polygon_type: t(['B', 'L'], 'int8'),
      polygon_on_route: t(['B', 'L'], 'bool'),
      polygon_tl_status: t(['B', 'L'], 'int8'),
      polygon_has_speed_limit: t(['B', 'L'], 'bool'),
      polygon_speed_limit: t(['B', 'L']),
    }),
    static_objects: dict({
      position: t(['B', 'S', 2]),
      heading: t(['B', 'S']),
      shape: t(['B', 'S', 2]),
      category: t(['B', 'S']),
      valid_mask: t(['B', 'S'], 'bool'),
    }),
    reference_line: dict({
      position: t(['B', 'R', 'Q', 2]),
      vector: t(['B', 'R', 'Q', 2]),
      orientation: t(['B', 'R', 'Q']),
      valid_mask: t(['B', 'R', 'Q'], 'bool'),
    }),
    current_state: t(['B', 7]),
  });
}

export function plutoForwardInputExample(): Project {
  const node = registry.create(
    'ForwardInput',
    { schema: JSON.stringify(plutoInputSchema()) },
    'data',
  );
  node.label = 'data';
  node.position = { x: 120, y: 180 };
  return {
    schemaVersion: 1,
    name: 'PLUTO · forward 输入字段示例',
    graph: { nodes: [node], edges: [] },
    variants: [],
  };
}
