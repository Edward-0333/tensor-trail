import json
from pathlib import Path
import tempfile
import unittest

import torch
from tensor_trail_runtime import describe_input, export_input_snapshot, update_project


class RuntimeInputTests(unittest.TestCase):
    def test_metadata_without_values_and_autograd_unchanged(self):
        value = torch.tensor([3.14159, 7.01234], requires_grad=True)
        data = {'agent': {'position': value, 'mask': torch.ones(0, 2, dtype=torch.bool)}}
        before = value.requires_grad
        schema, metadata = describe_input(data)
        self.assertEqual(schema['fields']['agent']['fields']['mask']['shape'], [0, 2])
        self.assertEqual(metadata[0]['requires_grad'], True)
        self.assertNotIn('3.14159', json.dumps([schema, metadata]))
        self.assertEqual(value.requires_grad, before)
        self.assertIsNone(value.grad)

    def test_scalar_and_nested_dictionary(self):
        schema, meta = describe_input({'nested': {}, 'scalar': torch.tensor(4, dtype=torch.int8)})
        self.assertEqual(schema['fields']['scalar']['shape'], [])
        self.assertEqual(meta[0]['dtype'], 'int8')
        self.assertEqual(meta[0]['path'], ['scalar'])

    def test_unsupported_values_and_keys_are_not_silently_dropped(self):
        for data in [{'x': [1, 2]}, {'x': None}, {1: torch.zeros(1)}, {'__proto__': torch.zeros(1)}]:
            with self.assertRaises((ValueError, TypeError)):
                describe_input(data)

    def test_update_preserves_graph_and_annotated_axes_with_exact_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snap = export_input_snapshot({'position': torch.zeros(2, 4)}, root/'snapshot.json')
            self.assertEqual(json.loads((root/'snapshot.json').read_text()), snap)
            project = {'schemaVersion': 1, 'name': 'pluto', 'variants': [], 'graph': {
                'nodes': [
                    {'id': 'input1', 'type': 'DictionaryInput', 'definitionVersion': 1,
                     'label': 'data', 'position': {'x': 200, 'y': 300}, 'tensorColor': '#123456',
                     'params': {'schema': json.dumps({'kind': 'dict', 'fields': {
                         'position': {'shape': ['B', 8], 'axes': ['batch', 'feature'], 'dtype': 'float32'}
                     }})}},
                    {'id': 'output', 'type': 'Output', 'definitionVersion': 1, 'label': 'result',
                     'params': {}, 'position': {'x': 500, 'y': 300}},
                ],
                'edges': [{'id': 'edge', 'source': 'input1', 'sourcePort': 'out', 'target': 'output', 'targetPort': 'in'}],
            }}
            path = root/'project.json'
            path.write_text(json.dumps(project))
            original = path.read_bytes()
            result = update_project(snap, path)
            self.assertEqual(Path(result['backup']).read_bytes(), original)
            updated = json.loads(path.read_text())
            self.assertEqual(updated['graph']['nodes'][0]['type'], 'ForwardInput')
            self.assertEqual(updated['graph']['edges'], project['graph']['edges'])
            self.assertEqual(updated['graph']['nodes'][1], project['graph']['nodes'][1])
            updated_schema = json.loads(updated['graph']['nodes'][0]['params']['schema'])
            self.assertEqual(updated_schema['fields']['position']['shape'], [2, 4])
            self.assertEqual(updated_schema['fields']['position']['axes'], ['batch', 'feature'])
            for key in ['id', 'label', 'position', 'tensorColor']:
                self.assertEqual(updated['graph']['nodes'][0][key], project['graph']['nodes'][0][key])

    def test_empty_project_and_ambiguous_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'project.json'
            snapshot = export_input_snapshot({'x': torch.zeros(2)}, Path(temporary)/'input.json')
            project = {'schemaVersion': 1, 'name': 'pluto', 'variants': [], 'graph': {'nodes': [], 'edges': []}}
            path.write_text(json.dumps(project))
            update_project(snapshot, path)
            self.assertEqual(json.loads(path.read_text())['graph']['nodes'][0]['type'], 'ForwardInput')
            project['graph']['nodes'] = [
                {'id': 'a', 'type': 'ForwardInput', 'label': 'first'},
                {'id': 'b', 'type': 'ForwardInput', 'label': 'second'},
            ]
            path.write_text(json.dumps(project))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Multiple inputs'):
                update_project(snapshot, path)
            self.assertEqual(path.read_bytes(), before)

    def test_existing_network_input_keeps_type_notes_and_layout_on_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'project.json'
            snapshot = export_input_snapshot({'mask': torch.zeros(2, 49, 101, dtype=torch.bool)}, Path(temporary)/'input.json')
            node = {'id': 'network', 'type': 'ForwardInput', 'definitionVersion': 1,
                    'label': 'data', 'position': {'x': 420, 'y': 200},
                    'params': {'text': '保留的备注', 'schema': ''}}
            project = {'schemaVersion': 1, 'name': 'pluto', 'variants': [], 'graph': {'nodes': [node], 'edges': []}}
            path.write_text(json.dumps(project))
            for _ in range(2):
                update_project(snapshot, path, node_id='network')
                updated = json.loads(path.read_text())['graph']['nodes'][0]
                self.assertEqual(updated['type'], 'ForwardInput')
                self.assertEqual(updated['params']['text'], '保留的备注')
                self.assertEqual(updated['position'], node['position'])
                self.assertEqual(json.loads(updated['params']['schema']), snapshot['schema'])

    def test_captured_model_hyperparameters_seed_missing_values_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = export_input_snapshot({'x': torch.zeros(2)}, root/'input.json', context={
                'model_config': {'history_steps': 21, 'dropout': 0.1, 'enabled': False,
                                 '_target_': 'Model', 'nested': {'hidden': 8}, 'name': 'test'}
            })
            path = root/'project.json'
            project = {'schemaVersion': 1, 'name': 'test', 'variants': [],
                       'hyperparameters': {'history_steps': {'type': 'int', 'value': 10}},
                       'graph': {'nodes': [], 'edges': []}}
            path.write_text(json.dumps(project))
            update_project(snapshot, path)
            parameters = json.loads(path.read_text())['hyperparameters']
            self.assertEqual(parameters['history_steps']['value'], 10)
            self.assertEqual(parameters['dropout']['type'], 'float')
            self.assertEqual(parameters['enabled']['value'], False)
            self.assertEqual(parameters['name']['type'], 'string')
            self.assertNotIn('_target_', parameters)
            self.assertNotIn('nested', parameters)


if __name__ == '__main__':
    unittest.main()
