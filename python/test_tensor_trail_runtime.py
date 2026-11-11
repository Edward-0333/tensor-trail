import json
from pathlib import Path
import tempfile
import unittest

import torch
from tensor_trail_runtime import describe_input, export_input_snapshot, update_project, register_runtime_snapshot


class RuntimeInputTests(unittest.TestCase):
    def test_metadata_without_values_and_autograd_unchanged(self):
        value = torch.tensor([3.14159, 7.01234], requires_grad=True)
        data = {'agent': {'position': value, 'mask': torch.ones(0, 2, dtype=torch.bool)}}
        before = value.requires_grad
        schema, metadata = describe_input(data)
        self.assertEqual(schema['fields']['agent']['fields']['mask']['shape'], [0, 2])
        self.assertEqual(metadata[0]['requires_grad'], True)
        self.assertEqual(schema['fields']['agent']['fields']['position']['autograd'],
                         {'requiresGrad': True, 'source': 'runtime'})
        self.assertEqual(schema['fields']['agent']['fields']['mask']['autograd'],
                         {'requiresGrad': False, 'source': 'runtime'})
        self.assertNotIn('3.14159', json.dumps([schema, metadata]))
        self.assertEqual(value.requires_grad, before)
        self.assertIsNone(value.grad)

    def test_scalar_and_nested_dictionary(self):
        schema, meta = describe_input({'nested': {}, 'scalar': torch.tensor(4, dtype=torch.int8)})
        self.assertEqual(schema['fields']['scalar']['shape'], [])
        self.assertEqual(meta[0]['dtype'], 'int8')
        self.assertEqual(meta[0]['path'], ['scalar'])

    def test_unsupported_values_and_keys_are_not_silently_dropped(self):
        for data in [{'x': [1, 2]}, {'x': None}, {True: torch.zeros(1)}, {'__proto__': torch.zeros(1)}]:
            with self.assertRaises((ValueError, TypeError)):
                describe_input(data)

    def test_integer_time_keys_keep_their_type_and_tensor_axes(self):
        states = {i: torch.zeros(1, 1, 21) for i in range(6)}
        schema, metadata = describe_input({'temporal_ego_states': states})
        temporal = schema['fields']['temporal_ego_states']
        self.assertEqual(temporal['keyType'], 'int')
        self.assertEqual(list(temporal['fields']), ['0', '1', '2', '3', '4', '5'])
        self.assertEqual(temporal['fields']['0']['shape'], [1, 1, 21])
        self.assertEqual(metadata[0]['path'], ['temporal_ego_states', 0])
        strings, _ = describe_input({'0': torch.zeros(1)})
        self.assertNotIn('keyType', strings)

    def test_mixed_boolean_and_unsafe_integer_keys_are_rejected(self):
        for value in [{0: torch.zeros(1), '0': torch.zeros(1)},
                      {False: torch.zeros(1)}, {2 ** 53: torch.zeros(1)}]:
            with self.assertRaises(ValueError):
                describe_input({'temporal': value})

    def test_project_refresh_preserves_integer_key_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = export_input_snapshot({'states': {0: torch.zeros(1, 1, 21)}}, root/'input.json')
            path = root/'project.json'
            path.write_text(json.dumps({'schemaVersion': 1, 'name': 'test', 'variants': [],
                                        'graph': {'nodes': [], 'edges': []}}), encoding='utf-8')
            update_project(snapshot, path)
            project = json.loads(path.read_text(encoding='utf-8'))
            schema = project['graph']['nodes'][0]['params']['schema']
            self.assertEqual(schema['fields']['states']['keyType'], 'int')
            self.assertNotIn('autograd', schema['fields']['states']['fields']['0'])
            run = json.loads((root / project['runtime']['snapshots'][0]['file']).read_text())
            self.assertEqual(run['bindings'][0]['value'], snapshot['schema'])

    def test_update_preserves_graph_and_annotated_axes_with_exact_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snap = export_input_snapshot({'position': torch.zeros(2, 4)}, root/'snapshot.json')
            self.assertEqual(json.loads((root/'snapshot.json').read_text(encoding='utf-8')), snap)
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
            path.write_text(json.dumps(project), encoding='utf-8')
            original = path.read_bytes()
            result = update_project(snap, path)
            self.assertEqual(Path(result['backup']).read_bytes(), original)
            updated = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(updated['graph']['nodes'][0]['type'], 'ForwardInput')
            self.assertEqual(updated['graph']['edges'], project['graph']['edges'])
            self.assertEqual(updated['graph']['nodes'][1], project['graph']['nodes'][1])
            updated_schema = updated['graph']['nodes'][0]['params']['schema']
            self.assertEqual(updated_schema['fields']['position']['shape'], ['B', 8])
            self.assertEqual(updated_schema['fields']['position']['axes'], ['batch', 'feature'])
            self.assertNotIn('autograd', updated_schema['fields']['position'])
            runtime = json.loads(Path(result['snapshot']).read_text())
            self.assertEqual(runtime['bindings'][0]['value']['fields']['position']['shape'], [2, 4])
            self.assertEqual(runtime['bindings'][0]['value']['fields']['position']['autograd'],
                             {'requiresGrad': False, 'source': 'runtime'})
            for key in ['id', 'label', 'position', 'tensorColor']:
                self.assertEqual(updated['graph']['nodes'][0][key], project['graph']['nodes'][0][key])

    def test_empty_project_and_ambiguous_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'project.json'
            snapshot = export_input_snapshot({'x': torch.zeros(2)}, Path(temporary)/'input.json')
            project = {'schemaVersion': 1, 'name': 'pluto', 'variants': [], 'graph': {'nodes': [], 'edges': []}}
            path.write_text(json.dumps(project), encoding='utf-8')
            update_project(snapshot, path)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['graph']['nodes'][0]['type'], 'ForwardInput')
            project['graph']['nodes'] = [
                {'id': 'a', 'type': 'ForwardInput', 'label': 'first'},
                {'id': 'b', 'type': 'ForwardInput', 'label': 'second'},
            ]
            path.write_text(json.dumps(project), encoding='utf-8')
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
            path.write_text(json.dumps(project), encoding='utf-8')
            for _ in range(2):
                update_project(snapshot, path, node_id='network')
                updated = json.loads(path.read_text(encoding='utf-8'))['graph']['nodes'][0]
                self.assertEqual(updated['type'], 'ForwardInput')
                self.assertEqual(updated['params']['text'], '保留的备注')
                self.assertEqual(updated['position'], node['position'])
                self.assertEqual(updated['params']['schema']['fields']['mask']['shape'], [2, 49, 101])
                self.assertNotIn('autograd', updated['params']['schema']['fields']['mask'])

    def test_runs_are_immutable_and_do_not_overwrite_static_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'project.json'
            path.write_text(json.dumps({'schemaVersion': 1, 'name': 'test', 'variants': [],
                                        'graph': {'nodes': [], 'edges': []}}))
            snap = export_input_snapshot({'x': torch.zeros(2, 3)}, root/'input.json')
            first = update_project(snap, path, run_id='batch2')
            saved_run = Path(first['snapshot']).read_bytes()
            before = path.read_bytes()
            snap['schema']['fields']['x']['shape'] = [3, 3]
            with self.assertRaisesRegex(ValueError, 'immutable'):
                update_project(snap, path, run_id='batch2')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(Path(first['snapshot']).read_bytes(), saved_run)
            update_project(snap, path, run_id='batch3')
            project = json.loads(path.read_text())
            self.assertEqual(len(project['runtime']['snapshots']), 2)
            self.assertEqual(project['graph']['nodes'][0]['params']['schema']['fields']['x']['shape'], [2, 3])
            with self.assertRaises(ValueError):
                update_project(snap, path, run_id='../outside')

    def test_register_multi_input_snapshot_without_touching_static_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'project.json'
            graph = {'nodes': [{'id': 'img', 'type': 'Input'},
                               {'id': 'kwargs', 'type': 'ForwardInput'}], 'edges': []}
            path.write_text(json.dumps({'schemaVersion': 1, 'name': 'test', 'variants': [], 'graph': graph}))
            schema, _ = describe_input({'img': torch.zeros(2, 6, 3, 4, 4),
                                        'kwargs': {'states': {0: torch.zeros(2, 1, 21)}}})
            snapshot = {'format': 'tensor-trail.runtime-snapshot', 'version': 1,
                        'id': 'inputs', 'name': 'inputs', 'capturedAt': '2026-11-10T00:00:00Z',
                        'context': {'batch': 2}, 'bindings': [
                            {'nodePath': ['img'], 'nodeType': 'Input', 'value': schema['fields']['img']},
                            {'nodePath': ['kwargs'], 'nodeType': 'ForwardInput', 'value': schema['fields']['kwargs']},
                        ]}
            result = register_runtime_snapshot(snapshot, path)
            self.assertEqual(json.loads(path.read_text())['graph'], graph)
            self.assertEqual(json.loads(Path(result['snapshot']).read_text()), snapshot)
            before = path.read_bytes()
            snapshot['id'] = 'bad'
            snapshot['bindings'][0]['nodePath'] = ['missing']
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                register_runtime_snapshot(snapshot, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root/'project-data/runs/bad.json').exists())

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
            path.write_text(json.dumps(project), encoding='utf-8')
            update_project(snapshot, path)
            parameters = json.loads(path.read_text(encoding='utf-8'))['hyperparameters']
            self.assertEqual(parameters['history_steps']['value'], 10)
            self.assertEqual(parameters['dropout']['type'], 'float')
            self.assertEqual(parameters['enabled']['value'], False)
            self.assertEqual(parameters['name']['type'], 'string')
            self.assertNotIn('_target_', parameters)
            self.assertNotIn('nested', parameters)


if __name__ == '__main__':
    unittest.main()
