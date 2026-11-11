"""Export real PyTorch input metadata and apply it to a Tensor Trail project.

The web editor does not import this optional Python adapter. Tensor values,
weights and gradients are never serialized.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import math
import re
from pathlib import Path
import tempfile
from uuid import uuid4


def describe_input(data):
    import torch

    tensors = []
    count = 0

    def walk(value, path):
        nonlocal count
        count += 1
        if count > 256 or len(path) > 8:
            raise ValueError('Input exceeds editor limits: 256 entries / 8 levels')
        if isinstance(value, dict):
            fields = {}
            key_types = {type(key) for key in value}
            if key_types and key_types not in ({str}, {int}):
                raise ValueError(f'Unsupported or mixed dictionary key types at {path}')
            integer_keys = key_types == {int}
            for key, child in value.items():
                if integer_keys:
                    if abs(key) > 9007199254740991:
                        raise ValueError(f'Unsafe integer dictionary key at {path}: {key!r}')
                elif (not key.strip() or len(key) > 128
                      or key in {'__proto__', 'constructor', 'prototype'}):
                    raise ValueError(f'Unsupported dictionary key at {path}: {key!r}')
                fields[str(key)] = walk(child, path + [key])
            return {'kind': 'dict', **({'keyType': 'int'} if integer_keys else {}), 'fields': fields}
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{path}: expected dict/Tensor, got {type(value).__name__}')
        dtype = str(value.dtype).removeprefix('torch.')
        supported = {'float32', 'float16', 'float64', 'bfloat16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool'}
        if dtype not in supported or value.ndim > 8:
            raise ValueError(f'{path}: unsupported dtype/rank: {dtype}/{value.ndim}')
        shape = list(value.shape)
        tensors.append({
            'path': path, 'python_type': f'{type(value).__module__}.{type(value).__qualname__}',
            'shape': shape, 'dtype': dtype, 'device': str(value.device),
            'requires_grad': value.requires_grad, 'is_leaf': value.is_leaf,
        })
        return {'shape': shape, 'axes': [f'axis{i}' for i in range(value.ndim)], 'dtype': dtype,
                'autograd': {'requiresGrad': value.requires_grad, 'source': 'runtime'}}

    if not isinstance(data, dict):
        raise TypeError('This adapter currently supports dictionary forward inputs')
    schema = walk(data, [])
    if len(json.dumps(schema, ensure_ascii=False)) > 64_000:
        raise ValueError('Input schema exceeds the editor 64,000 character limit')
    return schema, tensors


def atomic_json(path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as output:
            temp_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temp_path, path)
        else:
            os.link(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def export_input_snapshot(data, destination, *, input_name='data', context=None):
    """Call at the real forward boundary; no detach/copy or tensor-value reads."""
    schema, tensors = describe_input(data)
    snapshot = {
        'format': 'tensor-trail.runtime-input', 'version': 1,
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'input_name': input_name, 'schema': schema, 'tensors': tensors,
        'context': context or {},
    }
    atomic_json(destination, snapshot)
    return snapshot


def _strip_runtime_metadata(value):
    if isinstance(value, dict):
        if value.get('autograd', {}).get('source') == 'runtime':
            del value['autograd']
        for child in value.values():
            _strip_runtime_metadata(child)
    elif isinstance(value, list):
        for child in value:
            _strip_runtime_metadata(child)


def _write_run(path, draft, before, snapshot):
    if (snapshot.get('format') != 'tensor-trail.runtime-snapshot' or snapshot.get('version') != 1
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', snapshot.get('id', ''))
            or snapshot['id'] in {'__proto__', 'constructor', 'prototype'}):
        raise ValueError('Invalid runtime snapshot identity')
    if not snapshot.get('bindings') or len(snapshot['bindings']) > 256:
        raise ValueError('Snapshot must have 1-256 input bindings')
    if not isinstance(snapshot.get('name'), str) or not 1 <= len(snapshot['name']) <= 200:
        raise ValueError('Snapshot name must have 1-200 characters')
    try:
        datetime.fromisoformat(snapshot['capturedAt'].replace('Z', '+00:00'))
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError('Invalid snapshot timestamp') from error
    if (not isinstance(snapshot.get('context'), dict)
            or len(json.dumps(snapshot['context'], ensure_ascii=False, allow_nan=False)) > 64_000):
        raise ValueError('Snapshot context must be an object within 64,000 characters')
    if len(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')) + 1 > 3_000_000:
        raise ValueError('Runtime snapshot exceeds 3 MB')
    seen = set()
    for binding in snapshot['bindings']:
        key = tuple(binding['nodePath'])
        if (not key or len(key) > 13 or key in seen
                or any(not isinstance(part, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', part)
                       or part in {'__proto__', 'constructor', 'prototype'} for part in key)):
            raise ValueError('Empty or duplicate input binding')
        seen.add(key)
        graph = draft['graph']
        for part in key[:-1]:
            graph = next(node for node in graph['nodes'] if node['id'] == part)['subgraph']
        node = next((node for node in graph['nodes'] if node['id'] == key[-1]), None)
        if (node is None or node['type'] != binding['nodeType'] or node.get('valueSource')
                or node['type'] not in {'Input', 'ForwardInput', 'DictionaryInput'}):
            raise ValueError(f'Runtime input binding no longer matches: {key}')
    relative = Path(path.stem + '-data') / 'runs' / (snapshot['id'] + '.json')
    destination = path.parent / relative
    # Never follow a sidecar-directory symlink outside the selected project directory.
    current = path.parent
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Snapshot path cannot contain symlinks')
    runtime = draft.setdefault('runtime', {'snapshots': []})
    reference = {'id': snapshot['id'], 'name': snapshot['name'], 'file': relative.as_posix()}
    old = next((ref for ref in runtime['snapshots'] if ref['id'] == snapshot['id']), None)
    if old and old != reference:
        raise ValueError('Snapshot ID already refers to another file or name')
    if not old:
        if len(runtime['snapshots']) >= 100:
            raise ValueError('Project already has 100 snapshot references')
        runtime['snapshots'].append(reference)
    runtime['active'] = snapshot['id']
    runtime.pop('embedded', None)
    if len(json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')) + 1 > 3_000_000:
        raise ValueError('Project exceeds 3 MB')
    if path.read_bytes() != before:
        raise RuntimeError('Project changed during export; retry after saving the editor')
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != snapshot:
            raise ValueError('Snapshot files are immutable; use a new run ID')
    else:
        atomic_json(destination, snapshot, overwrite=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = path.with_name(f'{path.stem}.before-runtime-{stamp}{path.suffix}')
    if path.read_bytes() != before:
        raise RuntimeError('Project changed during export; completed sidecar left unreferenced')
    with backup.open('xb') as output:
        output.write(before)
    # A completed immutable sidecar is committed before the project reference.
    atomic_json(path, draft)
    return {'project': str(path), 'backup': str(backup), 'snapshot': str(destination), 'run_id': snapshot['id']}


def register_runtime_snapshot(snapshot, project_path):
    """Register an immutable, already bound input snapshot without overwriting static contracts."""
    path = Path(project_path)
    before = path.read_bytes()
    draft = json.loads(before)
    if draft.get('schemaVersion') != 1:
        raise ValueError('Unsupported Tensor Trail project version')
    return _write_run(path, draft, before, snapshot)


def update_project(snapshot, project_path, *, node_id=None, run_id=None, run_name=None):
    """Register a sidecar run; initialize missing input contracts, never overwrite existing ones."""
    if snapshot.get('format') != 'tensor-trail.runtime-input' or snapshot.get('version') != 1:
        raise ValueError('Unsupported runtime input snapshot')
    if snapshot.get('schema', {}).get('kind') != 'dict':
        raise ValueError('Snapshot must describe a dictionary')
    path = Path(project_path)
    before = path.read_bytes()
    project = json.loads(before)
    if project.get('schemaVersion') != 1:
        raise ValueError('Unsupported Tensor Trail project version')
    draft = deepcopy(project)
    nodes = draft['graph']['nodes']
    name = snapshot.get('input_name', 'data')
    supported = {'DictionaryInput', 'ForwardInput', 'DictInput', 'ListInput'}
    candidates = [node for node in nodes if node['id'] == node_id] if node_id else [
        node for node in nodes if node['type'] in supported and (node['label'] == name or node['id'] == name)
    ]
    if node_id and not candidates:
        raise ValueError(f'Input node {node_id!r} does not exist')
    if not candidates and not node_id:
        candidates = [node for node in nodes if node['type'] in supported]
    if len(candidates) > 1:
        raise ValueError('Multiple inputs: specify --node-id')
    if candidates:
        node = candidates[0]
        if node['type'] not in supported or node.get('subgraph'):
            raise ValueError('Target is not a supported root input node')
    else:
        if any(node['id'] == 'data' for node in nodes):
            raise ValueError('Node ID data is occupied; specify --node-id')
        node = {'id': 'data', 'type': 'ForwardInput', 'definitionVersion': 1,
                'label': name, 'params': {}, 'position': {'x': 120, 'y': 180}}
        nodes.append(node)
    node['type'] = 'ForwardInput'
    node['params'].setdefault('text', '')
    schema = deepcopy(snapshot['schema'])
    # Keep manually supplied axis meanings when ranks still match.
    try:
        previous = node['params'].get('schema', {})
        if isinstance(previous, str):
            previous = json.loads(previous)
        if not isinstance(previous, dict):
            previous = {}
    except (ValueError, TypeError):
        previous = {}

    def retain_axes(current, old):
        if not isinstance(old, dict):
            return
        if current.get('kind') == 'dict':
            for key, child in current['fields'].items():
                retain_axes(child, old.get('fields', {}).get(key, {}))
        elif len(old.get('axes', [])) == len(current['shape']) and 'axes' in old:
            current['axes'] = old['axes']
    retain_axes(schema, previous)
    # Seed project-level literals from captured model config. Existing user edits win.
    config = snapshot.get('context', {}).get('model_config', {})
    captured_parameters = {}
    for key, value in config.items():
        if (not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', key)
                or key.startswith('_') or key in {'self', 'constructor', 'prototype'}):
            continue
        if isinstance(value, bool):
            kind = 'bool'
        elif isinstance(value, int) and abs(value) <= 9007199254740991:
            kind = 'int'
        elif isinstance(value, float) and math.isfinite(value):
            kind = 'float'
        elif isinstance(value, str) and len(value) <= 1999:
            kind = 'string'
        else:
            continue
        captured_parameters[key] = {'type': kind, 'value': value,
                                    'source': '训练输入采集 ' + snapshot.get('captured_at', '')[:64]}
    if captured_parameters:
        parameters = draft.setdefault('hyperparameters', {})
        for key, parameter in captured_parameters.items():
            if key not in parameters and len(parameters) < 128:
                parameters[key] = parameter

    contract = deepcopy(previous if previous.get('kind') == 'dict' else schema)
    _strip_runtime_metadata(contract)
    node['params']['schema'] = contract
    if len(json.dumps(contract, ensure_ascii=False)) > 64_000:
        raise ValueError('Input schema exceeds 64,000 characters')
    runtime = {'format': 'tensor-trail.runtime-snapshot', 'version': 1,
               'id': run_id or ('run-' + uuid4().hex), 'name': run_name or name,
               'capturedAt': snapshot['captured_at'], 'context': snapshot.get('context', {}),
               'bindings': [{'nodePath': [node['id']], 'nodeType': node['type'], 'value': schema}]}
    result = _write_run(path, draft, before, runtime)
    return {**result, 'node_id': node['id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--node-id')
    parser.add_argument('--run-id')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    print(json.dumps(update_project(snapshot, args.project, node_id=args.node_id,
                                   run_id=args.run_id, run_name=args.run_name), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
