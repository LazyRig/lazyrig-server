"""Explicit HETERO assignments for the stateful GenAI language graph.

The IR view uses operation extensions only while reading. They return ordinary
native operations, preserving variables through the IR reader's own visitor.
No Python/custom operation survives into the compiled model. Weights stay in
the original, hardlinked BIN; planning reads sizes, never tensor contents.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


EXTENSION_VERSION = 'lazyrig_affinity_v1'
TOLERANCE = 5.0
WEIGHT_TRANSFER_THRESHOLD = 64 * 1024


class ManualAffinityError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.placement_preflight = report


def scope_compiled_cache(properties, assignments):
    """Separate native caches whose IR bytes match but affinities differ."""
    scope = 'lazyrig-manual-' + hashlib.sha256(json.dumps(
        [EXTENSION_VERSION, assignments], sort_keys=True).encode()).hexdigest()[:24]
    pending = [properties]
    while pending:
        values = pending.pop()
        for key, value in values.items():
            if key == 'CACHE_DIR' and isinstance(value, str) and value:
                values[key] = str(Path(value) / scope)
            elif isinstance(value, dict):
                pending.append(value)


def prepare_xml(xml):
    from .state_order import MAX_XML_BYTES
    if len(xml) > MAX_XML_BYTES or b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
        raise ValueError('Invalid manual split language XML')
    root = ET.fromstring(xml)
    layers = root.find('layers')
    if root.tag != 'net' or layers is None:
        raise ValueError('Manual splitting requires an OpenVINO language graph')
    top_level = set(layers.findall('layer'))
    if any(n not in top_level and n.get('type') in {'ReadValue', 'Assign'} for n in root.iter('layer')):
        raise ValueError('Manual splitting does not yet support state variables inside nested subgraphs')
    count = 0
    for node in layers.findall('layer'):
        kind = node.get('type')
        if kind not in {'Result', 'Assign', 'Parameter'}:
            continue
        if node.get('version') not in {'opset1', 'opset3', 'opset6'}:
            raise ValueError('Manual splitting cannot prepare ' + str(kind) + ' from ' + str(node.get('version')))
        name = node.get('name')
        if not name:
            raise ValueError('Manual splitting requires named language operations')
        data = node.find('data')
        if data is None:
            data = ET.SubElement(node, 'data')
        data.set('lazyrig_name', name)
        data.set('lazyrig_opset', node.get('version'))
        node.set('version', EXTENSION_VERSION)
        count += 1
    if not count:
        raise ValueError('Manual split graph has no inputs or outputs')
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def partition_records(records, selected, shares):
    """Partition a topological sequence without splitting state or large weights.

    Records contain names, parent names, byte sizes, constant-expression status
    and optional state IDs. The dynamic search considers every permissible
    boundary; no GPU ratio search or enlarged acceptance tolerance is used.
    """
    names = [r['name'] for r in records]
    if not names or len(names) != len(set(names)) or any(not n for n in names):
        raise ValueError('Manual split requires unique operation names')
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError('Manual split requires distinct selected devices')
    requested = [float(shares[d]) for d in selected]
    if any(not math.isfinite(v) or v <= 0 for v in requested) or abs(sum(requested) - 100) > .0001:
        raise ValueError('Active manual weight shares must be positive and total 100%')
    positions = {name: i for i, name in enumerate(names)}
    forbidden = [0] * (len(records) + 1)
    prefix = [0]
    states = defaultdict(list)
    constant_only, origins = {}, {}

    def keep_together(start, end):
        if start < end:
            forbidden[start + 1] += 1
            forbidden[end + 1] -= 1

    for i, record in enumerate(records):
        size = int(record['weight_bytes'])
        if size < 0:
            raise ValueError('Manual split encountered an invalid constant size')
        prefix.append(prefix[-1] + size)
        parents = record['parents']
        if any(p not in positions or positions[p] >= i for p in parents):
            raise ValueError('Manual split graph is not topologically ordered')
        if record.get('variable') is not None:
            states[record['variable']].append((i, record['kind']))
        is_constant = record['kind'] == 'Constant' or (
            bool(parents) and all(constant_only[p] for p in parents)
            and record['kind'] not in {'ReadValue', 'Assign', 'Result'})
        inherited = [origins[p] for p in parents if origins[p] is not None]
        origin = min(inherited) if inherited else None
        if record['kind'] == 'Constant' and size >= WEIGHT_TRANSFER_THRESHOLD:
            origin = i
        if not is_constant and origin is not None:
            # A compressed weight and its dequantization/consumer belong to
            # one device; otherwise a split can transfer expanded weights.
            keep_together(origin, i)
            origin = None
        constant_only[record['name']] = is_constant
        origins[record['name']] = origin
    for variable, members in states.items():
        if len(members) != 2 or {kind for _, kind in members} != {'ReadValue', 'Assign'}:
            raise ValueError('Manual split requires one read and update per cache variable: ' + str(variable))
        keep_together(min(i for i, _ in members), max(i for i, _ in members))
    total = prefix[-1]
    if not total:
        raise ValueError('Manual split graph contains no constant weights')
    # Equivalent boundaries with the same weight total need only one entry.
    candidates = {0: 0}
    blocked = 0
    for p in range(1, len(records)):
        blocked += forbidden[p]
        if not blocked and 0 < prefix[p] < total:
            candidates[prefix[p]] = p
    candidates[total] = len(records)
    weights = sorted(candidates)

    def find_path(tolerance):
        reachable = [0]
        links = []
        for device_index, wanted in enumerate(requested):
            previous_weights = [weights[p] for p in reachable]
            link = {}
            ends = [len(weights) - 1] if device_index == len(selected) - 1 else range(1, len(weights) - 1)
            lower_share = max(1, (wanted - tolerance) * total / 100)
            upper_share = (wanted + tolerance) * total / 100
            for end in ends:
                lower = bisect_left(previous_weights, weights[end] - upper_share - 1e-6)
                upper = bisect_right(previous_weights, weights[end] - lower_share + 1e-6)
                if lower >= upper:
                    continue
                ideal = weights[end] - wanted * total / 100
                near = bisect_left(previous_weights, ideal, lower, upper)
                indices = {max(lower, min(upper - 1, near)), max(lower, min(upper - 1, near - 1))}
                best = min(indices, key=lambda j: abs(previous_weights[j] - ideal))
                link[end] = reachable[best]
            if not link:
                return None
            reachable = sorted(link)
            links.append(link)
        current = len(weights) - 1
        path = [current]
        for link in reversed(links):
            current = link[current]
            path.append(current)
        return list(reversed(path))

    path = find_path(TOLERANCE)
    if path is None:
        raise ValueError('No explicit whole-state partition meets the requested weight shares within '
                         '5 percentage points per device. No model was compiled.')
    low, high = 0.0, TOLERANCE
    for _ in range(16):
        mid = (low + high) / 2
        better = find_path(mid)
        if better is None:
            low = mid
        else:
            high, path = mid, better
    assignments = {}
    for device, start, end in zip(selected, path, path[1:]):
        for record in records[candidates[weights[start]]:candidates[weights[end]]]:
            assignments[record['name']] = device
    if len(assignments) != len(records):
        raise ValueError('Explicit manual split left operations unassigned')
    return assignments, {'method': 'explicit_affinity', 'candidate_boundaries': len(weights) - 2,
                         'state_variables': len(states), 'tolerance_percentage_points': TOLERANCE,
                         'maximum_deviation_percentage_points': round(high, 4)}


def _constant_bytes(node):
    if node.get_type_name() == 'Constant':
        return int(node.get_byte_size())
    if not hasattr(node, 'get_function'):
        return 0
    bodies = ([node.get_function(0), node.get_function(1)] if node.get_type_name() == 'If'
              else [node.get_function()])
    # Bodies remain wholly with their parent operation. Their weights also
    # contribute to that device's share; only metadata is accessed.
    return sum(_constant_bytes(n) for body in bodies if body is not None for n in body.get_ordered_ops())


def graph_records(graph):
    return [{'name': n.get_friendly_name(), 'kind': n.get_type_name(),
             'parents': [p.get_node().get_friendly_name() for p in n.input_values()],
             'weight_bytes': _constant_bytes(n),
             **({'variable': n.get_variable_id()} if n.get_type_name() in {'ReadValue', 'Assign'} else {})}
            for n in graph.get_ordered_ops()]


class AffinityReader:
    def __init__(self, ov):
        self.ov = ov
        self.assignments = None
        self.seen = set()
        self.covered = set()
        self.projection_count = 0
        self.extensions = [self._extension(kind) for kind in ('Parameter', 'Assign', 'Result')]

    def _mark(self, node):
        stack = [node]
        while stack:
            current = stack.pop()
            identity = current.get_instance_id()
            if identity in self.seen:
                continue
            self.seen.add(identity)
            name = current.get_friendly_name()
            if name not in self.assignments:
                if 'affinity' not in current.get_rt_info():
                    raise ValueError('Unassigned manual split operation: ' + name)
            else:
                current.get_rt_info()['affinity'] = self.assignments[name]
                self.covered.add(name)
            stack.extend(p.get_node() for p in current.input_values())

    def _projection(self, value):
        """Prepare GenAI's final-token projection, keeping generated ops placed.

        The shape-preserving Reshape marks this projection as already prepared:
        GenAI 2026.4 does not insert a second, unassigned Slice through it.
        The native decoder still receives identical [batch, 1, vocab] logits.
        """
        from openvino import opset13 as ops
        last = value.get_node()
        matmul, axis, tail = None, 1, []
        if last.get_type_name() == 'MatMul':
            matmul = last
        elif last.get_type_name() in {'Add', 'Transpose'}:
            possible = last.input_value(0).get_node()
            if possible.get_type_name() == 'MatMul':
                matmul, tail = possible, [last]
                if last.get_type_name() == 'Transpose':
                    # Only the tiny permutation tensor is read, never weights.
                    order = last.input_value(1).get_node()
                    if order.get_type_name() != 'Constant' or order.get_byte_size() > 64:
                        raise ValueError('Unsupported output permutation for manual splitting')
                    axis = int(order.get_vector()[axis])
        elif last.get_type_name() == 'Multiply':
            tanh = last.input_value(0).get_node()
            if tanh.get_type_name() == 'Tanh':
                divide = tanh.input_value(0).get_node()
                if divide.get_type_name() == 'Divide':
                    possible = divide.input_value(0).get_node()
                    if possible.get_type_name() == 'MatMul':
                        matmul, tail = possible, [divide, tanh, last]
        if matmul is None:
            return value
        shape = matmul.input_value(0).get_partial_shape()
        if shape.rank.is_dynamic or shape.rank.get_length() != 3:
            return value
        device = self.assignments[matmul.get_friendly_name()]
        output_device = self.assignments[last.get_friendly_name()]
        if any(self.assignments[n.get_friendly_name()] != device for n in tail):
            raise ValueError('Manual split cannot separate the final language projection')
        names = value.get_names()
        constants = [ops.constant([v], dtype=self.ov.Type.i64) for v in (-1, -2, -1, axis)]
        sliced = ops.slice(matmul.input_value(0), *constants)
        for n in [*constants, sliced]:
            n.get_rt_info()['affinity'] = device
        matmul.input(0).replace_source_output(sliced.output(0))
        for n in [matmul, *tail]:
            n.validate_and_infer_types()
        shape_of = ops.shape_of(value)
        prepared = ops.reshape(value, shape_of, False)
        for n in (shape_of, prepared):
            n.get_rt_info()['affinity'] = output_device
        value.set_names(set())
        prepared.output(0).set_names(names)
        self.projection_count += 1
        return prepared.output(0)

    def _extension(self, kind):
        owner, ov = self, self.ov

        class Terminal(ov.Op):
            info = ov.DiscreteTypeInfo(kind, EXTENSION_VERSION)

            def __init__(self):
                super().__init__(self)
                self.native = None

            def get_type_info(self):
                return self.info

            def visit_attributes(self, visitor):
                from openvino.utils.node_factory import NodeFactory
                attrs = {'lazyrig_name': '', 'lazyrig_opset': ''}
                visitor.on_attributes(attrs)
                self.native = NodeFactory(attrs['lazyrig_opset']).create(kind)
                inputs = self.input_values()
                if kind == 'Result' and owner.assignments is not None:
                    inputs = [owner._projection(inputs[0])]
                self.native.set_arguments(inputs)
                self.native.visit_attributes(visitor)
                self.native.constructor_validate_and_infer_types()
                self.native.set_friendly_name(attrs['lazyrig_name'])
                if owner.assignments is not None:
                    owner._mark(self.native)
                return False

            def outputs(self):
                return self.native.outputs()

        return ov.OpExtension(Terminal)

    def verify_loaded(self, report):
        missing = set(self.assignments) - self.covered
        if missing:
            raise ValueError('Native manual-split loader missed operations: ' + ', '.join(sorted(missing)[:3]))
        report['partition']['native_loader_verified'] = True
        report['partition']['prepared_projections'] = self.projection_count


def prepare(core, ov, genai, model):
    # This path owns the SDPA projection rewrite, so new runtime minor versions
    # require an integration check instead of silently changing that contract.
    for runtime in (ov, genai):
        if not re.match(r'^2026\.4(?:\.|$|[-+])', str(getattr(runtime, '__version__', ''))):
            raise ValueError('Explicit manual splitting currently requires OpenVINO and GenAI 2026.4.x')
    from .placement_probe import _partition_report, _resolved_query_devices
    reader = AffinityReader(ov)
    core.add_extension(reader.extensions)
    filename = 'openvino_model.xml' if model['pipeline'] == 'llm' else 'openvino_language_model.xml'
    graph = core.read_model(str(Path(model['path']) / filename))
    selected = model['_placement']['devices']
    resolved = _resolved_query_devices(core, selected)
    records = graph_records(graph)
    try:
        assignments, partition = partition_records(records, selected, model['placement']['shares'])
    except ValueError as exc:
        raise ManualAffinityError(str(exc), {
            'status': 'rejected', 'scope': 'exported_language_graph',
            'target': model['_placement']['target'], 'runtime_version': str(ov.__version__),
            'requested_weight_shares': {d: model['placement']['shares'][d] for d in selected},
            'partition': {'method': 'explicit_affinity', 'tolerance_percentage_points': TOLERANCE},
            'note': str(exc)}) from exc
    report = _partition_report(graph, assignments, selected, resolved,
                               constant_sizes={r['name']: r['weight_bytes'] for r in records})
    reader.assignments = {name: resolved[device] for name, device in assignments.items()}
    report.update(status='preflight_verified', scope='exported_language_graph',
                  target=model['_placement']['target'], runtime_version=str(ov.__version__),
                  requested_weight_shares={d: model['placement']['shares'][d] for d in selected},
                  partition=partition,
                  note='Explicit operation assignments; exported constant weights are not measured VRAM.')
    if model.get('_state_order'):
        report['state_order'] = model['_state_order']
    del graph
    return reader, report
