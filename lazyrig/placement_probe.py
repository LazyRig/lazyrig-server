"""Read-only checks for approximate manual HETERO weight partitioning.

This checks the exported language graph before GenAI compiles it. It does not
measure VRAM or claim that GenAI's final transformed graph has identical splits.
Only constant metadata is inspected; weight tensors are never copied to Python.
"""
from collections.abc import Mapping
import copy
from pathlib import Path
import math
import re
import time


MAX_CALIBRATION_QUERIES = 9
CALIBRATION_TOLERANCE_PERCENTAGE_POINTS = 5.0
CALIBRATION_BUDGET_SECONDS = 60.0
_MIN_QUERY_RATIO = 0.000001


class ManualSplitCalibrationError(ValueError):
    def __init__(self, report):
        self.placement_preflight = report
        requested = ', '.join(f'{device} {share:g}%' for device, share in report['requested_weight_shares'].items())
        measured = ', '.join(f'{device} {value["weight_share_percent"]:g}%' for device, value in report['devices'].items())
        calibration = report['calibration']
        unused = [device for device, value in report['devices'].items() if not value['weight_bytes']]
        missing = ('; no model weights were assigned to ' + ', '.join(unused)) if unused else ''
        super().__init__(f'No manual split within {calibration["tolerance_percentage_points"]:g} percentage points '
                         f'per device was found after {calibration["query_count"]} preflight checks '
                         f'({calibration["stop_reason"]}). Requested: {requested}. Closest tested: {measured}'
                         f'{missing}. OpenVINO may restrict legal graph boundaries. '
                         'No model was compiled; adjust the shares or use automatic allocation.')


def _runtime_version():
    import openvino as ov
    version = str(getattr(ov, '__version__', 'unknown'))
    match = re.match(r'^(\d+)\.(\d+)(?:\.|$|[-+])', version)
    if not match or tuple(map(int, match.groups())) < (2026, 4):
        raise ValueError('Manual model splitting requires OpenVINO 2026.4 or newer; '
                         'the installed runtime reports ' + version)
    return version


def _query_properties(properties, selected):
    # PIPELINE_PARALLEL would overwrite the explicit QUERY_MODEL_RATIO values.
    if properties.get('MODEL_DISTRIBUTION_POLICY'):
        raise ValueError('Manual model splitting cannot combine weight shares with '
                         'MODEL_DISTRIBUTION_POLICY')
    per_device = properties.get('DEVICE_PROPERTIES', {})
    result = {}
    for device in selected:
        if device.split('.')[0] != 'GPU':
            continue
        values = per_device.get(device, {})
        ratio = values.get('QUERY_MODEL_RATIO')
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or
                not math.isfinite(ratio) or not 0 < ratio <= 1):
            raise ValueError('Manual model splitting has no valid weight-share target for ' + device)
        result[device] = {'QUERY_MODEL_RATIO': ratio}
    return {'DEVICE_PROPERTIES': result}


def _resolved_query_devices(core, selected):
    """Resolve only the GPU plugin's documented generic-device alias.

    Core.available_devices exposes "GPU" when a plugin has just one device,
    while the GPU plugin's query_model reports its context name ("GPU.0", for
    example). DEVICE_ID proves which physical device a generic GPU denotes.
    An explicit single AVAILABLE_DEVICES entry is a safe fallback when that
    property cannot be read. Never infer aliases from a prefix or device count
    in the selected plan: that could hide assignment to a different GPU.
    """
    resolved = {device: device for device in selected}
    if 'GPU' in resolved:
        def valid_id(value):
            return isinstance(value, str) and re.fullmatch(r'\d+(?:\.\d+)*', value) is not None

        try:
            device_id = core.get_property('GPU', 'DEVICE_ID')
        except Exception:
            device_id = None
        if not valid_id(device_id):
            try:
                device_ids = core.get_property('GPU', 'AVAILABLE_DEVICES')
            except Exception:
                device_ids = None
            device_id = (device_ids[0] if isinstance(device_ids, (list, tuple)) and
                         len(device_ids) == 1 and valid_id(device_ids[0]) else None)
        if valid_id(device_id):
            resolved['GPU'] = 'GPU.' + device_id
        elif any(device.startswith('GPU.') for device in selected):
            raise ValueError('Cannot verify manual model splitting: the generic GPU alias '
                             'could not be resolved alongside explicit GPU devices')
    if len(set(resolved.values())) != len(resolved):
        raise ValueError('Cannot verify manual model splitting: selected device aliases '
                         'refer to the same physical device: ' + repr(resolved))
    return resolved


def _partition_report(graph, assignments, selected, resolved_query_devices=None, *, require_all_devices=True,
                      constant_sizes=None):
    """Validate original coverage while allowing HETERO's partition boundaries.

    HETERO query_model_update queries rewritten subgraphs and merges their maps.
    Its result can therefore include generated Parameters/Results that do not
    exist in the input graph. These entries must still use selected devices,
    but only original graph nodes contribute to weight/operation accounting.
    """
    nodes = {}
    for node in graph.get_ordered_ops():
        name = node.get_friendly_name()
        if not isinstance(name, str) or not name or name in nodes:
            raise ValueError('Cannot verify manual model splitting: the language graph '
                             'has missing or duplicate operation names')
        nodes[name] = node
    if not nodes or not isinstance(assignments, Mapping):
        raise ValueError('Cannot verify manual model splitting: OpenVINO returned no valid operation mapping')
    devices = {device: {'operations': 0, 'weight_bytes': 0} for device in selected}
    resolved = resolved_query_devices or {device: device for device in selected}
    accepted_labels = {label: device for device, label in resolved.items()}
    accepted_labels.update({device: device for device in selected})
    normalized_assignments = {}
    for name, device in assignments.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Cannot verify manual model splitting: invalid query operation name')
        if not isinstance(device, str) or device not in accepted_labels:
            raise ValueError('Cannot verify manual model splitting: operation ' + name +
                             ' was assigned outside the selected devices: returned ' + repr(device) +
                             '; selected ' + repr(selected) + '; resolved query devices ' + repr(resolved))
        normalized_assignments[name] = accepted_labels[device]
    missing = set(nodes) - set(assignments)
    if missing:
        raise ValueError('Cannot verify manual model splitting; unassigned operations: ' +
                         ', '.join(sorted(missing)[:3]))
    additional = {device: 0 for device in selected}
    for name in set(assignments) - set(nodes):
        additional[normalized_assignments[name]] += 1
    input_devices = {}
    embedding_inputs = []
    for name, node in nodes.items():
        device = normalized_assignments[name]
        devices[device]['operations'] += 1
        if node.get_type_name() == 'Parameter':
            try:
                tensor_names = sorted(node.output(0).get_names())
            except (AttributeError, RuntimeError):
                tensor_names = []
            input_devices[name] = {'device': device, 'query_device': resolved[device], 'tensor_names': tensor_names}
            if name == 'inputs_embeds' or 'inputs_embeds' in tensor_names:
                embedding_inputs.append(resolved[device])
        if constant_sizes is not None or node.get_type_name() == 'Constant':
            # Constant.get_byte_size handles packed int4/int2 storage correctly.
            # Accessing .data or get_data here would risk a large weight copy.
            size = int(constant_sizes.get(name, 0)) if constant_sizes is not None else int(node.get_byte_size())
            if size < 0:
                raise ValueError('Cannot verify manual model splitting: invalid constant weight size')
            devices[device]['weight_bytes'] += size
    total = sum(value['weight_bytes'] for value in devices.values())
    if not total:
        raise ValueError('Cannot verify manual model splitting: the language graph contains no constant weights')
    unused = [device for device, value in devices.items() if not value['weight_bytes']]
    if unused and require_all_devices:
        raise ValueError('OpenVINO could not honor the manual split: no model weights were assigned to ' +
                         ', '.join(unused) + '. Adjust the weight shares or use automatic allocation.')
    for value in devices.values():
        value['weight_share_percent'] = round(100 * value['weight_bytes'] / total, 4)
    return {'weight_bytes': total, 'devices': devices,
            'additional_query_operations': additional,
            'resolved_query_devices': resolved,
            'input_devices': input_devices,
            **({'language_input_device': embedding_inputs[0]} if len(embedding_inputs) == 1 else {})}


def _next_ratios(ratios, requested_ratios, attempts, selected, requested):
    """Adjust a conditional ratio using measured cumulative weight shares.

    Downstream ratios act only on the graph left by preceding GPUs. Observations
    therefore bracket one coordinate only when the preceding ratios match.
    Native graph boundaries may create plateaus or nonmonotonic responses; no
    candidate is accepted without measuring the resulting original weights.
    """
    adjustable = [device for device in selected[:-1] if device in ratios]
    current = attempts[-1]['measured_weight_shares']
    coordinate = None
    for index, device in enumerate(selected):
        if not current[device] or abs(current[device] - requested[device]) > CALIBRATION_TOLERANCE_PERCENTAGE_POINTS:
            coordinate = min(index, len(selected) - 2)
            break
    if coordinate is None:
        return None
    if coordinate < 0 or selected[coordinate] not in adjustable:
        return None
    device = selected[coordinate]
    prefix = selected[:coordinate]
    target = sum(requested[d] for d in selected[:coordinate + 1])
    observations = []
    for attempt in attempts:
        if all(attempt['ratios'].get(d) == ratios.get(d) for d in prefix):
            observed = sum(attempt['measured_weight_shares'][d] for d in selected[:coordinate + 1])
            observations.append((attempt['ratios'][device], observed))
    actual = sum(current[d] for d in selected[:coordinate + 1])
    lower, upper = _MIN_QUERY_RATIO, 1.0
    for ratio, observed in observations:
        if observed < target:
            lower = max(lower, ratio)
        elif observed > target:
            upper = min(upper, ratio)
    tried = {ratio for ratio, _ in observations}
    same_output = [(ratio, observed) for ratio, observed in observations if abs(observed - actual) < .000001]
    edge = _MIN_QUERY_RATIO if actual > target else 1.0
    neighbor = max(_MIN_QUERY_RATIO, ratios[device] * .95)
    if len(observations) == 1 and current[device] == 0 and neighbor < ratios[device]:
        # Native ratio queries can return zero at an otherwise usable boundary:
        # a normalized stateful graph gives 0% at .5 but 50% at .475. Before
        # treating zero as proof that the ratio must increase, measure one
        # nearby lower ratio. This uses the existing global query/time budget.
        candidate = neighbor
    elif len(same_output) >= 2 and actual != target:
        if any(ratio == edge for ratio, _ in same_output):
            return None  # This coordinate's requested direction has a tested plateau.
        candidate = edge
    else:
        candidate = (lower + upper) / 2
    if lower >= upper or candidate in tried:
        # A nonmonotonic response invalidates a binary-search bracket. Explore
        # an unsampled interval, still under the same global query budget.
        points = sorted({_MIN_QUERY_RATIO, 1.0, *tried})
        gaps = sorted(((b - a, (a + b) / 2) for a, b in zip(points, points[1:])), reverse=True)
        candidate = next((mid for width, mid in gaps if width > .000001 and mid not in tried), None)
        if candidate is None:
            return None
    result = dict(ratios)
    result[device] = candidate
    for later in selected[coordinate + 1:]:
        if later in result:
            result[later] = requested_ratios[later]
    return result


def _calibrate(core, graph, target, selected, resolved, query_properties, requested, deadline):
    initial_ratios = {device: values['QUERY_MODEL_RATIO']
                      for device, values in query_properties['DEVICE_PROPERTIES'].items()}
    ratios = dict(initial_ratios)
    attempts, best, best_score, best_ratios = [], None, None, None
    stop_reason = 'query budget reached'
    for _ in range(MAX_CALIBRATION_QUERIES):
        if time.monotonic() >= deadline:
            if best is None:
                raise TimeoutError('Manual split preflight deadline reached before querying the model')
            stop_reason = 'calibration time budget reached'
            break
        properties = {'DEVICE_PROPERTIES': {device: {'QUERY_MODEL_RATIO': value} for device, value in ratios.items()}}
        assignments = core.query_model(graph, target, properties)
        report = _partition_report(graph, assignments, selected, resolved, require_all_devices=False)
        measured = {device: value['weight_share_percent'] for device, value in report['devices'].items()}
        deviations = {device: measured[device] - requested[device] for device in selected}
        score = (max(abs(value) for value in deviations.values()), sum(abs(value) for value in deviations.values()))
        attempts.append({'ratios': dict(ratios), 'measured_weight_shares': measured,
                         'maximum_deviation_percentage_points': score[0]})
        if best_score is None or score < best_score:
            best, best_score, best_ratios = report, score, dict(ratios)
        if score[0] <= CALIBRATION_TOLERANCE_PERCENTAGE_POINTS and all(v['weight_bytes'] for v in report['devices'].values()):
            best, best_score, best_ratios = report, score, dict(ratios)
            stop_reason = 'within tolerance'
            break
        candidate = _next_ratios(ratios, initial_ratios, attempts, selected, requested)
        if candidate is None:
            stop_reason = 'tested graph-boundary plateau'
            break
        ratios = candidate
    best['requested_weight_shares'] = dict(requested)
    best['calibrated_properties'] = {'DEVICE_PROPERTIES': {
        device: {'QUERY_MODEL_RATIO': value} for device, value in best_ratios.items()}}
    best['calibration'] = {'query_count': len(attempts), 'max_queries': MAX_CALIBRATION_QUERIES,
                           'tolerance_percentage_points': CALIBRATION_TOLERANCE_PERCENTAGE_POINTS,
                           'requested_ratios': initial_ratios, 'selected_ratios': best_ratios,
                           'maximum_deviation_percentage_points': best_score[0],
                           'stop_reason': stop_reason, 'attempts': attempts}
    return best, stop_reason == 'within tolerance'


def verify_manual_split(core, model, properties, *, deadline=None):
    """Return a preflight report, or {} unless this is a manual multi-device plan.

    The caller separately checks QUERY_MODEL_RATIO plugin capability. Query only
    placement-owned settings: GenAI options such as ATTENTION_BACKEND are not
    OpenVINO plugin properties. The graph is released before pipeline loading.
    """
    placement = model.get('placement', {})
    plan = model.get('_placement', {})
    selected = plan.get('devices', [])
    if placement.get('mode') != 'manual' or len(selected) < 2:
        return {}
    version = _runtime_version()
    target = 'HETERO:' + ','.join(selected)
    if plan.get('target') != target or len(set(selected)) != len(selected):
        raise ValueError('Cannot verify manual model splitting: invalid device allocation target')
    if any(placement.get('shares', {}).get(device, 0) <= 0 for device in selected):
        raise ValueError('Cannot verify manual model splitting: active devices must have positive weight shares')
    pipeline = model.get('pipeline')
    if pipeline not in {'llm', 'vlm', 'vlm-text'}:
        raise ValueError('Manual model splitting preflight supports text model exports only')
    filename = 'openvino_model.xml' if pipeline == 'llm' else 'openvino_language_model.xml'
    path = Path(model['path']) / filename
    if not path.is_file():
        raise ValueError('Cannot verify manual model splitting: missing language graph ' + filename)
    query_properties = _query_properties(properties, selected)
    resolved = _resolved_query_devices(core, selected)
    budget_end = time.monotonic() + CALIBRATION_BUDGET_SECONDS
    if deadline is not None:
        budget_end = min(budget_end, deadline)
    graph = core.read_model(str(path))
    try:
        requested = {device: placement['shares'][device] for device in selected}
        report, calibrated = _calibrate(core, graph, target, selected, resolved, query_properties, requested, budget_end)
    finally:
        # No graph or Tensor escapes into the report or model configuration.
        del graph
    report = {'status': 'preflight_verified' if calibrated else 'calibration_failed', 'scope': 'exported_language_graph',
            'runtime_version': version, 'target': target, **report,
            'note': 'Exported-language-graph preflight only. Weight shares are approximate; '
                    'GenAI transformations may change the compiled partition. '
                    'Constant-weight bytes are not VRAM usage.'}
    if '_state_order' in model:
        report['state_order'] = copy.deepcopy(model['_state_order'])
    if not calibrated:
        raise ManualSplitCalibrationError(report)
    return report
