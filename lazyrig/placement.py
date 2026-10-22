"""Device placement plans for native GenAI workers.

HETERO partitions one model; AUTO/MULTI replication is deliberately not used.
Manual shares target Constant-weight bytes, not transformer layer counts or a
hard VRAM limit. Explicit assignments keep state and weight consumers together.
"""
import copy
import math
from pathlib import Path

from .device_info import canonical_device
from .mtp import devices as mtp_devices


_MANAGED = {'MODEL_DISTRIBUTION_POLICY', 'QUERY_MODEL_RATIO', 'ATTENTION_BACKEND'}


def _kind(device):
    return device.split('.')[0]


def has_managed_properties(model):
    def contains(values):
        return isinstance(values, dict) and any(
            key in _MANAGED or (isinstance(value, dict) and contains(value))
            for key, value in values.items())
    return contains(model.get('plugin_config', {}))


def devices(model):
    return set(model.get('_placement', {}).get('devices') or
               model.get('placement', {}).get('devices') or [model['device']])


def reservations(model):
    allocation = model.get('_placement', {}).get('reservations')
    return dict(allocation) if allocation is not None else {model['device']: float(model['memory_gib'])}


def default_placement(model, config):
    primary = canonical_device(model['device'], config['devices'])
    return {'mode': 'auto', 'devices': [primary], 'shares': {},
            'allow_fallback': False, 'shared_memory': 'budget'}


def migrate_manual_allocations(config):
    """Retire saved percentages while preserving selected devices and priority."""
    for model in config.get('models', []):
        allocation = model.get('placement')
        if isinstance(allocation, dict) and allocation.get('mode') == 'manual':
            allocation.update(mode='auto', shares={})
    return config


def require_automatic_allocation(model):
    if model.get('placement', {}).get('mode') == 'manual':
        raise ValueError('Manual device percentages are unavailable. Use automatic allocation for the selected devices.')


def validate_placement(model, config):
    if 'placement' not in model:
        return
    p = model['placement']
    if not isinstance(p, dict) or set(p) - {'mode', 'devices', 'shares', 'allow_fallback', 'shared_memory'}:
        raise ValueError('Unknown device placement setting')
    if p.get('mode') not in {'auto', 'single', 'split', 'manual'}:
        raise ValueError('Device mode must be auto, single or split')
    if not isinstance(p.get('allow_fallback', False), bool):
        raise ValueError('Whole-model fallback must be true or false')
    if p.get('shared_memory', 'budget') not in {'budget', 'allow'}:
        raise ValueError('Shared-memory policy must be budget or allow')
    selected = p.get('devices')
    if not isinstance(selected, list) or not selected or len(selected) > 16 or any(not isinstance(d, str) for d in selected):
        raise ValueError('Select between one and sixteen devices')
    known = {d['id'] for d in config['devices']}
    selected = [canonical_device(d, config['devices']) for d in selected]
    if len(set(selected)) != len(selected):
        raise ValueError('Select each physical device only once')
    if any(d not in known for d in selected):
        raise ValueError('Selected device is not configured')
    if selected[0] != canonical_device(model['device'], config['devices']):
        raise ValueError('The primary device must be first in the allocation')
    if p['mode'] == 'single' and len(selected) != 1:
        raise ValueError('Single device mode requires exactly one selected device')
    if p['mode'] == 'split':
        if len(selected) < 2:
            raise ValueError('Split mode requires at least two selected devices')
        if model.get('mtp', {}).get('enabled'):
            raise ValueError('Split mode is unavailable while MTP is enabled. Disable MTP explicitly or choose Automatic / Single device.')
        hardware = {d['id']: d for d in config['devices']}
        host_stages = [d for d in selected if _kind(d) == 'CPU' or hardware[d].get('device_type') == 'integrated']
        if len(host_stages) > 1:
            raise ValueError('Select at most one shared-memory stage: an integrated GPU or CPU. They share system RAM; Automatic can try them as alternatives.')
    if len(selected) > 1:
        if model['pipeline'] == 'image':
            raise ValueError('Image pipelines currently use one device; multi-device splitting applies to text models')
        if any(_kind(d) not in {'GPU', 'CPU'} for d in selected):
            raise ValueError('Model splitting currently supports GPU devices and CPU only')
        if any(_kind(d) == 'CPU' for d in selected[:-1]):
            raise ValueError('CPU must be the last device, after the GPUs')
        if sum(_kind(d) == 'CPU' for d in selected) > 1:
            raise ValueError('Select only one CPU fallback')
    if has_managed_properties(model):
        raise ValueError('Remove MODEL_DISTRIBUTION_POLICY, QUERY_MODEL_RATIO and ATTENTION_BACKEND from advanced OpenVINO properties before using device allocation')
    shares = p.get('shares', {})
    if not isinstance(shares, dict):
        raise ValueError('Model weight shares must be an object')
    if p['mode'] == 'manual':
        if model.get('mtp', {}).get('enabled') and len(selected) > 1:
            raise ValueError('Disable MTP before setting a manual multi-device split')
        if set(shares) != set(selected):
            raise ValueError('Set a weight share for every selected device')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100 for v in shares.values()):
            raise ValueError('Model weight shares must be percentages between 0 and 100')
        if abs(sum(shares.values()) - 100) > .0001 or shares[selected[0]] <= 0:
            raise ValueError('Weight shares must total 100%, with a positive primary-device share')
    p['devices'] = selected
    p.setdefault('shares', {})


def info(model, config):
    text = model['pipeline'] != 'image'
    # These flags describe the model's controls, not its saved device choice.
    # A user editing an NPU model can select a GPU without saving/reopening first.
    # Draft selection and validate_placement still enforce device compatibility.
    multi = text and not has_managed_properties(model)
    manual = False
    reason = (
              'Image pipelines currently use one device.' if not text else
              'Remove custom placement properties to use these controls.' if has_managed_properties(model) else
              'Splitting is disabled while MTP is enabled. The main model stays on one device; whole-model fallback requires permission.' if model.get('mtp', {}).get('enabled') else
              'Automatic tries the primary, then a selected-device split. Moving the whole model to another device requires permission. OpenVINO decides the partition; budgets are not partition limits.')
    weights = stored_weights_gib(model)
    warning = ('The reservation is below the stored language weights. Full-GPU loads reserve at least '
               f'{weights:.1f} GiB; runtime and cache memory need additional headroom.'
               if weights is not None and weights > float(model['memory_gib']) and not offloads_weights(model) else '')
    return {'default_devices': default_placement(model, config)['devices'],
            'manual_supported': manual, 'multi_device_supported': multi,
            'custom_properties': has_managed_properties(model),
            'unit': 'weight_share', 'reason': reason, 'stored_weights_gib': weights,
            'reservation_floor_gib': weights if not offloads_weights(model) else None,
            'reservation_warning': warning, 'split': split_status(model, config)}


def lease_devices(model, config):
    selected = devices(model)
    if '_placement' not in model and model.get('placement', {}).get('mode') == 'manual':
        shares = model['placement'].get('shares', {})
        selected = {device for device in selected if shares.get(device, 0) > 0}
    selected |= mtp_devices(model)
    # A residency slot belongs to an alias even when automatic placement changes.
    return {canonical_device(d, config['devices']) for d in selected} | {'@model:' + model['id']}


def _plan(model, config, selected, label, properties=None, notes=(), shared_memory_attempt=False):
    result = copy.deepcopy(model)
    target = selected[0] if len(selected) == 1 else 'HETERO:' + ','.join(selected)
    budgets = {d['id']: float(d['budget_gib']) for d in config['devices']}
    if len(selected) == 1 and selected[0] == canonical_device(model['device'], config['devices']):
        amount = float(model['memory_gib'])
        weights = stored_weights_gib(model)
        if _kind(selected[0]) == 'GPU' and weights is not None and not offloads_weights(model):
            amount = max(amount, weights)
        reserved = {selected[0]: amount}
    else:
        # HETERO's public GenAI interface does not report per-device peak usage.
        # Reserve each participating device's budget instead of inventing free VRAM.
        reserved = {d: budgets[d] for d in selected}
    if shared_memory_attempt:
        # An explicit over-budget attempt gets an exclusive scheduling slot.
        # This books the whole planning budget; it is NOT a usage estimate.
        reserved = {d: budgets[d] for d in selected}
    result['_placement'] = {'target': target, 'devices': list(selected), 'reservations': reserved,
                            'properties': properties or {}, 'label': label, 'notes': list(notes),
                            'shared_memory_attempt': shared_memory_attempt}
    return result


def _mixed_properties():
    # Continuous batching currently only supports homogeneous GPU execution sets.
    # Stateful SDPA permits CPU/GPU HETERO. It does not use CB cache controls.
    return {'ATTENTION_BACKEND': 'SDPA'}


def _ratios(selected, shares):
    properties = {}
    for index, device in enumerate(selected):
        share = float(shares[device])
        remaining = sum(float(shares[d]) for d in selected[index:])
        if _kind(device) == 'GPU':
            properties[device] = {'QUERY_MODEL_RATIO': min(1.0, share / remaining)}
    return {'DEVICE_PROPERTIES': properties}


def prepare(model, config):
    """Ordered, bounded load attempts. Caller leases the whole candidate pool."""
    if 'placement' not in model:
        return [copy.deepcopy(model)]
    validate_placement(model, config)
    p = model['placement']
    hardware = {d['id']: d for d in config['devices']}
    selected = [d for d in p['devices'] if hardware[d].get('detected') is not False]
    if p['mode'] in {'single', 'split'} and selected != p['devices']:
        missing = ', '.join(d for d in p['devices'] if d not in selected)
        raise ValueError(f'Selected placement was not attempted: {missing} is not detected. Refresh devices or change the selection.')
    if p['mode'] == 'auto' and p['devices'][0] not in selected and not p.get('allow_fallback', False):
        raise ValueError('The primary device is not detected. Select an available primary or explicitly allow whole-model fallback.')
    if not selected:
        raise ValueError('None of the selected devices is currently detected. Refresh devices and choose an available device.')
    primary = selected[0]
    if p['mode'] == 'manual':
        active = [d for d in selected if p['shares'][d] > 0]
        properties = _ratios(active, p['shares']) if len(active) > 1 else {}
        split = len(active) > 1
        if split:
            properties.update(_mixed_properties())
        notes = ['Targets are approximate weight shares, not exact layer assignments.']
        if split:
            notes.append('Manual splitting uses SDPA; continuous-batching cache settings are inactive for this worker.')
        return [_plan(model, config, active, 'Manual weight split' if len(active) > 1 else 'Single device', properties, notes)]
    if p['mode'] == 'split':
        ordered = [d for d in selected if _kind(d) != 'CPU' and hardware[d].get('device_type') != 'integrated']
        ordered += [d for d in selected if d not in ordered]
        properties = {'MODEL_DISTRIBUTION_POLICY': 'PIPELINE_PARALLEL'}
        if any(_kind(d) == 'CPU' for d in ordered):
            properties.update(_mixed_properties())
        return [_plan(model, config, ordered, 'Split across selected devices', properties,
                      ['OpenVINO chooses the partition. Device budgets are planning targets, not per-partition memory limits.',
                       'CPU splits use SDPA; continuous-batching cache settings are inactive.' if 'CPU' in ordered else
                       'Native compilation and the inference probe must both succeed.'])]
    plans = [_plan(model, config, [primary], 'Primary device' if primary == p['devices'][0] else
                   'Whole-model fallback · primary not detected')]
    gpu = [d for d in selected if _kind(d) == 'GPU']
    discrete = [d for d in gpu if hardware[d].get('device_type') != 'integrated']
    integrated = [d for d in gpu if hardware[d].get('device_type') == 'integrated']
    cpu = [d for d in selected if _kind(d) == 'CPU']
    mtp = model.get('mtp', {}).get('enabled', False)
    if not mtp and p['mode'] == 'auto':
        gpu_group = discrete + integrated[:1]
        if len(gpu_group) > 1:
            plans.append(_plan(model, config, gpu_group, 'Automatic GPU split', {'MODEL_DISTRIBUTION_POLICY': 'PIPELINE_PARALLEL'}))
        # iGPU and CPU share host memory: offer them as alternative final
        # stages, never count both as independent memory pools in one split.
        if discrete and cpu:
            plans.append(_plan(model, config, discrete + cpu, 'Automatic GPU / CPU split',
                               {'MODEL_DISTRIBUTION_POLICY': 'PIPELINE_PARALLEL', **_mixed_properties()},
                               ['CPU/GPU splitting uses SDPA; continuous-batching cache settings are inactive for this worker.']))
            if len(discrete) > 1:
                plans.append(_plan(model, config, [discrete[0], cpu[0]], 'Primary GPU / CPU split',
                                   {'MODEL_DISTRIBUTION_POLICY': 'PIPELINE_PARALLEL', **_mixed_properties()},
                                   ['Try the primary GPU with CPU if the multi-GPU path fails.']))
    # A shared-memory override is a distinct, explicit native attempt. In Auto
    # it follows splitting, and only exists if the regular primary is screened
    # out. Never repeat a native attempt which has already failed on this GPU.
    if primary == p['devices'][0] and p.get('shared_memory') == 'allow' and weight_budget_check(plans[0], config):
        shared = _plan(model, config, [primary], 'Primary GPU · shared-memory attempt',
                       notes=['The planning budget is exceeded with explicit permission. The driver decides whether system RAM can be used; host-memory guards remain active.'],
                       shared_memory_attempt=True)
        if p['mode'] == 'single':
            plans = [shared]
        else:
            plans.append(shared)
    if p['mode'] == 'auto' and p.get('allow_fallback', False):
        plans += [_plan(model, config, [device], 'Whole-model GPU fallback') for device in gpu if device != primary]
        if cpu and cpu[0] != primary:
            plans.append(_plan(model, config, cpu, 'Whole-model CPU fallback'))
    return plans


def stored_weights_gib(model):
    """Known language-weight file size only; never imply measured runtime use."""
    if not model.get('path') or model.get('pipeline') not in {'llm', 'vlm', 'vlm-text'}:
        return None
    name = 'openvino_model.bin' if model['pipeline'] == 'llm' else 'openvino_language_model.bin'
    try:
        return (Path(model['path']) / name).stat().st_size / 1024**3
    except OSError:
        return None


def offloads_weights(model):
    def contains(values):
        return isinstance(values, dict) and any(
            (key == 'OFFLOAD_RATIO' and value != 0) or
            (isinstance(value, dict) and contains(value)) for key, value in values.items())
    return contains(model.get('plugin_config', {}))


def split_status(model, config):
    p = model.get('placement', {})
    selected = p.get('devices', [model['device']])
    if model.get('mtp', {}).get('enabled'):
        return {'status': 'disabled_mtp', 'reason': 'Not attempted: MTP requires the main model on one device. Disable MTP explicitly to split.'}
    if p.get('mode') == 'single':
        return {'status': 'single_mode', 'reason': 'Not attempted: Single device mode is selected.'}
    if model['pipeline'] == 'image' or len(selected) < 2 or has_managed_properties(model):
        return {'status': 'not_selected', 'reason': 'Not attempted: no supported multi-device selection is configured.'}
    hardware = {d['id']: d for d in config['devices']}
    available = [d for d in selected if hardware.get(d, {}).get('detected') is not False]
    if len(available) < 2 or not any(_kind(d) == 'GPU' and hardware.get(d, {}).get('device_type') != 'integrated' for d in available):
        return {'status': 'unavailable', 'reason': 'Not attempted: a split needs a detected discrete GPU and another selected GPU or CPU. An integrated GPU and CPU share system RAM.'}
    return {'status': 'eligible', 'reason': 'Native compilation and an inference probe determine compatibility. Device budgets do not set partition sizes.'}


def retryable_load_error(exc):
    # Only loading failures from an already-cleaned-up native child qualify.
    from .worker import NativeWorkerError
    from .worker_status import WorkerProcessError
    from .memory import MemoryPressureError
    if getattr(exc, 'error_code', None) == 'attention_cache_budget_exceeded':
        return False  # Changing devices does not correct invalid cache settings.
    if isinstance(exc, MemoryPressureError):
        return True
    if isinstance(exc, (NativeWorkerError, WorkerProcessError)) and getattr(exc, 'worker_stage', None) == 'validation':
        return True
    if not isinstance(exc, NativeWorkerError):
        return False
    message = str(exc).lower()
    allocation = ('cl_out_of_resources', 'cl_mem_object_allocation_failure', 'out of memory',
                  'out_of_memory', 'bad_alloc', 'failed to allocate', 'cannot allocate memory',
                  'not enough memory', 'insufficient memory')
    unsupported = ('no supported devices', 'no device is available', 'unsupported device',
                   'device is not registered', 'not registered in the openvino runtime',
                   'not supported by any device', 'failed to create plugin', 'no suitable plugin')
    gpu_compile = ('[gpu]' in message or 'intel_gpu' in message) and any(token in message for token in (
        'not supported', 'unsupported', 'cl_build_program_failure', 'cl_invalid_device', 'no kernel'))
    return (message.startswith('memoryerror:') or gpu_compile or
            any(token in message for token in (*allocation, *unsupported)))


def weight_budget_check(model, config):
    """Cheap conservative screening of Auto's full-GPU load, without reading weights.

    Stored language weights alone are not a runtime-memory estimate. But when
    they exceed the configured GPU budget, skip this candidate. The operator
    may raise that budget to permit driver-managed shared-memory loading.
    Detected dGPU memory describes dedicated VRAM, not a hard allocation limit.
    Manual placement and explicit native weight-offload settings remain under
    operator control. Native loading and host-memory guards still decide fit.
    """
    selected = list(devices(model))
    if (model.get('placement', {}).get('mode') not in {'auto', 'single'} or
            model.get('_placement', {}).get('shared_memory_attempt') or
            model['pipeline'] not in {'llm', 'vlm', 'vlm-text'} or
            len(selected) != 1 or _kind(selected[0]) != 'GPU'):
        return None
    if offloads_weights(model):
        return None
    weights = stored_weights_gib(model)
    if weights is None:
        return None  # Let the native loader report missing/inaccessible files.
    device = canonical_device(selected[0], config['devices'])
    candidates = [d for d in config['devices'] if canonical_device(d['id'], config['devices']) == device]
    if not candidates:
        return None
    hardware = next((d for d in candidates if d['id'] == device), candidates[0])
    capacity = float(hardware['budget_gib'])
    if weights <= capacity:
        return None
    return {'device': device, 'weights_gib': round(weights, 3), 'budget_gib': capacity}


def load_failure_reason(exc):
    """Bounded diagnostic categories; never echo arbitrary native error text."""
    from .memory import MemoryPressureError
    from .worker import NativeWorkerError
    from .gpu_errors import kernel_build_failed
    if getattr(exc, 'error_code', None) == 'attention_cache_budget_exceeded':
        return 'attention_cache_budget_exceeded'
    if isinstance(exc, MemoryPressureError):
        return 'host_memory_pressure'
    # The probe runs after compilation; retain that distinction even when its
    # native error happens to contain GPU kernel terminology.
    if getattr(exc, 'worker_stage', None) == 'validation':
        return 'split_inference_failed'
    if isinstance(exc, NativeWorkerError) and kernel_build_failed(exc):
        return 'gpu_kernel_build_failed'
    return 'native_load_rejected'


def runtime_properties(model, core):
    """Check the private GPU ratio capability before relying on it.

    QUERY_MODEL_RATIO is intentionally version/capability gated. A runtime
    lacking it must reject a manual split, never silently run it on one GPU.
    """
    result = copy.deepcopy(model.get('_placement', {}).get('properties', {}))
    for device, values in result.get('DEVICE_PROPERTIES', {}).items():
        if 'QUERY_MODEL_RATIO' not in values:
            continue
        try:
            supported = core.get_property(device, 'INTERNAL_SUPPORTED_PROPERTIES')
            names = {str(x) for x in supported}
        except Exception as exc:
            raise ValueError('This OpenVINO GPU plugin cannot verify manual model splitting on ' + device) from exc
        if 'QUERY_MODEL_RATIO' not in names:
            raise ValueError('This OpenVINO GPU plugin does not support manual model splitting on ' + device)
    return result
