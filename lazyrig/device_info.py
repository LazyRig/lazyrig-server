"""Runtime-reported hardware metadata, separate from configured memory budgets.

No total-memory value here is a measurement of currently free GPU memory. Missing
properties stay unknown; device names do not establish vendor compatibility.
"""
import math
import re

from . import memory


def support_metadata(device_id, name):
    """Development support scope, not a model-compatibility test or device gate."""
    intel = bool(re.search(r'\bIntel\b', name, re.IGNORECASE))
    # CPU support follows the enumerated runtime plugin, not the vendor name.
    supported = device_id.split('.')[0] == 'CPU' or (intel and _gpu(device_id))
    return {'support_level': 'supported' if supported else 'experimental',
            'support_note': ('Model, runtime and driver compatibility still applies.'
                             if supported else
                             'Model, runtime and driver compatibility has not been validated by LazyRig.')}


def preferred_device(configured, weights_gib=None):
    """Choose one conservative default, without replacing an explicit choice.

    Budgets are capacity hints, not free-memory readings or a promise of fit.
    Experimental accelerators remain available for explicit selection.
    """
    available = [d for d in configured if d.get('detected') is not False]
    gpus = [d for d in available if _gpu(d['id']) and d.get('support_level') == 'supported']
    def capacity(d):
        limits = [v for v in (_positive_number(d.get('budget_gib')),
                             _positive_number(d.get('total_memory_gib'))) if v is not None]
        return min(limits) if limits else 0
    if weights_gib is not None:
        gpus = [d for d in gpus if capacity(d) > weights_gib]
    gpus.sort(key=lambda d: (d.get('device_type') != 'discrete', -capacity(d), d['id']))
    cpu = sorted((d for d in available if d['id'].split('.')[0] == 'CPU'),
                 key=lambda d: (d.get('support_level') != 'supported', d['id']))
    chosen = (gpus or cpu or available)
    return chosen[0]['id'] if chosen else None


def _property(core, device_id, key):
    try:
        return core.get_property(device_id, key)
    except Exception:
        # Optional metadata support differs between runtime plugins and drivers.
        return None


def _positive_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _gpu(device_id):
    return device_id == 'GPU' or device_id.startswith('GPU.')


def _device_type(core, device_id, name):
    if device_id == 'CPU' or device_id.startswith('CPU.'):
        return 'cpu'
    if not _gpu(device_id):
        return 'other'
    reported = _property(core, device_id, 'DEVICE_TYPE')
    if reported is not None and str(reported).strip():
        # OpenVINO returns device.Type enums; some plugins expose their strings.
        label = str(getattr(reported, 'name', reported)).rsplit('.', 1)[-1].upper()
        return {'DISCRETE': 'discrete', 'INTEGRATED': 'integrated'}.get(label, 'other')
    markers = {value.lower() for value in re.findall(r'\((dGPU|iGPU)\)', name, re.IGNORECASE)}
    if markers == {'dgpu'}:
        return 'discrete'
    if markers == {'igpu'}:
        return 'integrated'
    return 'other'


def discover_devices(core):
    """Return metadata for enumerated devices without altering saved settings.

    Enumeration failure propagates to the caller's runtime-discovery handling.
    Optional property failures affect only that field, never another device.
    Total capacities use GiB; unknown and currently available memory are None.
    """
    devices = []
    for device_id in dict.fromkeys(core.available_devices):
        name = _property(core, device_id, 'FULL_DEVICE_NAME')
        name = str(name).strip() if name is not None else ''
        name = name or device_id
        device_type = _device_type(core, device_id, name)
        total = None
        if device_type == 'cpu':
            try:
                total = _positive_number(memory.system_memory_gib())
            except Exception:
                pass
        elif _gpu(device_id):
            size = _positive_number(_property(core, device_id, 'GPU_DEVICE_TOTAL_MEM_SIZE'))
            if size is not None:
                total = size / 1024**3
        devices.append({'id': device_id, 'name': name, 'device_type': device_type,
                        'total_memory_gib': total, 'available_memory_gib': None,
                        'detected': True, 'canonical_id': device_id,
                        **support_metadata(device_id, name)})
    return devices


def canonical_device(device_id, configured_devices):
    """Resolve the generic GPU alias only to a device that actually exists.

    GPU.0 is OpenVINO's default GPU when present. Without it, a sole GPU may be
    selected. Unknown or ambiguous identifiers remain unchanged for validation;
    explicit GPU indices and other device names are never reassigned.
    """
    if device_id != 'GPU':
        return device_id
    ids = {device['id'] for device in configured_devices}
    if 'GPU.0' in ids:
        return 'GPU.0'
    if 'GPU' in ids:
        return 'GPU'
    gpus = [candidate for candidate in ids if _gpu(candidate)]
    return gpus[0] if len(gpus) == 1 else device_id
