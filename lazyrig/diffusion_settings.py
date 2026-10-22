"""Image sampler configuration; separate from the LLM/KV SchedulerConfig."""
import copy
import json
import math
import tempfile
from pathlib import Path
from .image_sizes import is_qwen_image_21

QWEN_REFERENCE = {
    'num_train_timesteps': 1000, 'shift': 1.0, 'use_dynamic_shifting': True,
    'base_shift': 0.5, 'max_shift': 0.9, 'base_image_seq_len': 256,
    'max_image_seq_len': 8192, 'shift_terminal': 0.02,
}

FLOW_CONTROLS = [
    {'key': 'use_dynamic_shifting', 'label': 'Dynamic shifting', 'kind': 'boolean'},
    {'key': 'shift', 'label': 'Static shift', 'kind': 'number'},
    {'key': 'base_shift', 'label': 'Base shift', 'kind': 'number'},
    {'key': 'max_shift', 'label': 'Maximum shift', 'kind': 'number'},
    {'key': 'base_image_seq_len', 'label': 'Base image sequence length', 'kind': 'integer'},
    {'key': 'max_image_seq_len', 'label': 'Maximum image sequence length', 'kind': 'integer'},
    {'key': 'shift_terminal', 'label': 'Terminal shift', 'kind': 'number'},
    {'key': 'num_train_timesteps', 'label': 'Training timesteps', 'kind': 'integer'},
]


def export_config(model):
    if not model.get('path'):
        return {}
    try:
        value = json.loads((Path(model['path']) / 'scheduler/scheduler_config.json').read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def describe(model):
    source = export_config(model)
    qwen = is_qwen_image_21(model)
    flow = qwen or source.get('_class_name') == 'FlowMatchEulerDiscreteScheduler'
    types = ['AUTO', 'FLOW_MATCH_EULER_DISCRETE'] if flow else ['AUTO']
    if source.get('_class_name') in {'PNDMScheduler', 'DDIMScheduler', 'EulerDiscreteScheduler',
                                    'EulerAncestralDiscreteScheduler', 'LCMScheduler', 'LMSDiscreteScheduler'}:
        types = ['AUTO', 'DDIM', 'EULER_DISCRETE', 'EULER_ANCESTRAL_DISCRETE', 'PNDM', 'LCM']
    options = model.get('diffusion_scheduler', {})
    return {'export_config': source, 'effective_config': {**source, **options.get('config', {})},
            'selected_type': options.get('type', 'AUTO'), 'types': types,
            # Qwen calls calculate_shift then set_timesteps_with_mu directly.
            # Do not expose time_shift_type: changing it has no effect there.
            'controls': copy.deepcopy(FLOW_CONTROLS) if qwen else [],
            'qwen_reference': copy.deepcopy(QWEN_REFERENCE) if qwen else None,
            'note': 'AUTO uses scheduler/scheduler_config.json from this export. Overrides are applied to a temporary copy; model files are not changed.'}


def validate_settings(model):
    options = model.get('diffusion_scheduler', {})
    if not isinstance(options, dict) or set(options) - {'type', 'config'}:
        raise ValueError('diffusion_scheduler accepts type and config')
    if options and model.get('pipeline') != 'image':
        raise ValueError('diffusion_scheduler is only for image models')
    overrides = options.get('config', {})
    if not isinstance(overrides, dict):
        raise ValueError('diffusion_scheduler.config must be a JSON object')
    info = describe(model)
    if options.get('type', 'AUTO') not in info['types']:
        raise ValueError('Unsupported diffusion scheduler for this export; choose ' + ', '.join(info['types']))
    if set(overrides) - {c['key'] for c in info['controls']}:
        raise ValueError('Unsupported diffusion scheduler override: ' + ', '.join(sorted(set(overrides) - {c['key'] for c in info['controls']})))
    if not overrides and options.get('type', 'AUTO') == 'AUTO':
        return
    if not info['export_config']:
        raise ValueError('Diffusion scheduler overrides require scheduler/scheduler_config.json in the export')
    effective = info['effective_config']
    if is_qwen_image_21(model):
        for control in FLOW_CONTROLS:
            key = control['key']
            value = effective.get(key, QWEN_REFERENCE[key])
            if key == 'shift_terminal' and value is None:
                continue
            kind = control['kind']
            if kind == 'boolean':
                if type(value) is not bool:
                    raise ValueError(key + ' must be true or false')
            elif kind == 'integer':
                if type(value) is not int or not 1 <= value <= 1048576:
                    raise ValueError(key + ' must be an integer from 1 to 1048576')
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(key + ' must be a finite number')
        if effective.get('shift', 1.0) <= 0:
            raise ValueError('Diffusion shift must be greater than zero')
        terminal = effective.get('shift_terminal')
        if terminal is not None and not 0 <= terminal < 1:
            raise ValueError('shift_terminal must be at least 0 and below 1, or null')
        if effective.get('max_image_seq_len', 8192) <= effective.get('base_image_seq_len', 256):
            raise ValueError('max_image_seq_len must be greater than base_image_seq_len')


def configure_pipeline(genai, pipeline, model):
    options = model.get('diffusion_scheduler', {})
    if options.get('type', 'AUTO') == 'AUTO' and not options.get('config'):
        return  # Native constructor already loaded the export's scheduler.
    validate_settings(model)
    selected = options.get('type', 'AUTO')
    scheduler_type = getattr(genai.Scheduler.Type, selected, None)
    if scheduler_type is None:
        raise ValueError('Installed OpenVINO GenAI does not support scheduler ' + selected)
    merged = {**export_config(model), **options.get('config', {})}
    with tempfile.TemporaryDirectory(prefix='lazyrig-diffusion-') as directory:
        path = Path(directory) / 'scheduler_config.json'
        path.write_text(json.dumps(merged), encoding='utf-8')
        scheduler = genai.Scheduler.from_config(str(path), scheduler_type)
    pipeline.set_scheduler(scheduler)
