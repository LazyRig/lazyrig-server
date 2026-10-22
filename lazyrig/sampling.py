"""Text sampling overrides and explicit model-card presets."""
import math

FIELDS = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
          'frequency_penalty', 'repetition_penalty')
WORKER_FIELDS = tuple(key for key in FIELDS if key != 'temperature') + ('rng_seed', 'stop_strings')

QWEN_PRESETS = [
    {'id': 'qwen3.8-thinking', 'label': 'Qwen3.8 · Thinking',
     'template_context': {'enable_thinking': True},
     'generation': {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 0.0, 'repetition_penalty': 1.0}},
    {'id': 'qwen3.8-instruct', 'label': 'Qwen3.8 · Instruct / non-thinking',
     'template_context': {'enable_thinking': False},
     'generation': {'temperature': .7, 'top_p': .80, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 1.5, 'repetition_penalty': 1.0}},
]


def validate(values):
    for key in FIELDS:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + ' must be a finite number')
        if key == 'top_k' and (not isinstance(value, int) or not 0 <= value <= 1000000):
            raise ValueError('top_k must be an integer from 0 to 1000000; 0 disables filtering')
        if key == 'temperature' and not 0 <= value <= 2:
            raise ValueError('temperature must be 0–2')
        if key == 'top_p' and not 0 < value <= 1:
            raise ValueError('top_p must be greater than 0 and at most 1')
        if key == 'min_p' and not 0 <= value <= 1:
            raise ValueError('min_p must be 0–1')
        if key in {'presence_penalty', 'frequency_penalty'} and not -2 <= value <= 2:
            raise ValueError(key + ' must be -2–2')
        if key == 'repetition_penalty' and value <= 0:
            raise ValueError('repetition_penalty must be positive')


def apply(config, values):
    for key in WORKER_FIELDS:
        if key in values:
            if not hasattr(config, key):
                raise ValueError('Installed OpenVINO GenAI does not support ' + key)
            setattr(config, key, set(values[key]) if key == 'stop_strings' else values[key])


def snapshot(config):
    """Report native resolved values, including defaults inherited from the export."""
    result = {key: getattr(config, key) for key in FIELDS if hasattr(config, key)}
    result.update(source='openvino', do_sample=bool(config.do_sample))
    if not config.do_sample:
        result['temperature'] = 0.0  # Effective API temperature for greedy decoding.
    if result.get('top_k') == (1 << 64) - 1:
        result['top_k'] = 0  # Native unlimited sentinel; avoid an unsafe JSON integer.
    return result
