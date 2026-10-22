"""Output dimensions shared by image generation, editing and model settings."""
import json
import math
import re
from pathlib import Path

# Width, height pairs published on Qwen/Qwen-Image-2.1's model card.
# These are presets, not an exhaustive allow-list or a hard model limit.
QWEN_IMAGE_21_PRESETS = (
    ('1:1', 2048, 2048), ('4:3', 2400, 1792), ('3:4', 1792, 2400),
    ('3:2', 2528, 1696), ('2:3', 1696, 2528),
    ('16:9', 2752, 1536), ('9:16', 1536, 2752),
)


def is_qwen_image_21(model):
    if model.get('pipeline') != 'image':
        return False
    if model.get('runtime_profile') == 'qwen-image-21':
        return True
    architecture = model.get('inspection', {}).get('architecture', '')
    if model.get('path'):
        try:
            architecture = json.loads((Path(model['path']) / 'model_index.json').read_text()).get('_class_name', architecture)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return 'qwenimage21' in str(architecture).lower().replace('_', '')


def pixel_integer(value, name):
    # Never truncate fractional dimensions or accept booleans as pixels.
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        return int(value)
    raise ValueError(name + ' must be a whole number of pixels')


def constraints(model):
    qwen = is_qwen_image_21(model)
    # OpenVINO QwenImage21Pipeline::check_image_size uses 2 * VAE scale
    # factor. Qwen-Image-2.1's VAE factor is 16, hence a 32-pixel step.
    # Preserve the existing conservative policy for other image adapters.
    return {
        'multiple_of': 32 if qwen else 64,
        'minimum': 256,
        'configured_max_width': pixel_integer(model.get('max_width', 1024), 'Maximum width'),
        'configured_max_height': pixel_integer(model.get('max_height', 1024), 'Maximum height'),
        'presets': [{'aspect_ratio': ratio, 'width': width, 'height': height}
                    for ratio, width, height in QWEN_IMAGE_21_PRESETS] if qwen else [],
        'note': 'Maximum width/height are server settings, not model limits. Presets are not an allow-list.',
    }


def validate_dimensions(model, values):
    info = constraints(model)
    width = pixel_integer(values.get('width', 512), 'Image width')
    height = pixel_integer(values.get('height', 512), 'Image height')
    size = f'{width}x{height}'
    if width < info['minimum'] or height < info['minimum']:
        raise ValueError(f'LazyRig requires image width and height of at least {info["minimum"]} pixels; received {size}')
    step = info['multiple_of']
    if width % step or height % step:
        label = 'Qwen-Image-2.1' if is_qwen_image_21(model) else 'This image adapter'
        raise ValueError(f'{label} requires width and height to be multiples of {step} pixels; received {size}')
    max_width, max_height = info['configured_max_width'], info['configured_max_height']
    if width > max_width or height > max_height:
        raise ValueError(f'Image size {size} exceeds the configured maximum {max_width}x{max_height}. '
                         'Increase Maximum width/height in Models > Settings to allow this size. '
                         'These are server limits, not a model aspect-ratio allow-list.')
    return width, height


def validate_settings(model):
    info = constraints(model)
    for axis in ('width', 'height'):
        if info['configured_max_' + axis] < info['minimum']:
            raise ValueError('Maximum ' + axis + f' must be at least {info["minimum"]} pixels')
    defaults = {axis: min(512, info['configured_max_' + axis]) for axis in ('width', 'height')}
    validate_dimensions(model, {**defaults, **model.get('generation', {})})
