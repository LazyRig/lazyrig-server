"""Validate the image options exposed by LazyRig before loading a worker."""
import math
from .image_sizes import is_qwen_image_21

NATIVE_OPTIONS = ('negative_prompt', 'guidance_scale', 'max_sequence_length')


def validate_options(model, values):
    if 'guidance_scale' in values:
        value = values['guidance_scale']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('guidance_scale must be a finite number from 0 to 100')
    if 'negative_prompt' in values:
        if not isinstance(values['negative_prompt'], str):
            raise ValueError('negative_prompt must be a string')
        if values.get('guidance_scale', 4 if is_qwen_image_21(model) else 7.5) <= 1:
            raise ValueError('negative_prompt requires guidance_scale greater than 1 in OpenVINO; disable the negative prompt or increase guidance')
    if 'max_sequence_length' in values:
        if not is_qwen_image_21(model):
            raise ValueError('LazyRig currently exposes max_sequence_length for Qwen-Image-2.1 only')
        value = values['max_sequence_length']
        if type(value) is not int or not 1 <= value <= 1048576:
            raise ValueError('max_sequence_length must be an integer from 1 to 1048576')
