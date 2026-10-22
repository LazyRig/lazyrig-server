"""Validate request-scoped HF chat template options without changing residency."""
import copy
import json
from .response_format import validate_format, is_json

EFFORTS = {'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}
RESERVED = {'messages', 'tools', 'add_generation_prompt', 'chat_template', 'bos_token', 'eos_token', 'pad_token'}


def template_overrides(body):
    value = body.get('chat_template_kwargs')
    if value is None:
        value = {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError('chat_template_kwargs must be a JSON object or null')
    if RESERVED & set(value):
        raise ValueError('chat_template_kwargs cannot override: ' + ', '.join(sorted(RESERVED & set(value))))
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('chat_template_kwargs must contain finite JSON values') from exc
    result = copy.deepcopy(value)
    for key in ('enable_thinking', 'preserve_thinking', 'add_vision_id'):
        if key in result and not isinstance(result[key], bool):
            raise ValueError(key + ' must be a boolean')
    effort = body.get('reasoning_effort')
    if effort is None or effort == 'default':
        effort = result.get('reasoning_effort')
    if effort is None or effort == 'default':
        result.pop('reasoning_effort', None)
    elif not isinstance(effort, str) or effort not in EFFORTS:
        raise ValueError('reasoning_effort must be default, none, minimal, low, medium, high or xhigh')
    elif effort == 'none':
        result.pop('reasoning_effort', None)
        result['enable_thinking'] = False
    else:
        result['reasoning_effort'] = effort
        # An explicit thinking=false is stronger than a positive effort level.
        result.setdefault('enable_thinking', True)
    timings = body.get('timings_per_token', False)
    if not isinstance(timings, bool):
        raise ValueError('timings_per_token must be a boolean')
    return result


def template_model(model, body):
    result = copy.deepcopy(model)
    result['template_context'] = {**model.get('template_context', {}), **template_overrides(body)}
    result['response_format'] = validate_format(body.get('response_format'))
    if is_json(result['response_format']):
        result['template_context']['enable_thinking'] = False
    return result
