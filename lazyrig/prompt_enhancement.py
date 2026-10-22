"""An optional, isolated LLM pass before image inference. Never rewrites chat history."""
import copy
import hashlib
import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

from .image_sizes import constraints, validate_dimensions
from .mtp import request_options as mtp_request
from .reasoning import split_reasoning
from .request_options import template_model
from .sampling import WORKER_FIELDS
from .terminal import Reporter
from .vision import is_vlm
from .worker import Cancelled
from .deadlines import cancellation_requested, check_deadline, time_budget

FIELDS = {'enhance_prompt', 'enhancement_model'}
PROMPT_DIRECTORY = Path(__file__).parent / 'prompts'
PROMPT_FILES = {False: 'image_generation.txt', True: 'image_editing.txt'}
# Old URLs resolve to LazyRig's new defaults, never to legacy files on disk.
PROMPT_ROUTES = {f'/prompts/{name}': name for name in PROMPT_FILES.values()} | {
    '/prompts/system_prompt_t2i.txt': PROMPT_FILES[False],
    '/prompts/system_prompt_edit.txt': PROMPT_FILES[True],
}


def enabled(body):
    value = body.get('enhance_prompt', False)
    if not isinstance(value, bool):
        raise ValueError('enhance_prompt must be a boolean')
    return value


@lru_cache(maxsize=2)
def instructions(editing):
    """Read LazyRig's independently authored, Apache-2.0 enhancement defaults."""
    return (PROMPT_DIRECTORY / PROMPT_FILES[bool(editing)]).read_text(encoding='utf-8')


def response_format(editing):
    names = ['rewritten_prompt', 'wh_ratio'] + (['ratio_follow'] if editing else [])
    return {'type': 'json_schema', 'json_schema': {'name': 'image_prompt', 'strict': True,
            'schema': {'type': 'object', 'properties': {k: {'type': 'string'} for k in names},
                       'required': names, 'additionalProperties': False}}}


def parse(text, editing):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: '+key)
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError) as exc:
        raise ValueError('Prompt enhancement returned invalid JSON; no image was generated') from exc
    names = {'rewritten_prompt', 'wh_ratio'} | ({'ratio_follow'} if editing else set())
    if not isinstance(value, dict) or set(value) != names or any(not isinstance(v, str) for v in value.values()):
        raise ValueError('Prompt enhancement returned the wrong fields; no image was generated')
    if not 1 <= len(value['rewritten_prompt'].strip()) <= 8000:
        raise ValueError('Enhanced prompt must contain 1–8000 characters')
    ratio, follow = value['wh_ratio'], value.get('ratio_follow', '')
    if bool(ratio) == bool(follow) or (follow and (not editing or follow != '<image1>')):
        raise ValueError('Enhanced dimensions need one aspect ratio or ratio_follow=<image1> for an edit')
    if ratio and not re.fullmatch(r'[1-9][0-9]{0,4}:[1-9][0-9]{0,4}', ratio):
        raise ValueError('Enhanced aspect ratio must be positive integers in W:H format')
    if editing and any(c in value['rewritten_prompt'] for c in '\r\n'):
        raise ValueError('Enhanced edit prompt must be one continuous paragraph')
    return value


def dimensions(model, params, result, fixed_size=False):
    """Explicit dimensions win; otherwise fit the ratio within adapter limits."""
    if fixed_size:
        return params['width'], params['height']
    info = constraints(model)
    if result.get('ratio_follow'):
        source = params['input_images'][0]
        ratio = source['width']/source['height']
        area = source['width']*source['height']
        try:
            return validate_dimensions(model, source)
        except ValueError:
            pass
    else:
        a, b = map(int, result['wh_ratio'].split(':'))
        ratio = a/b
        area = params['width']*params['height']
        for preset in info['presets']:
            x, y = map(int, preset['aspect_ratio'].split(':'))
            if a*y == b*x:
                try:
                    return validate_dimensions(model, preset)
                except ValueError:
                    break
    step, minimum = info['multiple_of'], info['minimum']
    max_w = info['configured_max_width']//step*step
    max_h = info['configured_max_height']//step*step
    ideal_h = min(math.sqrt(area/ratio), max_h, max_w/ratio)
    ideal_w = ideal_h*ratio
    def candidates(value, upper):
        return {max(minimum, min(upper, int(value//step)*step + offset*step)) for offset in (0, 1)}
    widths, heights = candidates(ideal_w, max_w), candidates(ideal_h, max_h)
    width, height = min(((w, h) for w in widths for h in heights),
                        key=lambda wh: abs(math.log((wh[0]/wh[1])/ratio)) +
                        .05*abs(math.log((wh[0]*wh[1])/area)))
    return validate_dimensions(model, {'width': width, 'height': height})


def run(service, job, image_model, llm, prompt, params, editing, fixed_size, cancelled, emit):
    if editing and not is_vlm(llm):
        raise ValueError('Edit prompt enhancement requires a vision-language model to inspect the source image. '
                         'Select a VLM or disable prompt enhancement.')
    if cancelled.is_set():
        raise Cancelled()
    record = {'enabled': True, 'status': 'running', 'model': llm['id'],
              'operation': 'edit' if editing else 'generate', 'original_prompt': prompt}
    job['prompt_enhancement'] = record
    job['phase'] = 'Enhancing edit prompt' if editing else 'Enhancing image prompt'
    emit('enhancement_started', model=llm['id'], operation=record['operation'])
    service.store.save_job(job)
    reporter = Reporter(service.config, job)
    try:
        fmt = response_format(editing)
        model = template_model(llm, {'response_format': fmt, 'reasoning_effort': 'none'})
        content = prompt
        if job.get('kind') == 'chat':
            record['original_request'] = job['prompt']
            content = ('User’s latest request (preserve its explicit requirements):\n'+job['prompt']+
                       '\n\nResolved image task from the conversation:\n'+prompt)
        if editing:
            content = [{'type': 'text', 'text': content}, {'type': 'image_url', 'image_url': {
                'url': 'lazyrig-input:'+params['input_images'][0]['sha256']}}]
        messages = [{'role': 'system', 'content': instructions(editing)}, {'role': 'user', 'content': content}]
        budget = min(2048, int(llm['context_limit']))
        prepared = service.tokens.prepare(model, messages, budget, False, add_system=False)
        check_deadline(cancelled)
        if prepared['context']['fits'] is False:
            raise ValueError('Prompt enhancement instructions exceed this LLM’s context budget; select a model with a larger context')
        worker = service.scheduler.ensure(llm, cancelled, emit)
        job['phase'] = 'Enhancing edit prompt' if editing else 'Enhancing image prompt'
        service.store.save_job(job)
        generation = {k: v for k, v in llm.get('generation', {}).items() if k in WORKER_FIELDS}
        generation['temperature'] = llm.get('generation', {}).get('temperature', .7)
        # A chat's custom stop strings and output budget must not cut off the JSON pass.
        for key in ('stop_strings', 'max_tokens'):
            generation.pop(key, None)
        generation['stop_strings'] = []
        mtp = mtp_request(llm, {})
        if mtp['enabled']:
            generation.update(temperature=0, num_assistant_tokens=mtp['num_assistant_tokens'])
        reporter.start_text(llm['id'], 'prompt_enhancement', prepared['context'])
        final, last_notice = None, 0
        for item in worker.run({'action': 'text', 'prepared': prepared, 'tools_enabled': False,
                'max_tokens': budget, 'temperature': generation.get('temperature', .7),
                'generation': generation, 'response_format': fmt, 'timings_per_token': True},
                time_budget(service.config.get('request_timeout_seconds', 3600), cancelled), cancelled):
            if cancelled.is_set():
                raise Cancelled()
            if item['kind'] in {'text', 'result'}:
                now = time.monotonic()
                if now-last_notice >= .25:
                    emit('enhancement_progress', generated_tokens=item.get('generated_tokens', 0))
                    last_notice = now
                if item['kind'] == 'result':
                    final = item
        if final is None:
            raise ValueError('Prompt enhancement worker returned no result')
        if final.get('generated_tokens', 0) >= budget:
            raise ValueError('Prompt enhancement reached its output limit; no image was generated')
        rewritten = parse(split_reasoning(final['text'], prepared.get('thinking_open', False), final=True)['content'], editing)
        width, height = dimensions(image_model, params, rewritten, fixed_size)
        effective = {**params, 'width': width, 'height': height}
        record.update(rewritten, status='completed', width=width, height=height, explicit_size=fixed_size,
                      timings=final.get('timings'), instructions_sha256=hashlib.sha256(instructions(editing).encode()).hexdigest())
        usage = job.setdefault('usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        count = final.get('input_tokens', prepared['context']['input_tokens'])
        usage['prompt_tokens'] += count
        usage['completion_tokens'] += final.get('generated_tokens', 0)
        usage['total_tokens'] = usage['prompt_tokens']+usage['completion_tokens']
        job.setdefault('timing_passes', []).append({'stage': 'prompt_enhancement', 'timings': final.get('timings')})
        reporter.text({**prepared['context'], 'input_tokens': count,
                       'generated_tokens': final.get('generated_tokens', 0),
                       'used_tokens': count+final.get('generated_tokens', 0)}, final.get('timings'), final=True)
        emit('prompt_enhanced', enhancement=copy.deepcopy(record))
        service.store.save_job(job)
        if cancelled.is_set():
            raise Cancelled()
        return rewritten['rewritten_prompt'], effective
    except BaseException as exc:
        record.update(status='cancelled' if cancellation_requested(cancelled) else 'failed', error=str(exc))
        try:
            emit('enhancement_failed', error=str(exc))
            service.store.save_job(job)
        finally:
            service.scheduler.unload(llm['id'], emit)
        raise
