"""Frontend extension regressions; fixture models are not full Qwen inference."""
import copy
import json
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from lazyrig.context import TokenizerService
from lazyrig.metrics import GenerationTimer
from lazyrig.request_options import template_model, template_overrides
from lazyrig.templates import render_chat_details


# Original, minimal template fixture exercising the model's declared effort levels.
TEMPLATE = """{% if enable_thinking is undefined or enable_thinking is true %}
{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{% if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
{{ raise_exception('invalid effort') }}{% endif %}
reasoning={{ resolved_reasoning_effort }}
{% endif %}{{ messages[-1].content }}
{% if add_generation_prompt %}assistant:{% endif %}"""


class RequestOptionTests(unittest.TestCase):
    def test_precedence_default_none_and_validation(self):
        model = {'template_context': {'enable_thinking': False, 'reasoning_effort': 'medium', 'custom': 'saved'}}
        self.assertEqual(template_model(model, {'reasoning_effort': 'default'})['template_context'], model['template_context'])
        self.assertEqual(template_model(model, {'reasoning_effort': None})['template_context'], model['template_context'])
        result = template_model(model, {'reasoning_effort': 'high', 'chat_template_kwargs': {'reasoning_effort': 'low', 'enable_thinking': False, 'custom': 'request'}})
        self.assertEqual(result['template_context'], {'enable_thinking': False, 'reasoning_effort': 'high', 'custom': 'request'})
        self.assertFalse(template_overrides({'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': True}})['enable_thinking'])
        self.assertTrue(template_overrides({'reasoning_effort': 'low'})['enable_thinking'])
        self.assertEqual(model['template_context']['custom'], 'saved')
        invalid = [{'chat_template_kwargs': []}, {'chat_template_kwargs': {'messages': []}},
                   {'chat_template_kwargs': {'tools': []}}, {'chat_template_kwargs': {'enable_thinking': 'false'}},
                   {'chat_template_kwargs': {'x': float('inf')}}, {'reasoning_effort': 'max'},
                   {'reasoning_effort': []}, {'timings_per_token': 'true'}, {'timings_per_token': None}]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValueError):
                template_overrides(body)

    def test_qwen_aliases_use_declared_template_levels_and_exact_prompt_ids(self):
        class Tokenizer:
            def __init__(self, path): pass
            def encode(self, text, add_special_tokens):
                assert add_special_tokens is False
                row = types.SimpleNamespace(tolist=lambda: list(text.encode()))
                return types.SimpleNamespace(input_ids=types.SimpleNamespace(data=[row]))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'openvino_genai': types.SimpleNamespace(Tokenizer=Tokenizer)}):
            path = Path(temp, 'chat_template.jinja')
            path.write_text(TEMPLATE)
            model = {'path': temp, 'pipeline': 'llm', 'context_limit': 4096, 'template_context': {}}
            service = TokenizerService('openvino')
            for effort, resolved in [('high', 'xhigh'), ('minimal', 'low'), ('medium', 'medium'), ('none', None)]:
                prepared = service.prepare(template_model(model, {'reasoning_effort': effort}), [{'role': 'user', 'content': 'Hola'}], 20, False, add_system=False)
                self.assertEqual(prepared['token_ids'], list(prepared['text'].encode()))
                self.assertEqual(prepared['context']['input_tokens'], len(prepared['token_ids']))
                if resolved:
                    self.assertIn('reasoning='+resolved, prepared['text'])
                    self.assertEqual(prepared['template_context']['reasoning_effort'], resolved)
                else:
                    self.assertNotIn('reasoning=', prepared['text'])
                self.assertEqual(bool(prepared['template_notes']), effort in {'high', 'minimal'})
            path.write_text('{{ reasoning_effort }}')
            rendered = render_chat_details(temp, object(), [], extra_context={'reasoning_effort': 'high'})
            self.assertEqual(rendered['text'], 'high')
            self.assertEqual(rendered['template_notes'], [])


class TimingTests(unittest.TestCase):
    def test_elapsed_time_batch_counts_and_optional_native_metrics(self):
        now = [10.0]
        timer = GenerationTimer(100, clock=lambda: now[0])
        self.assertIsNone(timer.snapshot(0)['prompt_ms'])
        now[0] = 10.5
        timer.observe(2)
        self.assertIsNone(timer.snapshot(2)['predicted_per_second'])
        now[0] = 11.0
        timer.observe(7)
        timer.finish()
        now[0] = 100.0
        perf = types.SimpleNamespace(get_ttft=lambda: types.SimpleNamespace(mean=499),
                                     get_tpot=lambda: types.SimpleNamespace(mean=float('nan')))
        measured = timer.snapshot(7, perf=perf)
        self.assertEqual(measured['prompt_ms'], 500)
        self.assertEqual(measured['prompt_per_second'], 200)
        self.assertEqual(measured['predicted_ms'], 500)
        self.assertEqual(measured['x_generation_ms'], 1000)
        self.assertEqual(measured['x_decode_tokens'], 5)
        self.assertEqual(measured['predicted_per_second'], 10)
        self.assertEqual(measured['x_openvino_metrics'], {'ttft_ms': 499})
        json.dumps(measured, allow_nan=False)
        zero = GenerationTimer(0, clock=lambda: 1)
        zero.observe(1)
        self.assertIsNone(zero.snapshot(1)['predicted_per_second'])
        json.dumps(zero.snapshot(1), allow_nan=False)


class ExtensionHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **extra):
        return {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 32,
                'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_effort': 'high',
                'timings_per_token': True, **extra}

    def test_key_generator_auth_draft_rotation_and_admin_origin(self):
        draft = self.json('/api/server/api-key', {})
        key = draft['api_key']
        self.assertRegex(key, r'^lr_[A-Za-z0-9_-]{43}$')
        self.assertFalse(draft['saved'])
        self.assertFalse(self.service.config.get('api_key'))
        self.json('/api/server', {'api_key': key, 'browser_access': 'api-key'})
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/api/server/api-key', {})
        self.assertEqual(err.exception.code, 401)
        headers = {'Authorization': 'Bearer '+key}
        with self.request('/api/server/api-key', {}, headers) as response:
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            replacement = json.load(response)['api_key']
        self.assertNotEqual(key, replacement)
        with self.request('/v1/models', headers=headers) as response:
            self.assertEqual(response.status, 200)  # Generating a draft preserves the live key.
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/api/server/api-key', {}, {**headers, 'Origin': 'https://external.example'})
        self.assertEqual(err.exception.code, 400)
        with self.request('/api/server', {'api_key': replacement}, headers) as response:
            self.assertNotIn(replacement, response.read().decode())
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/v1/models', headers=headers)
        self.assertEqual(err.exception.code, 401)
        with self.request('/v1/models', headers={'Authorization': 'Bearer '+replacement}) as response:
            self.assertEqual(response.status, 200)

    def test_all_fields_sync_sse_async_context_and_resident_isolation(self):
        body = self.body()
        model_before = self.service.model('qwen-chat')
        context = self.json('/api/context', body)
        result = self.json('/v1/chat/completions', body)
        self.assertEqual(result['x_context']['prompt_sha256'], context['prompt_sha256'])
        self.assertEqual(result['x_context']['input_tokens'], context['input_tokens'])
        self.assertFalse(result['x_template_context']['enable_thinking'])
        self.assertEqual(result['timings']['predicted_n'], result['usage']['completion_tokens'])
        self.assertEqual(result['timings']['x_source'], 'estimated-test')
        self.assertEqual(len(result['x_timing_passes']), 1)
        resident = self.service.scheduler.resident['qwen-chat'][1]
        with self.request('/v1/chat/completions', {**body, 'stream': True, 'reasoning_effort': 'none'}) as response:
            wire = response.read().decode()
        chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        text_chunks = [c for c in chunks if c.get('choices', [{}])[0].get('delta', {}).get('content')]
        self.assertTrue(text_chunks)
        self.assertTrue(all(c['timings'] and c['timings']['predicted_n'] > 0 for c in text_chunks))
        final = next(c for c in chunks if c.get('choices', [{}])[0].get('finish_reason'))
        self.assertFalse(final['x_template_context']['enable_thinking'])
        self.assertEqual(len(final['x_timing_passes']), 1)
        self.assertIn('data: [DONE]', wire)
        receipt = self.json('/api/chat/jobs', {**body, 'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_effort': 'low'})
        wait_job(self.service, receipt['id'])
        saved = self.json(receipt['result_url'])
        self.assertTrue(saved['x_template_context']['enable_thinking'])
        self.assertEqual(saved['x_template_context']['reasoning_effort'], 'low')
        self.assertEqual(self.json(receipt['result_url'])['timings'], saved['timings'])
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1], resident)
        self.assertEqual(self.service.model('qwen-chat'), model_before)
        plain = self.json('/v1/chat/completions', self.body(timings_per_token=False, reasoning_effort=None, chat_template_kwargs=None))
        self.assertNotIn('timings', plain)
        self.assertEqual(plain['x_template_context'], model_before['template_context'])

    def test_invalid_extension_is_rejected_before_enqueue(self):
        before = len(self.service.store.jobs())
        for change in [{'chat_template_kwargs': []}, {'timings_per_token': 1}, {'reasoning_effort': 'unsupported'}]:
            with self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/api/chat/jobs', self.body(**change))
            self.assertEqual(err.exception.code, 400)
        self.assertEqual(len(self.service.store.jobs()), before)

    def test_template_settings_survive_image_handoff_and_retry_is_local(self):
        original = self.service.model('qwen-chat')
        workers = FixtureWorkers([('<think>Still deciding', 64), (qwen_call(), 45), ('Image ready.', 4)])
        self.service.scheduler.factory = workers
        result = self.json('/v1/chat/completions', self.body(max_tokens=64, auto_image=True,
            image_model='qwen-image', reasoning_effort='low', chat_template_kwargs={'enable_thinking': True, 'custom': 'keep'}))
        calls = [c for c in workers.calls if c['action'] == 'text']
        self.assertEqual(len(calls), 3)
        self.assertEqual([c['prepared']['template_context']['enable_thinking'] for c in calls], [True, False, True])
        self.assertTrue(all(c['prepared']['template_context']['custom'] == 'keep' for c in calls))
        self.assertTrue(all(c['prepared']['template_context']['reasoning_effort'] == 'low' for c in calls))
        self.assertEqual(result['choices'][0]['message']['content'], 'Image ready.')
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertEqual([p['stage'] for p in result['x_timing_passes']], ['initial', 'retry', 'continuation'])
        self.assertEqual(self.service.model('qwen-chat'), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
