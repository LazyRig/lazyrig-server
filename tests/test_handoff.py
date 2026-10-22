"""Qwen-format regression tests. Workers are controlled fixtures, not inference."""
from support import fixture_png
import copy
import json
import threading
import unittest
import urllib.error
from pathlib import Path
import test_http
from test_core import wait_job
from lazyrig.context import IMAGE_TOOL, parse_image_tool
from lazyrig.tool_calls import parse_tool_calls, ToolCallError


def qwen_call(prompt="A red bicycle by a lake"):
    return "<tool_call>\n<function=generate_image>\n<parameter=prompt>\n" + prompt + "\n</parameter>\n</function>\n</tool_call>"


class ToolFormatTests(unittest.TestCase):
    def test_qwen_xml_with_preamble_and_both_reasoning_forms(self):
        prompt = 'A café at dawn.\nA sign reads "Hello" & a cyclist passes.'
        call = qwen_call(prompt)
        for text, opened in [(call, False), ("I'll create that image.\n" + call, False),
                             ("<think>I should use the tool.</think>\n" + call, False),
                             ("I should use the tool.</think>\n" + call, True),
                             ("</think>\n" + call + "<|im_end|>", True), (call + "</s>", False)]:
            with self.subTest(opened=opened, text=text[:30]):
                self.assertEqual(parse_image_tool(text, opened), {"prompt": prompt})

    def test_legacy_json_and_json_strings_containing_markers(self):
        prompt = 'A sign saying </tool_call> and a literal "quote"'
        call = {"name": "generate_image", "arguments": {"prompt": prompt}}
        for arguments in [call['arguments'], json.dumps(call['arguments'])]:
            self.assertEqual(parse_image_tool('<tool_call>'+json.dumps({**call, 'arguments': arguments})+'</tool_call>'), {'prompt': prompt})

    def test_quoted_mixed_unknown_duplicate_and_incomplete_calls_never_dispatch(self):
        call = qwen_call()
        quoted = ['```xml\n'+call+'\n```', 'An example: '+call, '"'+call+'"',
                  '`'+call+'`', '> '+call, '    '+call]
        for text in quoted:
            with self.subTest(quoted=text[:40]):
                self.assertIsNone(parse_image_tool(text))
        invalid = [call+'\nHere is the result', call+call, call[:-6],
                   call.replace('generate_image', 'shell'),
                   call.replace('</parameter>', '</parameter><parameter=prompt>other</parameter>'),
                   '<tool_call>{"name":[],"arguments":{}}</tool_call>',
                   '<tool_call>{"name":"generate_image","arguments":{"prompt":"A","prompt":"B"}}</tool_call>',
                   '<tool_call>{"name":"generate_image","arguments":{"prompt":1e999}}</tool_call>']
        for text in invalid:
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                parse_image_tool(text)
        self.assertIsNone(parse_image_tool('<think>'+call+'</think>Just an explanation.'))
        self.assertIsNone(parse_image_tool('No image was requested.'))
        with self.assertRaises(ToolCallError) as err:
            parse_image_tool('Still reasoning', True)
        self.assertTrue(err.exception.incomplete)

    def test_tool_preamble_can_use_inline_code_quotes_and_closed_fences(self):
        call = qwen_call()
        for prefix in ["I'll use `generate_image`.\n", 'The image will say "Hello".\n',
                       '```text\nA description\n```\nNow I will generate it.\n',
                       'Example:\n```xml\n'+call+'\n```\nNow executing:\n']:
            with self.subTest(prefix=prefix):
                self.assertEqual(parse_image_tool(prefix+call), {'prompt': 'A red bicycle by a lake'})

    def test_client_schema_preserves_xml_string_types(self):
        tool = {'type': 'function', 'function': {'name': 'lookup', 'parameters': {'properties': {
            'label': {'type': 'string'}, 'count': {'type': 'integer'}, 'enabled': {'type': 'boolean'}}}}}
        text = '<tool_call><function=lookup><parameter=label>123</parameter><parameter=count>2</parameter><parameter=enabled>true</parameter></function></tool_call>'
        parsed = parse_tool_calls(text, [tool])
        self.assertEqual(parsed['calls'][0]['arguments'], {'label': '123', 'count': 2, 'enabled': True})


class FixtureWorkers:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.lifecycle = []

    def __call__(self, mode, model, timeout, cancelled):
        owner = self
        owner.lifecycle.append(('load', model['id']))

        class Worker:
            alive = True

            def __init__(self):
                self.process = self

            def is_alive(self):
                return self.alive

            def close(self):
                self.alive = False
                owner.lifecycle.append(('close', model['id']))

            def run(self, payload, timeout, cancelled):
                owner.calls.append(copy.deepcopy(payload))
                if payload['action'] == 'image':
                    fixture_png(payload['output_path'], 'HANDOFF TEST FIXTURE', payload['width'], payload['height'])
                    yield {'kind': 'progress', 'step': 1, 'steps': 1}
                else:
                    text, count = owner.responses.pop(0)
                    yield {'kind': 'result', 'text': text, 'generated_tokens': count,
                           'input_tokens': payload['prepared']['context']['input_tokens'], 'accuracy': 'estimated-test'}
        return Worker()


class HandoffTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def fixture(self, responses, thinking=False):
        self.workers = FixtureWorkers(responses)
        self.service.scheduler.factory = self.workers
        self.service.config['models'][1]['generation'] = {
            'width': 384, 'height': 256, 'steps': 7, 'seed': 19, 'guidance_scale': 3.5, 'negative_prompt': 'blur'}
        if thinking:
            self.service.config['models'][0]['template_context']['enable_thinking'] = True
            prepare = self.service.tokens.prepare
            def opened(model, *args, **kwargs):
                value = prepare(model, *args, **kwargs)
                value['thinking_open'] = model.get('template_context', {}).get('enable_thinking', True)
                return value
            self.service.tokens.prepare = opened

    def chat(self, **extra):
        return self.json('/v1/chat/completions', {'model': 'qwen-chat', 'messages': [
            {'role': 'user', 'content': 'Generate an image of a red bicycle by a lake'}],
            'auto_image': True, 'image_model': 'qwen-image', 'max_tokens': 512, **extra})

    def check_roundtrip(self, result, resident=('qwen-chat',)):
        self.assertEqual(len(result['x_artifacts']), 1)
        job = wait_job(self.service, result['x_job_id'])
        self.assertEqual(job['status'], 'completed', job['error'])
        self.assertEqual(set(self.service.scheduler.resident), set(resident))
        images = [c for c in self.workers.calls if c['action'] == 'image']
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]['prompt'], 'A red bicycle by a lake')
        self.assertEqual([images[0][k] for k in ('width', 'height', 'steps', 'seed', 'guidance_scale')], [384, 256, 7, 19, 3.5])
        self.assertEqual(images[0]['negative_prompt'], 'blur')
        self.assertNotIn('max_tokens', images[0])
        history = self.service.store.messages(job['session_id'])
        self.assertEqual([m['role'] for m in history], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(history[1]['tool_calls'][0]['id'], history[2]['tool_call_id'])
        self.assertTrue(json.loads(history[2]['content'])['ok'])
        return job

    def test_qwen_roundtrip_with_image_initially_unloaded(self):
        self.fixture([("I'll generate it now.\n"+qwen_call(), 90), ('Image saved.', 8)])
        self.check_roundtrip(self.chat())
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
                         ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_qwen_roundtrip_with_image_initially_loaded(self):
        self.fixture([(qwen_call(), 90), ('Image saved.', 8)])
        self.service.scheduler.ensure(self.service.model('qwen-image'), threading.Event(), lambda *a, **k: None)
        self.check_roundtrip(self.chat())
        self.assertEqual(self.workers.lifecycle[:3], [('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_roundtrip_when_both_models_fit_and_image_can_be_reused(self):
        self.fixture([(qwen_call(), 90), ('Image saved.', 8)])
        self.service.config['devices'][0].update(policy='budget', budget_gib=40)
        self.service.scheduler.ensure(self.service.model('qwen-image'), threading.Event(), lambda *a, **k: None)
        self.check_roundtrip(self.chat(), resident=('qwen-chat', 'qwen-image'))
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-image'), ('load', 'qwen-chat')])

    def test_truncated_reasoning_retries_once_without_thinking_then_executes_once(self):
        self.fixture([('I need to consider how to create this image...', 256),
                      (qwen_call(), 90), ('The image has been saved.', 12)], thinking=True)
        result = self.chat(max_tokens=256)
        job = self.check_roundtrip(result)
        self.assertEqual(result['x_tool_attempts'], 2)
        self.assertEqual(job['tool_attempts'][0]['code'], 'incomplete_reasoning')
        calls = [c for c in self.workers.calls if c['action'] == 'text']
        self.assertEqual([c['max_tokens'] for c in calls], [256, 256, 256])
        self.assertFalse(calls[1]['prepared']['thinking_open'])
        self.assertTrue(self.service.model('qwen-chat')['template_context']['enable_thinking'])
        self.assertEqual(result['usage']['completion_tokens'], 256+90+12)

    def test_incomplete_calls_fail_with_diagnostics_and_no_image_execution(self):
        self.fixture([(qwen_call()[:-6], 256), (qwen_call()[:-6], 256)])
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.chat(max_tokens=256)
        result = json.load(err.exception)
        self.assertEqual(err.exception.code, 500)
        self.assertIn('256-token output limit', result['error']['message'])
        self.assertEqual(len(result['x_tool_diagnostics']), 2)
        self.assertEqual(result['x_artifacts'], [])
        self.assertFalse(any(c['action'] == 'image' for c in self.workers.calls))

    def test_undeclared_tool_is_not_retried_even_at_output_limit(self):
        self.fixture([(qwen_call().replace('generate_image', 'shell'), 256)])
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.chat(max_tokens=256)
        self.assertIn('Unsupported tool', json.load(err.exception)['error']['message'])
        self.assertEqual(len(self.workers.calls), 1)

    def test_plain_text_does_not_trigger_an_image(self):
        self.fixture([('Hello there.', 4)])
        result = self.chat()
        self.assertFalse(result['x_artifacts'])
        self.assertEqual(len(self.workers.calls), 1)

    def test_client_tool_calls_are_normalized_without_server_execution(self):
        self.fixture([('I will call the tool.\n'+qwen_call(), 90)])
        result = self.chat(auto_image=False, tools=[IMAGE_TOOL])
        choice = result['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(choice['message']['content'], 'I will call the tool.')
        self.assertEqual(json.loads(choice['message']['tool_calls'][0]['function']['arguments']), {'prompt': 'A red bicycle by a lake'})
        self.assertFalse(result['x_artifacts'])

    def test_explicit_image_tool_buffers_routing_and_streams_final_response(self):
        self.fixture([(qwen_call(), 90), ('Image saved.', 8)])
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Make a bicycle image'}],
                'stream': True, 'auto_image': True, 'image_model': 'qwen-image'}
        with self.request('/v1/chat/completions', body) as response:
            chunks = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: {')]
        content = ''.join(c['choices'][0]['delta'].get('content', '') or '' for c in chunks if c.get('choices'))
        self.assertEqual(content, 'Image saved.')
        self.assertEqual(len(chunks[-1]['x_artifacts']), 1)
        self.assertNotIn('<tool_call>', content)


if __name__ == '__main__':
    unittest.main(verbosity=2)
