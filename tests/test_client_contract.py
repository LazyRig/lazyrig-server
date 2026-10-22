"""Public-client workflows, durable admission and demand-based residency."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
import urllib.error

import test_http
from support import Service
from test_core import wait_job
from test_handoff import FixtureWorkers
from test_cancellation import GatedImages
from test_image_edits import EditWorkers, export_fixture
from test_vision_mtp import image_url
from lazyrig.server import handler
from lazyrig.tool_calls import parse_tool_calls, ToolCallError
from lazyrig.response_format import format_messages, check_json_output

ART = {'type': 'function', 'function': {'name': 'make_art', 'parameters': {
    'type': 'object', 'properties': {'prompt': {'type': 'string'}}, 'required': ['prompt']}}}
CALL = '{"function":"make_art","args":{"prompt":"A silver train"}}'
BODY = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Draw a silver train'}], 'tools': [ART]}


class ClientContractTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def keyed(self, body=BODY, key='b'*32):
        with self.request('/v1/chat/completions', body, {'Prefer': 'respond-async', 'Idempotency-Key': key}) as response:
            self.assertEqual(response.status, 202)
            return json.load(response)

    def test_concurrent_chat_retries_and_restart_keep_one_job(self):
        workers = FixtureWorkers([(CALL, 40)])
        self.service.scheduler.factory = workers
        with ThreadPoolExecutor(max_workers=5) as pool:
            receipts = list(pool.map(lambda _: self.keyed(), range(5)))
        self.assertEqual({r['id'] for r in receipts}, {'b'*32})
        result = wait_job(self.service, receipts[0]['id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(len(workers.calls), 1)
        self.assertEqual(len(self.service.store.sessions()), 1)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.keyed({**BODY, 'max_tokens': 2})
        self.assertEqual(error.exception.code, 409)
        config = copy.deepcopy(self.service.config)
        self.service.close()
        self.service = Service(config)
        self.server.RequestHandlerClass = handler(self.service)
        replay = self.keyed()
        self.assertEqual(replay['id'], receipts[0]['id'])
        self.assertEqual(self.json(replay['result_url'])['choices'][0]['finish_reason'], 'tool_calls')
        self.assertFalse(self.service.running)

    def test_cancelled_chat_retry_never_requeues(self):
        workers = GatedImages(); self.service.scheduler.factory = workers
        image = self.json('/api/image/jobs', {'model': 'qwen-image', 'prompt': 'Hold the device'})
        self.assertTrue(workers.started.wait(2))
        try:
            receipt = self.keyed()
            self.json(receipt['cancel_url'], {})
            self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'cancelled')
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.keyed()
            self.assertEqual(error.exception.code, 410)
            self.assertEqual(error.exception.headers['X-Should-Retry'], 'false')
            self.assertEqual(workers.started_actions, ['image'])
        finally:
            self.service.cancel(image['id']); workers.release.set()

    def test_failed_tool_protocol_is_a_saved_nonretryable_model_error(self):
        self.service.scheduler.factory = FixtureWorkers([(CALL.replace('make_art', 'image_gen'), 40)])
        receipt = self.keyed()
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error_code'], 'undeclared_tool')
        for path in (receipt['result_url'],):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, 502)
            self.assertEqual(error.exception.headers['X-Should-Retry'], 'false')
            self.assertEqual(json.load(error.exception)['error']['code'], 'undeclared_tool')
        self.assertFalse(job['artifacts'])

    def test_bare_json_tool_sse_does_not_leak_into_content(self):
        self.service.scheduler.factory = FixtureWorkers([(CALL, 40)])
        with self.request('/v1/chat/completions', {**BODY, 'stream': True}) as response:
            chunks = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertFalse(any(c.get('error') for c in chunks))
        deltas = [c['choices'][0]['delta'] for c in chunks if c.get('choices')]
        self.assertEqual(''.join(d.get('content', '') for d in deltas), '')
        calls = [c for d in deltas for c in d.get('tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['function']['name'], 'make_art')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_instruction_endpoints_and_rendering_preserve_exact_files(self):
        fmt = {'type': 'json_object'}
        for name, legacy in (('image_generation.txt', 'system_prompt_t2i.txt'),
                             ('image_editing.txt', 'system_prompt_edit.txt')):
            raw = (Path(__file__).resolve().parents[1]/'lazyrig'/'prompts'/name).read_bytes()
            for route in (name, legacy):
                with self.request('/prompts/'+route) as response:
                    self.assertEqual(response.read(), raw)
            messages = [{'role': 'system', 'content': raw.decode()}, {'role': 'user', 'content': 'A train'}]
            formatted = format_messages(messages, fmt)
            self.assertEqual(formatted[0], messages[0])
            self.assertIn('matching this schema', formatted[-1]['content'])
        with self.assertRaises(ValueError):
            check_json_output(fmt, '{"rewritten_prompt":"A","rewritten_prompt":"B"}')

    def test_new_and_legacy_prompt_routes_require_key_for_external_frontends(self):
        self.service.config.update(api_key='prompt-contract-key-123', browser_access='api-key')
        for name in ('image_generation.txt', 'image_editing.txt',
                     'system_prompt_t2i.txt', 'system_prompt_edit.txt'):
            headers = {'Origin': 'https://frontend.example'}
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/prompts/'+name, headers=headers)
            self.assertEqual(error.exception.code, 401)
            headers['Authorization'] = 'Bearer prompt-contract-key-123'
            with self.request('/prompts/'+name, headers=headers) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], headers['Origin'])

    def test_sdk_tool_generate_edit_and_continuation_use_public_contract(self):
        try:
            from openai import OpenAI, DefaultHttpxClient
        except ImportError:
            self.skipTest('Install openai to run the optional real-SDK compatibility test')
        self.service.config['models'][1]['path'] = str(export_fixture(Path(self.temp.name)/'image-export'))
        self.service.config['models'][1]['runtime_profile'] = 'qwen-image-21'
        workers = EditWorkers([(CALL, 40), ('Your image is ready.', 8)])
        self.service.scheduler.factory = workers
        with OpenAI(base_url=self.base+'/v1', api_key='test-key', max_retries=0,
                    http_client=DefaultHttpxClient(trust_env=False)) as client:
            first = client.chat.completions.create(**BODY)
            message = first.choices[0].message
            self.assertEqual(first.choices[0].finish_reason, 'tool_calls')
            self.assertEqual([c['action'] for c in workers.calls], ['text'])
            generated = client.images.generate(model='qwen-image', prompt='A silver train', size='512x512', response_format='url')
            wait_job(self.service, generated.x_job_id)
            self.assertFalse(self.service.scheduler.resident)  # No eager text reload.
            self.assertTrue(generated.data[0].url.startswith(self.base+'/artifacts/'))
            import base64
            edited = client.images.edit(model='qwen-image', prompt='Add rain', image=('input.png', base64.b64decode(image_url().split(',')[1]), 'image/png'), size='512x512')
            wait_job(self.service, edited.x_job_id)
            self.assertTrue(edited.data[0].b64_json)
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
            messages = [*BODY['messages'], {'role': 'assistant', 'content': message.content,
                'tool_calls': [call.model_dump() for call in message.tool_calls]},
                {'role': 'tool', 'tool_call_id': message.tool_calls[0].id, 'content': json.dumps({'url': edited.data[0].url, 'status': 'completed'})}]
            chunks = list(client.chat.completions.create(model='qwen-chat', messages=messages, tools=[ART], stream=True))
            self.assertEqual(''.join(c.choices[0].delta.content or '' for c in chunks if c.choices), 'Your image is ready.')
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image', 'image_edit', 'text'])
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
            ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])
        self.assertEqual(workers.calls[-1]['prepared']['messages'][-1]['tool_call_id'], message.tool_calls[0].id)


class BareToolEnvelopeTests(unittest.TestCase):
    def test_envelope_names_are_declared_and_quoted_examples_are_not_calls(self):
        self.assertEqual(parse_tool_calls(CALL, [ART])['calls'], [{'name': 'make_art', 'arguments': {'prompt': 'A silver train'}}])
        for call in ('{"arguments":{"prompt":"A silver train"},"name":"make_art"}',
                     '{"type":"function","function":{"name":"make_art","arguments":"{\\"prompt\\":\\"A silver train\\"}"}}'):
            self.assertEqual(parse_tool_calls(call, [ART])['calls'], parse_tool_calls(CALL, [ART])['calls'])
        for text in ('```json\n'+CALL+'\n```', 'Example: '+CALL, '{"name":"a person"}', '<think>'+CALL+'</think>Done.'):
            self.assertFalse(parse_tool_calls(text, [ART])['calls'])
        for text in (CALL.replace('make_art', 'image_gen'), CALL[:-2], CALL.replace('"prompt":', '"prompt":"duplicate","prompt":')):
            with self.assertRaises(ToolCallError):
                parse_tool_calls(text, [ART])


if __name__ == '__main__':
    unittest.main()
