"""Reasoning/format regressions with explicit transport and model fixtures."""
import copy
import json
import sys
import time
import types
import unittest
import urllib.error
from unittest.mock import patch
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from lazyrig.context import IMAGE_TOOL
from lazyrig.reasoning import split_reasoning
from lazyrig.response_format import validate_format, configure_output
from lazyrig.worker import _child

TITLE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'session_title', 'strict': True,
    'schema': {'type': 'object', 'properties': {'title': {'type': 'string'}},
               'required': ['title'], 'additionalProperties': False}}}
RAW = '<think>Check responsiveness.</think>\n\nWorking. What do you need?'


class ReasoningTests(unittest.TestCase):
    def test_every_stream_boundary_matches_final_channels(self):
        cases = [(RAW, False, 'Working. What do you need?', 'Check responsiveness.'),
                 (RAW.removeprefix('<think>'), True, 'Working. What do you need?', 'Check responsiveness.'),
                 (' \n'+RAW+'<|im_end|>\n', False, 'Working. What do you need?', 'Check responsiveness.'),
                 ('<think></think>\nOkay', False, 'Okay', ''),
                 ('<think>One</think>\n<think>Two</think>Answer', False, 'Answer', 'One\nTwo'),
                 ('  A plain response\n', False, '  A plain response\n', ''),
                 ('The literal `<think>` is an opening tag.', False, 'The literal `<think>` is an opening tag.', '')]
        for raw, implicit, answer, thought in cases:
            previous = {'content': '', 'reasoning_content': ''}
            for end in range(len(raw)+1):
                current = split_reasoning(raw[:end], implicit, final=False)
                for field in previous:
                    self.assertTrue(current[field].startswith(previous[field]), (raw, end, field, previous, current))
                previous = {k: current[k] for k in previous}
            final = split_reasoning(raw, implicit)
            self.assertEqual(final['content'], answer)
            self.assertEqual(final['reasoning_content'], thought)
            self.assertFalse(final['reasoning_incomplete'])
            for field in previous:
                self.assertTrue(final[field].startswith(previous[field]))

    def test_incomplete_reasoning_never_becomes_answer_and_plain_prose_is_not_guessed(self):
        for raw, implicit in [('<think>Still checking', False), ('Still checking', True), ('<think>Still checking</thi', False)]:
            value = split_reasoning(raw, implicit)
            self.assertEqual(value['content'], '')
            self.assertTrue(value['reasoning_incomplete'])
            self.assertIn('Still checking', value['reasoning_content'])
        self.assertEqual(split_reasoning('Let me think about that.')['content'], 'Let me think about that.')


class ResponseHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **extra):
        return {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'test'}], 'max_tokens': 256, **extra}

    def fixture(self, responses, implicit=False):
        workers = FixtureWorkers([(text, 25) for text in responses])
        self.service.scheduler.factory = workers
        if implicit:
            original = self.service.tokens.prepare
            def prepare(*args, **kwargs):
                value = original(*args, **kwargs)
                value['thinking_open'] = True
                return value
            self.service.tokens.prepare = prepare
        return workers

    def test_plain_tool_enabled_and_streaming_replies_have_identical_channels(self):
        workers = self.fixture([RAW]*3)
        plain = self.json('/v1/chat/completions', self.body())
        tool_enabled = self.json('/v1/chat/completions', self.body(tools=[IMAGE_TOOL]))
        with self.request('/v1/chat/completions', self.body(stream=True, stream_options={'include_usage': True})) as response:
            records = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: {')]
        deltas = [v['choices'][0]['delta'] for v in records if v.get('choices')]
        message = plain['choices'][0]['message']
        self.assertEqual(message['content'], 'Working. What do you need?')
        self.assertEqual(message['reasoning_content'], 'Check responsiveness.')
        self.assertEqual(tool_enabled['choices'][0]['message'], message)
        for field in ('content', 'reasoning_content'):
            self.assertEqual(''.join(d.get(field, '') for d in deltas), message[field])
        self.assertFalse(any('<think>' in d.get('content', '') or 'responsiveness' in d.get('content', '') for d in deltas))
        job = wait_job(self.service, plain['x_job_id'])
        self.assertEqual(job['text'], RAW)  # Original diagnostic output remains available.
        self.assertEqual(self.service.store.messages(job['session_id'])[-1], message)
        self.assertEqual(records[-1]['usage'], plain['usage'])
        self.assertEqual(len([e for e in workers.lifecycle if e[0] == 'load']), 1)

    def test_live_sse_handles_marker_fragments_without_content_leakage(self):
        workers = FixtureWorkers([])
        def factory(*args):
            worker = workers(*args)
            def run(payload, *unused):
                text = ''
                for index, piece in enumerate(['<th', 'ink>Analyze', ' the request.', '</thi', 'nk>\n', 'Working.', '<|im_', 'end|>']):
                    text += piece
                    yield {'kind':'text', 'text':text, 'generated_tokens':index+1, 'accuracy':'estimated-test'}
                    time.sleep(.008)
                yield {'kind':'result', 'text':text, 'generated_tokens':8, 'accuracy':'estimated-test'}
            worker.run = run
            return worker
        self.service.scheduler.factory = factory
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            values = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: {')]
        self.assertFalse(any('error' in value for value in values), values)
        deltas = [value['choices'][0]['delta'] for value in values if value.get('choices')]
        self.assertEqual(''.join(d.get('content','') for d in deltas), 'Working.')
        self.assertEqual(''.join(d.get('reasoning_content','') for d in deltas), 'Analyze the request.')
        self.assertGreater(len([d for d in deltas if d.get('reasoning_content')]), 1)

    def test_quoted_examples_remain_text_but_formatted_preambles_allow_tool_calls(self):
        example = 'Example:\n```xml\n'+qwen_call()+'\n```'
        actual = '<think>Use the image tool.</think>\nI will use `generate_image`.\n'+qwen_call()
        self.fixture([example, actual, example])
        first = self.json('/v1/chat/completions', self.body(tools=[IMAGE_TOOL]))
        self.assertEqual(first['choices'][0]['message']['content'], example)
        self.assertNotIn('tool_calls', first['choices'][0]['message'])
        second = self.json('/v1/chat/completions', self.body(tools=[IMAGE_TOOL]))
        message = second['choices'][0]['message']
        self.assertEqual(message['reasoning_content'], 'Use the image tool.')
        self.assertEqual(message['content'], 'I will use `generate_image`.')
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'generate_image')
        third = self.json('/v1/chat/completions', self.body(auto_image=True, image_model='qwen-image'))
        self.assertEqual(third['choices'][0]['message']['content'], example)
        self.assertFalse(third['x_artifacts'])

    def test_implicit_reasoning_async_retrieval_and_truncation(self):
        self.fixture([RAW.removeprefix('<think>'), 'Still thinking'], implicit=True)
        receipt = self.json('/api/chat/jobs', self.body())
        wait_job(self.service, receipt['id'])
        result = self.json(receipt['result_url'])
        self.assertEqual(result['choices'][0]['message']['content'], 'Working. What do you need?')
        self.assertEqual(self.json(receipt['result_url']), result)
        truncated = self.json('/v1/chat/completions', self.body(max_tokens=25, tools=[IMAGE_TOOL]))
        self.assertEqual(truncated['choices'][0]['finish_reason'], 'length')
        self.assertEqual(truncated['choices'][0]['message']['content'], '')
        self.assertEqual(truncated['choices'][0]['message']['reasoning_content'], 'Still thinking')
        self.assertTrue(truncated['x_reasoning_incomplete'])

    def test_separation_after_image_handoff_and_in_client_tool_response(self):
        self.fixture(['<think>Need an image.</think>'+qwen_call(), RAW,
                      '<think>Need an image.</think>'+qwen_call()])
        result = self.json('/v1/chat/completions', self.body(auto_image=True, image_model='qwen-image'))
        self.assertEqual(result['choices'][0]['message']['content'], 'Working. What do you need?')
        self.assertEqual(result['choices'][0]['message']['reasoning_content'], 'Check responsiveness.')
        self.assertEqual(len(result['x_artifacts']), 1)
        tool = self.json('/v1/chat/completions', self.body(tools=[IMAGE_TOOL]))['choices'][0]
        self.assertEqual(tool['finish_reason'], 'tool_calls')
        self.assertEqual(tool['message']['reasoning_content'], 'Need an image.')
        self.assertEqual(tool['message']['tool_calls'][0]['function']['name'], 'generate_image')

    def test_hermes_title_format_context_defaults_and_request_isolation(self):
        workers = self.fixture(['{"title":"Connection test"}', RAW, '{"ok":true}'])
        # Retired global settings cannot alter title generation or ordinary chat.
        self.service.config.update(default_auto_image=True, default_image_model='qwen-image')
        body = self.body(response_format=TITLE_FORMAT, reasoning_effort='none')
        before = copy.deepcopy(self.service.config['models'])
        context = self.json('/api/context', body)
        title = self.json('/v1/chat/completions', body)
        self.assertEqual(json.loads(title['choices'][0]['message']['content']), {'title': 'Connection test'})
        self.assertTrue(title['x_json_valid'])
        self.assertEqual(title['x_context']['prompt_sha256'], context['prompt_sha256'])
        self.assertEqual(workers.calls[0]['response_format'], TITLE_FORMAT)
        self.assertFalse(workers.calls[0]['tools_enabled'])
        self.assertFalse(workers.calls[0]['prepared']['template_context']['enable_thinking'])
        self.assertEqual(len(workers.calls), 1)
        resident = self.service.scheduler.resident['qwen-chat'][1]
        regular = self.json('/v1/chat/completions', self.body(response_format={'type':'text'}, auto_image=False))
        self.assertEqual(regular['choices'][0]['message']['content'], 'Working. What do you need?')
        receipt = self.json('/api/chat/jobs', self.body(response_format={'type':'json_object'}))
        wait_job(self.service, receipt['id'])
        self.assertTrue(self.json(receipt['result_url'])['x_json_valid'])
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1], resident)
        self.assertEqual(self.service.config['models'], before)

    def test_bad_formats_rejected_before_queue_and_invalid_output_not_success(self):
        workers = self.fixture(['not JSON', '{"title":'])
        for fmt in ['json', {}, {'type':'xml'}, {'type':'json_schema','json_schema':{}},
                    {'type':'json_schema','json_schema':{**TITLE_FORMAT['json_schema'],'strict':'true'}}]:
            with self.subTest(fmt=fmt), self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/api/chat/jobs', self.body(response_format=fmt))
            self.assertEqual(err.exception.code, 400)
        self.assertFalse(workers.calls)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/v1/chat/completions', self.body(response_format=TITLE_FORMAT, tools=[IMAGE_TOOL]))
        self.assertEqual(err.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/v1/chat/completions', self.body(response_format=TITLE_FORMAT))
        self.assertEqual(err.exception.code, 502)
        rejected = json.load(err.exception)
        self.assertEqual(rejected['error']['code'], 'invalid_response_format')
        self.assertFalse(rejected['x_retryable'])
        self.assertIn('valid JSON', rejected['error']['message'])
        limited = self.json('/v1/chat/completions', self.body(response_format=TITLE_FORMAT, max_tokens=25))
        self.assertEqual(limited['choices'][0]['finish_reason'], 'length')
        self.assertFalse(limited['x_json_valid'])


class WorkerContractTests(unittest.TestCase):
    def test_worker_preserves_markers_and_resets_native_constraint_between_requests(self):
        calls, decode_flags, stream_flags = [], [], []
        class Structured:
            def __init__(self, json_schema=''): self.json_schema = json_schema
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                decode_flags.append(skip_special_tokens)
                return ''.join(chr(i) for i in ids)
        class TextStreamer:
            def __init__(self, tokenizer, callback, params):
                stream_flags.append(params['skip_special_tokens'])
                self.callback = callback
            def write(self, ids): self.callback(''.join(chr(i) for i in ids))
            def end(self): pass
        class Pipeline:
            def __init__(self, *args, **kwargs):
                self.config = types.SimpleNamespace(structured_output_config=Structured())
            def get_tokenizer(self): return Tokenizer()
            def get_generation_config(self): return self.config
            def generate(self, inputs, generation_config, streamer):
                schema = generation_config.structured_output_config.json_schema if generation_config.structured_output_config else ''
                calls.append(schema)
                raw = '{"title":"Connection test"}' if schema else RAW
                ids = list(map(ord, raw))
                for token in ids: streamer.write([token])
                streamer.end()
                return types.SimpleNamespace(tokens=[ids])
        fake = types.SimpleNamespace(LLMPipeline=Pipeline, VLMPipeline=Pipeline, Text2ImagePipeline=Pipeline,
                                     TextStreamer=TextStreamer, StreamerBase=object,
                                     StreamingStatus=types.SimpleNamespace(RUNNING=0), StructuredOutputConfig=Structured)
        prepared = {'context':{'input_tokens':2}, 'text':'test', 'token_ids':[1,2]}
        requests = [{'action':'text','prepared':prepared,'max_tokens':256,'response_format':fmt} for fmt in ({'type':'text'},TITLE_FORMAT,{'type':'text'})]
        class Connection:
            def __init__(self): self.responses=[]
            def poll(self, timeout=0): return False  # No out-of-band cancellation in this fixture.
            def recv(self):
                if requests: return requests.pop(0)
                raise EOFError
            def send(self, data): self.responses.append(data)
            def close(self): pass
        conn = Connection()
        with patch.dict(sys.modules, {'openvino_genai':fake, 'openvino':types.SimpleNamespace(Tensor=lambda value:value)}):
            _child(conn, 'openvino', {'pipeline':'llm','path':'fixture','device':'CPU'})
        self.assertFalse(any(x['kind']=='error' for x in conn.responses), conn.responses[-1])
        results = [x for x in conn.responses if x['kind']=='result']
        self.assertEqual([x['text'] for x in results], [RAW,'{"title":"Connection test"}',RAW])
        self.assertEqual(calls[0], '')
        self.assertEqual(json.loads(calls[1]), TITLE_FORMAT['json_schema']['schema'])
        self.assertEqual(calls[2], '')
        self.assertEqual(decode_flags, [False]*3)
        self.assertEqual(stream_flags, [False]*3)

    def test_installed_native_structured_config_accepts_title_and_resets(self):
        try:
            import openvino_genai as genai
        except ImportError:
            self.skipTest('Optional native OpenVINO runtime is not installed')
        config = genai.GenerationConfig(max_new_tokens=64)
        configure_output(genai, config, validate_format(TITLE_FORMAT))
        self.assertIsInstance(config.structured_output_config, genai.StructuredOutputConfig)
        config.validate()
        configure_output(genai, config, {'type':'text'})
        self.assertIsNone(config.structured_output_config)
        config.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
