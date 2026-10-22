"""Sampling persistence, request precedence, native binding and result diagnostics."""
import copy
import json
import sys
import types
import unittest
import urllib.error
from unittest.mock import patch
import test_http
from test_handoff import FixtureWorkers
from lazyrig.config import restore_settings
from lazyrig.worker import _child


THINKING = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
            'presence_penalty': 0.0, 'repetition_penalty': 1.0}
INSTRUCT = {**THINKING, 'temperature': .7, 'top_p': .8, 'presence_penalty': 1.5}


class SamplingHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_official_presets_persist_and_request_overrides_reach_worker(self):
        presets = self.json('/api/state')['sampling_presets']
        self.assertEqual([p['generation'] for p in presets], [THINKING, INSTRUCT])
        workers = FixtureWorkers([('Done.', 2)] * 3)
        self.service.scheduler.factory = workers
        saved = {**INSTRUCT, 'frequency_penalty': .25}
        self.json('/api/settings', {'model': 'qwen-chat', 'generation': saved})
        self.assertEqual(restore_settings(self.service.config)['models'][0]['generation'], saved)
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 32}
        overrides = {'temperature': 1.0, 'top_p': .95, 'top_k': 7,
                     'min_p': .1, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}
        for payload, expected in [(body, saved), ({**body, **overrides}, {**saved, **overrides}), (body, saved)]:
            result = self.json('/v1/chat/completions', payload)
            call = workers.calls[-1]
            self.assertEqual(call['temperature'], expected['temperature'])
            self.assertEqual(call['generation'], {k: v for k, v in expected.items() if k != 'temperature'})
            self.assertEqual(self.service.store.job(result['x_job_id'])['generation'], call['generation'])
        self.assertEqual(self.service.model('qwen-chat')['generation'], saved)

    def test_invalid_sampling_is_rejected_before_admission_or_settings_changes(self):
        workers = FixtureWorkers([])
        self.service.scheduler.factory = workers
        original = copy.deepcopy(self.service.config)
        cases = [('presence_penalty', 2.01), ('frequency_penalty', -2.01), ('min_p', -.01),
                 ('min_p', 1.01), ('presence_penalty', True), ('frequency_penalty', None),
                 ('min_p', '0.1'), ('top_k', 2.5), ('temperature', False), ('repetition_penalty', 0)]
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}]}
        for key, value in cases:
            for route, payload in [('/v1/chat/completions', {**body, key: value}),
                                   ('/api/settings', {'model': 'qwen-chat', 'generation': {key: value}})]:
                with self.subTest(route=route, key=key, value=value), self.assertRaises(urllib.error.HTTPError) as error:
                    self.request(route, payload)
                self.assertEqual(error.exception.code, 400)
                detail = json.loads(error.exception.read())['error']
                self.assertIn(key, detail['message'] if isinstance(detail, dict) else detail)
        self.assertEqual(self.service.config, original)
        self.assertFalse(workers.calls)
        self.assertFalse(self.service.store.jobs())

    def test_effective_sampling_survives_json_sse_and_saved_results(self):
        workers = FixtureWorkers([('Done.', 2)] * 2)
        actual = {**INSTRUCT, 'frequency_penalty': 0.0, 'do_sample': True, 'source': 'fixture'}
        def factory(*args):
            worker = workers(*args); run = worker.run
            def with_settings(*params):
                yield {'kind': 'sampling', 'settings': actual.copy()}
                yield from run(*params)
            worker.run = with_settings
            return worker
        self.service.scheduler.factory = factory
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 32}
        response = self.json('/v1/chat/completions', body)
        self.assertEqual(response['x_sampling'], actual)
        with self.request('/v1/chat/completions', {**body, 'stream': True}) as stream:
            jid = stream.headers['X-Job-Id']
            chunks = [json.loads(line[6:]) for line in stream.read().decode().splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertEqual(chunks[-1]['x_sampling'], actual)
        self.assertEqual(self.json('/api/jobs/' + jid + '/result')['x_sampling'], actual)
        self.assertEqual(self.json('/api/jobs/' + jid)['sampling'], actual)


class NativeSamplingContractTests(unittest.TestCase):
    def test_worker_applies_explicit_zeros_and_preserves_unset_export_defaults(self):
        try:
            import openvino_genai as genai
        except ImportError:
            self.skipTest('Optional OpenVINO GenAI binding')
        calls, events = [], []
        class Tokenizer:
            def decode(self, tokens, **unused): return 'Done.'
        class Decoder:
            def __init__(self, tokenizer, callback, options): self.callback = callback
            def write(self, tokens): self.callback('Done.')
            def end(self): pass
        class Pipeline:
            def __init__(self, *args, **kwargs): pass
            def get_tokenizer(self): return Tokenizer()
            def get_generation_config(self):
                config = genai.GenerationConfig()
                config.set_eos_token_id(2)
                config.min_p, config.presence_penalty = .15, -.5
                return config
            def generate(self, inputs, generation_config, streamer, **unused):
                generation_config.validate()
                calls.append(generation_config)
                streamer.write(99)
                return types.SimpleNamespace(tokens=[[99]])
        payload = {'action': 'text', 'max_tokens': 4, 'temperature': 1.0,
                   'prepared': {'token_ids': [1], 'context': {'input_tokens': 1}}}
        requests = iter([{**payload, 'generation': {**THINKING, 'frequency_penalty': .3}},
                         {**payload, 'temperature': 0, 'generation': {}}])
        class Connection:
            def send(self, item): events.append(item)
            def poll(self, timeout=0): return False  # No out-of-band cancellation in this fixture.
            def recv(self):
                try: return next(requests)
                except StopIteration: raise EOFError()
            def close(self): pass
        fake = types.SimpleNamespace(LLMPipeline=Pipeline, VLMPipeline=Pipeline, Text2ImagePipeline=Pipeline,
                                     TextStreamer=Decoder, StreamerBase=object,
                                     StreamingStatus=types.SimpleNamespace(RUNNING=0))
        with patch.dict(sys.modules, {'openvino_genai': fake}):
            _child(Connection(), 'openvino', {'pipeline': 'llm', 'path': '', 'device': 'CPU'})
        self.assertFalse([e for e in events if e['kind'] == 'error'], events)
        self.assertEqual(len(calls), 2)
        for key, value in THINKING.items():
            self.assertAlmostEqual(getattr(calls[0], key), value)
        self.assertAlmostEqual(calls[0].frequency_penalty, .3)
        self.assertFalse(calls[1].do_sample)
        self.assertAlmostEqual(calls[1].min_p, .15)
        self.assertAlmostEqual(calls[1].presence_penalty, -.5)
        snapshots = [e['settings'] for e in events if e['kind'] == 'sampling']
        self.assertEqual([s['temperature'] for s in snapshots], [1.0, 0.0])
        self.assertEqual([s['source'] for s in snapshots], ['openvino', 'openvino'])


if __name__ == '__main__': unittest.main(verbosity=2)
