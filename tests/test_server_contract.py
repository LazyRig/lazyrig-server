"""Contract and lifecycle tests; inference here is explicitly simulated."""
import base64
import copy
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import test_http
from test_core import wait_job
from lazyrig.config import SERVER_DEFAULT, restore_settings, validate
from lazyrig.downloads import extract_zip, source_spec
from support import Service

class ServerContractTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_public_chat_and_context_preflight(self):
        messages = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}]
        expected = self.json('/api/context', {"model": "qwen-chat", "messages": messages, "max_tokens": 24})
        result = self.json('/v1/chat/completions', {"model": "qwen-chat", "messages": messages, "max_tokens": 24})
        self.assertEqual(result['object'], 'chat.completion')
        self.assertEqual(result['x_context']['input_tokens'], expected['input_tokens'])
        self.assertEqual(result['usage']['total_tokens'], result['x_context']['used_tokens'])
        self.assertEqual(result['x_mode'], 'openvino')
        self.assertEqual(result['choices'][0]['finish_reason'], 'length')
        self.assertTrue(result['choices'][0]['message']['content'])
        self.assertIsNone(result['x_context']['cache_occupancy'])
        self.assertEqual(len(self.json('/v1/models')['data']), 2)

    def test_stream_reconstructs_response_and_usage(self):
        with self.request('/v1/chat/completions', {"model": "qwen-chat", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 24, "stream": True, "stream_options": {"include_usage": True}}) as response:
            jid = response.headers['X-Job-Id']
            self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
            data = [line[6:] for line in response.read().decode().splitlines() if line.startswith('data: ')]
        self.assertEqual(data[-1], '[DONE]')
        chunks = [json.loads(x) for x in data[:-1]]
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
        job = self.json('/api/jobs/' + jid)
        self.assertEqual(text, job['text'])
        self.assertEqual(chunks[-1]['usage']['total_tokens'], job['context']['used_tokens'])
        self.assertEqual(chunks[-1]['choices'], [])

    def test_output_budget_default_and_explicit_override_precedence(self):
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}]}
        for settings, request, expected in [({}, {}, 8192),
                ({'max_tokens': 512}, {}, 512),
                ({'max_tokens': 512}, {'max_tokens': 64}, 64),
                ({'max_tokens': 512}, {'max_tokens': 64, 'max_completion_tokens': 32}, 32),
                ({}, {}, 8192)]:
            with self.subTest(settings=settings, request=request):
                self.json('/api/settings', {'model': 'qwen-chat', 'generation': settings})
                preflight = self.json('/api/context', {**body, **request})
                result = self.json('/v1/chat/completions', {**body, **request})
                job = self.json('/api/jobs/' + result['x_job_id'])
                self.assertEqual(preflight['output_budget'], expected)
                self.assertEqual(result['x_context']['output_budget'], expected)
                self.assertEqual(job['request']['max_tokens'], expected)

    def test_session_chat_uses_same_default_and_keeps_context_guard(self):
        sid = self.json('/api/sessions', {'title': 'Output budget default'})['id']
        body = {'session_id': sid, 'model': 'qwen-chat', 'prompt': 'Hello', 'tools_enabled': False}
        preflight = self.json('/api/context', {**body, 'draft': 'Hello'})
        job = wait_job(self.service, self.json('/api/jobs', body)['id'])
        self.assertEqual(preflight['output_budget'], 8192)
        self.assertEqual(job['context']['output_budget'], 8192)
        self.assertEqual(job['status'], 'completed', job['error'])
        self.service.config['models'][0]['context_limit'] = 8192
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.json('/v1/chat/completions', {'model': 'qwen-chat',
                'messages': [{'role': 'user', 'content': 'Hello'}]})
        self.assertEqual(error.exception.code, 400)
        self.assertIn('Context overflow', error.exception.read().decode())

    def test_image_tool_preference_is_unique_persisted_and_keeps_worker(self):
        other = copy.deepcopy(self.service.config['models'][1])
        other['id'] = 'other-image'
        self.service.config['models'].append(other)
        wait_job(self.service, self.json('/api/models/qwen-image/load', {})['id'])
        worker = self.service.scheduler.resident['qwen-image'][1]
        for mid in ['qwen-image', 'other-image']:
            self.json('/api/settings', {'model': mid, 'default_image_tool': True})
            restored = restore_settings(self.service.config)
            self.assertEqual([m['id'] for m in restored['models'] if m.get('default_image_tool')], [mid])
            self.assertIs(self.service.scheduler.resident['qwen-image'][1], worker)
        self.json('/api/settings', {'model': 'other-image', 'default_image_tool': False})
        self.assertFalse(any(m.get('default_image_tool') for m in restore_settings(self.service.config)['models']))

    def test_legacy_output_budget_upgrades_once_without_overwriting_custom_limits(self):
        legacy = copy.deepcopy(self.service.config)
        legacy.pop('output_token_defaults_version', None)
        legacy['models'][0]['generation'] = {'max_tokens': 1024, 'temperature': .4}
        for mid, budget, context in [('custom-chat', 512, 32768), ('small-chat', 1024, 4096)]:
            model = copy.deepcopy(legacy['models'][0])
            model.update(id=mid, generation={'max_tokens': budget}, context_limit=context)
            legacy['models'].append(model)
        self.service.settings_path.write_text(json.dumps(legacy))
        restored = restore_settings(self.service.config)
        self.assertEqual(restored['models'][0]['generation'], {'max_tokens': 8192, 'temperature': .4})
        self.assertEqual([m['generation']['max_tokens'] for m in restored['models'][2:]], [512, 1024])
        self.assertEqual(restored['output_token_defaults_version'], 1)
        restored['models'][0]['generation']['max_tokens'] = 1024
        self.service.settings_path.write_text(json.dumps(restored))
        self.assertEqual(restore_settings(self.service.config)['models'][0]['generation']['max_tokens'], 1024)

    def test_invalid_image_tool_preference_is_atomic_and_api_chat_stays_plain(self):
        self.json('/api/settings', {'model': 'qwen-image', 'default_image_tool': True})
        original = copy.deepcopy(self.service.config)
        for body in [{'model': 'qwen-chat', 'default_image_tool': True},
                     {'model': 'qwen-image', 'default_image_tool': 'true'}]:
            with self.subTest(body=body), self.assertRaises(urllib.error.HTTPError) as error:
                self.json('/api/settings', body)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(self.service.config, original)
        result = self.json('/v1/chat/completions', {'model': 'qwen-chat',
            'messages': [{'role': 'user', 'content': 'Generate an image'}], 'max_tokens': 64})
        self.assertEqual(result['x_artifacts'], [])
        self.assertFalse(result['choices'][0]['message'].get('tool_calls'))
        self.assertIsNone(self.json('/api/jobs/' + result['x_job_id'])['image_model'])

    def test_direct_images_restore_loaded_llm_and_use_model_defaults(self):
        self.service.config['models'][1]['restore_text_after_image'] = True
        wait_job(self.service, self.json('/api/models/qwen-chat/load', {})['id'])
        self.service.config['models'][1]['generation'] = {'width': 256, 'height': 384, 'steps': 1}
        result = self.json('/v1/images/generations', {"model": "qwen-image", "prompt": "a lake", "response_format": "b64_json"})
        self.assertIn(b'TEST OUTPUT', base64.b64decode(result['data'][0]['b64_json']))
        self.assertEqual(result['x_media_type'], 'image/png')
        job = wait_job(self.service, result['x_job_id'])
        self.assertEqual((job['request']['width'], job['request']['height'], job['request']['steps']), (256, 384, 1))
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        kinds = [(e['kind'], e.get('model')) for e in self.service.store.events(job['id']) if e['kind'] in {'loading', 'unloaded'}]
        self.assertEqual(kinds, [('unloaded', 'qwen-chat'), ('loading', 'qwen-image'), ('unloaded', 'qwen-image'), ('loading', 'qwen-chat')])

    def test_auto_image_reports_total_usage_separately_from_context(self):
        result = self.json('/v1/chat/completions', {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Draw an image of a lake'}], 'auto_image': True, 'image_model': 'qwen-image', 'steps': 1, 'max_tokens': 512})
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertGreater(result['usage']['prompt_tokens'], result['x_context']['input_tokens'])
        self.assertEqual(result['x_context']['scope'], 'current inference')
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})

    def test_bad_requests_fail_before_enqueue(self):
        base = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hi'}]}
        for extra in [{'logprobs': True}, {'stream_options': None}, {'tools': [7]}, {'max_tokens': 999999}, {'stream': 'yes'}, {'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}]}]:
            with self.subTest(extra=extra), self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/v1/chat/completions', {**base, **extra})
            self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.service.running)
        self.assertFalse(self.service.scheduler.resident)

    def test_api_key_cors_and_persisted_server_settings(self):
        key = 'test-key-at-least-16-characters'
        result = self.json('/api/server', {'api_key': key, 'allowed_origins': ['https://example.test'], 'port': 9876})
        self.assertTrue(result['server']['restart_required'])
        self.assertNotIn('api_key', result['server'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/v1/models')
        self.assertEqual(error.exception.code, 401)
        with self.request('/v1/models', headers={'Authorization': 'Bearer ' + key, 'Origin': 'https://example.test'}) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://example.test')
        persisted = json.loads(self.service.settings_path.read_text())
        self.assertEqual(persisted['port'], 9876)
        self.assertEqual(persisted['api_key'], key)
        with self.request('/api/health') as response:
            self.assertEqual(response.status, 200)

    def test_empty_inspection_path_is_rejected_without_scanning(self):
        with patch.object(Path, 'rglob', side_effect=AssertionError('Unexpected directory scan')):
            for empty_path in ('', '   ', None):
                with self.subTest(path=empty_path):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        self.request('/api/models/inspect', {'path': empty_path})
                    self.assertEqual(error.exception.code, 400)
                    self.assertEqual(json.load(error.exception)['error']['message'], 'Choose a model folder first.')

    def test_folder_registration_update_disable_and_remove(self):
        model = Path(self.temp.name) / 'model'
        fixture_export(model)
        report = self.json('/api/models/inspect', {'path': str(model)})
        self.assertEqual(report['pipeline'], 'llm')
        registered = self.json('/api/models', {'id': 'local-model', 'path': str(model), 'device': 'TEST.CPU', 'memory_gib': 1, 'context_limit': 1024})
        self.assertEqual(registered['path'], str(model))
        folders = self.json('/api/folders?path=' + urllib.parse.quote(self.temp.name))
        self.assertIn(str(model), [d['path'] for d in folders['folders']])
        self.json('/api/settings', {'model': 'local-model', 'enabled': False, 'generation': {'temperature': .25, 'max_tokens': 32}})
        self.assertNotIn('local-model', [m['id'] for m in self.json('/v1/models')['data']])
        self.assertEqual(json.loads(self.service.settings_path.read_text())['models'][-1]['generation']['temperature'], .25)
        self.json('/api/models/local-model/remove', {})
        self.assertTrue(model.is_dir())
        self.assertFalse(any(m['id'] == 'local-model' for m in self.service.config['models']))

    def test_zip_download_inspection_and_nonoverwrite(self):
        root = Path(self.temp.name)
        fixture_export(root / 'source-model')
        with zipfile.ZipFile(root / 'model.zip', 'w') as z:
            for f in (root / 'source-model').iterdir():
                z.write(f, 'export/' + f.name)
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args): pass
        host = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(root)))
        thread = threading.Thread(target=host.serve_forever, daemon=True); thread.start()
        try:
            destination = root / 'downloaded'
            body = {'url': 'http://127.0.0.1:' + str(host.server_port) + '/model.zip', 'destination': str(destination)}
            job = self.json('/api/downloads', body)
            end = time.monotonic() + 10
            while time.monotonic() < end:
                state = next(j for j in self.json('/api/downloads')['downloads'] if j['id'] == job['id'])
                if state['status'] in {'completed', 'failed'}: break
                time.sleep(.03)
            self.assertEqual(state['status'], 'completed', state.get('error'))
            self.assertTrue((destination / 'openvino_model.xml').is_file())
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/downloads', body)
            self.assertEqual(error.exception.code, 400)
            self.assertFalse(list(root.glob('.lazyrig-download-*')))
        finally:
            host.shutdown(); host.server_close(); thread.join(2)

    def test_incomplete_download_is_retried_and_cancelled_without_publishing(self):
        manifest = {'sha': 'test-commit', 'siblings': [{'rfilename': 'openvino_model.xml', 'size': 10}]}
        def remote(request, **kwargs):
            value = json.dumps(manifest).encode() if '/api/models/' in request.full_url else b''
            response = io.BytesIO(value)
            response.headers = {'Content-Length': str(len(value))}
            return response
        destination = Path(self.temp.name) / 'incomplete-export'
        with patch('lazyrig.downloads.urllib.request.urlopen', side_effect=remote):
            job = self.service.downloads.start({'url': 'https://huggingface.co/test/export', 'destination': str(destination)})
            end = time.monotonic() + 5
            while time.monotonic() < end:
                state = next(j for j in self.service.downloads.snapshot() if j['id'] == job['id'])
                if state.get('retry'): break
                time.sleep(.01)
            self.assertIn('Incomplete transfer', state.get('retry', ''))
            self.service.downloads.cancel(job['id'])
            while job['id'] in self.service.downloads.running and time.monotonic() < end:
                time.sleep(.01)
            state = next(j for j in self.service.downloads.snapshot() if j['id'] == job['id'])
            self.assertEqual(state['status'], 'cancelled')
        self.assertFalse(destination.exists())
        self.assertFalse(list(Path(self.temp.name).glob('.lazyrig-download-*')))

class DownloadValidationTests(unittest.TestCase):
    def test_archive_traversal_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / 'unsafe.zip', 'w') as z:
                z.writestr('safe.txt', 'ok')
                z.writestr('../escape', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                extract_zip(root / 'unsafe.zip', root / 'target', 10000)
            self.assertFalse((root / 'target').exists())

    def test_hf_revision_subfolder_and_configuration_validation(self):
        spec = source_spec('https://huggingface.co/org/model/tree/rev/openvino')
        self.assertEqual((spec['repo'], spec['revision'], spec['subfolder']), ('org/model', 'rev', 'openvino'))
        for url in ['file:///etc/passwd', 'https://user:pass@example.test/m.zip', 'https://huggingface.co/org/model/resolve/main/model.bin']:
            with self.assertRaises(ValueError): source_spec(url)
        for change in [{'host': '0.0.0.0'}, {'download_max_gib': -1}, {'allowed_origins': '*'}]:
            with self.assertRaises(ValueError): validate({**copy.deepcopy(SERVER_DEFAULT), **change})

def fixture_export(root):
    root.mkdir()
    for name in ['openvino_model', 'openvino_tokenizer', 'openvino_detokenizer']:
        (root / (name + '.xml')).write_text('<net />')
        (root / (name + '.bin')).write_bytes(b'fixture, not real weights')
    (root / 'config.json').write_text(json.dumps({'max_position_embeddings': 2048, 'model_type': 'test'}))

if __name__ == '__main__': unittest.main(verbosity=2)
