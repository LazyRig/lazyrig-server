"""Global settings cannot silently turn a client's chat into an image job."""
import copy
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import test_http
from support import DEFAULT
from test_handoff import FixtureWorkers, qwen_call
from test_core import wait_job
from lazyrig.config import restore_settings
from lazyrig.context import IMAGE_TOOL


class ImageRoutingTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **extra):
        return {'model': 'qwen-chat', 'messages': [
            {'role': 'user', 'content': 'Generate an image of a bicycle'}], **extra}

    def legacy_config(self):
        # Even stale in-memory keys must not activate routing or select a model.
        self.service.config.update(default_auto_image=True, default_image_model='missing-model')

    def test_ordinary_chat_and_context_ignore_legacy_global_settings(self):
        workers = FixtureWorkers([('Use the image endpoint to generate a picture.', 12)])
        self.service.scheduler.factory = workers
        self.legacy_config()
        context = self.json('/api/context', self.body())
        result = self.json('/v1/chat/completions', self.body())
        self.assertEqual(result['x_context']['prompt_sha256'], context['prompt_sha256'])
        self.assertFalse(result['x_artifacts'])
        self.assertEqual(len(workers.calls), 1)
        self.assertFalse(workers.calls[0]['tools_enabled'])
        self.assertFalse(json.loads(workers.calls[0]['prepared']['text'])['tools'])
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])

    def test_client_tool_is_returned_to_client_and_never_executed_by_server(self):
        workers = FixtureWorkers([(qwen_call(), 90)])
        self.service.scheduler.factory = workers
        self.legacy_config()
        result = self.json('/v1/chat/completions', self.body(tools=[IMAGE_TOOL]))
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'], 'generate_image')
        self.assertFalse(result['x_artifacts'])
        self.assertEqual(len(workers.calls), 1)
        self.assertFalse(workers.calls[0]['tools_enabled'])
        self.assertEqual(json.loads(workers.calls[0]['prepared']['text'])['tools'], [IMAGE_TOOL])

    def test_explicit_opt_in_uses_registered_image_model_without_global_default(self):
        workers = FixtureWorkers([(qwen_call(), 90), ('Image saved.', 8)])
        self.service.scheduler.factory = workers
        self.legacy_config()
        result = self.json('/v1/chat/completions', self.body(auto_image=True, steps=1, width=256, height=256))
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image', 'text'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})

    def test_retired_server_settings_are_rejected_without_mutating_configuration(self):
        before = copy.deepcopy(self.service.config)
        for field, value in [('default_auto_image', True), ('default_image_model', 'qwen-image')]:
            with self.subTest(field=field), self.assertRaises(urllib.error.HTTPError) as error:
                self.json('/api/server', {field: value, 'terminal_metrics': False})
            self.assertEqual(error.exception.code, 400)
            self.assertIn('Global image routing settings were removed', json.load(error.exception)['error']['message'])
            self.assertEqual(self.service.config, before)
        state = self.json('/api/state')['server']
        self.assertNotIn('default_auto_image', state)
        self.assertNotIn('default_image_model', state)

    def test_status_polling_and_unaware_chat_during_image_swap(self):
        workers = FixtureWorkers([('LLM is ready.', 5), ('The waiting chat can now run.', 8)])
        entered, release = threading.Event(), threading.Event()

        def factory(*args):
            worker = workers(*args)
            run = worker.run

            def gated(request, *args):
                if request['action'] == 'image':
                    # Denoising is finished, but the PNG has not been saved yet.
                    yield {'kind': 'progress', 'step': 1, 'steps': 1}
                    entered.set()
                    if not release.wait(8):
                        raise AssertionError('Image test barrier was not released')
                yield from run(request, *args)

            worker.run = gated
            return worker

        self.service.scheduler.factory = factory
        self.json('/v1/chat/completions', self.body())
        receipt = self.json('/api/image/jobs', {'model': 'qwen-image', 'prompt': 'A bicycle',
                                              'size': '256x256', 'steps': 1})
        try:
            self.assertTrue(entered.wait(3))
            with ThreadPoolExecutor(max_workers=1) as pool:
                waiting_chat = pool.submit(self.json, '/v1/chat/completions', self.body())
                try:
                    deadline = time.monotonic()+3
                    while time.monotonic() < deadline:
                        with self.service.scheduler.condition:
                            queued = bool(self.service.scheduler.waiting)
                        if queued:
                            break
                        time.sleep(.01)
                    self.assertTrue(queued, 'The chat request must wait for the occupied device')
                    status = self.json(receipt['status_url'])
                    self.assertEqual(status['progress']['step'], 1)
                    self.assertNotEqual(status['status'], 'completed')
                    self.assertFalse(status['artifacts'])
                    self.assertFalse(waiting_chat.done())
                    self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
                    self.assertNotIn(('close', 'qwen-image'), workers.lifecycle)
                finally:
                    release.set()
                reply = waiting_chat.result(timeout=8)
                self.assertEqual(reply['choices'][0]['message']['content'], 'The waiting chat can now run.')
        finally:
            release.set()
        image = wait_job(self.service, receipt['id'])
        self.assertEqual(image['status'], 'completed')
        result = self.json(receipt['result_url'])
        self.assertTrue(result['x_result_ready'])
        self.assertTrue(result['x_handoff_complete'])
        with self.request(image['artifacts'][0]['url']) as response:
            self.assertEqual(response.read(8), b'\x89PNG\r\n\x1a\n')
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image', 'text'])
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
                         ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])


class SettingsMigrationTests(unittest.TestCase):
    def test_old_settings_drop_retired_keys_and_preserve_models_and_residency(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(DEFAULT)
            config['data_dir'] = directory
            saved = copy.deepcopy(config)
            saved.update(default_auto_image=True, default_image_model='qwen-image', terminal_metrics=False)
            saved['devices'][0].update(policy='budget', budget_gib=40)
            saved['models'][0]['generation'] = {'temperature': 0.7}
            expected = restore_settings(saved)
            for legacy_value in (True, 'obsolete-format'):
                saved['default_auto_image'] = legacy_value
                Path(directory, 'settings.json').write_text(json.dumps(saved), encoding='utf-8')
                restored = restore_settings(config)
                self.assertNotIn('default_auto_image', restored)
                self.assertNotIn('default_image_model', restored)
                self.assertEqual(restored['models'], expected['models'])
                self.assertEqual(restored['devices'], expected['devices'])
                self.assertFalse(restored['terminal_metrics'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
