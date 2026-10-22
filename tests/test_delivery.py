"""Response delivery and recovery: real HTTP, controlled model workers."""
import http.client
import json
import socket
import struct
import sys
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from lazyrig.context import IMAGE_TOOL
from lazyrig.server import handler
from support import Service
from lazyrig.worker import Cancelled


BODY = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Generate a bicycle image'}],
        'auto_image': True, 'image_model': 'qwen-image', 'max_tokens': 512}


class GatedWorkers(FixtureWorkers):
    def __init__(self):
        super().__init__([(qwen_call(), 80), ('The bicycle image is ready.', 12)])
        self.waiting = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args):
        worker = super().__call__(*args)
        run = worker.run
        def gated(payload, timeout, cancelled):
            if payload['action'] == 'text' and not payload['tools_enabled']:
                self.waiting.set()
                while not self.release.wait(.01):
                    if cancelled.is_set():
                        raise Cancelled()
            yield from run(payload, timeout, cancelled)
        worker.run = gated
        return worker


class DeliveryTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_async_chat_shows_image_before_continuation_and_recovers_saved_result(self):
        workers = GatedWorkers()
        self.service.scheduler.factory = workers
        receipt = self.json('/api/chat/jobs', BODY)
        try:
            self.assertEqual(receipt['object'], 'inference.job')
            self.assertTrue(workers.waiting.wait(3))
            with self.request(receipt['result_url']) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(response.headers['X-Job-Id'], receipt['id'])
                self.assertEqual(response.headers['Location'], receipt['result_url'])
                pending = json.load(response)
            self.assertEqual(len(pending['x_artifacts']), 1)
            self.assertEqual(pending['phase'], 'Preparing final response')
            self.assertFalse(workers.release.is_set())
        finally:
            workers.release.set()
        job = wait_job(self.service, receipt['id'])
        result = self.json(receipt['result_url'])
        self.assertEqual(result['choices'][0]['message']['content'], 'The bicycle image is ready.')
        self.assertEqual(result, self.json(receipt['result_url']))
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertEqual(len([c for c in workers.calls if c['action'] == 'image']), 1)
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        # Persisted results from the previous server are available after restart.
        config = self.service.config
        self.service.close()
        self.service = Service(config)
        self.server.RequestHandlerClass = handler(self.service)
        self.assertEqual(result, self.json(receipt['result_url']))
        self.assertFalse(self.service.scheduler.resident)

    def test_windows_abort_after_completed_chat_does_not_send_500_or_destroy_result(self):
        workers = FixtureWorkers([(qwen_call(), 80), ('The bicycle image is ready.', 12)])
        self.service.scheduler.factory = workers
        writes, errors = [], []
        Base = handler(self.service)
        class Writer:
            def __init__(self, wrapped): self.wrapped = wrapped
            def __getattr__(self, name): return getattr(self.wrapped, name)
            def write(self, data):
                if b'"object": "chat.completion"' in data:
                    raise ConnectionAbortedError(10053, 'Injected Windows aborted connection')
                return self.wrapped.write(data)
        class AbortedHandler(Base):
            def setup(self):
                super().setup()
                self.wfile = Writer(self.wfile)
            def send_response(self, code, message=None):
                writes.append((self.command, self.path, code))
                return super().send_response(code, message)
        self.server.RequestHandlerClass = AbortedHandler
        self.server.handle_error = lambda *args: errors.append('unhandled exception')
        with self.request('/v1/chat/completions', BODY) as response:
            jid = response.headers['X-Job-Id']
            with self.assertRaises(http.client.IncompleteRead):
                response.read()
        job = wait_job(self.service, jid)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['text'], 'The bicycle image is ready.')
        self.assertEqual(writes, [('POST', '/v1/chat/completions', 200)])
        self.assertFalse(errors)
        self.server.RequestHandlerClass = Base
        result = self.json('/api/jobs/'+jid+'/result')
        self.assertEqual(result['choices'][0]['message']['content'], job['text'])
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertTrue(any(e['kind'] == 'client_disconnected' for e in self.service.store.events(jid)))

    def test_disconnected_background_events_continue_restoration_and_result_recovery(self):
        workers = GatedWorkers()
        self.service.scheduler.factory = workers
        errors = []
        self.server.handle_error = lambda *args: errors.append('unhandled exception')
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        receipt = self.json('/api/chat/jobs', BODY)
        jid = receipt['id']
        conn.request('GET', receipt['events_url'])
        sock = conn.sock
        response = conn.getresponse()
        try:
            self.assertTrue(workers.waiting.wait(3))
            # Consume progress until the image is available, before final text.
            while True:
                line = response.readline().decode()
                if line.startswith('data: {') and json.loads(line[6:]).get('kind') == 'artifact':
                    break
            self.assertFalse(workers.release.is_set())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('hh' if sys.platform == 'win32' else 'ii', 1, 0))
            response.close(); conn.close()
        finally:
            workers.release.set()
        job = wait_job(self.service, jid)
        self.assertEqual(job['status'], 'completed', job['error'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.json('/api/jobs/'+jid+'/result')['choices'][0]['message']['content'], 'The bicycle image is ready.')
        self.assertFalse(errors)

    def test_async_preflight_runs_in_job_and_failures_keep_a_retrievable_record(self):
        started = threading.Event()
        proceed = threading.Event()
        prepare = self.service.tokens.prepare
        def blocked(*args, **kwargs):
            started.set()
            proceed.wait(5)
            return prepare(*args, **kwargs)
        self.service.config['models'][0]['context_limit'] = 512
        with patch.object(self.service.tokens, 'prepare', side_effect=blocked):
            try:
                receipt = self.json('/api/chat/jobs', {**BODY, 'auto_image': False, 'max_tokens': 500,
                                                      'messages': [{'role': 'user', 'content': 'x'*2000}]})
                self.assertTrue(started.wait(2))
                self.assertFalse(proceed.is_set())
            finally:
                proceed.set()
            job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'failed')
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request(receipt['result_url'])
        self.assertIn('Context overflow', json.load(err.exception)['error']['message'])

    def test_prefer_async_image_and_stable_client_tool_ids(self):
        self.service.scheduler.factory = FixtureWorkers([(qwen_call(), 80)])
        with self.request('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'a bicycle', 'steps': 1}, {'Prefer': 'respond-async'}) as response:
            self.assertEqual(response.status, 202)
            receipt = json.load(response)
        wait_job(self.service, receipt['id'])
        self.assertEqual(len(self.json(receipt['result_url'])['x_artifacts']), 1)
        result = self.json('/v1/chat/completions', {**BODY, 'auto_image': False, 'tools': [IMAGE_TOOL]})
        self.assertEqual(result, self.json('/api/jobs/'+result['x_job_id']+'/result'))


class BrowserAccessTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_any_frontend_mode_requires_key_and_keeps_admin_origin_restrictions(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/server', {'browser_access': 'api-key'})
        key = 'browser-access-regression-key'
        self.json('/api/server', {'browser_access': 'api-key', 'api_key': key})
        for origin in ('null', 'https://different.example', 'http://localhost:5371'):
            with self.subTest(origin=origin):
                with self.assertRaises(urllib.error.HTTPError) as err:
                    self.request('/v1/models', headers={'Origin': origin})
                self.assertEqual(err.exception.code, 401)
                self.assertEqual(err.exception.headers['Access-Control-Allow-Origin'], origin)
                with self.request('/v1/models', headers={'Origin': origin, 'Authorization': 'Bearer '+key}) as response:
                    self.assertEqual(response.status, 200)
                with self.assertRaises(urllib.error.HTTPError) as err:
                    self.request('/api/state', headers={'Origin': origin, 'Authorization': 'Bearer '+key})
                self.assertEqual(err.exception.code, 400)
                self.assertIsNone(err.exception.headers.get('Access-Control-Allow-Origin'))

    def test_host_rejection_includes_actual_host_in_diagnostics(self):
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/v1/models', headers={'Host': 'my-pc.local:8765'})
        self.assertEqual(err.exception.code, 400)
        error = self.json('/api/state')['server']['connection_errors'][-1]
        self.assertIn('my-pc.local:8765', error['reason'])
        self.assertIn('0.0.0.0', error['reason'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
