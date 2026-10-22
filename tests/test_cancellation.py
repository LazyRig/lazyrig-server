"""Client stops must release workers without replaying accepted image requests."""
import copy
import http.client
import json
import threading
import unittest
import urllib.error
from pathlib import Path

import test_connections
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from test_image_edits import export_fixture, multipart
from test_vision_mtp import image_url
from lazyrig.server import handler
from lazyrig.worker import Cancelled
from support import Service


IMAGE = {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 40}


class GatedImages(FixtureWorkers):
    def __init__(self, responses=()):
        super().__init__(responses)
        self.started = threading.Event()
        self.release = threading.Event()
        self.started_actions = []

    def __call__(self, *args):
        worker = super().__call__(*args)
        run = worker.run
        def gated(payload, timeout, cancelled):
            if payload['action'] in {'image', 'image_edit'}:
                self.started_actions.append(payload['action'])
                self.started.set()
                while not self.release.wait(.01):
                    if cancelled.is_set():
                        raise Cancelled()
                if cancelled.is_set():
                    raise Cancelled()
            yield from run(payload, timeout, cancelled)
        worker.run = gated
        return worker


class CancellationTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def image(self, key='a'*32, **kw):
        with self.request('/api/image/jobs', IMAGE, {'Idempotency-Key': key, **kw}) as response:
            return json.load(response)

    def assert_cancelled_result(self, jid):
        for path in ('/api/jobs/'+jid+'/result', '/artifacts/'+jid+'.png'):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, 410)
            self.assertEqual(error.exception.headers['X-Should-Retry'], 'false')
            data = json.load(error.exception)
            self.assertEqual(data['error']['type'], 'request_cancelled')
            self.assertFalse(data['x_retryable'])
            self.assertEqual(data['x_status'], 'cancelled')

    def cancelled_replay(self, jid):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.image(jid)
        self.assertEqual(error.exception.code, 410)
        self.assertEqual(error.exception.headers['X-Should-Retry'], 'false')
        result = json.load(error.exception)
        self.assertEqual(result['x_job_id'], jid)
        self.assertEqual(result['x_status'], 'cancelled')

    def test_explicit_stop_and_retries_remain_cancelled_across_restart(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        receipt = self.image()
        self.assertTrue(workers.started.wait(2))
        reply = self.json(receipt['cancel_url'], {})
        self.assertTrue(reply['cancel_requested'])
        self.json(receipt['cancel_url'], {})
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(self.service.scheduler.resident)
        self.assertFalse(self.service.scheduler.busy)
        self.assertEqual(len([e for e in self.service.store.events(job['id']) if e['kind']=='cancelling']), 1)
        self.assert_cancelled_result(job['id'])
        self.cancelled_replay(job['id'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/v1/images/generations', IMAGE, {'Idempotency-Key': job['id']})
        self.assertEqual(error.exception.code, 410)
        self.assertEqual(workers.started_actions, ['image'])
        config = copy.deepcopy(self.service.config)
        self.service.close()
        self.service = Service(config)
        self.server.RequestHandlerClass = handler(self.service)
        self.cancelled_replay(job['id'])
        self.assertFalse(self.service.running)
        self.assertFalse(self.service.scheduler.resident)

    def test_cancel_queued_job_does_not_start_it_or_cancel_its_predecessor(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        first = self.image()
        self.assertTrue(workers.started.wait(2))
        second = self.image('b'*32)
        self.json(second['cancel_url'], {})
        self.assertEqual(wait_job(self.service, second['id'])['status'], 'cancelled')
        self.cancelled_replay(second['id'])
        self.assertEqual(workers.started_actions, ['image'])
        self.assertEqual(self.service.store.job(first['id'])['status'], 'running')
        workers.release.set()
        self.assertEqual(wait_job(self.service, first['id'])['status'], 'completed')
        self.assertFalse(self.service.scheduler.waiting)

    def test_stop_during_model_load_never_starts_inference(self):
        entered = threading.Event()
        def loading(mode, model, timeout, cancelled):
            entered.set()
            if not cancelled.wait(3):
                raise RuntimeError('Cancellation never reached model loading')
            raise Cancelled()
        self.service.scheduler.factory = loading
        receipt = self.image()
        self.assertTrue(entered.wait(2))
        self.json(receipt['cancel_url'], {})
        self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'cancelled')
        self.assertFalse(self.service.scheduler.resident)
        self.assertFalse(any(e['kind']=='image_started' for e in self.service.store.events(receipt['id'])))

    def test_stop_during_final_unload_wins_over_completion_and_keeps_saved_png(self):
        # This scenario requires an unload; Automatic now keeps a healthy image.
        self.service.config["models"][1]["keep_image_model_loaded"] = False
        workers = FixtureWorkers([])
        self.service.scheduler.factory = workers
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        entered, release = threading.Event(), threading.Event()
        def factory(*args):
            worker = workers(*args)
            close = worker.close
            if args[1]['task'] == 'text-to-image':
                def closing():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Unload not released by test')
                    close()
                worker.close = closing
            return worker
        self.service.scheduler.factory = factory
        receipt = self.image()
        try:
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.service.store.job(receipt['id'])['result_ready'])
            self.assertTrue(self.json(receipt['cancel_url'], {})['cancel_requested'])
        finally:
            release.set()
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(workers.lifecycle.count(('load', 'qwen-chat')), 1)
        result = self.json(receipt['result_url'])
        self.assertTrue(result['x_result_ready'])
        self.assertFalse(result['x_handoff_complete'])
        self.assertEqual(result['x_status'], 'cancelled')
        with self.request('/artifacts/'+job['id']+'.png') as response:
            self.assertTrue(response.read().startswith(b'\x89PNG'))

    def test_restart_preserves_stop_intent_during_cleanup(self):
        from lazyrig.storage import Store
        job = self.service._new_job(None, 'qwen-image', 'Interrupted stop', {})
        job.update(kind='image', status='running', cancel_requested=True)
        self.service.store.save_job(job)
        # Release the previous owner before modeling recovery. Two live stores
        # must never be allowed to perform recovery on the same data directory.
        self.service.close()
        reopened = Store(self.temp.name)
        try:
            self.assertEqual(reopened.job(job['id'])['status'], 'cancelled')
        finally:
            reopened.close()

    def test_completed_job_cannot_be_cancelled_retroactively(self):
        self.service.scheduler.factory = FixtureWorkers([])
        receipt = self.image()
        wait_job(self.service, receipt['id'])
        reply = self.json(receipt['cancel_url'], {})
        self.assertEqual(reply['status'], 'completed')
        self.assertFalse(reply['cancel_requested'])
        self.assertTrue(self.json(receipt['result_url'])['x_handoff_complete'])

    def test_cancel_endpoint_requires_the_same_bearer_key(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        self.service.config['api_key'] = 'cancellation-test-api-key'
        headers = {'Authorization': 'Bearer cancellation-test-api-key'}
        receipt = self.image(**headers)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(receipt['cancel_url'], {})
        self.assertEqual(error.exception.code, 401)
        self.assertFalse(self.service.store.job(receipt['id']).get('cancel_requested'))
        with self.request(receipt['cancel_url'], {}, headers) as response:
            self.assertTrue(json.load(response)['cancel_requested'])
        self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'cancelled')

    def test_sync_image_and_edit_abort_before_response_headers_cancel_workers(self):
        import base64
        self.service.config['models'][1]['path'] = str(export_fixture(Path(self.temp.name)/'edit'))
        uploads = multipart([('model', 'qwen-image'), ('prompt', 'A lake')],
                            [('image', base64.b64decode(image_url().split(',', 1)[1]))])
        for index, (path, content_type, payload) in enumerate([
            ('/v1/images/generations', 'application/json', json.dumps(IMAGE).encode()),
            ('/v1/images/edits', 'application/json', json.dumps({**IMAGE, 'image': image_url()}).encode()),
            ('/v1/images/edits', uploads[0], uploads[1]),
        ]):
            with self.subTest(path=path, content_type=content_type):
                workers = GatedImages()
                self.service.scheduler.factory = workers
                jid = str(index+1)*32
                conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
                try:
                    conn.request('POST', path, payload, {'Content-Type': content_type, 'Idempotency-Key': jid})
                    self.assertTrue(workers.started.wait(2))
                finally:
                    conn.close()
                job = wait_job(self.service, jid, timeout=3)
                self.assertEqual(job['status'], 'cancelled')
                self.assertEqual(job['cancel_reason'], 'client_disconnect')
                self.assertFalse(job['artifacts'])
                self.assertFalse(self.service.scheduler.resident)
                self.assertEqual(len(workers.started_actions), 1)

    def test_sync_explicit_cancel_returns_410_on_original_connection(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        jid = 'c'*32
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request('POST', '/v1/images/generations', json.dumps(IMAGE),
                         {'Content-Type': 'application/json', 'Idempotency-Key': jid})
            self.assertTrue(workers.started.wait(2))
            self.json('/api/jobs/'+jid+'/cancel', {})
            response = conn.getresponse()
            self.assertEqual(response.status, 410)
            self.assertEqual(response.headers['X-Should-Retry'], 'false')
            self.assertEqual(json.load(response)['x_status'], 'cancelled')
        finally:
            conn.close()

    def test_sync_chat_abort_cancels_managed_image_without_llm_continuation(self):
        workers = GatedImages([(qwen_call(), 80)])
        self.service.scheduler.factory = workers
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Draw a lake'}],
                'auto_image': True, 'image_model': 'qwen-image', 'stream': True}
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        response = None
        try:
            conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            jid = response.headers['X-Job-Id']
            self.assertTrue(workers.started.wait(2))
        finally:
            if response is not None:
                response.close()
            conn.close()
        job = wait_job(self.service, jid, timeout=3)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(workers.lifecycle.count(('load', 'qwen-chat')), 1)
        self.assertFalse(self.service.scheduler.resident)
        self.assertEqual(workers.started_actions, ['image'])

    def test_retrieval_connection_does_not_own_a_background_job(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        receipt = self.image()
        self.assertTrue(workers.started.wait(2))
        self.assertEqual(receipt['disconnect_policy'], 'continue')
        # Even a synchronous keyed replay is just an observer of a job that
        # was originally accepted for background execution.
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        conn.request('POST', '/v1/images/generations', json.dumps(IMAGE),
                     {'Content-Type': 'application/json', 'Idempotency-Key': receipt['id']})
        conn.close()
        workers.release.set()
        self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'completed')
        self.assertEqual(workers.started_actions, ['image'])


@unittest.skipUnless(test_connections.shutil.which('openssl'), 'OpenSSL test certificate required')
class TLSCancellationTests(unittest.TestCase):
    setUp = test_connections.TLSConnectionTests.setUp
    tearDown = test_connections.TLSConnectionTests.tearDown

    def test_https_abort_before_headers_cancels_image(self):
        workers = GatedImages()
        self.service.scheduler.factory = workers
        conn = http.client.HTTPSConnection('127.0.0.1', self.server.server_port, context=self.client, timeout=5)
        jid = 'd'*32
        try:
            conn.request('POST', '/v1/images/generations', json.dumps(IMAGE),
                         {'Content-Type': 'application/json', 'Idempotency-Key': jid})
            self.assertTrue(workers.started.wait(2))
        finally:
            conn.close()
        job = wait_job(self.service, jid, timeout=3)
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(self.service.scheduler.resident)


if __name__ == '__main__':
    unittest.main()
