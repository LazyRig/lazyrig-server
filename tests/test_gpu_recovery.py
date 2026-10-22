"""Bounded native-GPU recovery contracts; no physical GPU is exercised."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

import test_http
from support import fixture_png
from test_core import wait_job
from test_chat_image_tools import Workers, rewritten, source_image
from test_handoff import qwen_call
from test_image_edits import export_fixture
from lazyrig.worker import Cancelled, NativeWorkerError, ProcessWorker
from lazyrig.deadlines import DeadlineEvent

ERROR = '[GPU] CL_OUT_OF_RESOURCES exception'


class FaultWorkers(Workers):
    def __init__(self, texts=(), failures=1, error=None):
        super().__init__(texts)
        self.failures = failures
        self.error = error or NativeWorkerError(ERROR)
        self.attempts, self.signals, self.timeouts = [], [], []
        self.before_failure = None
        self.after_close = None
        self.reload_gate = None
        self.reloading = threading.Event()
        self.image_loads = 0
        self.close_error = False

    def __call__(self, *args):
        model, signal = args[1], args[3]
        if model['pipeline'] == 'image':
            self.image_loads += 1
            if self.image_loads == 2 and self.reload_gate:
                self.reloading.set()
                while not self.reload_gate.wait(.01):
                    if signal.is_set():
                        raise Cancelled()
        worker = super().__call__(*args)
        original, close = worker.run, worker.close

        def run(payload, timeout, cancel):
            if payload['action'] in {'image', 'image_edit'}:
                self.attempts.append(copy.deepcopy(payload))
                self.signals.append(cancel)
                self.timeouts.append(timeout)
                if len(self.attempts) <= self.failures:
                    self.calls.append(copy.deepcopy(payload))
                    yield {'kind': 'progress', 'step': 2, 'steps': payload['steps']}
                    if self.before_failure:
                        self.before_failure(payload, cancel)
                    raise self.error
            yield from original(payload, timeout, cancel)

        def cleanup():
            if model['pipeline'] == 'image':
                if self.close_error:
                    raise RuntimeError('Cleanup unconfirmed')
                close()
                if self.after_close:
                    self.after_close(signal)
            else:
                close()
        worker.run, worker.close = run, cleanup
        return worker


class GPURecoveryTests(unittest.TestCase):
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def setUp(self):
        test_http.HTTPTests.setUp(self)
        self.service.config['devices'][0]['id'] = 'GPU.1'
        for model in self.service.config['models']:
            model['device'] = 'GPU.1'
        self.service.config['models'][1].update(
            path=str(export_fixture(Path(self.temp.name) / 'image-export')),
            runtime_profile='qwen-image-21', generation={'width': 512, 'height': 512, 'steps': 4})
        self.body = {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 4, 'seed': -1}
        self.workers = FaultWorkers()
        self.service.scheduler.factory = self.workers

    def tearDown(self):
        self.workers.close_error = False
        self.workers.after_close = None
        if self.workers.reload_gate:
            self.workers.reload_gate.set()
        test_http.HTTPTests.tearDown(self)

    def post(self, path='/v1/images/generations', body=None, key='gpu-recovery', async_=False):
        with self.request(path, self.body if body is None else body,
                          {'Idempotency-Key': key, **({'Prefer': 'respond-async'} if async_ else {})}) as response:
            return json.load(response)

    def failed(self, **kwargs):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(**kwargs)
        error = caught.exception
        return error, json.loads(error.read())

    def test_generation_recovers_once_preserves_seed_and_retains_healthy_worker(self):
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            result = self.post()
            job = wait_job(self.service, result['x_job_id'])
        self.assertEqual(job['status'], 'completed', job['error'])
        self.assertEqual(len(self.service.store.jobs()), 1)
        self.assertEqual(len(job['artifacts']), 1)
        self.assertEqual(self.workers.attempts[0], self.workers.attempts[1])
        self.assertGreaterEqual(self.workers.attempts[0]['seed'], 0)
        self.assertIs(self.workers.signals[0], self.workers.signals[1])
        self.assertLessEqual(self.workers.timeouts[1], self.workers.timeouts[0])
        self.assertEqual(result['x_image_recovery']['status'], 'succeeded')
        self.assertEqual(result['x_image_recovery']['retry_count'], 1)
        self.assertEqual(result['x_image_recovery']['first_error'], ERROR)
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-image')])
        self.assertTrue(any('automatic retry 1/1' in line for line in logs.output))
        self.assertEqual(self.post()['x_job_id'], job['id'])
        self.assertEqual(len(self.workers.attempts), 2)
        followup = self.post(key='healthy-followup')
        wait_job(self.service, followup['x_job_id'])
        self.assertEqual(self.workers.image_loads, 2)
        self.assertIsNone(followup['x_image_recovery'])
        with self.request('/api/jobs/' + job['id'] + '/events') as response:
            events = response.read().decode()
        self.assertEqual(events.count('"kind": "image_retry"'), 1)
        self.assertEqual(events.count('"kind": "artifact"'), 1)

    def test_edit_reuses_exact_captured_source_and_options(self):
        result = self.post('/v1/images/edits', {**self.body, 'image': source_image(), 'seed': 12345,
                                             'size': '512x256', 'guidance_scale': 4.0})
        self.assertEqual(result['x_operation'], 'edit')
        self.assertEqual(self.workers.attempts[0], self.workers.attempts[1])
        self.assertEqual(self.workers.attempts[1]['seed'], 12345)
        self.assertEqual(result['x_artifacts'][0]['source_image_sha256'], self.workers.attempts[1]['input_images'][0]['sha256'])
        self.assertEqual([m['_image_operation'] for m in self.workers.loaded_models], ['edit', 'edit'])

    def test_direct_enhancement_is_not_repeated(self):
        def enhance(service, job, model, enhancer, prompt, params, *args):
            return 'Enhanced lake', {**params, 'width': 768}
        with patch('lazyrig.service.enhance_image_prompt', side_effect=enhance) as enhancing:
            result = self.post(body={**self.body, 'enhance_prompt': True, 'enhancement_model': 'qwen-chat'})
        enhancing.assert_called_once()
        self.assertEqual(self.workers.attempts[0], self.workers.attempts[1])
        self.assertEqual(result['x_artifacts'][0]['prompt'], 'Enhanced lake')
        self.assertEqual(result['x_artifacts'][0]['width'], 768)

    def test_managed_tool_retries_image_only_and_continues_once(self):
        self.workers.responses = [(t, 100) for t in (qwen_call(), rewritten(), 'The image is saved.')]
        result = self.post('/v1/chat/completions', {'model': 'qwen-chat', 'auto_image': True, 'image_model': 'qwen-image',
            'enhance_prompt': True, 'messages': [{'role': 'user', 'content': 'Draw a bicycle.'}]})
        self.assertEqual([c['action'] for c in self.workers.calls], ['text', 'text', 'image', 'image', 'text'])
        self.assertEqual(self.workers.attempts[0], self.workers.attempts[1])
        self.assertEqual(result['x_image_recovery']['status'], 'succeeded')
        job = wait_job(self.service, result['x_job_id'])
        history = self.service.store.messages(job['session_id'])
        self.assertEqual(sum('tool_calls' in m for m in history), 1)
        self.assertEqual(sum(m['role'] == 'tool' for m in history), 1)
        self.assertEqual(len(job['artifacts']), 1)

    def test_exhausted_retry_fails_once_and_preserves_original_diagnostic(self):
        self.workers.failures = 10
        response, result = self.failed()
        self.assertEqual(response.code, 502)
        self.assertEqual(response.headers['X-Should-Retry'], 'false')
        self.assertEqual(result['error']['code'], 'gpu_resource_error')
        self.assertFalse(result['x_retryable'])
        self.assertEqual(result['x_image_recovery']['status'], 'failed')
        self.assertEqual(result['x_image_recovery']['first_error'], ERROR)
        self.assertEqual(len(self.workers.attempts), 2)
        self.assertFalse(self.service.scheduler.resident)
        self.failed()
        self.assertEqual(len(self.workers.attempts), 2)

    def test_native_allocation_error_is_recoverable(self):
        self.workers.error = NativeWorkerError('[GPU] CL_MEM_OBJECT_ALLOCATION_FAILURE')
        self.assertEqual(self.post()['x_image_recovery']['status'], 'succeeded')

    def test_unrelated_native_error_is_not_retried(self):
        self.workers.error = NativeWorkerError('Invalid input shape')
        _, result = self.failed()
        self.assertIsNone(result['x_image_recovery'])
        self.assertEqual(len(self.workers.attempts), 1)

    def test_matching_text_in_non_native_error_is_not_retried(self):
        self.workers.error = RuntimeError(ERROR)
        self.failed()
        self.assertEqual(len(self.workers.attempts), 1)

    def test_cpu_error_is_not_retried(self):
        self.service.config['models'][1]['device'] = 'TEST.CPU'
        self.failed()
        self.assertEqual(len(self.workers.attempts), 1)

    def test_timeout_is_not_retried(self):
        self.workers.error = TimeoutError('Operation deadline exceeded')
        self.failed()
        self.assertEqual(len(self.workers.attempts), 1)

    def test_unconfirmed_cleanup_prevents_replacement(self):
        self.workers.close_error = True
        _, result = self.failed()
        self.assertIn('Cleanup unconfirmed', result['error']['message'])
        self.assertEqual(self.workers.image_loads, 1)
        self.assertEqual(self.service.scheduler.states['qwen-image'], 'cleanup-failed')
        self.assertEqual(result['x_image_recovery']['status'], 'failed')

    def test_stop_during_cleanup_prevents_retry(self):
        self.workers.after_close = lambda signal: signal.set()
        response, result = self.failed()
        self.assertEqual(response.code, 410)
        self.assertEqual(result['x_image_recovery']['status'], 'cancelled')
        self.assertEqual(self.workers.image_loads, 1)

    def test_original_deadline_is_not_extended_during_recovery(self):
        self.workers.after_close = lambda signal: setattr(signal, 'deadline', time.monotonic() - 1)
        response, result = self.failed()
        self.assertEqual(response.code, 504)
        self.assertEqual(result['error']['code'], 'request_timeout')
        self.assertEqual(self.workers.image_loads, 1)

    def test_saved_output_is_never_generated_again(self):
        self.workers.before_failure = lambda payload, _: fixture_png(payload['output_path'])
        _, result = self.failed()
        self.assertIsNone(result['x_image_recovery'])
        self.assertEqual(len(self.workers.attempts), 1)
        self.assertTrue(Path(self.workers.attempts[0]['output_path']).is_file())

    def test_reporting_failure_is_not_mistaken_for_native_failure(self):
        event = self.service.store.event
        def fail(jid, kind, *args, **kwargs):
            if kind == 'progress':
                raise RuntimeError('Storage write failed: ' + ERROR)
            return event(jid, kind, *args, **kwargs)
        with patch.object(self.service.store, 'event', side_effect=fail):
            self.failed()
        self.assertEqual(len(self.workers.attempts), 1)

    def test_reconnects_share_recovery_and_progress_is_reset(self):
        self.workers.reload_gate = threading.Event()
        receipt = self.post(async_=True)
        self.assertTrue(self.workers.reloading.wait(3))
        job = self.json(receipt['status_url'])
        self.assertIsNone(job['progress'])
        self.assertEqual(job['image_recovery']['status'], 'retrying')
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(lambda _: self.post(async_=True), range(4)))
        self.assertEqual({r['id'] for r in receipts}, {receipt['id']})
        self.assertEqual(len(self.workers.attempts), 1)
        waiting = self.service.control('qwen-chat', 'load')
        self.assertFalse(any(m['id'] == 'qwen-chat' for m in self.workers.loaded_models))
        self.workers.reload_gate.set()
        self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'completed')
        self.assertEqual(wait_job(self.service, waiting['id'])['status'], 'completed')
        self.assertEqual(len(self.workers.attempts), 2)
        self.assertEqual(len(self.service.store.image_jobs()), 1)


class NativeErrorBoundaryTests(unittest.TestCase):
    def test_child_error_is_typed_and_worker_is_closed_before_propagation(self):
        class Connection:
            def send(self, payload): pass
            def poll(self, timeout): return True
            def recv(self): return {'kind': 'error', 'error': ERROR}
        worker = ProcessWorker.__new__(ProcessWorker)
        worker.conn = Connection()
        closed = []
        worker.close = lambda: closed.append(True)
        with self.assertRaises(NativeWorkerError):
            list(worker.run({'action': 'image'}, 5, DeadlineEvent(5)))
        self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
