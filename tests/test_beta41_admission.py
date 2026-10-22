"""Production admission/finalization regressions; deterministic, no native model."""
import copy
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from lazyrig.admission import AdmissionError
from lazyrig.compat import start_chat, start_image, wait_job, job_error_status, job_receipt
from lazyrig.deadlines import DeadlineEvent, RequestDeadline
from lazyrig.worker import Cancelled
from support import DEFAULT, Service

CHAT = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 32}
IMAGE = {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 1}


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=self.temp.name, terminal_metrics=False)
        self.service = Service(config, restore=False)
        self.releases = []

    def tearDown(self):
        for release in self.releases:
            release.set()
        self.service.close()
        self.temp.cleanup()

    def launch_gate(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        def operation(cancel, emit):
            entered.set()
            while not release.wait(.01):
                if cancel.is_set():
                    raise Cancelled()
        job = self.service._launch(self.service._new_job(None, 'qwen-chat', 'gate', {}), operation)
        self.assertTrue(entered.wait(1))
        return job, release

    def wait_terminal(self, jid):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = self.service.job(jid)
            if job['status'] in {'completed', 'cancelled', 'failed'}:
                return job
            time.sleep(.01)
        self.fail('Job never became terminal')

    def test_overload_rejected_before_capture_or_api_session(self):
        self.service.config['max_jobs'] = 1
        self.launch_gate()
        with patch.object(self.service.images, 'capture') as capture:
            for call in (lambda: start_chat(self.service, CHAT), lambda: start_image(self.service, IMAGE)):
                with self.assertRaises(AdmissionError) as raised:
                    call()
                self.assertEqual(raised.exception.status, 503)
                self.assertEqual(raised.exception.code, 'queue_full')
                self.assertTrue(raised.exception.retryable)
            capture.assert_not_called()
        self.assertEqual(self.service.store.sessions(), [])
        self.assertEqual(self.service._preparing, 0)
        self.assertEqual(self.service._reservations, 0)

    def test_slow_keyed_preparation_does_not_block_unrelated_cancel(self):
        active, _ = self.launch_gate()
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        self.releases.append(release)
        original = self.service.images.capture
        results, errors = [], []
        def capture(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        def prepare():
            try:
                results.append(start_chat(self.service, CHAT, idempotency_key='slow-key'))
            except Exception as exc:
                errors.append(exc)
        with patch.object(self.service.images, 'capture', side_effect=capture):
            thread = threading.Thread(target=prepare)
            thread.start()
            self.assertTrue(entered.wait(1))
            def stop():
                self.service.cancel(active['id'])
                cancelled.set()
            stopper = threading.Thread(target=stop)
            stopper.start()
            try:
                self.assertTrue(cancelled.wait(.5), 'Cancellation blocked behind preparation')
            finally:
                release.set()
                thread.join(3)
                stopper.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(self.wait_terminal(active['id'])['status'], 'cancelled')

    def test_matching_retries_prepare_and_launch_exactly_once(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        original = self.service.images.capture
        results, errors, captures = [], [], []
        def capture(*args):
            captures.append(True)
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        def submit():
            try:
                results.append(start_chat(self.service, CHAT, idempotency_key='same-key'))
            except Exception as exc:
                errors.append(exc)
        with patch.object(self.service.images, 'capture', side_effect=capture):
            first = threading.Thread(target=submit)
            first.start()
            self.assertTrue(entered.wait(1))
            second = threading.Thread(target=submit)
            second.start()
            time.sleep(.04)
            release.set()
            first.join(3)
            second.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], results[1]['id'])
        self.assertEqual(len(captures), 1)
        self.assertEqual(len(self.service.store.sessions()), 1)
        self.assertEqual(self.service._key_locks, {})

    def test_preparation_limit_rejects_before_second_capture(self):
        self.service.config['max_preparing'] = 1
        with self.service.admission():
            self.assertFalse(self.service.readiness()['ready'])
            errors = []
            def independent():
                try:
                    start_chat(self.service, CHAT)
                except Exception as exc:
                    errors.append(exc)
            with patch.object(self.service.images, 'capture') as capture:
                thread = threading.Thread(target=independent)
                thread.start()
                thread.join(1)
                capture.assert_not_called()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], AdmissionError)
            self.assertEqual(errors[0].code, 'preparation_full')
        self.assertEqual(self.service._reservations, 0)

    def test_invalid_preparation_releases_capacity(self):
        with self.assertRaises(ValueError):
            start_chat(self.service, {**CHAT, 'n': 2})
        self.assertEqual(self.service._reservations, 0)
        self.assertEqual(self.service._preparing, 0)
        self.assertEqual(self.service.store.sessions(), [])

    def test_preparation_deadline_expires_before_session_persistence(self):
        self.service.config['request_timeout_seconds'] = .02
        original = self.service.images.capture
        def slow(*args):
            time.sleep(.04)
            return original(*args)
        with patch.object(self.service.images, 'capture', side_effect=slow):
            with self.assertRaises(RequestDeadline):
                start_chat(self.service, CHAT)
        self.assertEqual(self.service.store.sessions(), [])
        self.assertEqual(self.service._reservations, 0)

    def test_queue_time_is_included_in_end_to_end_budget(self):
        self.service.config['request_timeout_seconds'] = .05
        with self.service.scheduler.lease(['TEST.GPU'], threading.Event()):
            job = start_chat(self.service, CHAT, preflight=False)
            terminal = self.wait_terminal(job['id'])
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['error_code'], 'request_timeout')
            self.assertEqual(job_error_status(terminal), 504)
            self.assertFalse(self.service.scheduler.resident)
        self.assertEqual(self.service.running, {})
        self.assertEqual(self.service.active_jobs, {})
        self.assertEqual(self.service.scheduler.waiting, [])

    def test_phases_share_one_budget(self):
        self.service.config['request_timeout_seconds'] = .055
        phases = []
        # Exercise the budget, not the host's ability to schedule a 35ms sleep.
        # Patch only the deadline module's clock; polling/cleanup keep real time.
        with patch('lazyrig.deadlines.time') as clock:
            clock.monotonic.return_value = 100.0
            def operation(cancel, emit):
                for name in ('first', 'second', 'third'):
                    cancel.check()
                    phases.append(name)
                    clock.monotonic.return_value += .035
                    cancel.check()
            job = self.service._launch(self.service._new_job(None, 'qwen-chat', 'phases', {}), operation)
            terminal = self.wait_terminal(job['id'])
        self.assertEqual(terminal['error_code'], 'request_timeout')
        self.assertEqual(phases, ['first', 'second'])

    def test_final_storage_failure_releases_slots_and_recovers_pending_terminal(self):
        job = self.service._new_job(None, 'qwen-chat', 'finish', {})
        self.service.store.save_job(job)
        signal = DeadlineEvent(timeout=60)
        self.service.running[job['id']] = (signal, None)
        self.service.active_jobs[job['id']] = job
        with patch.object(self.service.store, 'save_job', side_effect=OSError('simulated disk full')):
            with self.assertLogs('lazyrig.service', level='ERROR'):
                self.service._finish_job(job, signal, Mock())
            self.assertEqual(self.service.running, {})
            self.assertEqual(self.service.active_jobs, {})
            fallback = self.service.job(job['id'])
            self.assertEqual(fallback['status'], 'completed')
            self.assertFalse(fallback['durable'])
            self.assertFalse(job_receipt(fallback)['x_durable'])
            self.assertEqual(wait_job(self.service, job['id'])['status'], 'completed')
            self.assertIn('storage_unhealthy', self.service.readiness()['reasons'])
            with self.assertRaises(AdmissionError) as raised:
                start_chat(self.service, CHAT)
            self.assertEqual(raised.exception.code, 'storage_unhealthy')
        deadline = time.monotonic() + 3
        while self.service._pending_terminal and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(self.service._pending_terminal, {})
        self.assertTrue(self.service.store.job(job['id'])['durable'])
        self.assertNotIn('storage_unhealthy', self.service.readiness()['reasons'])

    def test_reporter_failure_does_not_prevent_durable_finalization(self):
        job = self.service._new_job(None, 'qwen-chat', 'finish', {})
        self.service.running[job['id']] = (DeadlineEvent(timeout=60), None)
        self.service.active_jobs[job['id']] = job
        reporter = Mock()
        reporter.finish.side_effect = OSError('closed terminal')
        with self.assertLogs('lazyrig.service', level='ERROR'):
            self.service._finish_job(job, self.service.running[job['id']][0], reporter)
        self.assertEqual(self.service.running, {})
        self.assertEqual(self.service.store.job(job['id'])['status'], 'completed')

    def test_drain_rejects_new_jobs_but_preserves_running_job(self):
        job, release = self.launch_gate()
        self.service.begin_drain()
        with self.assertRaises(AdmissionError) as raised:
            start_image(self.service, IMAGE)
        self.assertEqual(raised.exception.code, 'server_unavailable')
        self.assertIn('server_draining', self.service.readiness()['reasons'])
        release.set()
        self.assertEqual(self.wait_terminal(job['id'])['status'], 'completed')

    def test_readiness_checks_runtime_model_and_failed_cleanup(self):
        self.assertTrue(self.service.readiness('qwen-chat')['ready'])
        self.assertIn('model_unavailable', self.service.readiness('missing')['reasons'])
        self.service.detected['available'] = False
        self.service.scheduler.states['qwen-chat'] = 'cleanup-failed'
        reasons = self.service.readiness()['reasons']
        self.assertIn('runtime_unavailable', reasons)
        self.assertIn('worker_cleanup_failed', reasons)

    def test_disk_pressure_blocks_new_work_before_capture(self):
        health = {**self.service.store.health(), 'free_disk_bytes': 1024}
        with patch.object(self.service.store, 'health', return_value=health):
            with patch.object(self.service.images, 'capture') as capture:
                with self.assertRaises(AdmissionError) as raised:
                    start_chat(self.service, CHAT)
                capture.assert_not_called()
            self.assertEqual(raised.exception.code, 'disk_pressure')
            self.assertIn('disk_pressure', self.service.readiness()['reasons'])
        self.assertEqual(self.service.store.sessions(), [])

    def test_retention_disabled_by_default_and_explicitly_bounded(self):
        with patch.object(self.service.store, 'compact_events', return_value={'jobs': 1, 'text_events_removed': 12}) as compact:
            self.assertEqual(self.service._compact_old_events()['jobs'], 0)
            compact.assert_not_called()
            self.service.config['event_retention_days'] = 2
            before = time.time() - 2 * 86400
            self.assertEqual(self.service._compact_old_events()['jobs'], 1)
            self.assertAlmostEqual(compact.call_args.args[0], before, delta=1)
            self.assertEqual(compact.call_args.kwargs, {'limit': 20})

    def test_close_signals_preparation_and_key_waiters_do_not_touch_closed_store(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        original = self.service.images.capture
        errors = []
        def capture(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        def submit():
            try:
                start_chat(self.service, CHAT, idempotency_key='closing-key')
            except Exception as exc:
                errors.append(exc)
        with patch.object(self.service.images, 'capture', side_effect=capture):
            first, second = threading.Thread(target=submit), threading.Thread(target=submit)
            first.start()
            self.assertTrue(entered.wait(1))
            signal = next(iter(self.service._preparation_signals))
            second.start()
            deadline = time.monotonic() + 1
            while next(iter(self.service._key_locks.values()))[1] < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            closing = threading.Thread(target=self.service.close)
            closing.start()
            self.assertTrue(signal.wait(1))
            self.assertTrue(closing.is_alive())
            self.assertTrue(self.service.store.health()['ok'], 'Storage closed beneath active preparation')
            release.set()
            for thread in (first, second, closing):
                thread.join(3)
                self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(exc, (Cancelled, AdmissionError)) for exc in errors), errors)
        self.assertTrue(all(not isinstance(exc, AdmissionError) or exc.code == 'server_unavailable'
                            for exc in errors), errors)
        self.assertEqual(self.service._preparing, 0)
        self.assertEqual(self.service._key_locks, {})
        self.assertEqual(self.service._preparation_signals, set())

    def test_initial_event_failure_leaves_persisted_retry_terminal_without_worker(self):
        original = self.service.store.event
        def fail_queued(jid, kind, **data):
            if kind == 'queued':
                raise OSError('simulated queued event failure')
            return original(jid, kind, **data)
        with patch.object(self.service.store, 'event', side_effect=fail_queued), \
                patch.object(self.service, '_run') as run:
            with self.assertRaises(AdmissionError):
                start_chat(self.service, CHAT, idempotency_key='failed-admission')
            retry = start_chat(self.service, CHAT, idempotency_key='failed-admission')
            run.assert_not_called()
        self.assertEqual(retry['status'], 'failed')
        self.assertEqual(retry['error_code'], 'storage_failure')
        self.assertEqual(self.service.store.job(retry['id'])['status'], 'failed')
        self.assertEqual(len(self.service.store.jobs()), 1)
        self.assertEqual(self.service.running, {})
        self.assertEqual(self.service.active_jobs, {})
        self.assertEqual(self.service._reservations, 0)
        self.assertEqual(self.service._preparing, 0)

    def test_initial_event_failure_uses_terminal_fallback_if_final_write_also_fails(self):
        original = self.service.store.save_job
        def save_initial_only(job, retry_identity=None):
            if retry_identity is not None:
                return original(job, retry_identity=retry_identity)
            raise OSError('simulated terminal write failure')
        with patch.object(self.service.store, 'save_job', side_effect=save_initial_only), \
                patch.object(self.service.store, 'event', side_effect=OSError('simulated event failure')), \
                patch.object(self.service, '_run') as run:
            with self.assertLogs('lazyrig.service', level='ERROR'), self.assertRaises(AdmissionError):
                start_chat(self.service, CHAT, idempotency_key='failed-admission-fallback')
            retry = start_chat(self.service, CHAT, idempotency_key='failed-admission-fallback')
            self.assertEqual(retry['status'], 'failed')
            self.assertFalse(retry['durable'])
            self.assertEqual(self.service.store.job(retry['id'])['status'], 'queued')
            run.assert_not_called()
        self.assertEqual(self.service.running, {})
        self.assertEqual(self.service.active_jobs, {})
        self.assertEqual(self.service._reservations, 0)
        deadline = time.monotonic() + 3
        while self.service._pending_terminal and time.monotonic() < deadline:
            time.sleep(.02)
        retry = start_chat(self.service, CHAT, idempotency_key='failed-admission-fallback')
        self.assertEqual(retry['status'], 'failed')
        self.assertEqual(retry['error_code'], 'storage_failure')
        self.assertTrue(retry['durable'])
        self.assertEqual(len(self.service.store.jobs()), 1)

    def test_unknown_visual_tokens_do_not_hide_known_streamed_text_overflow(self):
        # Exercise a stale/third-party preflight with unknown fit: streaming must
        # preserve what the subsequently computed text lower bound already proves.
        model = self.service.model('qwen-chat')
        prepared = self.service.tokens.prepare(model, CHAT['messages'], 32, False, add_system=False)
        prepared['context'].update(input_tokens=model['context_limit'], fits=None)
        prepared['images'] = [{'sha256': 'fixture-source'}]
        sid = self.service.store.session('VLM reconciliation')
        job = self.service.submit({'session_id': sid, 'model': model['id'], 'prompt': 'Hello',
                                   'max_tokens': 32, 'tools_enabled': False, 'prepared': prepared}, captured=True)
        terminal = self.wait_terminal(job['id'])
        self.assertFalse(terminal['context']['fits'])
        self.assertFalse(terminal['context']['visual_tokens_included'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
