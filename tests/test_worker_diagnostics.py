"""Real disposable-process exits and diagnostic privacy; no model or GPU loads."""
import os
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyrig.terminal import Reporter
from lazyrig.deadlines import DeadlineEvent, RequestDeadline
from lazyrig.worker import Cancelled, ExternalProcess, NativeWorkerError, ProcessWorker
from lazyrig.worker_status import WorkerProcessError, exit_status
from lazyrig.worker_diagnostics import create_fault_log, enable_fault_log, read_fault_log, FAULT_LOG_BYTES


def _exit_child(conn, mode, model, job_name):
    if os.name != 'nt':
        os.setsid()
    action = model['test_action']
    if action in {'delayed_validation_exit', 'fatal_validation'}:
        conn.send({'kind': 'worker_diagnostics', 'runtime': {'openvino': 'test fixture'},
                   'validation': {'checkpoint': 'generate_started', 'reported_tokens': 0, 'expected_tokens': 2}})
        conn.send({'kind': 'load_stage', 'stage': 'validation'})
        if action == 'fatal_validation':
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            stream = enable_fault_log(model['_worker_fault_log'])
            if stream is None:
                os._exit(99)
            os.abort()
        conn.close()
        time.sleep(.6)  # Longer than the old 0.2-second diagnostic window.
        os._exit(26)
    if action == 'load_exit':
        os._exit(23)
    if action == 'validation_exit':
        conn.send({'kind': 'load_stage', 'stage': 'validation'})
        os._exit(25)
    conn.send({'kind': 'ready', 'pid': os.getpid()})
    conn.recv()
    if action == 'native_error':
        conn.send({'kind': 'error', 'error': 'native failure marker'})
        os._exit(31)
    if action == 'disconnect':
        conn.close()
        time.sleep(10)  # The parent must report unknown status, then terminate us.
    os._exit(24)


class WorkerDiagnosticTests(unittest.TestCase):
    def worker(self, action):
        with patch('lazyrig.worker._spawn_child', _exit_child):
            return ProcessWorker('fixture', {'test_action': action}, 10, threading.Event())

    def test_real_load_exit_code_survives_constructor_cleanup(self):
        with self.assertRaises(WorkerProcessError) as caught:
            self.worker('load_exit')
        error = caught.exception
        self.assertEqual(error.error_code, 'worker_process_exited')
        self.assertEqual(error.worker_exit['exit_code'], 23)
        self.assertEqual(error.worker_exit['stage'], 'loading')
        self.assertTrue(error.worker_exit['exit_observed'])
        self.assertIn('exit code 23', str(error))
        self.assertIn('load_memory', error.__dict__)

    def test_real_split_validation_exit_is_classified_before_cleanup(self):
        from lazyrig.placement import retryable_load_error, load_failure_reason
        with self.assertRaises(WorkerProcessError) as caught:
            self.worker('validation_exit')
        error = caught.exception
        self.assertEqual(error.worker_exit['exit_code'], 25)
        self.assertEqual(error.worker_exit['stage'], 'validation')
        self.assertTrue(retryable_load_error(error))
        self.assertEqual(load_failure_reason(error), 'split_inference_failed')

    def test_delayed_natural_exit_keeps_code_runtime_and_probe_evidence(self):
        with self.assertRaises(WorkerProcessError) as caught:
            self.worker('delayed_validation_exit')
        error = caught.exception
        self.assertEqual(error.worker_exit['exit_code'], 26)
        self.assertTrue(error.worker_exit['exit_observed'])
        self.assertEqual(error.worker_diagnostics['runtime']['openvino'], 'test fixture')
        self.assertEqual(error.worker_diagnostics['validation']['reported_tokens'], 0)
        self.assertEqual(error.worker_diagnostics['fault_log']['status'], 'empty')

    @unittest.skipIf(os.name == 'nt', 'Isolated POSIX abort fixture; Windows handler requires native validation')
    def test_fatal_fault_survives_broken_ipc_and_temporary_file_cleanup(self):
        paths = []
        def create():
            path = create_fault_log()
            paths.append(path)
            return path
        with patch('lazyrig.worker.create_fault_log', create), self.assertRaises(WorkerProcessError) as caught:
            self.worker('fatal_validation')
        error = caught.exception
        self.assertEqual(error.worker_stage, 'validation')
        self.assertEqual(error.worker_exit['exit_code'], -6)
        fault = error.worker_diagnostics['fault_log']
        self.assertEqual(fault['status'], 'captured')
        self.assertIn('Fatal Python error: Aborted', fault['text'])
        self.assertIn('_exit_child', fault['text'])
        self.assertTrue(paths)
        self.assertTrue(all(not os.path.exists(path) for path in paths))

    def test_fault_capture_is_bounded_and_removes_terminal_controls(self):
        path = create_fault_log()
        self.addCleanup(os.unlink, path)
        with open(path, 'wb') as stream:
            stream.write(b'Fatal\x1b[0m\x00\n' + b'x' * (FAULT_LOG_BYTES + 100))
        report = read_fault_log(path)
        self.assertTrue(report['truncated'])
        self.assertLessEqual(len(report['text']), FAULT_LOG_BYTES)
        self.assertNotIn('\x1b', report['text'])
        self.assertNotIn('\x00', report['text'])
        self.assertEqual(read_fault_log(None), {'status': 'unavailable'})

    def test_exit_grace_respects_operation_deadline(self):
        now = [10.0]
        waits = []
        def join(timeout):
            waits.append(timeout)
            now[0] += timeout
        worker = ProcessWorker.__new__(ProcessWorker)
        worker.process = SimpleNamespace(is_alive=lambda: True, join=join, pid=123, exitcode=None)
        with patch('lazyrig.worker.time.monotonic', lambda: now[0]):
            error = worker._exit_error(threading.Event(), 'validation', operation_deadline=10.12)
        self.assertAlmostEqual(sum(waits), .12)
        self.assertLessEqual(max(waits), .05)
        self.assertFalse(error.worker_exit['exit_observed'])
        self.assertIsNone(error.worker_exit['exit_code'])

    def test_real_generation_exit_retains_code_after_worker_is_closed(self):
        worker = self.worker('generation_exit')
        self.addCleanup(worker.close)
        with self.assertRaises(WorkerProcessError) as caught:
            list(worker.run({'action': 'test'}, 5, threading.Event()))
        self.assertEqual(caught.exception.worker_exit['exit_code'], 24)
        self.assertEqual(caught.exception.worker_exit['stage'], 'generation')
        self.assertFalse(worker.process.is_alive())
        self.assertTrue(worker._closed)

    def test_closed_pipe_does_not_report_the_later_cleanup_termination_as_a_crash(self):
        worker = self.worker('disconnect')
        self.addCleanup(worker.close)
        with self.assertRaises(WorkerProcessError) as caught:
            list(worker.run({'action': 'test'}, 5, threading.Event()))
        error = caught.exception
        self.assertEqual(error.error_code, 'worker_connection_closed')
        self.assertFalse(error.worker_exit['exit_observed'])
        self.assertIsNone(error.worker_exit['exit_code'])
        self.assertIn('exit code unavailable before cleanup', str(error))
        self.assertFalse(worker.process.is_alive())

    def test_native_exception_buffered_before_exit_is_preserved(self):
        worker = self.worker('native_error')
        self.addCleanup(worker.close)
        with self.assertRaisesRegex(NativeWorkerError, 'native failure marker'):
            list(worker.run({'action': 'test'}, 5, threading.Event()))
        self.assertFalse(worker.process.is_alive())

    def test_external_process_status_is_captured_without_pipe_eof(self):
        process = subprocess.Popen([sys.executable, '-c', 'import os;os._exit(27)'],
                                   start_new_session=os.name != 'nt')
        process.wait(5)
        worker = ProcessWorker.__new__(ProcessWorker)
        worker.process = ExternalProcess(process)
        worker.conn = SimpleNamespace(poll=lambda timeout: False, close=lambda: None)
        self.addCleanup(worker.close)
        self.assertEqual(worker.process.exitcode, 27)
        with self.assertRaises(WorkerProcessError) as caught:
            next(worker._receive(5, threading.Event()))
        worker.close()
        self.assertEqual(caught.exception.worker_exit['exit_code'], 27)

    def test_cancel_during_exit_grace_stays_cancelled(self):
        cancel = threading.Event()
        worker = ProcessWorker.__new__(ProcessWorker)
        waits = []
        def join(timeout):
            waits.append(timeout)
            cancel.set()
        worker.process = SimpleNamespace(is_alive=lambda: True, join=join, pid=123, exitcode=None)
        with self.assertRaises(Cancelled):
            worker._exit_error(cancel, 'generation')
        self.assertEqual(len(waits), 1)
        self.assertLessEqual(waits[0], .2)

    def test_windows_signed_and_unsigned_status_have_same_hex(self):
        for code in (3221225477, -1073741819):
            report = WorkerProcessError(code, platform='nt').worker_exit
            self.assertIn('0xC0000005', exit_status(report))
            self.assertIn(str(code), exit_status(report))
            self.assertNotIn('signal', exit_status(report))
        self.assertEqual(exit_status({'platform': 'posix', 'exit_code': -15}), 'signal 15 (SIGTERM)')

    def test_request_deadline_during_exit_grace_stays_a_timeout(self):
        cancel = DeadlineEvent(1)
        worker = ProcessWorker.__new__(ProcessWorker)
        def join(timeout):
            cancel.deadline = time.monotonic() - 1
        worker.process = SimpleNamespace(is_alive=lambda: True, join=join, pid=123, exitcode=None)
        with self.assertRaises(RequestDeadline):
            worker._exit_error(cancel, 'generation')

    def test_zero_exit_code_is_still_an_incomplete_exchange(self):
        error = WorkerProcessError(0, stage='loading')
        self.assertEqual(error.error_code, 'worker_process_exited')
        self.assertIn('exit code 0', str(error))

    def test_terminal_exposes_only_validated_numeric_status(self):
        reports = [WorkerProcessError(-1073741819, platform='nt').worker_exit,
                   {'exit_code': 'PRIVATE STATUS', 'platform': 'PRIVATE PLATFORM', 'pid': 'PRIVATE PID'},
                   {'exit_code': 2**100, 'platform': 'nt'}, {'exit_code': True}, None]
        for report in reports:
            with self.subTest(report=report):
                job = {'id': 'a'*32, 'model': 'llm', 'error_code': 'worker_process_exited',
                       'worker_exit': report, 'error': 'PRIVATE NATIVE ERROR', 'request': 'PRIVATE PROMPT'}
                with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
                    Reporter({'terminal_metrics': True}, job).finish('failed')
                output = '\n'.join(logs.output)
                self.assertNotIn('PRIVATE', output)
                self.assertNotIn(str(2**100), output)
                self.assertIn('worker_process_exited', output)
                self.assertIn('0xC0000005' if report is reports[0] else 'exit code unavailable', output)


if __name__ == '__main__':
    unittest.main()
