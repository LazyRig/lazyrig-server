"""Clean stops retain a worker; unacknowledged stops cannot leave stale replies."""
import os
import threading
import time
import unittest
from unittest.mock import patch

from lazyrig.worker import Cancelled, ProcessWorker
import test_core
from test_core import wait_job
from test_handoff import FixtureWorkers


def _stop_child(conn, mode, model, job_name):
    if os.name != 'nt':
        os.setsid()
    conn.send({'kind': 'ready', 'pid': os.getpid(), 'cooperative_text_stop': True})
    try:
        while True:
            request = conn.recv()
            if request['action'] == 'cancel_text':
                if model['behavior'] == 'unresponsive':
                    time.sleep(10)
                elif model['behavior'] == 'disconnect':
                    return
                elif model['behavior'] == 'error':
                    conn.send({'kind': 'error', 'error': 'native cancellation failed'})
                    return
                else:
                    conn.send({'kind': 'cancelled', 'worker_reusable': True})
            elif request.get('stop_test'):
                conn.send({'kind': 'text', 'text': 'partial'})
                if model['behavior'] == 'result_race':
                    conn.send({'kind': 'result', 'text': 'old result'})
            else:
                conn.send({'kind': 'result', 'text': 'next result', 'pid': os.getpid()})
    finally:
        conn.close()


class TextStopWorkerTests(unittest.TestCase):
    def worker(self, behavior):
        with patch('lazyrig.worker._spawn_child', _stop_child):
            worker = ProcessWorker('fixture', {'behavior': behavior}, 10, threading.Event())
        self.addCleanup(worker.close)
        return worker

    def stop(self, worker):
        cancelled = threading.Event()
        with self.assertRaises(Cancelled) as caught:
            for message in worker.run({'action': 'text', 'stop_test': True}, 10, cancelled):
                self.assertEqual(message['kind'], 'text')
                cancelled.set()
        return caught.exception

    def test_clean_and_racing_result_stop_keep_pid_and_drain_old_messages(self):
        for behavior in ('cooperative', 'result_race'):
            with self.subTest(behavior=behavior):
                worker = self.worker(behavior)
                pid = worker.process.pid
                for _ in range(3):
                    self.assertTrue(self.stop(worker).worker_reusable)
                    self.assertTrue(worker.process.is_alive())
                    result = list(worker.run({'action': 'text'}, 5, threading.Event()))
                    self.assertEqual(result, [{'kind': 'result', 'text': 'next result', 'pid': pid}])

    def test_timeout_disconnect_or_error_during_stop_requires_shutdown(self):
        for behavior in ('unresponsive', 'disconnect', 'error'):
            with self.subTest(behavior=behavior), patch('lazyrig.worker.TEXT_STOP_GRACE_SECONDS', .15):
                worker = self.worker(behavior)
                self.assertFalse(self.stop(worker).worker_reusable)
                self.assertFalse(worker.process.is_alive())

    def test_cancel_before_send_keeps_idle_worker(self):
        worker = self.worker('cooperative')
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled) as caught:
            list(worker.run({'action': 'text', 'stop_test': True}, 5, cancel))
        self.assertTrue(caught.exception.worker_reusable)
        self.assertEqual(list(worker.run({'action': 'text'}, 5, threading.Event()))[0]['text'], 'next result')

    def test_broken_cancel_send_cannot_claim_worker_reusable(self):
        worker = self.worker('cooperative')
        cancel = threading.Event()
        stream = worker.run({'action': 'text', 'stop_test': True}, 5, cancel)
        self.assertEqual(next(stream)['kind'], 'text')
        cancel.set()
        with patch.object(worker.conn, 'send', side_effect=BrokenPipeError()), self.assertRaises(Cancelled) as caught:
            next(stream)
        self.assertFalse(caught.exception.worker_reusable)
        self.assertFalse(worker.process.is_alive())


class TextStopServiceTests(unittest.TestCase):
    setUp = test_core.ServiceTests.setUp
    tearDown = test_core.ServiceTests.tearDown
    submit = test_core.ServiceTests.submit

    def test_acknowledged_stop_keeps_model_and_next_chat_uses_same_worker(self):
        workers = FixtureWorkers([('Next response.', 5)])
        started = threading.Event()
        stopped = [False]
        def factory(*args):
            worker = workers(*args)
            original = worker.run
            def run(payload, timeout, cancelled):
                if not stopped[0]:
                    started.set()
                    while not cancelled.wait(.01):
                        pass
                    stopped[0] = True
                    raise Cancelled(worker_reusable=True)
                yield from original(payload, timeout, cancelled)
            worker.run = run
            return worker
        self.service.scheduler.factory = factory
        jid = self.submit()['id']
        self.assertTrue(started.wait(3))
        self.service.cancel(jid)
        job = wait_job(self.service, jid)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(job['error'], 'Stopped by client.')
        self.assertTrue(self.service.scheduler.is_loaded(self.service.model('qwen-chat')))
        self.assertTrue(any(e['kind'] == 'generation_stopped' and e['worker_retained']
                            for e in self.service.store.events(jid)))
        next_job = wait_job(self.service, self.submit()['id'])
        self.assertEqual(next_job['status'], 'completed', next_job['error'])
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])


if __name__ == '__main__':
    unittest.main()
