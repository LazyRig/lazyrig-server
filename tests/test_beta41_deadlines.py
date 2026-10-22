"""Budget propagation and cleanup, without model downloads or GPU dependencies."""
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from lazyrig import conversion
from lazyrig.deadlines import (DeadlineEvent, RequestDeadline, cancellation_requested,
                               current_deadline, deadline_context, time_budget)
from lazyrig.scheduler import Scheduler
from lazyrig.vision import ImageStore, fetch_image
from lazyrig.worker import ProcessWorker


class DeadlineTests(unittest.TestCase):
    def test_one_budget_across_phases_and_final_cancel_inspection(self):
        signal = DeadlineEvent(.08)
        with deadline_context(signal):
            self.assertIs(current_deadline(), signal)
            first = time_budget(5)
            time.sleep(.015)
            self.assertLess(time_budget(5), first)
            with deadline_context(DeadlineEvent(3)):
                self.assertIsNot(current_deadline(), signal)
            self.assertIs(current_deadline(), signal)
        self.assertIsNone(current_deadline())
        signal.deadline = time.monotonic() - 1
        with self.assertRaises(RequestDeadline): signal.is_set()
        self.assertFalse(cancellation_requested(signal))
        signal.set()
        self.assertTrue(signal.is_set())
        self.assertTrue(cancellation_requested(signal))

    def test_queue_expiry_removes_ticket_and_next_job_can_enter(self):
        scheduler = Scheduler({'devices': [], 'models': []})
        result = []
        signal = DeadlineEvent(.04)
        def queued():
            try:
                with scheduler.lease(['CPU'], signal): result.append('entered')
            except RequestDeadline: result.append('expired')
        with scheduler.lease(['CPU'], threading.Event()):
            thread = threading.Thread(target=queued); thread.start(); thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, ['expired'])
            self.assertEqual(scheduler.waiting, [])
        with scheduler.lease(['CPU'], DeadlineEvent(1)):
            self.assertEqual(scheduler.busy, {'CPU'})
        self.assertEqual(scheduler.busy, set())

    def test_native_wait_uses_total_deadline_and_closes_worker(self):
        worker = object.__new__(ProcessWorker)
        worker.conn = SimpleNamespace(send=lambda request: None, poll=lambda timeout: time.sleep(timeout) or False)
        worker.process = SimpleNamespace(is_alive=lambda: True)
        worker.close = Mock()
        started = time.monotonic()
        with self.assertRaises(RequestDeadline):
            list(worker.run({'action': 'text'}, 20, DeadlineEvent(.04)))
        self.assertLess(time.monotonic() - started, .5)
        worker.close.assert_called_once()

    def test_expired_lock_acquisition_does_not_leak_converter_lock(self):
        lock = threading.Lock()
        with self.assertRaises(RequestDeadline):
            conversion._acquire(lock, DeadlineEvent(deadline=time.monotonic() - 1))
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()

    def test_event_write_failure_cannot_prevent_native_cleanup(self):
        scheduler = Scheduler({'terminal_metrics': False})
        worker = SimpleNamespace(close=Mock(), worker_pid=42)
        scheduler.resident['model'] = ({'device': 'CPU'}, worker)
        scheduler.states['model'] = 'loaded'
        scheduler.last_used['model'] = 1
        emit = Mock(side_effect=OSError('disk unavailable'))
        with self.assertRaisesRegex(OSError, 'disk unavailable'):
            scheduler.unload('model', emit)
        worker.close.assert_called_once()
        self.assertNotIn('model', scheduler.resident)
        self.assertNotIn('model', scheduler.last_used)
        self.assertEqual(scheduler.states['model'], 'unloaded')

    def test_cleanup_failure_remains_visible_when_reporting_also_fails(self):
        scheduler = Scheduler({'terminal_metrics': False})
        worker = SimpleNamespace(close=Mock(side_effect=RuntimeError('worker still alive')))
        scheduler.resident['model'] = ({'device': 'CPU'}, worker)
        with self.assertLogs('lazyrig.runtime', level='ERROR'), \
             self.assertRaisesRegex(RuntimeError, 'worker still alive'):
            scheduler.unload('model', Mock(side_effect=OSError('disk unavailable')))
        self.assertIn('model', scheduler.resident)
        self.assertEqual(scheduler.states['model'], 'cleanup-failed')

    def test_profile_probe_timeout_is_not_mistaken_for_bad_environment(self):
        with patch.object(conversion, 'run_logged', side_effect=RequestDeadline()):
            with self.assertRaises(RequestDeadline):
                conversion._profile_compatible('python', Path('requirements.txt'), DeadlineEvent(2), lambda x: None, Path('unused.log'))

    def test_slow_image_download_has_one_budget_and_preserves_timeout_type(self):
        response = SimpleNamespace(status=200, getheader=lambda key: None,
                                   read1=lambda count: time.sleep(.018) or b'x')
        connection = SimpleNamespace(request=lambda *a, **kw: None, getresponse=lambda: response, close=Mock())
        sock = SimpleNamespace(settimeout=Mock())
        lookup = [(2,1,6,'',('8.8.8.8',80))]
        with patch('lazyrig.vision.socket.getaddrinfo',return_value=lookup), \
             patch('lazyrig.vision.socket.create_connection',return_value=sock), \
             patch('lazyrig.vision.http.client.HTTPConnection',return_value=connection), \
             deadline_context(DeadlineEvent(.04)):
            with self.assertRaises(RequestDeadline): fetch_image('http://public.example/image')
        connection.close.assert_called_once()
        self.assertTrue(all(call.args[0] <= .04 for call in sock.settimeout.call_args_list))

    def test_decode_expiry_does_not_publish_input_or_become_invalid_image(self):
        data = io.BytesIO()
        Image.new('RGB', (16,16), 'red').save(data,format='PNG')
        original = Image.Image.save
        def slow_save(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(.04)
            return result
        with tempfile.TemporaryDirectory() as directory:
            store = ImageStore(directory)
            with patch.object(Image.Image,'save',slow_save), deadline_context(DeadlineEvent(.02)):
                with self.assertRaises(RequestDeadline): store.save(data.getvalue(),'auto')
            self.assertEqual(list(store.directory.iterdir()), [])


if __name__ == '__main__': unittest.main()
