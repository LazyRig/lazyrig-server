"""Host-memory safeguards use simulated readings, never real RAM exhaustion."""
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lazyrig import memory
from lazyrig import windows_job
from lazyrig.worker import ProcessWorker, WorkerCleanupError


GIB = 1024**3


def snapshot(available=20, total=64):
    return {'total_gib': total, 'available_gib': available, 'source': 'test physical RAM'}


class MemoryReadingTests(unittest.TestCase):
    def test_windows_reports_available_physical_not_commit_or_pagefile(self):
        def query(pointer):
            value = pointer._obj
            self.assertEqual(value.length, 64)
            value.total_physical = 64 * GIB
            value.available_physical = 7 * GIB
            value.available_pagefile = 300 * GIB
            return 1
        kernel = SimpleNamespace(GlobalMemoryStatusEx=query)
        with patch.object(memory, 'os', SimpleNamespace(name='nt')), patch.object(
                memory.ctypes, 'windll', SimpleNamespace(kernel32=kernel), create=True):
            value = memory.host_memory_snapshot()
            self.assertEqual(value['total_gib'], 64)
            self.assertEqual(value['available_gib'], 7)
            kernel.GlobalMemoryStatusEx = lambda pointer: 0
            self.assertIsNone(memory.host_memory_snapshot())

    def _files(self, values):
        def read(path, *args, **kwargs):
            # The simulated Linux filesystem also runs under Windows pathlib.
            key = path.as_posix()
            if key not in values:
                raise FileNotFoundError(str(path))
            return values[key]
        return patch.object(Path, 'read_text', read)

    def test_linux_memavailable_does_not_mistake_reclaimable_cache_for_pressure(self):
        values = {'/proc/meminfo': 'MemTotal: 67108864 kB\nMemFree: 512 kB\nMemAvailable: 8388608 kB\n'}
        with patch.object(memory, 'os', SimpleNamespace(name='posix')), self._files(values):
            self.assertEqual(memory.host_memory_snapshot(), {
                'total_gib': 64, 'available_gib': 8, 'source': 'Linux physical RAM'})

    def test_cgroup_v2_honors_nested_and_parent_bounds(self):
        values = {'/proc/meminfo': 'MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n',
                  '/proc/self/cgroup': '0::/workers/native\n',
                  '/sys/fs/cgroup/memory.max': 'max',
                  '/sys/fs/cgroup/workers/memory.max': str(16 * GIB),
                  '/sys/fs/cgroup/workers/memory.current': str(15 * GIB),
                  '/sys/fs/cgroup/workers/memory.stat': f'inactive_file {GIB}\n',
                  '/sys/fs/cgroup/workers/native/memory.max': str(8 * GIB),
                  '/sys/fs/cgroup/workers/native/memory.current': str(4 * GIB)}
        with patch.object(memory, 'os', SimpleNamespace(name='posix')), self._files(values):
            value = memory.host_memory_snapshot()
        self.assertEqual(value['total_gib'], 8)
        self.assertEqual(value['available_gib'], 2)
        self.assertIn('cgroup', value['source'])

    def test_cgroup_namespace_parent_paths_do_not_escape_mount(self):
        values = {'/proc/meminfo': 'MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n',
                  '/proc/self/cgroup': '0::/../../outside\n',
                  '/sys/fs/cgroup/memory.max': str(8 * GIB),
                  '/sys/fs/cgroup/memory.current': str(8 * GIB)}
        with patch.object(memory, 'os', SimpleNamespace(name='posix')), self._files(values):
            value = memory.host_memory_snapshot()
        self.assertEqual(value['total_gib'], 8)
        self.assertEqual(value['available_gib'], 0)

    def test_missing_or_malformed_statistics_are_nonblocking(self):
        for data in ('MemTotal: 1024 kB\n', 'MemTotal: nope kB\nMemAvailable: 10 kB\n',
                     'MemTotal: 1024 kB\nMemAvailable: 2048 kB\n'):
            with self.subTest(data=data), patch.object(memory, 'os', SimpleNamespace(name='posix')), \
                    self._files({'/proc/meminfo': data}):
                self.assertIsNone(memory.host_memory_snapshot())
        with patch.object(memory, 'host_memory_snapshot', return_value=None):
            memory.LoadMemoryGuard().check(force=True)


class LoadMemoryGuardTests(unittest.TestCase):
    def test_headroom_scales_with_ram_with_lower_and_upper_bounds(self):
        for total, expected in ((8, 1), (64, 6.4), (256, 8)):
            with self.subTest(total=total), patch.object(memory, 'host_memory_snapshot',
                    return_value=snapshot(expected - .1, total)):
                with self.assertRaises(memory.MemoryPressureError) as caught:
                    memory.LoadMemoryGuard().check()
                self.assertEqual(caught.exception.headroom_gib, expected)
                self.assertEqual(caught.exception.available_gib, expected - .1)
                self.assertIn('not a hard memory-allocation limit', str(caught.exception))

    def test_sampling_is_throttled_and_forced_boundary_check_is_fresh(self):
        with patch.object(memory.time, 'monotonic', side_effect=[0, .05, .099, .1, .101]), \
                patch.object(memory, 'host_memory_snapshot', side_effect=[snapshot(), snapshot(), snapshot(1)]) as read:
            guard = memory.LoadMemoryGuard()
            guard.check()
            guard.check()
            guard.check()
            self.assertEqual(read.call_count, 1)
            guard.check()
            self.assertEqual(read.call_count, 2)
            with self.assertRaises(memory.MemoryPressureError):
                guard.check(force=True)
            self.assertEqual(read.call_count, 3)

    def test_diagnostics_record_floor_cap_and_minimum_without_stale_unknown_values(self):
        with patch.object(memory, 'host_memory_snapshot',
                          side_effect=[snapshot(20), snapshot(12), None]):
            guard = memory.LoadMemoryGuard()
            guard.check(force=True)
            self.assertEqual(guard.allocation_limit_bytes, int((20 - 6.4) * GIB))
            guard.check(force=True)
            self.assertEqual(guard.minimum_available_gib, 12)
            guard.check(force=True)
            self.assertIsNone(guard.allocation_limit_bytes)
            self.assertIsNone(guard.last_snapshot)
            self.assertEqual(guard.diagnostics()['status'], 'unavailable')
            self.assertEqual(guard.diagnostics()['minimum_available_gib'], 12)
            self.assertEqual(guard.diagnostics()['sample_count'], 3)

    def test_no_available_allocation_budget_is_rejected(self):
        with patch.object(memory, 'host_memory_snapshot', return_value=snapshot(6.4)):
            with self.assertRaises(memory.MemoryPressureError):
                memory.LoadMemoryGuard().check(force=True)

    def test_observer_context_restores_outer_listener_and_copies_event(self):
        outer, inner = [], []
        with memory.observe_load_memory(outer.append):
            memory.publish_load_memory({'stage': 'initial'})
            with memory.observe_load_memory(inner.append):
                memory.publish_load_memory({'stage': 'inner'})
            data = {'stage': 'ready'}
            memory.publish_load_memory(data)
            data['stage'] = 'changed'
        memory.publish_load_memory({'stage': 'outside'})
        self.assertEqual(outer, [{'stage': 'initial'}, {'stage': 'ready'}])
        self.assertEqual(inner, [{'stage': 'inner'}])


class WindowsCommitLimitTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.CreateJobObjectW.return_value = 123
        self.api.SetInformationJobObject.return_value = 1
        self.installed = []
        def capture(handle, info_class, pointer, size):
            limits = pointer._obj
            self.installed.append({'handle': handle, 'class': info_class, 'size': size,
                                   'flags': limits.basic.flags, 'job_memory': limits.job_memory,
                                   'process_memory': limits.process_memory})
            return 1
        self.api.SetInformationJobObject.side_effect = capture
        patcher = patch.object(windows_job, '_api', return_value=self.api)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_job_commit_cap_is_installed_with_kill_on_close_and_exact_byte_count(self):
        job = windows_job.WindowsJob(memory_limit_bytes=37 * GIB)
        limit = self.installed[0]
        self.assertEqual(limit['flags'], 0x2200)
        self.assertEqual(limit['job_memory'], 37 * GIB)
        self.assertEqual(limit['process_memory'], 0)
        self.assertEqual(limit['class'], 9)
        self.assertEqual(limit['size'], windows_job.c.sizeof(windows_job.ExtendedLimits))
        self.assertEqual(job.memory_limit_bytes, 37 * GIB)

    def test_unknown_memory_keeps_kill_on_close_without_inventing_a_cap(self):
        job = windows_job.WindowsJob()
        self.assertEqual(self.installed[0]['flags'], 0x2000)
        self.assertEqual(self.installed[0]['job_memory'], 0)
        self.assertIsNone(job.memory_limit_bytes)

    def test_peak_commit_uses_os_measurement_and_closed_handle_is_unknown(self):
        job = windows_job.WindowsJob(memory_limit_bytes=37 * GIB)
        def query(handle, info_class, pointer, size, returned):
            self.assertEqual((handle, info_class), (123, 9))
            info = windows_job.c.cast(pointer, windows_job.c.POINTER(windows_job.ExtendedLimits)).contents
            info.peak_job_memory = 11 * GIB
            return 1
        self.api.QueryInformationJobObject.side_effect = query
        self.assertEqual(job.peak_commit_bytes(), 11 * GIB)
        job.handle = None
        self.assertIsNone(job.peak_commit_bytes())

    def test_invalid_caps_fail_before_creating_a_job(self):
        for value in (0, -1, True, 1.5, float('inf'), 1 << (8 * windows_job.c.sizeof(windows_job.c.c_size_t))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                windows_job.WindowsJob(memory_limit_bytes=value)
        self.api.CreateJobObjectW.assert_not_called()

    def test_limit_installation_failure_closes_handle_and_aborts(self):
        self.api.SetInformationJobObject.side_effect = OSError('cannot install commit limit')
        with self.assertRaisesRegex(OSError, 'cannot install'):
            windows_job.WindowsJob(memory_limit_bytes=GIB)
        self.api.CloseHandle.assert_called_once_with(123)


class WorkerLoadGuardTests(unittest.TestCase):
    def setUp(self):
        self.cancelled = threading.Event()
        self.model = {'pipeline': 'llm', 'path': '/unused', 'device': 'CPU'}
        self.error = memory.MemoryPressureError(64, 2, 6.4)
        self.guard = Mock()
        self.guard.diagnostics.return_value = {**snapshot(), 'status': 'available', 'headroom_gib': 6.4,
                                              'minimum_available_gib': 20, 'sample_count': 2}
        self.guard.allocation_limit_bytes = int((20 - 6.4) * GIB)
        self.connection = Mock()
        self.connection.poll.return_value = True
        self.connection.recv.return_value = {'kind': 'ready', 'pid': 123}
        self.child = Mock()
        self.process = Mock(pid=123)
        self.context = Mock()
        self.context.Pipe.return_value = (self.connection, self.child)
        self.context.Process.return_value = self.process
        for target, value in (
                ('lazyrig.worker.LoadMemoryGuard', Mock(return_value=self.guard)),
                ('lazyrig.worker.mp.get_context', Mock(return_value=self.context)),
                ('lazyrig.windows_job.WindowsJob', Mock())):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pressure_before_spawn_starts_no_worker(self):
        self.guard.check.side_effect = self.error
        with self.assertRaises(memory.MemoryPressureError):
            ProcessWorker('openvino', self.model, 5, self.cancelled)
        self.context.Process.assert_not_called()

    def test_pressure_during_load_is_cleaned_up_before_propagation(self):
        self.guard.check.side_effect = [None, None, self.error]
        closed = []
        with patch.object(ProcessWorker, 'close', lambda worker: closed.append(worker)), \
                self.assertRaises(memory.MemoryPressureError):
            ProcessWorker('openvino', self.model, 5, self.cancelled)
        self.process.start.assert_called_once()
        self.assertEqual(len(closed), 1)
        self.connection.recv.assert_not_called()

    def test_pressure_at_ready_boundary_rejects_worker_and_cleans_up(self):
        self.guard.check.side_effect = [None, None, None, self.error]
        with patch.object(ProcessWorker, 'close') as close, self.assertRaises(memory.MemoryPressureError):
            ProcessWorker('openvino', self.model, 5, self.cancelled)
        self.connection.recv.assert_called_once()
        self.assertEqual(self.guard.check.call_args.kwargs, {'force': True})
        close.assert_called_once()

    def test_cleanup_failure_remains_distinct_and_cannot_look_like_retryable_pressure(self):
        self.guard.check.side_effect = [None, None, self.error]
        with patch.object(ProcessWorker, 'close', side_effect=RuntimeError('still alive')), \
                self.assertRaises(WorkerCleanupError) as caught:
            ProcessWorker('openvino', self.model, 5, self.cancelled)
        self.assertNotIsInstance(caught.exception, memory.MemoryPressureError)
        self.assertIn('cleanup is unconfirmed', str(caught.exception))
        self.assertIs(caught.exception.worker.process, self.process)
        self.assertEqual(caught.exception.load_memory['stage'], 'error')

    def test_generation_does_not_apply_loading_memory_guard(self):
        worker = ProcessWorker('openvino', self.model, 5, self.cancelled)
        checks_after_load = self.guard.check.call_count
        self.guard.check.side_effect = self.error
        self.connection.recv.return_value = {'kind': 'result', 'text': 'ok'}
        self.assertEqual(list(worker.run({'action': 'text'}, 5, self.cancelled)),
                         [{'kind': 'result', 'text': 'ok'}])
        self.assertEqual(self.guard.check.call_count, checks_after_load)

    def test_windows_worker_receives_capped_job_before_spawn_and_reports_actual_limit(self):
        events = []
        job = SimpleNamespace(name='bounded-worker', memory_limit_bytes=self.guard.allocation_limit_bytes,
                              peak_commit_bytes=lambda: 7 * GIB)
        with patch('lazyrig.worker.os.name', 'nt'), \
                patch.object(windows_job, 'WindowsJob', return_value=job) as create, \
                memory.observe_load_memory(events.append):
            worker = ProcessWorker('openvino', self.model, 5, self.cancelled)
        create.assert_called_once_with(memory_limit_bytes=self.guard.allocation_limit_bytes)
        self.assertEqual(self.context.Process.call_args.kwargs['args'][-1], 'bounded-worker')
        self.assertEqual([event['stage'] for event in events], ['initial', 'ready'])
        self.assertEqual(worker.load_memory['job_commit_limit_bytes'], job.memory_limit_bytes)
        self.assertEqual(worker.load_memory['job_peak_commit_bytes'], 7 * GIB)
        self.assertIn('worker lifetime', worker.load_memory['job_commit_limit_scope'])

    def test_unknown_windows_sample_is_explicit_in_report(self):
        self.guard.diagnostics.return_value = {'status': 'unavailable', 'source': None,
                                              'total_gib': None, 'available_gib': None,
                                              'headroom_gib': None, 'minimum_available_gib': None,
                                              'sample_count': 2}
        self.guard.allocation_limit_bytes = None
        job = SimpleNamespace(name='uncapped-worker', memory_limit_bytes=None)
        with patch('lazyrig.worker.os.name', 'nt'), \
                patch.object(windows_job, 'WindowsJob', return_value=job) as create:
            worker = ProcessWorker('openvino', self.model, 5, self.cancelled)
        create.assert_called_once_with(memory_limit_bytes=None)
        self.assertEqual(worker.load_memory['status'], 'unavailable')
        self.assertIsNone(worker.load_memory['job_commit_limit_bytes'])

    def test_pressure_exception_includes_measured_diagnostics_after_cleanup(self):
        self.guard.check.side_effect = [None, None, self.error]
        with patch.object(ProcessWorker, 'close') as close, self.assertRaises(memory.MemoryPressureError) as caught:
            ProcessWorker('openvino', self.model, 5, self.cancelled)
        close.assert_called_once()
        self.assertEqual(caught.exception.load_memory['stage'], 'error')
        self.assertIn('minimum_available_gib', caught.exception.load_memory)


if __name__ == '__main__':
    unittest.main()
