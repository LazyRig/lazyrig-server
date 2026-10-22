"""Windows ownership contracts; the native descendant test runs on Windows."""
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from lazyrig.windows_job import WindowsJob, join_worker_job, BasicLimits, ExtendedLimits, Accounting


class FakeKernel:
    def __init__(self): self.calls = []; self.active = 2; self.stuck = False
    def CreateJobObjectW(self, security, name): self.calls.append(('create', name)); return 100
    def SetInformationJobObject(self, handle, kind, info, size):
        assert kind == 9 and info._obj.basic.flags == 0x2000 and size == 144
        self.calls.append(('kill_on_close',)); return 1
    def OpenJobObjectW(self, rights, inherit, name):
        assert rights == 1 and not inherit
        self.calls.append(('open_child', name)); return 200
    def GetCurrentProcess(self): return -1
    def AssignProcessToJobObject(self, job, process): self.calls.append(('assign', job, process)); return 1
    def TerminateJobObject(self, handle, code): self.calls.append(('terminate',)); return 1
    def QueryInformationJobObject(self, handle, kind, info, size, returned):
        info._obj.active_processes = self.active
        if not self.stuck: self.active = max(0, self.active - 1)
        self.calls.append(('active', info._obj.active_processes)); return 1
    def CloseHandle(self, handle): self.calls.append(('close', handle)); return 1


class WindowsJobTests(unittest.TestCase):
    def test_x64_layout_join_and_drain_before_handle_release(self):
        self.assertEqual([ctypes.sizeof(t) for t in (BasicLimits, ExtendedLimits, Accounting)], [64, 144, 48])
        api = FakeKernel()
        with patch('lazyrig.windows_job._api', return_value=api):
            job = WindowsJob(); join_worker_job(job.name); job.close(); job.close()
        self.assertIn(('assign', 200, -1), api.calls)
        self.assertIn(('close', 200), api.calls)
        self.assertEqual(api.calls[-2:], [('active', 0), ('close', 100)])

    def test_unconfirmed_termination_keeps_ownership_for_retry(self):
        api = FakeKernel(); api.stuck = True
        with patch('lazyrig.windows_job._api', return_value=api):
            job = WindowsJob()
            with self.assertRaises(RuntimeError): job.terminate(timeout=0)
            self.assertEqual(job.handle, 100)
            self.assertNotIn(('close', 100), api.calls)
            api.stuck = False; job.close()

    @unittest.skipUnless(os.name == 'nt', 'Native Windows Job Object test')
    def test_native_job_terminates_worker_and_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            started = Path(directory) / 'started'
            job = WindowsJob()
            script = ('import sys,subprocess,time,pathlib;from lazyrig.windows_job import join_worker_job;'
                      'join_worker_job(sys.argv[1]);'
                      'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
                      'pathlib.Path(sys.argv[2]).write_text(str(p.pid));time.sleep(60)')
            process = subprocess.Popen([sys.executable, '-c', script, job.name, str(started)])
            try:
                deadline = time.monotonic() + 8
                while not started.exists() and time.monotonic() < deadline: time.sleep(.05)
                self.assertTrue(started.exists())
                self.assertGreaterEqual(job.active_processes(), 2)
                job.terminate()
                self.assertEqual(job.active_processes(), 0)
                process.wait(5)
            finally:
                job.close()
                if process.poll() is None: process.kill()
                process.wait(5)


if __name__ == '__main__': unittest.main()
