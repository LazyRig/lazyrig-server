"""Real process cleanup, including a descendant that ignores SIGTERM."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from lazyrig.worker import ProcessWorker, ExternalProcess


class WorkerLifetimeTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX process-session regression; Windows uses a Job Object')
    def test_close_stops_descendant_after_leader_exit_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / 'ready.json'
            child = ('import socket,time,signal,json,pathlib,sys;'
                     'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                     's=socket.socket();s.bind(("127.0.0.1",0));s.listen();'
                     'pathlib.Path(sys.argv[1]).write_text(json.dumps({"port":s.getsockname()[1]}));time.sleep(60)')
            parent = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]);time.sleep(60)'
            process = subprocess.Popen([sys.executable, '-c', parent, child, str(ready)], start_new_session=True)
            worker = ProcessWorker.__new__(ProcessWorker)
            worker.process = ExternalProcess(process)
            worker.conn = types.SimpleNamespace(close=lambda: None)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline: time.sleep(.02)
                self.assertTrue(ready.exists())
                port = json.loads(ready.read_text())['port']
                with socket.create_connection(('127.0.0.1', port), timeout=1): pass
                worker.close(); worker.close()
                self.assertFalse(worker.process.is_alive())
                deadline = time.monotonic() + 3
                while True:
                    with socket.socket() as connection:
                        code = connection.connect_ex(('127.0.0.1', port))
                    if code: break
                    self.assertLess(time.monotonic(), deadline, 'Model descendant survived worker close')
                    time.sleep(.02)
            finally:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(5)


if __name__ == '__main__': unittest.main()
