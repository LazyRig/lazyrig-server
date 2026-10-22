"""Cancel real diffusion after its first step, then reuse the device for a new job."""
import argparse
import copy
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig import __version__
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service, TERMINAL
from lazyrig.worker import ProcessWorker


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-native-stop-') as directory:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': directory,
                  'devices': [{'id': 'CPU', 'name': 'CPU', 'policy': 'exclusive', 'budget_gib': 8}]}
        service = Service(config)
        workers = []
        def factory(*args):
            worker = ProcessWorker(*args)
            workers.append(worker)
            return worker
        service.scheduler.factory = factory
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=http.serve_forever, daemon=True).start()
        base = 'http://127.0.0.1:' + str(http.server_port)

        def api(path, body=None, key=None):
            request = urllib.request.Request(base+path,
                None if body is None else json.dumps(body).encode(),
                {'Content-Type': 'application/json', **({'Idempotency-Key': key} if key else {})})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        def until(jid, predicate, seconds=120):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                job = api('/api/jobs/'+jid)
                if predicate(job):
                    return job
                if job['status'] in TERMINAL:
                    raise AssertionError(job)
                time.sleep(.02)
            raise TimeoutError('Native cancellation test timed out')

        try:
            service.register({'id': 'native-stop-image', 'path': str(Path(args.image_dir).resolve()),
                              'device': 'CPU', 'memory_gib': 2, 'keep_image_model_loaded': True,
                              'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                              'generation': {'width': 256, 'height': 256, 'steps': 100}})
            body = {'model': 'native-stop-image', 'prompt': 'A mountain lake'}
            receipt = api('/api/image/jobs', body, key='native-stop-test')
            progress = until(receipt['id'], lambda j: bool(j.get('progress')))
            started = time.monotonic()
            assert api(receipt['cancel_url'], {})['cancel_requested']
            job = until(receipt['id'], lambda j: j['status'] in TERMINAL, seconds=15)
            assert job['status'] == 'cancelled', job
            assert not job['artifacts']
            assert not service.scheduler.resident
            assert not service.scheduler.busy
            assert not workers[0].process.is_alive()
            cancellation_seconds = time.monotonic() - started
            for path, payload, key in [(receipt['result_url'], None, None),
                                       ('/api/image/jobs', body, 'native-stop-test')]:
                try:
                    api(path, payload, key)
                    raise AssertionError('Cancelled request was accepted as new work')
                except urllib.error.HTTPError as error:
                    assert error.code == 410
                    assert error.headers['X-Should-Retry'] == 'false'
                    assert json.load(error)['x_status'] == 'cancelled'
            assert len(workers) == 1
            # Deliberately new work must load cleanly after the stopped worker exits.
            fresh = api('/api/image/jobs', {**body, 'steps': 1}, key='native-fresh-test')
            completed = until(fresh['id'], lambda j: j['status'] in TERMINAL)
            assert completed['status'] == 'completed', completed
            with urllib.request.urlopen(base+completed['artifacts'][0]['url'], timeout=5) as response:
                assert response.read().startswith(b'\x89PNG')
            assert len(workers) == 2
            evidence = {'version': __version__, 'runtime': service.detected,
                        'scope': 'Real tiny classic diffusion on CPU, not full Qwen/GPU validation',
                        'step_at_stop': progress['progress'], 'cancellation_seconds': cancellation_seconds,
                        'cancelled_status': job['status'], 'cancelled_worker_exited': True,
                        'keyed_replay_http': 410, 'native_runs': len(workers),
                        'fresh_request_status': completed['status']}
            Path(args.evidence).write_text(json.dumps(evidence, indent=2))
            print(json.dumps(evidence, indent=2))
        finally:
            http.shutdown()
            http.server_close()
            service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--evidence', required=True)
    run(parser.parse_args())
