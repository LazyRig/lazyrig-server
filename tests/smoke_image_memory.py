"""Measure repeated real image requests with supplied exports (optional psutil)."""
import argparse
import base64
import copy
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from PIL import Image
from lazyrig import __version__
from lazyrig.compat import start_image
from lazyrig.config import SERVER_DEFAULT
from lazyrig.service import Service
from lazyrig.worker import ProcessWorker


def observed_pid(pid):
    # Containers may expose host /proc while Python runs in a nested PID
    # namespace. Match that namespace, never assume an unrelated host PID.
    if sys.platform == 'linux':
        namespace = Path('/proc/self/ns/pid').stat().st_ino
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal(): continue
            try:
                if (entry / 'ns/pid').stat().st_ino != namespace: continue
                row = next(line for line in (entry / 'status').read_text().splitlines() if line.startswith('NSpid:'))
                if int(row.split()[-1]) == pid: return int(entry.name)
            except (OSError, StopIteration): continue
        raise RuntimeError('Cannot locate owned process in /proc: ' + str(pid))
    return pid


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-memory-') as directory:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': directory}
        service = Service(config)
        processes = []
        def factory(*a, **kw):
            worker = ProcessWorker(*a, **kw)
            process = worker.process
            pid = getattr(worker, 'worker_pid', None) or getattr(process, 'pid', None) or process.process.pid
            processes.append(psutil.Process(observed_pid(pid)))
            return worker
        service.scheduler.factory = factory
        parent = psutil.Process(observed_pid(os.getpid()))
        def alive():
            return [p for p in processes if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
        def rss(process):
            try: return process.memory_info().rss
            except psutil.NoSuchProcess: return 0
        def complete(job):
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                current = service.store.job(job['id'])
                if current['status'] in {'completed', 'failed', 'cancelled'} and job['id'] not in service.running:
                    assert current['status'] == 'completed', current['error']
                    return current
                time.sleep(.05)
            raise TimeoutError('Request did not finish')
        try:
            service.register({'id': 'memory-image', 'path': str(Path(args.image_dir).resolve()), 'device': 'CPU',
                'memory_gib': 2, 'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                'generation': {'width': 256, 'height': 256, 'steps': 6, 'strength': .8}})
            source = io.BytesIO(); Image.new('RGB', (256, 256), 'red').save(source, format='PNG')
            request = {'model': 'memory-image', 'prompt': 'Add blue clouds.',
                       'image': 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()}
            samples = []
            for index in range(args.requests):
                editing = index % 3 != 2
                body = request if editing else {k: v for k, v in request.items() if k != 'image'}
                job = complete(start_image(service, body, editing=editing))
                live = alive()
                sample = {'request': index + 1, 'operation': job['operation'], 'parent_rss_bytes': rss(parent),
                          'live_model_pids': [p.pid for p in live], 'model_rss_bytes': sum(rss(p) for p in live),
                          'registered_residents': list(service.scheduler.resident)}
                samples.append(sample)
                assert bool(live) == args.expect_retained, sample
                assert bool(service.scheduler.resident) == args.expect_retained, sample
                artifact = Path(directory) / 'artifacts' / job['artifacts'][0]['name']
                with Image.open(artifact) as im: assert im.format == 'PNG' and im.size == (256, 256)
                print(json.dumps(sample), flush=True)
            growth = max(s['parent_rss_bytes'] for s in samples[1:]) - samples[0]['parent_rss_bytes']
            assert growth < 64 * 1024**2, ('Unexpected parent memory growth', growth)
            service.scheduler.close()
            assert not alive(), [p.pid for p in alive()]
            evidence = {'version': __version__, 'runtime': service.detected, 'expected_retained': args.expect_retained,
                        'scope': 'Actual tiny diffusion CPU inference; full Qwen and Windows not measured',
                        'samples': samples, 'parent_rss_growth_after_first_bytes': growth,
                        'tracked_pids': [p.pid for p in processes], 'live_model_pids_after_final_unload': []}
            Path(args.evidence).write_text(json.dumps(evidence, indent=2))
        finally:
            service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--requests', type=int, default=6)
    parser.add_argument('--expect-retained', action='store_true', help='Reproduce 0.3.7 default retention')
    parser.add_argument('--evidence', required=True)
    run(parser.parse_args())
