"""Native CPU sampler replacement using a supplied small classic image export."""
import argparse
import base64
import copy
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from lazyrig.config import SERVER_DEFAULT
from lazyrig.service import Service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    source = Path(args.image_dir).resolve(); config_path = source / 'scheduler/scheduler_config.json'
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': directory,
                  'devices': [{'id': 'CPU', 'name': 'CPU', 'policy': 'exclusive', 'budget_gib': 8}]}
        service = Service(config); results = []
        try:
            service.register({'id': 'native-image', 'path': str(source), 'device': 'CPU', 'memory_gib': 2,
                              'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                              'diffusion_scheduler': {'type': 'DDIM', 'config': {}},
                              'generation': {'width': 320, 'height': 256, 'steps': 6, 'guidance_scale': 4, 'negative_prompt': 'blur'}})
            buffer = io.BytesIO(); Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG')
            for editing in (False, True):
                request = {'model': 'native-image', 'prompt': 'A blue lake with a mountain.'}
                if editing: request['image'] = buffer.getvalue()
                job = service.submit_image(request, editing=editing); deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    job = service.store.job(job['id'])
                    if job['status'] in {'completed', 'failed', 'cancelled'}: break
                    time.sleep(.1)
                assert job['status'] == 'completed', job.get('error')
                path = service.store.directory / 'artifacts' / job['artifacts'][0]['name']
                with Image.open(path) as image: assert image.size == (320, 256) and image.format == 'PNG'
                assert not service.scheduler.resident
                results.append({'operation': 'edit' if editing else 'generate', 'size': [320, 256],
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'workers_after': 0})
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == before
            evidence = {'scope': 'Native OpenVINO CPU inference with tiny Stable Diffusion and explicit DDIM sampler. Not full Qwen inference.',
                        'runtime': service.detected, 'scheduler': 'DDIM', 'export_scheduler_unchanged': True, 'results': results}
            Path(args.evidence).write_text(json.dumps(evidence, indent=2)); print(json.dumps(evidence, indent=2))
        finally: service.close()


if __name__ == '__main__': main()
