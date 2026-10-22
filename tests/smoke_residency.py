"""Optional native check with supplied exports; no weight downloads or quality claim."""
import argparse
import copy
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service
from lazyrig.terminal import LOGGER, configure_logging


def run(args):
    configure_logging()
    output = io.StringIO()
    capture = logging.StreamHandler(output)
    LOGGER.addHandler(capture)
    try:
        with tempfile.TemporaryDirectory(prefix='lazyrig-residency-') as directory:
            config = copy.deepcopy(SERVER_DEFAULT)
            config.update(data_dir=directory, terminal_metrics=True)
            config['devices'] = [{'id': args.device, 'name': args.device, 'policy': 'budget', 'budget_gib': args.budget}]
            service = Service(config)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def api(path, body):
                request = urllib.request.Request('http://127.0.0.1:' + str(server.server_port) + path,
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=600) as response:
                        return json.load(response)
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(exc.read().decode()) from exc
            def workers():
                return {mid: worker for mid, (_, worker) in service.scheduler.resident.items()}
            try:
                for mid, path in [('chat-a', args.llm_dir), ('chat-b', args.llm_dir), ('image', args.image_dir)]:
                    api('/api/models', {'id': mid, 'path': str(Path(path).resolve()), 'device': args.device,
                        'memory_gib': args.reservation, 'context_limit': 2048,
                        'plugin_config': {'INFERENCE_NUM_THREADS': 2} if args.device == 'CPU' else {}})
                api('/api/settings', {'model': 'image', 'keep_image_model_loaded': True})
                body = {'messages': [{'role': 'user', 'content': 'Say hello.'}], 'temperature': 0,
                        'max_tokens': 16, 'auto_image': False}
                chats = [api('/v1/chat/completions', {**body, 'model': mid}) for mid in ('chat-a', 'chat-b')]
                original = workers()
                assert set(original) == {'chat-a', 'chat-b'}, original
                images = []
                for _ in range(2):
                    images.append(api('/v1/images/generations', {'model': 'image', 'prompt': 'A mountain lake',
                        'size': '256x256', 'steps': 2}))
                    assert not images[-1]['x_warning'], images[-1]
                    current = workers()
                    assert set(current) == {'chat-a', 'chat-b', 'image'}, current
                    assert all(current[mid] is worker for mid, worker in original.items())
                    original = current
                chats.append(api('/v1/chat/completions', {**body, 'model': 'chat-a'}))
                assert all(workers()[mid] is worker and worker.process.is_alive() for mid, worker in original.items())
                for chat in chats:
                    assert chat['x_mode'] == 'openvino' and chat['x_context']['accuracy'] == 'exact', chat
                    assert chat['usage']['prompt_tokens'] > 0 and chat['usage']['completion_tokens'] > 0, chat
                from PIL import Image
                artifacts = list((Path(directory) / 'artifacts').glob('*.png'))
                assert len(artifacts) == 2, artifacts
                for artifact in artifacts:
                    with Image.open(artifact) as image:
                        assert image.size == (256, 256) and image.format == 'PNG'
                terminal = output.getvalue()
                assert ' t/s (native)' in terminal and 'context ' in terminal and 'image step ' in terminal, terminal
                assert ' t/s (TTFT estimate)' in terminal and 'prompt n/a' not in terminal, terminal
                evidence = {'scope': 'Native CPU/device pipeline and residency; tiny fixtures do not test model quality',
                    'runtime': service.detected, 'device': args.device,
                    'retained_workers': service.scheduler.snapshot()['workers'],
                    'chat_usage': [chat['usage'] for chat in chats], 'png_images': 2,
                    'same_workers_reused': True, 'terminal': terminal}
            finally:
                server.shutdown(); server.server_close(); thread.join(3); service.close()
            assert all(not worker.process.is_alive() for worker in original.values())
            evidence['all_workers_closed'] = True
            if args.evidence:
                Path(args.evidence).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            print('PASS: two native LLM workers and an image worker retained, reused and closed; terminal metrics; two PNGs.')
    finally:
        LOGGER.removeHandler(capture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--budget', type=float, default=8)
    parser.add_argument('--reservation', type=float, default=2)
    parser.add_argument('--evidence')
    run(parser.parse_args())
