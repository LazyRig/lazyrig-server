"""Opt-in integration test: uses supplied OpenVINO exports, never downloads weights.
Example: python tests/smoke_openvino.py --llm-dir /models/chat --image-dir /models/image
The tiny Stable Diffusion fixture checks the pipeline, not image quality.
"""
import argparse
import copy
import io
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-smoke-') as directory:
        config = copy.deepcopy(SERVER_DEFAULT)
        config['data_dir'] = directory
        config['devices'] = [{'id': args.device, 'name': args.device, 'policy': 'exclusive', 'budget_gib': args.budget}]
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:' + str(http.server_port)
        def request(path, body, headers=None):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **(headers or {})})
            try:
                return urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(exc.read().decode()) from exc
        def api(path, body, headers=None):
            with request(path, body, headers) as response:
                return json.load(response)
        try:
            for mid, path in [('smoke-chat', args.llm_dir), ('smoke-image', args.image_dir)]:
                api('/api/models', {'id': mid, 'path': str(Path(path).resolve()), 'device': args.device,
                    'memory_gib': args.reservation, 'context_limit': 2048,
                    'plugin_config': {'INFERENCE_NUM_THREADS': 2} if args.device == 'CPU' else {}})
            if args.image_runtime_profile:
                service.config['models'][-1]['runtime_profile'] = args.image_runtime_profile
            body = {'model': 'smoke-chat', 'messages': [{'role': 'user', 'content': 'Name the capital of France. Answer briefly.'}], 'max_tokens': 24, 'temperature': 0}
            chat = api('/v1/chat/completions', body)
            assert chat['x_mode'] == 'openvino', chat
            assert chat['x_context']['accuracy'] == 'exact', chat
            assert chat['usage']['prompt_tokens'] > 0 and chat['usage']['completion_tokens'] > 0, chat
            preflight = api('/api/context', {'model': 'smoke-chat', 'messages': body['messages'], 'max_tokens': 24})
            assert preflight['input_tokens'] == chat['usage']['prompt_tokens'], (preflight, chat)
            with request('/v1/chat/completions', {**body, 'stream': True, 'stream_options': {'include_usage': True}}) as response:
                jid = response.headers['X-Job-Id']
                records = [line[6:] for line in response.read().decode().splitlines() if line.startswith('data: ')]
            assert records[-1] == '[DONE]', records
            chunks = [json.loads(line) for line in records[:-1]]
            assert not any('error' in c for c in chunks), chunks
            streamed = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
            assert streamed == service.store.job(jid)['content']
            assert chunks[-1]['usage'] == chat['usage'], (chunks[-1], chat)
            image_id = uuid.uuid4().hex
            expected_url = base + '/artifacts/' + image_id + '.png'
            image = api('/v1/images/generations', {'model': 'smoke-image', 'prompt': 'A mountain lake', 'size': '256x256', 'steps': args.steps},
                        {'Idempotency-Key': image_id})
            assert not image['x_warning'], image
            assert image['x_job_id'] == image_id and image['data'][0]['url'] == expected_url, image
            with urllib.request.urlopen(expected_url, timeout=30) as response:
                downloaded_image = response.read()
            # Image delivery can precede native cleanup and LLM restoration.
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                job = service.store.job(image['x_job_id'])
                if job['status'] in {'completed', 'failed', 'cancelled', 'interrupted'}:
                    assert job['status'] == 'completed', job
                    break
                time.sleep(.05)
            else:
                raise TimeoutError('Image model cleanup/restoration did not finish')
            lifecycle = [(e['kind'], e.get('model')) for e in service.store.events(image['x_job_id']) if e['kind'] in {'loading', 'loaded', 'unloaded'}]
            assert ('unloaded', 'smoke-chat') in lifecycle and lifecycle[-1] == ('loaded', 'smoke-chat'), lifecycle
            from PIL import Image
            artifact = next((Path(directory) / 'artifacts').glob('*.png'))
            with Image.open(artifact) as im:
                assert im.size == (256, 256) and im.format == 'PNG'
            with Image.open(io.BytesIO(downloaded_image)) as im:
                assert im.size == (256, 256) and im.format == 'PNG'
            assert downloaded_image == artifact.read_bytes()
            resumed = api('/v1/chat/completions', {**body, 'max_tokens': 12})
            evidence = {'runtime': service.detected, 'chat': chat, 'stream_text_matches': True, 'stream_usage_matches': True,
                        'image_format': 'PNG', 'image_size': [256, 256], 'predicted_url_matches': True,
                        'http_download_matches_saved_png': True, 'lifecycle': lifecycle, 'resumed': resumed,
                        'image_runtime_profile': service.config['models'][-1].get('runtime_profile', 'default')}
            if args.evidence:
                Path(args.evidence).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            print('PASS: real chat, exact preflight/runtime counts, SSE text/usage, PNG generation, unload/reload, resumed chat.')
            print('Runtime:', service.detected['openvino_version'], 'Device:', args.device)
            print('Chat usage:', chat['usage'])
            print('Lifecycle:', lifecycle)
        finally:
            http.shutdown()
            http.server_close()
            thread.join(3)
            service.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--budget', type=float, default=8)
    parser.add_argument('--reservation', type=float, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--evidence')
    parser.add_argument('--image-runtime-profile', choices=['default', 'qwen-image-21'], help='Optional override to test interpreter isolation with a small image export')
    run(parser.parse_args())
