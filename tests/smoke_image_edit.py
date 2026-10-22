"""Native image-edit HTTP smoke test with supplied exports; no weight download."""
import argparse
import base64
import copy
import hashlib
import io
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
from PIL import Image
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service
from test_image_edits import multipart


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-edit-native-') as directory:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': directory,
                  'devices': [{'id': 'CPU', 'name': 'CPU', 'policy': 'exclusive', 'budget_gib': 8}]}
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        base = 'http://127.0.0.1:' + str(http.server_port)

        def api(path, body=None, content_type='application/json'):
            data = body if isinstance(body, bytes) else None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(base + path, data, {'Content-Type': content_type})
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(exc.read().decode()) from exc

        def complete(receipt):
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                job = api('/api/jobs/' + receipt['id'])
                if job['status'] in {'completed', 'failed', 'cancelled'}:
                    assert job['status'] == 'completed', job
                    return job
                time.sleep(.1)
            raise TimeoutError('Native edit did not complete')

        def checked(result):
            assert not result['x_warning'], result
            inference=result['x_inference']
            assert inference['denoising']['step']==inference['denoising']['steps']>0, inference
            assert inference['seed']==result['x_seed']==result['x_artifacts'][0]['seed'], result
            assert inference['returned_normally'] and inference['visual_verification']=='not_performed'
            raw = base64.b64decode(result['data'][0]['b64_json'])
            with Image.open(io.BytesIO(raw)) as im:
                assert im.size == (256, 256) and im.format == 'PNG'
            return {'job_id': result['x_job_id'], 'operation': result['x_operation'],
                    'output_sha256': hashlib.sha256(raw).hexdigest(), 'artifact': result['x_artifacts'][0]}

        try:
            api('/api/models', {'id': 'native-image', 'path': str(Path(args.image_dir).resolve()),
                'device': 'CPU', 'memory_gib': 2, 'keep_image_model_loaded': True, 'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                'generation': {'width': 256, 'height': 256, 'steps': 6, 'strength': .8}})
            image_meta = next(m for m in api('/v1/models')['data'] if m['id'] == 'native-image')
            assert image_meta['capabilities']['image_editing'], image_meta
            fields = [('model', 'native-image'), ('prompt', 'Add blue clouds.'), ('size', 'auto'), ('n', '1')]
            buffer = io.BytesIO(); Image.new('RGB', (256, 256), 'red').save(buffer, format='PNG')
            content_type, upload = multipart(fields, [('image', buffer.getvalue())])
            generation = {'model': 'native-image', 'prompt': 'A mountain lake', 'size': 'auto', 'response_format': 'b64_json', 'seed': 42}
            before = checked(api('/v1/images/generations', generation))
            first = checked(api('/v1/images/edits', upload, content_type))
            source = first['artifact']['url']
            receipt = api('/api/image/edits', {'model': 'native-image', 'prompt': 'Add warm light.', 'image': base + source})
            complete(receipt)
            second = checked(api(receipt['result_url']))
            after = checked(api('/v1/images/generations', generation))
            assert before['output_sha256'] == after['output_sha256']
            assert first['output_sha256'] != before['output_sha256']
            events = {r['job_id']: [(e['kind'], e.get('model')) for e in service.store.events(r['job_id'])
                                  if e['kind'] in {'loading', 'loaded', 'unloaded'}]
                      for r in (before, first, second, after)}
            assert events[first['job_id']] == [('unloaded', 'native-image'), ('loading', 'native-image'), ('loaded', 'native-image')]
            assert events[second['job_id']] == []
            assert events[after['job_id']] == events[first['job_id']]
            evidence = {'runtime': service.detected, 'scope': 'Native tiny Stable Diffusion on CPU; execution, not edit quality or full Qwen validation',
                        'image_metadata': image_meta, 'outputs': [before, first, second, after], 'mode_switch_events': events}
            if args.llm_dir:
                api('/api/settings', {'model':'native-image','keep_image_model_loaded':False,'restore_text_after_image':True})
                api('/api/models', {'id': 'native-chat', 'path': str(Path(args.llm_dir).resolve()), 'device': 'CPU',
                    'context_limit': 2048, 'memory_gib': 2, 'plugin_config': {'INFERENCE_NUM_THREADS': 2}})
                chat = {'model': 'native-chat', 'messages': [{'role': 'user', 'content': 'Hello.'}], 'max_tokens': 4, 'temperature': 0}
                initial = api('/v1/chat/completions', chat)
                handoff = checked(api('/v1/images/edits', upload, content_type))
                complete({'id': handoff['job_id']})
                lifecycle = [(e['kind'], e.get('model')) for e in service.store.events(handoff['job_id'])
                             if e['kind'] in {'loading', 'loaded', 'unloaded'}]
                assert lifecycle == [('unloaded', 'native-chat'), ('loading', 'native-image'), ('loaded', 'native-image'),
                                     ('unloaded', 'native-image'), ('loading', 'native-chat'), ('loaded', 'native-chat')], lifecycle
                resumed = api('/v1/chat/completions', chat)
                assert resumed['usage']['completion_tokens'] > 0
                assert initial['choices'] == resumed['choices']
                evidence.update(restoration_events=lifecycle, resumed_chat=resumed, handoff_edit=handoff)
            Path(args.evidence).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            print('PASS: native multipart edit, RGB input, PNG output, artifact reuse, asynchronous retrieval, generation/edit switching and optional chat restoration.')
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', required=True, help='Complete classic diffusion export with VAE encoder')
    parser.add_argument('--llm-dir', help='Optional text or vision-language export for restoration check')
    parser.add_argument('--evidence', default='native-image-edit-evidence.json')
    run(parser.parse_args())
