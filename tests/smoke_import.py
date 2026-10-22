"""Opt-in network test: HF download -> conversion -> registration -> real chat.

This downloads weights and saves an export. Supply a new destination directory.
Example: python tests/smoke_import.py --url https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct --destination /models/smollm-test
"""
import argparse
import copy
import json
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.service import Service
from lazyrig.server import handler


def main(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-import-smoke-') as temporary:
        cfg = copy.deepcopy(SERVER_DEFAULT)
        cfg['data_dir'] = temporary
        service = Service(cfg)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        def api(path, body):
            req = urllib.request.Request(f'http://127.0.0.1:{http.server_port}'+path,
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=900) as response:
                return json.load(response)
        try:
            job = api('/api/downloads', {'url': args.url, 'destination': str(Path(args.destination).resolve()),
                                        'register': True, 'id': 'import-smoke', 'device': 'CPU',
                                        'precision': args.precision, 'memory_gib': 2})
            end, last = time.monotonic() + 1800, None
            while time.monotonic() < end:
                state = next(j for j in service.downloads.snapshot() if j['id'] == job['id'])
                progress = (state['status'], state.get('file'), state.get('bytes', 0)//(32*1024**2))
                if progress != last:
                    print(state['status'], state.get('file'), round(state['bytes']/1024**2, 1), 'MiB', flush=True)
                    last = progress
                if state['status'] in {'failed', 'cancelled', 'completed'}:
                    break
                time.sleep(.5)
            assert state['status'] == 'completed', state
            assert state['converted'] and state['model_id'] == 'import-smoke', state
            api('/api/settings', {'model': 'import-smoke', 'plugin_config': {'INFERENCE_NUM_THREADS': 2}})
            messages = [{'role': 'user', 'content': 'What is two plus two? Answer briefly.'}]
            preflight = api('/api/context', {'model': 'import-smoke', 'messages': messages, 'max_tokens': 24})
            chat = api('/v1/chat/completions', {'model': 'import-smoke', 'messages': messages, 'max_tokens': 24, 'temperature': 0})
            assert chat['x_mode'] == 'openvino' and chat['choices'][0]['message']['content'], chat
            assert preflight['input_tokens'] == chat['usage']['prompt_tokens'], (preflight, chat)
            evidence = {'python': sys.version, 'import': state, 'chat': chat}
            if args.evidence:
                Path(args.evidence).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            print('PASS: downloaded HF source, exported OpenVINO, registered automatically, generated real CPU chat with matching prompt counts.', flush=True)
            print(chat['choices'][0]['message']['content'])
        finally:
            http.shutdown(); http.server_close(); service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--precision', choices=['fp16', 'int8', 'int4'], default='fp16')
    parser.add_argument('--evidence')
    main(parser.parse_args())
