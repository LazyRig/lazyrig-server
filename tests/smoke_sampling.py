"""Native CPU sampling + HTTP test using an existing tiny LLM export; no download."""
import argparse
import copy
import json
import math
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-dir', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': temp,
                  'devices': [{'id': 'CPU', 'name': 'CPU', 'policy': 'exclusive', 'budget_gib': 8}]}
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        base = 'http://127.0.0.1:' + str(http.server_port)
        def request(path, body=None):
            return urllib.request.urlopen(urllib.request.Request(base + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type': 'application/json'}), timeout=120)
        def api(path, body=None):
            with request(path, body) as response: return json.load(response)
        def check(settings, expected):
            assert settings['source'] == 'openvino' and settings['do_sample'] is True, settings
            for key, value in expected.items():
                assert math.isclose(settings[key], value, abs_tol=1e-6), (key, settings, expected)
        try:
            defaults = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
                        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repetition_penalty': 1.0}
            api('/api/models', {'id': 'sampling-chat', 'path': str(Path(args.llm_dir).resolve()),
                'device': 'CPU', 'memory_gib': 1, 'context_limit': 2048, 'generation': defaults,
                'plugin_config': {'INFERENCE_NUM_THREADS': 2}})
            body = {'model': 'sampling-chat', 'messages': [{'role': 'user', 'content': 'Say hello.'}], 'max_tokens': 8}
            first = api('/v1/chat/completions', body); check(first['x_sampling'], defaults)
            overrides = {'temperature': .7, 'top_p': .8, 'top_k': 7, 'min_p': .1,
                         'presence_penalty': 1.5, 'frequency_penalty': .25}
            with request('/v1/chat/completions', {**body, **overrides, 'stream': True}) as response:
                jid = response.headers['X-Job-Id']
                lines = [line[6:] for line in response.read().decode().splitlines() if line.startswith('data: ')]
            assert lines[-1] == '[DONE]'
            chunks = [json.loads(line) for line in lines[:-1]]
            assert not any('error' in chunk for chunk in chunks), chunks
            actual = chunks[-1]['x_sampling']; check(actual, {**defaults, **overrides})
            recovered = api('/api/jobs/' + jid + '/result')
            assert recovered['x_sampling'] == actual
            streamed = ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks if chunk['choices'])
            assert streamed == recovered['choices'][0]['message']['content']
            third = api('/v1/chat/completions', body); check(third['x_sampling'], defaults)
            evidence = {'runtime': service.detected, 'checks': ['native sampled generation', 'request overrides including min_p/presence/frequency',
                        'SSE and saved result match', 'warm worker does not leak request overrides'],
                        'default_sampling': first['x_sampling'], 'override_sampling': actual,
                        'restored_sampling': third['x_sampling'], 'full_qwen_quality_tested': False}
            Path(args.evidence).write_text(json.dumps(evidence, indent=2))
            print(json.dumps(evidence, indent=2))
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()


if __name__ == '__main__': main()
