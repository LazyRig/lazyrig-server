"""Native tiny LLM/VLM JSON enhancement, with a fixture image worker.

The JSON fields are constrained to fixed test values: this verifies the native
structured-output/vision transport, not semantic rewriting or image quality.
"""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.service import Service
from lazyrig.worker import ProcessWorker
from lazyrig import prompt_enhancement
from lazyrig.compat import start_chat, completion
from support import FixtureWorker
from test_chat_image_tools import source_image
from test_image_edits import export_fixture
from test_core import wait_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-dir', required=True)
    parser.add_argument('--vlm-dir', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    original_format = prompt_enhancement.response_format

    def constrained(editing):
        value = original_format(editing)
        # Tiny random exports cannot follow rewriting instructions. Single-word
        # values exercise JSON transport without a multiword enum/tokenizer test.
        known = {'rewritten_prompt': 'blue', 'wh_ratio': '1:1', **({'ratio_follow': ''} if editing else {})}
        for key, expected in known.items():
            value['json_schema']['schema']['properties'][key]['enum'] = [expected]
        return value

    with tempfile.TemporaryDirectory() as directory:
        config = {**copy.deepcopy(SERVER_DEFAULT), 'data_dir': directory, 'request_timeout_seconds': 180,
                  'devices': [{'id': 'CPU', 'name': 'CPU', 'policy': 'exclusive', 'budget_gib': 8}]}
        native_results = []
        def factory(mode, model, timeout, cancel):
            cls = FixtureWorker if model['pipeline'] == 'image' else ProcessWorker
            worker = cls(mode, model, timeout, cancel)
            if cls is ProcessWorker:
                native_run = worker.run
                def run(*args, **kwargs):
                    for item in native_run(*args, **kwargs):
                        if item['kind'] == 'result':
                            native_results.append({'tokens': item.get('generated_tokens'), 'text': item.get('text', '')[:500]})
                        yield item
                worker.run = run
            return worker
        service = Service(config, worker_factory=factory)
        evidence = {'scope': 'Native CPU LLM/VLM with fixed JSON test values and fixture image worker; not semantic or image-quality validation',
                    'exports': {'llm': str(Path(args.llm_dir).resolve()), 'vlm': str(Path(args.vlm_dir).resolve())}, 'checks': []}
        try:
            service.register({'id': 'native-llm', 'path': str(Path(args.llm_dir).resolve()), 'device': 'CPU',
                'memory_gib': 1, 'context_limit': 32768, 'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                'generation': {'temperature': 0}})
            service.register({'id': 'native-vlm', 'path': str(Path(args.vlm_dir).resolve()), 'device': 'CPU',
                'memory_gib': 1, 'context_limit': 32768, 'plugin_config': {'INFERENCE_NUM_THREADS': 2},
                'generation': {'temperature': 0}})
            service.config['models'].append({'id': 'fixture-image', 'name': 'Fixture image', 'task': 'text-to-image',
                'pipeline': 'image', 'device': 'CPU', 'path': str(export_fixture(Path(directory)/'image')),
                'memory_gib': 1, 'max_width': 1024, 'max_height': 1024,
                'generation': {'width': 256, 'height': 256, 'steps': 1}, 'plugin_config': {}})
            for editing, model in [(False, 'native-llm'), (True, 'native-vlm')]:
                content = 'Make it blue.' if editing else 'A bicycle.'
                if editing:
                    content = [{'type': 'text', 'text': content}, {'type': 'image_url', 'image_url': {'url': source_image()}}]
                body = {'model': model, 'messages': [{'role': 'system', 'content': prompt_enhancement.instructions(editing)},
                    {'role': 'user', 'content': content}], 'response_format': constrained(editing),
                    'reasoning_effort': 'none', 'stop': [], 'max_tokens': 2048}
                receipt = start_chat(service, body, preflight=False)
                job = wait_job(service, receipt['id'], timeout=240)
                enhanced = None
                if job['status'] == 'completed':
                    choice = completion(job, body)['choices'][0]
                    assert choice['finish_reason'] != 'length'
                    enhanced = prompt_enhancement.parse(choice['message']['content'], editing)
                    image = service.submit_image({'model': 'fixture-image', 'prompt': enhanced['rewritten_prompt'],
                        'width': 256, 'height': 256, **({'image': source_image()} if editing else {})}, editing=editing)
                    saved = wait_job(service, image['id'])
                    assert saved['status'] == 'completed' and len(saved['artifacts']) == 1, saved['error']
                evidence['checks'].append({'operation': 'edit' if editing else 'generate',
                    'model': model, 'status': job['status'], 'error': job['error'],
                    'native_result': native_results[-1:] or None,
                    'enhancement': enhanced, 'usage': job.get('usage')})
            Path(args.evidence).write_text(json.dumps(evidence, indent=2))
            print(json.dumps({'checks': [c['operation'] for c in evidence['checks']], 'scope': evidence['scope']}))
            assert all(c['status'] == 'completed' for c in evidence['checks']), evidence['checks']
        finally:
            service.close()


if __name__ == '__main__':
    main()
