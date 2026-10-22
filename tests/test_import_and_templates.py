"""Import lifecycle and template regressions; fake exports are labelled fixtures."""
import copy
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from support import DEFAULT
from lazyrig.context import TokenizerService
from lazyrig.conversion import conversion_task, select_profile, run_logged
from lazyrig.downloads import hf_files, hf_documents
from support import Service
from lazyrig.templates import render_chat
from lazyrig.worker import Cancelled, ProcessWorker
from test_server_contract import fixture_export


class TemplateTests(unittest.TestCase):
    def test_undefined_thinking_and_exact_prompt_ids(self):
        fragment = """{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- set reasoning_instructions = resolved_reasoning_effort %}
{%- endif %}
{{ bos_token }}{{ reasoning_instructions }}:{{ messages[-1].content }}{{ eos_token }}"""
        class Tokenizer:
            def __init__(self, path): pass
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("Minja must not be called")
            def encode(self, text, add_special_tokens):
                assert add_special_tokens is False
                row = types.SimpleNamespace(tolist=lambda: list(text.encode()))
                return types.SimpleNamespace(input_ids=types.SimpleNamespace(data=[row]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'chat_template.jinja').write_text(fragment)
            (root / 'tokenizer_config.json').write_text(json.dumps({'bos_token': {'content': '<s>'}, 'eos_token': '</s>'}))
            model = {'path': temp, 'pipeline': 'llm', 'context_limit': 4096}
            service = TokenizerService('openvino')
            with patch.dict(sys.modules, {'openvino_genai': types.SimpleNamespace(Tokenizer=Tokenizer)}):
                for settings, expected in [({}, '<s>xhigh:Hola</s>'), ({'enable_thinking': False}, '<s>:Hola</s>'),
                                           ({'enable_thinking': True, 'reasoning_effort': 'low'}, '<s>low:Hola</s>')]:
                    prepared = service.prepare({**model, 'template_context': settings}, [{'role': 'user', 'content': 'Hola'}], 20, False, add_system=False)
                    self.assertEqual(prepared['text'], expected)
                    self.assertEqual(prepared['token_ids'], list(expected.encode()))
                    self.assertEqual(prepared['context']['input_tokens'], len(prepared['token_ids']))
                    self.assertEqual(prepared['context']['accuracy'], 'exact')

    def test_named_templates_generation_json_and_runtime_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fallback = types.SimpleNamespace(get_original_chat_template=lambda: '{{ messages[0].content }}')
            self.assertEqual(render_chat(root, fallback, [{'content': 'runtime template'}]), 'runtime template')
            (root / 'tokenizer_config.json').write_text(json.dumps({'chat_template': [
                {'name': 'default', 'template': 'default'},
                {'name': 'tool_use', 'template': '{% generation %}{{ tools|tojson }}{% endgeneration %}'}]}))
            self.assertEqual(json.loads(render_chat(root, fallback, [], [{'name': 'café'}])), [{'name': 'café'}])
            self.assertEqual(render_chat(root, fallback, []), 'default')
            (root / 'chat_template.jinja').write_text('file wins')
            self.assertEqual(render_chat(root, fallback, []), 'file wins')

    def test_tool_arguments_are_objects_without_mutating_history(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, 'chat_template.jinja').write_text('{{ messages[0].tool_calls[0].function.arguments.prompt }}')
            messages = [{'role': 'assistant', 'tool_calls': [{'function': {'arguments': '{"prompt":"a lake"}'}}]}]
            self.assertEqual(render_chat(temp, object(), messages), 'a lake')
            self.assertIsInstance(messages[0]['tool_calls'][0]['function']['arguments'], str)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = str(self.root / 'data')
        self.service = Service(config)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def remote(self, files, pipeline_tag='text-generation'):
        manifest = {'sha': 'pinned-commit', 'pipeline_tag': pipeline_tag,
                    'siblings': [{'rfilename': name, 'size': len(data)} for name, data in files.items()]}
        def open_url(request, **kwargs):
            url = request.full_url
            if '/api/models/' in url:
                data = json.dumps(manifest).encode()
            else:
                self.assertIn('/resolve/pinned-commit/', url)
                name = url.split('/resolve/pinned-commit/')[1].split('?')[0]
                data = files[name]
            response = io.BytesIO(data)
            response.headers = {'Content-Length': str(len(data))}
            return response
        return patch('lazyrig.downloads.urllib.request.urlopen', side_effect=open_url)

    def wait(self, jid):
        deadline = time.monotonic() + 10
        while jid in self.service.downloads.running and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertNotIn(jid, self.service.downloads.running)
        return next(j for j in self.service.downloads.snapshot() if j['id'] == jid)

    def test_source_download_convert_validate_register(self):
        files = {'config.json': b'{"model_type":"llama"}', 'model.safetensors': b'fixture, not actual weights',
                 'tokenizer_config.json': b'{}', 'modeling_remote.py': b'raise Exception("must not run")',
                 'LICENSE': b'Original model license', 'NOTICE': b'Original attribution',
                 'README.md': b'Model card with usage restrictions', 'USE_POLICY.md': b'Model use policy',
                 'LICENSE.py': b'raise Exception("must not download code disguised as a license")'}
        destination = self.root / 'converted'
        def export(source, output, profile, task, precision, cancel, log, logfile, phase):
            self.assertEqual(profile, 'standard')
            self.assertEqual(task, 'text-generation-with-past')
            self.assertEqual(precision, 'int4')
            self.assertFalse((source / 'modeling_remote.py').exists())
            self.assertFalse((source / 'LICENSE.py').exists())
            self.assertEqual((source / 'model.safetensors').read_bytes(), files['model.safetensors'])
            phase('converting'); log('Test converter fixture\n'); fixture_export(output)
        with self.remote(files), patch('lazyrig.conversion.ensure_profile'), patch('lazyrig.conversion.convert', side_effect=export):
            job = self.service.downloads.start({'url': 'https://huggingface.co/test/source', 'destination': str(destination),
                                               'precision': 'int4', 'register': True, 'device': 'TEST.CPU', 'id': 'converted'})
            result = self.wait(job['id'])
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['model_id'], 'converted')
        self.assertEqual(self.service.model('converted')['path'], str(destination))
        self.assertEqual(self.service.model('converted')['memory_gib'], 48)
        self.assertEqual(json.loads((destination / 'lazyrig-export.json').read_text())['revision'], 'pinned-commit')
        self.assertFalse((destination / 'model.safetensors').exists())
        for name in ('LICENSE', 'NOTICE', 'README.md', 'USE_POLICY.md'):
            self.assertEqual((destination / 'lazyrig-source-documents' / name).read_bytes(), files[name])
        provenance = json.loads((destination / 'lazyrig-export.json').read_text())
        self.assertEqual(len(provenance['source_documents']), 4)
        self.assertNotIn('license', provenance)  # Preservation is not a licensing decision.
        self.assertFalse(list(self.root.glob('.lazyrig-download-*')))

    def test_existing_export_skips_converter(self):
        fixture_export(self.root / 'fixture')
        files = {p.name: p.read_bytes() for p in (self.root / 'fixture').iterdir()}
        with self.remote(files), patch('lazyrig.conversion.ensure_profile') as prepare, patch('lazyrig.conversion.convert') as convert:
            job = self.service.downloads.start({'url': 'https://huggingface.co/test/export', 'destination': str(self.root / 'export')})
            result = self.wait(job['id'])
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertFalse(result['converted'])
        prepare.assert_not_called(); convert.assert_not_called()

    def test_nested_export_keeps_ancestor_notices_and_separate_model_card(self):
        fixture_export(self.root / 'fixture')
        files = {'variants/int4/' + p.name: p.read_bytes() for p in (self.root / 'fixture').iterdir()}
        documents = {'LICENSE': b'Repository terms', 'README.md': b'Repository card',
                     'licenses/Apache-2.0.txt': b'License text', 'variants/NOTICE': b'Variant attribution',
                     'variants/int4/README.md': b'Export-specific instructions'}
        files.update(documents)
        files['variants/int8/LICENSE'] = b'Unselected variant terms'
        destination = self.root / 'nested'
        with self.remote(files), patch('lazyrig.conversion.convert') as convert:
            job = self.service.downloads.start({'url': 'https://huggingface.co/test/export',
                                               'destination': str(destination)})
            result = self.wait(job['id'])
        self.assertEqual(result['status'], 'completed', result.get('error'))
        convert.assert_not_called()
        for name, data in documents.items():
            self.assertEqual((destination / 'lazyrig-source-documents' / name).read_bytes(), data)
        self.assertFalse((destination / 'lazyrig-source-documents/variants/int8').exists())
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            hf_documents({'siblings': [{'rfilename': '../LICENSE'}]}, '')

    def test_zip_keeps_license_outside_enclosing_export_without_overwriting(self):
        fixture_export(self.root / 'fixture')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for path in (self.root / 'fixture').iterdir():
                archive.writestr('export/' + path.name, path.read_bytes())
            archive.writestr('LICENSE', b'Archive terms')
            archive.writestr('export/lazyrig-source-documents/LICENSE', b'Previous terms')
        response = io.BytesIO(buffer.getvalue())
        response.headers = {'Content-Length': str(len(buffer.getvalue()))}
        destination = self.root / 'zip-export'
        with patch('lazyrig.downloads.urllib.request.urlopen', return_value=response):
            job = self.service.downloads.start({'url': 'https://example.test/export.zip',
                                               'destination': str(destination)})
            result = self.wait(job['id'])
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual((destination / 'lazyrig-source-documents/LICENSE').read_bytes(), b'Previous terms')
        self.assertEqual((destination / 'lazyrig-source-documents-2/LICENSE').read_bytes(), b'Archive terms')

    def test_failed_or_cancelled_conversion_never_publishes_or_registers(self):
        files = {'config.json': b'{"model_type":"llama"}', 'model.safetensors': b'test'}
        for error, expected in [(RuntimeError('unsupported architecture'), 'failed'), (Cancelled(), 'cancelled')]:
            destination = self.root / expected
            with self.remote(files), patch('lazyrig.conversion.ensure_profile'), patch('lazyrig.conversion.convert', side_effect=error):
                job = self.service.downloads.start({'url': 'https://huggingface.co/test/source', 'destination': str(destination),
                                                   'register': True, 'device': 'TEST.CPU', 'id': expected})
                result = self.wait(job['id'])
            self.assertEqual(result['status'], expected)
            self.assertFalse(destination.exists())
            self.assertNotIn(expected, [m['id'] for m in self.service.config['models']])
            self.assertFalse(list(self.root.glob('.lazyrig-download-*')))

    def test_duplicate_destination_blocked_during_conversion(self):
        files = {'config.json': b'{}', 'model.safetensors': b'test'}
        entered = threading.Event()
        def exporting(source, output, profile, task, precision, cancel, *args):
            entered.set(); cancel.wait(5); raise Cancelled()
        body = {'url': 'https://huggingface.co/test/source', 'destination': str(self.root / 'pending')}
        with self.remote(files), patch('lazyrig.conversion.ensure_profile'), patch('lazyrig.conversion.convert', side_effect=exporting):
            job = self.service.downloads.start(body)
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(ValueError, 'already being downloaded'):
                self.service.downloads.start(body)
            self.service.downloads.cancel(job['id']); self.wait(job['id'])

    def test_qwen_profile_and_unsupported_formats(self):
        self.assertEqual(select_profile({'_class_name': 'QwenImage21Pipeline'}), 'qwen-image-21')
        self.assertEqual(conversion_task({}, {'_class_name': 'QwenImage21Pipeline'}, None), 'text-to-image')
        with self.assertRaisesRegex(ValueError, 'Safetensors'):
            hf_files({'siblings': [{'rfilename': 'model.gguf'}]}, '')
        with self.assertRaisesRegex(ValueError, 'currently supports'):
            conversion_task({}, {}, 'automatic-speech-recognition')


class ProcessTests(unittest.TestCase):
    def test_cancel_stops_live_converter_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            started = root / 'started'
            cancel = threading.Event()
            script = "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('started');print('Export started',flush=True);time.sleep(60)"
            errors, logs = [], []
            def run():
                try: run_logged([sys.executable, '-c', script, str(started)], root / 'log', cancel, logs.append)
                except BaseException as exc: errors.append(exc)
            thread = threading.Thread(target=run); thread.start()
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline: time.sleep(.01)
            self.assertTrue(started.exists()); cancel.set(); thread.join(8)
            self.assertFalse(thread.is_alive()); self.assertIsInstance(errors[0], Cancelled)

    def test_external_interpreter_ipc_and_termination(self):
        worker = ProcessWorker.__new__(ProcessWorker)
        cancel = threading.Event()
        worker._external(sys.executable, 'openvino', {'pipeline': 'llm', 'path': '/missing-contract-model', 'device': 'CPU'}, cancel)
        try:
            with self.assertRaises(RuntimeError):
                next(worker._receive(20, cancel))
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())


if __name__ == '__main__': unittest.main()
