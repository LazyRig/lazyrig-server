"""Chat editing and enhancement contracts; controlled workers, not image-quality tests."""
import base64
import copy
import io
import json
import threading
import unittest
import urllib.error
from pathlib import Path

from PIL import Image
import test_http
from support import fixture_png
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from test_image_edits import export_fixture
from lazyrig.prompt_enhancement import dimensions, instructions, parse
from lazyrig.worker import Cancelled


def source_image(size=(256, 512), color='red'):
    stream = io.BytesIO()
    Image.new('RGB', size, color).save(stream, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def edit_call(image_id='image_1', prompt='Make the subject blue; preserve the rest.'):
    return '<tool_call>'+json.dumps({'name': 'edit_image', 'arguments': {'prompt': prompt, 'image_id': image_id}})+'</tool_call>'


def rewritten(edit=False, **extra):
    return json.dumps({'rewritten_prompt': 'Change the subject to deep blue and preserve the rest.' if edit else
                      'A wide watercolor illustration of an orange bicycle by a lake.',
                      'wh_ratio': '' if edit else '3:2', **({'ratio_follow': '<image1>'} if edit else {}), **extra})


class Workers(FixtureWorkers):
    def __init__(self, texts):
        super().__init__([(text, 100) for text in texts])
        self.gate = None
        self.entered = threading.Event()
        self.loaded_models = []

    def __call__(self, mode, model, timeout, cancelled):
        self.loaded_models.append(copy.deepcopy(model))
        worker = super().__call__(mode, model, timeout, cancelled)
        run = worker.run

        def tracked(payload, timeout, cancelled):
            enhancing = payload.get('response_format', {}).get('type') == 'json_schema'
            if (self.gate == 'enhancement' and enhancing) or self.gate == payload['action']:
                self.entered.set()
                while not cancelled.wait(.01):
                    pass
                raise Cancelled()
            if payload['action'] in {'image', 'image_edit'}:
                self.calls.append(copy.deepcopy(payload))
                fixture_png(payload['output_path'], width=payload['width'], height=payload['height'])
                yield {'kind': 'progress', 'step': 1, 'steps': 1}
                yield {'kind': 'result', 'path': payload['output_path']}
                return
            yield from run(payload, timeout, cancelled)
        worker.run = tracked
        return worker


class ChatImageTests(unittest.TestCase):
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def setUp(self):
        test_http.HTTPTests.setUp(self)
        self.service.config['models'][1].update(path=str(export_fixture(Path(self.temp.name)/'image-export')),
            runtime_profile='qwen-image-21', max_width=2752, max_height=2752,
            generation={'width': 512, 'height': 512, 'steps': 1})
        self.service.config['models'].append({**copy.deepcopy(self.service.config['models'][0]),
            'id': 'vision-chat', 'pipeline': 'vlm'})

    def fixture(self, *texts):
        workers = Workers(texts)
        self.service.scheduler.factory = workers
        return workers

    def body(self, **extra):
        return {'model': 'qwen-chat', 'auto_image': True, 'image_model': 'qwen-image',
                'messages': [{'role': 'user', 'content': 'Make this image blue.'}],
                'image_sources': [{'url': source_image(), 'description': 'Current source'}], **extra}

    def chat(self, **extra):
        return self.json('/v1/chat/completions', self.body(**extra))

    def test_text_llm_edits_an_upload_with_exact_source_and_tool_pairing(self):
        workers = self.fixture(edit_call(), 'The edit is saved.')
        result = self.chat()
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image_edit', 'text'])
        image = workers.calls[1]
        with Image.open(image['input_images'][0]['path']) as source:
            self.assertEqual(source.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result['x_artifacts'][0]['source_image_sha256'], image['input_images'][0]['sha256'])
        self.assertEqual(result['x_artifacts'][0]['operation'], 'edit')
        self.assertIsNone(result['x_prompt_enhancement'])
        self.assertFalse(workers.calls[0]['prepared']['images'])
        job = wait_job(self.service, result['x_job_id'])
        history = self.service.store.messages(job['session_id'])
        self.assertEqual(history[1]['tool_calls'][0]['function']['name'], 'edit_image')
        self.assertEqual(history[1]['tool_calls'][0]['id'], history[2]['tool_call_id'])
        self.assertTrue(json.loads(history[2]['content'])['ok'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual([m.get('_image_operation') for m in workers.loaded_models if m['pipeline']=='image'], ['edit'])

    def test_vlm_message_attachment_is_available_to_edit_without_extra_sources(self):
        workers = self.fixture(edit_call(), 'Done.')
        body = self.body(model='vision-chat', image_sources=[], messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Make it blue.'}, {'type': 'image_url', 'image_url': {'url': source_image()}}]}])
        preflight = self.json('/api/context', body)
        result = self.json('/v1/chat/completions', body)
        self.assertTrue(workers.calls[0]['prepared']['images'])
        self.assertEqual(preflight['prompt_sha256'], workers.calls[0]['prepared']['context']['prompt_sha256'])
        self.assertEqual(len(result['x_artifacts']), 1)

    def test_followup_edits_prior_generation_instead_of_generating_again(self):
        workers = self.fixture(qwen_call(), 'Saved.', edit_call(), 'Edited.')
        first = self.chat(image_sources=[], messages=[{'role': 'user', 'content': 'Draw a bicycle.'}])
        second = self.chat(image_sources=[{'url': first['x_artifacts'][0]['url'], 'description': 'Previous bicycle'}])
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image', 'text', 'text', 'image_edit', 'text'])
        self.assertNotEqual(first['x_artifacts'][0]['url'], second['x_artifacts'][0]['url'])
        self.assertEqual(second['x_artifacts'][0]['operation'], 'edit')

    def test_generation_enhancement_is_isolated_uses_lazyrig_prompt_and_supported_ratio(self):
        workers = self.fixture(qwen_call(), rewritten(), 'Saved.')
        self.service.config['models'][0]['generation'] = {'temperature': .95, 'stop_strings': ['STOP']}
        result = self.chat(image_sources=[], enhance_prompt=True, max_tokens=512)
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'text', 'image', 'text'])
        enhancer, image = workers.calls[1:3]
        self.assertEqual(enhancer['prepared']['messages'][0]['content'].split('\n\nReturn only JSON')[0], instructions(False))
        self.assertEqual(len(enhancer['prepared']['messages']), 2)
        self.assertFalse(enhancer['prepared']['template_context']['enable_thinking'])
        self.assertFalse(json.loads(enhancer['prepared']['text'])['tools'])
        self.assertEqual(enhancer['temperature'], .95)
        self.assertEqual(enhancer['generation']['stop_strings'], [])
        self.assertEqual(enhancer['max_tokens'], 2048)
        self.assertEqual((image['width'], image['height']), (2528, 1696))
        self.assertEqual(image['prompt'], json.loads(rewritten())['rewritten_prompt'])
        self.assertEqual(result['x_prompt_enhancement']['status'], 'completed')
        self.assertEqual(result['usage']['completion_tokens'], 300)
        self.assertNotIn(instructions(False), str(workers.calls[-1]['prepared']['messages']))
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
                         ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_edit_enhancement_receives_source_pixels_and_follows_input_size(self):
        workers = self.fixture(edit_call(), rewritten(True), 'Edited.')
        result = self.chat(model='vision-chat', enhance_prompt=True)
        enhancer, image = workers.calls[1:3]
        self.assertEqual(enhancer['prepared']['messages'][0]['content'].split('\n\nReturn only JSON')[0], instructions(True))
        self.assertEqual(enhancer['prepared']['images'][0]['sha256'], image['input_images'][0]['sha256'])
        self.assertEqual((image['width'], image['height']), (256, 512))
        self.assertEqual(result['x_prompt_enhancement']['ratio_follow'], '<image1>')
        self.assertEqual(set(self.service.scheduler.resident), {'vision-chat'})

    def test_explicit_size_overrides_enhanced_ratio_in_chat(self):
        workers = self.fixture(qwen_call(), rewritten(), 'Saved.')
        self.chat(image_sources=[], enhance_prompt=True, width=512, height=768)
        self.assertEqual((workers.calls[2]['width'], workers.calls[2]['height']), (512, 768))

    def test_stream_exposes_enhancement_metadata_but_only_streams_final_chat_text(self):
        self.fixture(qwen_call(), rewritten(), 'Your image is ready.')
        with self.request('/v1/chat/completions', self.body(image_sources=[], enhance_prompt=True, stream=True,
                stream_options={'include_usage': True})) as response:
            chunks = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: {')]
        content = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices'))
        self.assertEqual(content, 'Your image is ready.')
        final = next(c for c in reversed(chunks) if c.get('choices'))
        self.assertEqual(final['x_prompt_enhancement']['status'], 'completed')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 300)

    def test_context_accepts_own_absolute_artifact_sources(self):
        self.fixture(qwen_call(), 'Saved.', edit_call(), 'Edited.')
        first = self.chat(image_sources=[])
        body = self.body(image_sources=[{'url': self.base+first['x_artifacts'][0]['url']}])
        preflight = self.json('/api/context', body)
        result = self.json('/v1/chat/completions', body)
        self.assertTrue(preflight['fits'])
        self.assertEqual(result['x_artifacts'][0]['operation'], 'edit')

    def test_direct_enhancement_idempotency_and_saved_result_recovery(self):
        workers = self.fixture(rewritten())
        body = {'model': 'qwen-image', 'prompt': 'A bicycle', 'enhance_prompt': True, 'enhancement_model': 'qwen-chat'}
        with self.request('/api/image/jobs', body, {'Idempotency-Key': 'a'*32}) as response:
            receipt = json.load(response)
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'completed', job['error'])
        result = self.json(receipt['result_url'])
        with self.request('/api/image/jobs', body, {'Idempotency-Key': 'a'*32}) as response:
            self.assertEqual(json.load(response)['id'], receipt['id'])
        self.assertEqual(self.json(receipt['result_url']), result)
        self.assertEqual([c['action'] for c in workers.calls], ['text', 'image'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
        self.assertEqual(result['x_prompt_enhancement']['rewritten_prompt'], workers.calls[-1]['prompt'])
        image_worker = self.service.scheduler.resident['qwen-image'][1]
        later = self.json('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'Another bicycle', 'steps': 1})
        wait_job(self.service, later['x_job_id'])
        self.assertIs(self.service.scheduler.resident['qwen-image'][1], image_worker)
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'), ('load', 'qwen-image')])

    def test_enhanced_edit_rejects_nonvision_model_without_loading_image_worker(self):
        workers = self.fixture(edit_call())
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.chat(enhance_prompt=True)
        self.assertIn('vision-language', json.load(error.exception)['error']['message'])
        self.assertFalse(any(m['pipeline']=='image' for m in workers.loaded_models))
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.json('/api/image/edits', {'model': 'qwen-image', 'prompt': 'Blue', 'image': source_image(),
                'enhance_prompt': True, 'enhancement_model': 'qwen-chat'})
        self.assertEqual(error.exception.code, 400)

    def test_bad_json_never_falls_back_to_unenhanced_generation(self):
        workers = self.fixture('This is not JSON.')
        body = {'model': 'qwen-image', 'prompt': 'A bicycle', 'enhance_prompt': True, 'enhancement_model': 'qwen-chat'}
        receipt = self.json('/api/image/jobs', body)
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['prompt_enhancement']['status'], 'failed')
        self.assertIn('invalid JSON', job['error'])
        self.assertFalse(job['artifacts'])
        self.assertEqual([c['action'] for c in workers.calls], ['text'])
        self.assertFalse(any(m['pipeline']=='image' for m in workers.loaded_models))

    def test_unknown_source_never_dispatches_an_edit(self):
        workers = self.fixture(edit_call('other_job'))
        with self.assertRaises(urllib.error.HTTPError):
            self.chat()
        self.assertFalse(any(m['pipeline']=='image' for m in workers.loaded_models))

    def test_cancel_enhancement_prevents_image_load_and_keyed_replay(self):
        workers = self.fixture()
        workers.gate = 'enhancement'
        body = {'model': 'qwen-image', 'prompt': 'A bicycle', 'enhance_prompt': True, 'enhancement_model': 'qwen-chat'}
        with self.request('/api/image/jobs', body, {'Idempotency-Key': 'b'*32}) as response:
            receipt = json.load(response)
        self.assertTrue(workers.entered.wait(3))
        self.json(receipt['cancel_url'], {})
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(self.service.scheduler.resident)
        self.assertFalse(self.service.scheduler.busy)
        self.assertFalse(any(m['pipeline']=='image' for m in workers.loaded_models))
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/image/jobs', body, {'Idempotency-Key': 'b'*32})
        self.assertEqual(error.exception.code, 410)

    def test_cancel_chat_edit_stops_without_reloading_llm_or_requeueing(self):
        workers = self.fixture(edit_call())
        workers.gate = 'image_edit'
        receipt = self.json('/api/chat/jobs', self.body())
        self.assertTrue(workers.entered.wait(3))
        self.json(receipt['cancel_url'], {})
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(job['artifacts'])
        self.assertFalse(self.service.scheduler.resident)
        self.assertEqual(workers.lifecycle.count(('load', 'qwen-chat')), 1)


class EnhancementFormatTests(unittest.TestCase):
    def test_invalid_output_fields_ratios_and_follow_references_are_rejected(self):
        bad = ['{}', '{"rewritten_prompt":"a","rewritten_prompt":"b","wh_ratio":"1:1"}',
               rewritten(wh_ratio='1:0'), rewritten(wh_ratio=''), rewritten(rewritten_prompt=''),
               rewritten(True, wh_ratio='1:1'), rewritten(True, ratio_follow='<image2>'),
               rewritten(True, rewritten_prompt='First\nSecond')]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value, 'ratio_follow' in value)

    def test_generic_ratios_stay_within_adapter_bounds(self):
        model = {'pipeline': 'image', 'max_width': 1024, 'max_height': 768}
        for ratio in ('16:9', '9:16', '9:2', '1:99999'):
            width, height = dimensions(model, {'width': 512, 'height': 512}, {'wh_ratio': ratio})
            self.assertTrue(256 <= width <= 1024 and 256 <= height <= 768)
            self.assertEqual(width % 64, 0)
            self.assertEqual(height % 64, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
