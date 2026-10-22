"""Dimension, image option and diffusion scheduler integration regressions."""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
import test_image_edits as edits
from test_image_edits import export_fixture
from support import DEFAULT
from lazyrig.config import validate
from lazyrig.diffusion_settings import QWEN_REFERENCE, configure_pipeline, describe, validate_settings
from lazyrig.image_sizes import QWEN_IMAGE_21_PRESETS, constraints
from support import Service


def scheduler_file(root):
    path = Path(root) / 'scheduler/scheduler_config.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'_class_name': 'FlowMatchEulerDiscreteScheduler', **QWEN_REFERENCE,
                               'time_shift_type': 'exponential', 'use_karras_sigmas': False}))
    return path


class DimensionSettingsTests(unittest.TestCase):
    def model(self):
        return {**copy.deepcopy(DEFAULT['models'][1]), 'runtime_profile': 'qwen-image-21',
                'max_width': 2752, 'max_height': 2752}

    def test_official_presets_nonpreset_and_larger_configured_sizes(self):
        for ratio, width, height in QWEN_IMAGE_21_PRESETS:
            with self.subTest(ratio=ratio):
                model = {**self.model(), 'generation': {'width': width, 'height': height}}
                validate({**copy.deepcopy(DEFAULT), 'models': [model]})
                self.assertEqual(Service.image_parameters(None, model, {})['width'], width)
        model = self.model()
        for width, height in [(2752, 2752), (1728, 2560)]:
            self.assertEqual(Service.image_parameters(None, model, {'width': width, 'height': height})['height'], height)
        model.update(max_width=4096, max_height=4096)
        self.assertEqual(Service.image_parameters(None, model, {'width': 3072, 'height': 3072})['width'], 3072)

    def test_alignment_limits_and_fractional_dimensions(self):
        model = self.model()
        for value in [1697, 1696.5, True, None, [], float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Service.image_parameters(None, model, {'width': value, 'height': 2528})
        with self.assertRaisesRegex(ValueError, 'multiples of 32'):
            Service.image_parameters(None, model, {'width': 1697, 'height': 2528})
        model.update(max_width=1024, max_height=1024)
        with self.assertRaisesRegex(ValueError, 'configured maximum 1024x1024'):
            Service.image_parameters(None, model, {'width': 1696, 'height': 2528})
        model.pop('runtime_profile'); model.update(max_width=2752, max_height=2752)
        with self.assertRaisesRegex(ValueError, 'multiples of 64'):
            Service.image_parameters(None, model, {'width': 1696, 'height': 2528})

    def test_export_detection_and_new_registration_preserve_existing_limits(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.model(); model.pop('runtime_profile')
            model['path'] = str(export_fixture(Path(root) / 'export'))
            self.assertEqual(constraints(model)['multiple_of'], 32)
            service = Service({**copy.deepcopy(DEFAULT), 'data_dir': root})
            report = {'path': model['path'], 'task': 'text-to-image', 'pipeline': 'image',
                      'declared_context_limit': 32768, 'runtime_profile': 'qwen-image-21', 'architecture': 'QwenImage21Pipeline'}
            try:
                with patch('lazyrig.service.inspect_export', return_value=report):
                    new = service.register({'id': 'new-qwen', 'path': model['path'], 'device': 'TEST.GPU'})
                    self.assertEqual((new['max_width'], new['max_height']), (2752, 2752))
                    self.assertEqual(new['generation']['width'], 2048)
                self.assertEqual(service.model('qwen-image')['max_width'], 1024)
            finally:
                service.close()


class ImageSettingsHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json
    fixture = edits.ImageEditHTTPTests.fixture
    upload = edits.ImageEditHTTPTests.upload
    body = edits.ImageEditHTTPTests.body

    def setup_image(self):
        self.fixture(); scheduler_file(self.service.model('qwen-image')['path'])
        self.json('/api/settings', {'model': 'qwen-image', 'max_width': 2752, 'max_height': 2752})

    def test_all_presets_reach_generation_and_multipart_edit_unchanged(self):
        self.setup_image()
        for ratio, width, height in QWEN_IMAGE_21_PRESETS:
            for editing in (False, True):
                with self.subTest(ratio=ratio, editing=editing):
                    fields = {'model': 'qwen-image', 'prompt': 'A portrait', 'size': f'{width}x{height}', 'steps': 1}
                    if editing:
                        with self.upload(fields=list(fields.items())) as response:
                            result = json.load(response)
                    else:
                        result = self.json('/v1/images/generations', fields)
                    self.assertEqual((self.workers.calls[-1]['width'], self.workers.calls[-1]['height']), (width, height))
                    self.assertEqual((result['x_artifacts'][0]['width'], result['x_artifacts'][0]['height']), (width, height))
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})

    def test_saved_options_metadata_and_restart(self):
        self.setup_image()
        options = {'type': 'FLOW_MATCH_EULER_DISCRETE', 'config': {'base_shift': .6, 'shift_terminal': .03}}
        generation = {'width': 1696, 'height': 2528, 'steps': 1, 'guidance_scale': 4,
                      'negative_prompt': 'blur, text', 'max_sequence_length': 512}
        self.json('/api/settings', {'model': 'qwen-image', 'generation': generation, 'diffusion_scheduler': options})
        self.json('/v1/images/edits', self.body(size='auto'))
        for key, value in generation.items():
            if key != 'steps': self.assertEqual(self.workers.calls[-1][key], value)
        row = next(m for m in self.json('/v1/models')['data'] if m['id'] == 'qwen-image')
        self.assertEqual(row['image_size_constraints']['multiple_of'], 32)
        self.assertEqual(row['diffusion_scheduler_info']['effective_config']['base_shift'], .6)
        self.service.close(); restored = Service(self.service.config)
        try:
            self.assertEqual(restored.model('qwen-image')['generation'], generation)
            self.assertEqual(restored.model('qwen-image')['diffusion_scheduler'], options)
        finally: restored.close()

    def test_exact_nonpreset_size_reaches_png_for_generation_and_edit(self):
        self.setup_image()
        fields = {'model': 'qwen-image', 'prompt': 'A street photograph', 'size': '1376x928', 'steps': 1}
        for editing in (False, True):
            with self.subTest(editing=editing):
                if editing:
                    with self.upload(fields=list(fields.items())) as response:
                        result = json.load(response)
                else:
                    result = self.json('/v1/images/generations', fields)
                self.assertEqual((self.workers.calls[-1]['width'], self.workers.calls[-1]['height']), (1376, 928))
                artifact = result['x_artifacts'][0]
                self.assertEqual((artifact['width'], artifact['height']), (1376, 928))
                with self.request(artifact['url']) as response, Image.open(io.BytesIO(response.read())) as png:
                    self.assertEqual(png.size, (1376, 928))

    def test_invalid_requests_fail_before_worker_load_and_settings_are_atomic(self):
        self.setup_image()
        for size in ['1697x2528', '2816x1536', 123, None]:
            with self.subTest(size=size), self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'A portrait', 'size': size})
            self.assertEqual(err.exception.code, 400)
        before = copy.deepcopy(self.service.model('qwen-image'))
        for options in [{'type': 'DDIM'}, {'config': {'use_karras_sigmas': True}},
                        {'config': {'shift_terminal': 1}}, {'config': {'base_image_seq_len': 8192}}]:
            with self.subTest(options=options), self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/api/settings', {'model': 'qwen-image', 'diffusion_scheduler': options})
            self.assertEqual(err.exception.code, 400)
            self.assertEqual(self.service.model('qwen-image'), before)
        self.assertFalse(self.workers.calls); self.assertFalse(self.service.store.jobs())

    def test_async_and_llm_handoff_share_image_defaults(self):
        self.setup_image()
        self.json('/api/settings', {'model': 'qwen-image', 'generation': {
            'width': 1696, 'height': 2528, 'steps': 1, 'max_sequence_length': 512, 'negative_prompt': 'blur', 'guidance_scale': 4}})
        receipt = self.json('/api/image/jobs', {'model': 'qwen-image', 'prompt': 'A portrait'})
        self.assertEqual(wait_job(self.service, receipt['id'])['status'], 'completed')
        self.workers = FixtureWorkers([(qwen_call(), 40), ('Image complete.', 4)])
        self.service.scheduler.factory = self.workers
        result = self.json('/v1/chat/completions', {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Generate a portrait'}],
            'auto_image': True, 'image_model': 'qwen-image', 'max_tokens': 512})
        image = next(p for p in self.workers.calls if p['action'] == 'image')
        self.assertEqual((image['width'], image['height']), (1696, 2528))
        self.assertEqual(image['max_sequence_length'], 512); self.assertEqual(image['negative_prompt'], 'blur')
        self.assertEqual(result['choices'][0]['message']['content'], 'Image complete.')

    def test_native_option_validation_and_multipart_override(self):
        self.setup_image()
        for fields in [{'guidance_scale': 1, 'negative_prompt': 'blur'}, {'max_sequence_length': 1.5},
                       {'max_sequence_length': False}, {'max_sequence_length': 0}, {'guidance_scale': True}]:
            with self.subTest(fields=fields), self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/v1/images/edits', self.body(**fields))
            self.assertEqual(err.exception.code, 400)
        with self.upload(fields=[('model', 'qwen-image'), ('prompt', 'Add a scarf'), ('size', '1696x2528'),
                                 ('guidance_scale', 4), ('negative_prompt', 'blur'), ('max_sequence_length', 768)]) as response:
            result = json.load(response)
        self.assertEqual(self.workers.calls[-1]['max_sequence_length'], 768)
        self.assertEqual(result['x_artifacts'][0]['negative_prompt'], 'blur')


class NativeSchedulerTests(unittest.TestCase):
    def test_real_flow_scheduler_constructs_without_changing_export(self):
        import openvino_genai as genai
        with tempfile.TemporaryDirectory() as root:
            path = scheduler_file(root); before = path.read_bytes(); seen = []
            model = {'path': root, 'pipeline': 'image', 'runtime_profile': 'qwen-image-21',
                     'diffusion_scheduler': {'type': 'FLOW_MATCH_EULER_DISCRETE', 'config': QWEN_REFERENCE}}
            pipeline = SimpleNamespace(set_scheduler=seen.append)
            configure_pipeline(genai, pipeline, model)
            self.assertIsInstance(seen[0], genai.Scheduler)
            self.assertEqual(path.read_bytes(), before)
            model['diffusion_scheduler'] = {'type': 'AUTO', 'config': {}}
            configure_pipeline(genai, pipeline, model); self.assertEqual(len(seen), 1)
            model['diffusion_scheduler'] = {'config': {'shift_terminal': None}}
            configure_pipeline(genai, pipeline, model); self.assertEqual(len(seen), 2)

    def test_ignored_or_invalid_qwen_scheduler_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler_file(root); model = {'path': root, 'pipeline': 'image', 'runtime_profile': 'qwen-image-21'}
            info = describe(model)
            self.assertEqual(info['types'], ['AUTO', 'FLOW_MATCH_EULER_DISCRETE'])
            self.assertNotIn('time_shift_type', {c['key'] for c in info['controls']})
            for config in [{'shift': 0}, {'base_shift': float('nan')}, {'use_dynamic_shifting': 1},
                           {'num_train_timesteps': True}, {'time_shift_type': 'linear'}]:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    validate_settings({**model, 'diffusion_scheduler': {'config': config}})
