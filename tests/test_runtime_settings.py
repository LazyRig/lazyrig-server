"""Persist runtime controls and check the exact options delivered to workers."""
import copy
import json
import unittest
import urllib.error
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers
from lazyrig.config import restore_settings
from lazyrig.runtime_settings import CONTROLS, pipeline_properties


class RuntimeSettingsTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_precision_and_cache_settings_persist_and_reach_reloaded_worker(self):
        seen = []
        factory = FixtureWorkers([])
        def worker(mode, model, *args):
            seen.append(copy.deepcopy(model))
            return factory(mode, model, *args)
        self.service.scheduler.factory = worker
        wait_job(self.service, self.json('/api/models/qwen-chat/load', {})['id'])
        plugin = {'KV_CACHE_PRECISION': 'u4', 'KEY_CACHE_PRECISION': 'u8',
                  'VALUE_CACHE_PRECISION': 'u4', 'KEY_CACHE_GROUP_SIZE': 16,
                  'VALUE_CACHE_GROUP_SIZE': 64, 'KEY_CACHE_QUANT_MODE': 'BY_CHANNEL',
                  'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
                  'INFERENCE_PRECISION_HINT': 'f16', 'NUM_STREAMS': 'AUTO',
                  'CACHE_DIR': 'C:/models/compiled', 'CUSTOM_DEVICE_OPTION': 'preserved'}
        scheduler = {'cache_size': 7, 'enable_prefix_caching': True,
                     'max_num_batched_tokens': 128, 'max_num_seqs': 1,
                     'cache_interval_multiplier': 64, 'dynamic_split_fuse': True}
        state = self.json('/api/settings', {'model': 'qwen-chat', 'plugin_config': plugin, 'scheduler_config': scheduler})
        self.assertFalse(self.service.scheduler.resident)
        self.assertIn(('close', 'qwen-chat'), factory.lifecycle)
        self.assertEqual(len(state['runtime_controls']), len(CONTROLS))
        model = state['models'][0]
        self.assertEqual(model['plugin_config'], plugin)
        self.assertEqual(model['scheduler_config'], scheduler)
        restored = restore_settings(self.service.config)['models'][0]
        self.assertEqual(restored['plugin_config'], plugin)
        self.assertEqual(restored['scheduler_config'], scheduler)
        wait_job(self.service, self.json('/api/models/qwen-chat/load', {})['id'])
        self.assertEqual(seen[-1]['plugin_config'], plugin)
        self.assertEqual(seen[-1]['scheduler_config'], scheduler)
        self.json('/api/settings', {'model': 'qwen-chat', 'plugin_config': {}, 'scheduler_config': {}})
        self.assertEqual(self.service.model('qwen-chat')['plugin_config'], {})
        self.assertEqual(self.service.model('qwen-chat')['scheduler_config'], {})

    def test_invalid_settings_are_atomic_and_do_not_unload_existing_model(self):
        self.service.scheduler.factory = FixtureWorkers([])
        wait_job(self.service, self.json('/api/models/qwen-chat/load', {})['id'])
        original = copy.deepcopy(self.service.config)
        worker = self.service.scheduler.resident['qwen-chat']
        cases = [('plugin_config', {'KV_CACHE_PRECISION': 'int4'}),
                 ('plugin_config', {'KV_CACHE_PRECISION': 4}),
                 ('plugin_config', {'NUM_STREAMS': 0}),
                 ('plugin_config', {'INFERENCE_NUM_THREADS': -1}),
                 ('plugin_config', {'KEY_CACHE_GROUP_SIZE': 1.5}),
                 ('scheduler_config', {'cache_size': 1.5}),
                 ('scheduler_config', {'cache_size': -1}),
                 ('scheduler_config', {'max_num_batched_tokens': 0}),
                 ('scheduler_config', {'max_num_seqs': True}),
                 ('scheduler_config', {'enable_prefix_caching': 'true'}),
                 ('scheduler_config', {'cache_interval_multiplier': -2}),
                 ('scheduler_config', {'enable_prefix_caching': True, 'cache_interval_multiplier': 0}),
                 ('scheduler_config', {'cache_size': 1, 'num_kv_blocks': 32}),
                 ('scheduler_config', {'made_up_field': 1})]
        for scope, values in cases:
            with self.subTest(scope=scope, values=values), self.assertRaises(urllib.error.HTTPError) as err:
                self.request('/api/settings', {'model': 'qwen-chat', scope: values})
            self.assertEqual(err.exception.code, 400)
            self.assertEqual(self.service.config, original)
            self.assertIs(self.service.scheduler.resident['qwen-chat'], worker)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/api/settings', {'model': 'qwen-image', 'scheduler_config': {'cache_size': 1}})
        self.assertEqual(err.exception.code, 400)

    def test_native_scheduler_binding_preserves_values_and_allows_unset_defaults(self):
        try:
            import openvino_genai as genai
        except ImportError:
            self.skipTest('Optional installed OpenVINO GenAI runtime')
        model = {'pipeline': 'llm', 'plugin_config': {'KV_CACHE_PRECISION': 'u4'},
                 'scheduler_config': {'cache_size': 7, 'enable_prefix_caching': True,
                    'max_num_batched_tokens': 128, 'max_num_seqs': 1,
                    'cache_interval_multiplier': 64, 'dynamic_split_fuse': True}}
        saved = copy.deepcopy(model)
        properties = pipeline_properties(genai, model)
        for key, value in model['scheduler_config'].items():
            self.assertEqual(getattr(properties['scheduler_config'], key), value)
        self.assertEqual(properties['KV_CACHE_PRECISION'], 'u4')
        self.assertEqual(model, saved)
        self.assertEqual(pipeline_properties(genai, {'pipeline': 'llm'}), {})

    def test_cache_controls_default_to_one_sequence_without_overriding_saved_values(self):
        try:
            import openvino_genai as genai
        except ImportError:
            self.skipTest('Optional installed OpenVINO GenAI runtime')
        for pipeline in ('llm', 'vlm', 'vlm-text'):
            for options in ({'cache_size': 1}, {'cache_size': 0}, {'num_kv_blocks': 32},
                            {'enable_prefix_caching': False}, {'enable_prefix_caching': True},
                            {'cache_size': 1, 'max_num_seqs': 4}):
                with self.subTest(pipeline=pipeline, options=options):
                    model = {'pipeline': pipeline, 'device': 'GPU.1',
                             'scheduler_config': copy.deepcopy(options)}
                    saved = copy.deepcopy(model)
                    scheduler = pipeline_properties(genai, model)['scheduler_config']
                    self.assertEqual(scheduler.max_num_seqs, options.get('max_num_seqs', 1))
                    for key, value in options.items():
                        self.assertEqual(getattr(scheduler, key), value)
                    self.assertEqual(model, saved)
        # Do not force continuous batching onto a native-default or SDPA path.
        for model in ({'pipeline': 'llm'}, {'pipeline': 'image'},
                      {'pipeline': 'llm', 'scheduler_config': {'cache_size': 1},
                       'plugin_config': {'ATTENTION_BACKEND': 'SDPA'}}):
            self.assertNotIn('scheduler_config', pipeline_properties(genai, model))

    def test_mtp_uses_serial_sequence_default_and_preserves_fixed_cache_choice(self):
        from unittest.mock import patch
        try:
            import openvino_genai as genai
        except ImportError:
            self.skipTest('Optional installed OpenVINO GenAI runtime')
        for options in ({}, {'cache_size': 1}, {'max_num_seqs': 3}):
            with self.subTest(options=options), patch('lazyrig.runtime_settings.ensure_export'), \
                 patch.object(genai, 'draft_model', return_value='draft marker'):
                model = {'pipeline': 'vlm', 'path': '/fixture', 'device': 'GPU.1',
                         'mtp': {'enabled': True}, 'scheduler_config': options}
                scheduler = pipeline_properties(genai, model)['scheduler_config']
                self.assertEqual(scheduler.max_num_seqs, options.get('max_num_seqs', 1))
                self.assertEqual(scheduler.cache_size, options.get('cache_size', 0))
                self.assertFalse(scheduler.enable_prefix_caching)
