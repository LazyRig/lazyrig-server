import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyrig import placement
from lazyrig.runtime_settings import effective_cpu_thread_default, pipeline_properties
from lazyrig.worker import NativeWorkerError, Cancelled

try:
    import openvino as native_ov
    import openvino_genai as native_genai
except ImportError:
    native_ov = native_genai = None


def configuration():
    return {'devices': [
        {'id': 'GPU.0', 'device_type': 'integrated', 'detected': True, 'budget_gib': 12},
        {'id': 'GPU.1', 'device_type': 'discrete', 'detected': True, 'budget_gib': 14},
        {'id': 'GPU.2', 'device_type': 'discrete', 'detected': True, 'budget_gib': 20},
        {'id': 'GPU.3', 'device_type': 'discrete', 'detected': False, 'budget_gib': 20},
        {'id': 'CPU', 'device_type': 'cpu', 'detected': True, 'budget_gib': 54},
        {'id': 'NPU', 'device_type': 'other', 'detected': True, 'budget_gib': 8}]}


def text_model():
    return {'id': 'qwen', 'device': 'GPU.1', 'path': '/model', 'pipeline': 'llm',
            'task': 'text-generation', 'memory_gib': 14, 'plugin_config': {}}


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.config, self.model = configuration(), text_model()

    def test_defaults_use_only_explicit_primary_device(self):
        self.assertEqual(placement.default_placement(self.model, self.config)['devices'], ['GPU.1'])
        self.model['device'] = 'GPU.0'
        self.assertEqual(placement.default_placement(self.model, self.config)['devices'], ['GPU.0'])
        self.model['pipeline'] = 'image'
        self.assertEqual(placement.default_placement(self.model, self.config)['devices'], ['GPU.0'])

    def test_automatic_attempts_are_real_partition_targets(self):
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'GPU.2', 'CPU'], 'shares': {}}
        plans = placement.prepare(self.model, self.config)
        self.assertEqual([p['_placement']['target'] for p in plans],
                         ['GPU.1', 'HETERO:GPU.1,GPU.2', 'HETERO:GPU.1,GPU.2,CPU', 'HETERO:GPU.1,CPU'])
        self.assertEqual(plans[1]['_placement']['properties']['MODEL_DISTRIBUTION_POLICY'], 'PIPELINE_PARALLEL')
        self.assertEqual(plans[2]['_placement']['properties']['ATTENTION_BACKEND'], 'SDPA')
        self.assertEqual(placement.reservations(plans[2]), {'GPU.1': 14, 'GPU.2': 20, 'CPU': 54})
        self.assertEqual(placement.lease_devices(self.model, self.config), {'GPU.1', 'GPU.2', 'CPU', '@model:qwen'})

    def test_manual_shares_are_relative_to_remaining_weights(self):
        self.model['placement'] = {'mode': 'manual', 'devices': ['GPU.1', 'GPU.0', 'CPU'],
                                   'shares': {'GPU.1': 50, 'GPU.0': 30, 'CPU': 20}}
        plan, = placement.prepare(self.model, self.config)
        props = plan['_placement']['properties']
        self.assertEqual(props['DEVICE_PROPERTIES'], {'GPU.1': {'QUERY_MODEL_RATIO': .5}, 'GPU.0': {'QUERY_MODEL_RATIO': .6}})
        self.assertNotIn('MODEL_DISTRIBUTION_POLICY', props)
        self.assertEqual(props['ATTENTION_BACKEND'], 'SDPA')
        self.assertEqual(plan['device'], 'GPU.1')
        self.model['placement']['shares'] = {'GPU.1': 100, 'GPU.0': 0, 'CPU': 0}
        plan, = placement.prepare(self.model, self.config)
        self.assertEqual(plan['_placement']['target'], 'GPU.1')
        self.assertEqual(plan['_placement']['properties'], {})

    def test_cpu_last_duplicates_and_share_validation(self):
        for p in [
            {'mode': 'auto', 'devices': ['CPU', 'GPU.1']},
            {'mode': 'auto', 'devices': ['GPU', 'GPU.0']},
            {'mode': 'manual', 'devices': ['GPU.1', 'CPU'], 'shares': {'GPU.1': 80, 'CPU': 30}},
            {'mode': 'manual', 'devices': ['GPU.1', 'CPU'], 'shares': {'GPU.1': True, 'CPU': 99}},
            {'mode': 'auto', 'devices': ['GPU.1', 'NPU']},
        ]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                placement.validate_placement({**self.model, 'placement': p}, self.config)

    def test_cpu_primary_is_single_device_and_mixed_plans_keep_gpu_priority(self):
        self.model['device'] = 'CPU'
        self.model['placement'] = {'mode': 'auto', 'devices': ['CPU'], 'shares': {}}
        plan, = placement.prepare(self.model, self.config)
        self.assertEqual(plan['_placement']['target'], 'CPU')
        for mode in ('auto', 'manual'):
            self.model['placement'] = {'mode': mode, 'devices': ['CPU', 'GPU.1'],
                                       'shares': {'CPU': 50, 'GPU.1': 50}}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'CPU must be the last device'):
                placement.prepare(self.model, self.config)

    def test_model_capabilities_allow_changing_npu_selection_to_compatible_devices(self):
        self.model['device'] = 'NPU'
        self.model['placement'] = {'mode': 'auto', 'devices': ['NPU'], 'shares': {}}
        info = placement.info(self.model, self.config)
        self.assertTrue(info['multi_device_supported'])
        self.assertFalse(info['manual_supported'])
        self.assertEqual(info['default_devices'], ['NPU'])
        self.model['placement']['devices'] = ['NPU', 'GPU.1']
        with self.assertRaisesRegex(ValueError, 'GPU devices and CPU only'):
            placement.validate_placement(self.model, self.config)
        self.model['pipeline'] = 'image'
        self.assertFalse(placement.info(self.model, self.config)['multi_device_supported'])
        self.assertFalse(placement.info(self.model, self.config)['manual_supported'])

    def test_rounding_tolerance_cannot_create_zero_remaining_division(self):
        self.model['placement'] = {'mode': 'manual', 'devices': ['GPU.1', 'GPU.2'],
                                   'shares': {'GPU.1': 100, 'GPU.2': .00005}}
        plan, = placement.prepare(self.model, self.config)
        ratios = plan['_placement']['properties']['DEVICE_PROPERTIES']
        self.assertLess(ratios['GPU.1']['QUERY_MODEL_RATIO'], 1)
        self.assertEqual(ratios['GPU.2']['QUERY_MODEL_RATIO'], 1)

    def test_mtp_stays_enabled_and_avoids_unsupported_hetero_draft(self):
        self.model['mtp'] = {'enabled': True}
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'CPU'], 'shares': {}}
        self.assertEqual([p['_placement']['target'] for p in placement.prepare(self.model, self.config)], ['GPU.1'])
        self.model['placement']['allow_fallback'] = True
        self.assertEqual([p['_placement']['target'] for p in placement.prepare(self.model, self.config)], ['GPU.1', 'CPU'])
        self.assertFalse(placement.info(self.model, self.config)['manual_supported'])

    def test_legacy_and_custom_properties_are_preserved(self):
        before = copy.deepcopy(self.model)
        self.assertEqual(placement.prepare(self.model, self.config), [before])
        self.model['plugin_config'] = {'DEVICE_PROPERTIES': {'GPU': {'QUERY_MODEL_RATIO': .5}}}
        self.assertTrue(placement.info(self.model, self.config)['custom_properties'])
        self.model['placement'] = placement.default_placement(self.model, self.config)
        with self.assertRaisesRegex(ValueError, 'advanced OpenVINO'):
            placement.validate_placement(self.model, self.config)

    def test_runtime_ratio_capability_must_be_verified(self):
        class Core:
            def __init__(self, keys): self.keys = keys
            def get_property(self, device, key): return self.keys
        self.model['_placement'] = {'properties': {'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .5}}}}
        with self.assertRaisesRegex(ValueError, 'does not support'):
            placement.runtime_properties(self.model, Core([]))
        self.assertIn('DEVICE_PROPERTIES', placement.runtime_properties(self.model, Core(['QUERY_MODEL_RATIO'])))

    def test_retry_does_not_replay_cancellation_timeout_or_invalid_model(self):
        for exc in (Cancelled(), TimeoutError(), ValueError('out of memory'), NativeWorkerError('Tokenizer file missing'), NativeWorkerError('Invalid shape')):
            self.assertFalse(placement.retryable_load_error(exc), str(exc))
        for message in ('MemoryError: ', '[GPU] CL_MEM_OBJECT_ALLOCATION_FAILURE', 'CL_OUT_OF_RESOURCES', 'std::bad_alloc', '[GPU] operation not supported'):
            self.assertTrue(placement.retryable_load_error(NativeWorkerError(message)), message)

    def test_stateful_properties_preserve_custom_device_options_and_source(self):
        self.model['plugin_config'] = {'CACHE_DIR': 'cache', 'DEVICE_PROPERTIES': {'GPU.1': {'NUM_STREAMS': 1}}}
        self.model['scheduler_config'] = {'cache_size': 1}
        self.model['_placement'] = {'properties': {'ATTENTION_BACKEND': 'SDPA', 'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .5}}}}
        before = copy.deepcopy(self.model)
        props = pipeline_properties(object(), self.model)
        self.assertEqual(props['CACHE_DIR'], 'cache')
        self.assertEqual(props['DEVICE_PROPERTIES']['GPU.1'], {'NUM_STREAMS': 1, 'QUERY_MODEL_RATIO': .5})
        self.assertNotIn('scheduler_config', props)
        self.assertEqual(self.model, before)

    def test_vlm_manual_ratios_reach_exact_virtual_target_and_preserve_role_options(self):
        for pipeline in ('vlm', 'vlm-text'):
            with self.subTest(pipeline=pipeline):
                self.model['pipeline'] = pipeline
                self.model['placement'] = {'mode': 'manual', 'devices': ['GPU.1', 'CPU'],
                                           'shares': {'GPU.1': 50, 'CPU': 50}}
                self.model['plugin_config'] = {
                    'CACHE_DIR': 'cache',
                    'DEVICE_PROPERTIES': {'GPU.1': {'NUM_STREAMS': 1}, 'CPU': {'INFERENCE_NUM_THREADS': 6}},
                    'MODEL_PROPERTIES': {
                        'language_model': {'PERFORMANCE_HINT': 'LATENCY', 'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 4}}},
                        'vision_embeddings': {'INFERENCE_PRECISION_HINT': 'f32'},
                    }}
                plan, = placement.prepare(self.model, self.config)
                before = copy.deepcopy(plan)
                props = pipeline_properties(object(), plan)
                routed = props['DEVICE_PROPERTIES']['HETERO:GPU.1,CPU']['DEVICE_PROPERTIES']
                self.assertEqual(routed['GPU.1'], {'NUM_STREAMS': 1, 'QUERY_MODEL_RATIO': .5})
                self.assertEqual(routed['CPU'], {'INFERENCE_NUM_THREADS': 6})
                lm = props['MODEL_PROPERTIES']['language_model']
                self.assertEqual(lm['PERFORMANCE_HINT'], 'LATENCY')
                self.assertEqual(lm['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 4)
                self.assertEqual(lm['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'], .5)
                self.assertEqual(props['MODEL_PROPERTIES']['vision_embeddings'], {'INFERENCE_PRECISION_HINT': 'f32'})
                self.assertNotIn('ATTENTION_BACKEND', routed)
                self.assertNotIn('MODEL_DISTRIBUTION_POLICY', props)
                self.assertEqual(plan, before)

    def test_vlm_two_gpu_manual_ratios_use_full_target_without_cpu_fallback(self):
        self.model['pipeline'] = 'vlm'
        self.model['placement'] = {'mode': 'manual', 'devices': ['GPU.1', 'GPU.0'],
                                   'shares': {'GPU.1': 50, 'GPU.0': 50}}
        plan, = placement.prepare(self.model, self.config)
        props = pipeline_properties(object(), plan)
        self.assertEqual(props['DEVICE_PROPERTIES']['HETERO:GPU.1,GPU.0']['DEVICE_PROPERTIES'],
                         {'GPU.1': {'QUERY_MODEL_RATIO': .5}, 'GPU.0': {'QUERY_MODEL_RATIO': 1}})
        self.assertNotIn('CPU', props['DEVICE_PROPERTIES'])
        self.assertNotIn('MODEL_DISTRIBUTION_POLICY', props)

    def test_vlm_auto_policy_and_existing_target_specific_device_options_survive_routing(self):
        self.model['pipeline'] = 'vlm'
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'CPU'], 'shares': {}}
        self.model['plugin_config'] = {'DEVICE_PROPERTIES': {
            'GPU.1': {'NUM_STREAMS': 1},
            'HETERO:GPU.1,CPU': {'CACHE_DIR': 'target-cache',
                               'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 3}}}}}
        plan = placement.prepare(self.model, self.config)[1]
        props = pipeline_properties(object(), plan)
        self.assertEqual(props['MODEL_DISTRIBUTION_POLICY'], 'PIPELINE_PARALLEL')
        self.assertEqual(props['ATTENTION_BACKEND'], 'SDPA')
        self.assertEqual(props['DEVICE_PROPERTIES']['HETERO:GPU.1,CPU'], {
            'CACHE_DIR': 'target-cache', 'DEVICE_PROPERTIES': {
                'GPU.1': {'NUM_STREAMS': 1}, 'CPU': {'INFERENCE_NUM_THREADS': 3}}})

    def test_cpu_thread_default_applies_only_to_cpu_plan_without_model_wide_override(self):
        self.model['_cpu_inference_threads'] = 6
        self.assertIsNone(effective_cpu_thread_default(self.model, 6))
        self.assertNotIn('INFERENCE_NUM_THREADS', pipeline_properties(object(), self.model))
        for pipeline in ('llm', 'vlm', 'image'):
            with self.subTest(pipeline=pipeline):
                self.model.update(pipeline=pipeline, device='CPU')
                props = pipeline_properties(object(), self.model)
                self.assertEqual(props['INFERENCE_NUM_THREADS'], 6)
        self.model.update(pipeline='vlm', device='GPU.1')
        self.model['_placement'] = {'target': 'HETERO:GPU.1,CPU', 'devices': ['GPU.1', 'CPU'],
                                   'properties': {'ATTENTION_BACKEND': 'SDPA',
                                                  'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .5}}}}
        self.model['plugin_config'] = {'MODEL_PROPERTIES': {
            'language_model': {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 4}}}}}
        self.assertEqual(effective_cpu_thread_default(self.model, 6), 6)
        props = pipeline_properties(object(), self.model)
        self.assertEqual(props['DEVICE_PROPERTIES']['HETERO:GPU.1,CPU']['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 6)
        self.assertEqual(props['MODEL_PROPERTIES']['language_model']['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 4)
        self.assertNotIn('INFERENCE_NUM_THREADS', props['DEVICE_PROPERTIES']['GPU.1'])
        self.model['plugin_config']['MODEL_PROPERTIES']['language_model'] = {'INFERENCE_NUM_THREADS': 3}
        props = pipeline_properties(object(), self.model)
        self.assertEqual(props['MODEL_PROPERTIES']['language_model']['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 3)
        self.assertEqual(props['DEVICE_PROPERTIES']['HETERO:GPU.1,CPU']['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 6)
        for plugin in ({'INFERENCE_NUM_THREADS': 2},
                       {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 2}}},
                       {'DEVICE_PROPERTIES': {'HETERO:GPU.1,CPU': {'INFERENCE_NUM_THREADS': 2}}},
                       {'DEVICE_PROPERTIES': {'HETERO:GPU.1,CPU': {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 2}}}}}):
            with self.subTest(plugin=plugin):
                self.model['plugin_config'] = plugin
                self.assertIsNone(effective_cpu_thread_default(self.model, 6))

    @unittest.skipIf(native_genai is None, 'OpenVINO GenAI is not installed')
    def test_native_vlm_cpu_compile_receives_nested_device_settings(self):
        """A tiny real VLM compile catches the GenAI routing bug, without GPUs.

        The known CPU setting deliberately has an invalid value. If GenAI drops
        DEVICE_PROPERTIES, compilation succeeds and instead fails later on our
        intentionally incomplete VLM interface. The expected property error
        proves the setting reached the actual language-model compiler.
        """
        from openvino import opset13 as ops
        if 'CPU' not in native_ov.Core().available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        with tempfile.TemporaryDirectory() as folder:
            parameter = ops.parameter([1, -1, 4], native_ov.Type.f32, name='inputs_embeds')
            logits = ops.add(parameter, ops.constant(1, native_ov.Type.f32))
            logits.output(0).get_tensor().set_names({'logits'})
            native_ov.serialize(native_ov.Model([logits], [parameter]),
                                Path(folder) / 'openvino_language_model.xml')
            model = {**self.model, 'path': folder, 'pipeline': 'vlm',
                     'plugin_config': {'DEVICE_PROPERTIES': {
                         'CPU': {'INFERENCE_NUM_THREADS': 'LAZYRIG_INVALID_THREADS'}}},
                     '_placement': {'target': 'HETERO:CPU', 'devices': ['CPU'],
                                    'properties': {'ATTENTION_BACKEND': 'SDPA'}}}
            props = pipeline_properties(native_genai, model)
            with self.assertRaisesRegex(RuntimeError, 'Wrong value for property key INFERENCE_NUM_THREADS'):
                native_genai.VLMPipeline(folder, 'HETERO:CPU', **props)
            # A role-level override must beat the injected CPU device default
            # even after nested properties reach the native HETERO compiler.
            model['_cpu_inference_threads'] = 8
            model['plugin_config'] = {'MODEL_PROPERTIES': {
                'language_model': {'INFERENCE_NUM_THREADS': 'LAZYRIG_INVALID_THREADS'}}}
            props = pipeline_properties(native_genai, model)
            with self.assertRaisesRegex(RuntimeError, 'Wrong value for property key INFERENCE_NUM_THREADS'):
                native_genai.VLMPipeline(folder, 'HETERO:CPU', **props)

    def test_weight_screen_uses_language_file_and_honors_configured_budget(self):
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'CPU']}
        plan = placement.prepare(self.model, self.config)[0]
        hardware = self.config['devices'][1]
        hardware.update(budget_gib=24, total_memory_gib=16)
        with patch('lazyrig.placement.Path.stat', return_value=SimpleNamespace(st_size=18 * 1024**3)):
            # Dedicated VRAM is metadata, not a cap on an explicit shared-RAM budget.
            self.assertIsNone(placement.weight_budget_check(plan, self.config))
            hardware['detected'] = False
            self.assertIsNone(placement.weight_budget_check(plan, self.config))
            hardware['budget_gib'] = 14
            self.assertEqual(placement.weight_budget_check(plan, self.config)['budget_gib'], 14)
            hardware.update(detected=True, total_memory_gib=32)
            self.assertEqual(placement.weight_budget_check(plan, self.config)['budget_gib'], 14)
        checked = []
        def stat(path):
            checked.append(path.name)
            return SimpleNamespace(st_size=18 * 1024**3)
        plan['pipeline'] = 'vlm'
        with patch('lazyrig.placement.Path.stat', stat):
            placement.weight_budget_check(plan, self.config)
        self.assertEqual(checked, ['openvino_language_model.bin'])

    def test_weight_screen_preserves_manual_native_offload_and_missing_exports(self):
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'CPU']}
        plans = placement.prepare(self.model, self.config)
        with patch('lazyrig.placement.Path.stat', return_value=SimpleNamespace(st_size=18 * 1024**3)):
            self.assertIsNotNone(placement.weight_budget_check(plans[0], self.config))
            for plan in plans[1:]:
                self.assertIsNone(placement.weight_budget_check(plan, self.config))
            for plugin in ({'OFFLOAD_RATIO': .5}, {'DEVICE_PROPERTIES': {'GPU.1': {'OFFLOAD_RATIO': .5}}}):
                plan = {**plans[0], 'plugin_config': plugin}
                self.assertIsNone(placement.weight_budget_check(plan, self.config))
            plan = {**plans[0], 'placement': {'mode': 'manual'}}
            self.assertIsNone(placement.weight_budget_check(plan, self.config))
        with patch('lazyrig.placement.Path.stat', side_effect=FileNotFoundError):
            self.assertIsNone(placement.weight_budget_check(plans[0], self.config))


if __name__ == '__main__':
    unittest.main()
