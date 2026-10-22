import copy
from pathlib import Path
import tempfile
import unittest

from lazyrig.runtime_settings import pipeline_properties, pipeline_target

try:
    import openvino as ov
    import openvino_genai as genai
except ImportError:
    ov = genai = None


def model():
    return {
        'id': 'vlm', 'device': 'GPU.1', 'pipeline': 'vlm', 'path': '/model',
        'placement': {'mode': 'manual'},
        '_placement': {
            'target': 'HETERO:GPU.1,GPU.0', 'devices': ['GPU.1', 'GPU.0'],
            'properties': {'ATTENTION_BACKEND': 'SDPA', 'DEVICE_PROPERTIES': {
                'GPU.1': {'QUERY_MODEL_RATIO': .5}, 'GPU.0': {'QUERY_MODEL_RATIO': 1}}}},
        '_placement_preflight': {'status': 'preflight_verified', 'language_input_device': 'GPU.1'},
        'plugin_config': {},
    }


def native_role_properties(properties, role, target):
    """The pinned GenAI get_model_properties contract, before native compile."""
    result = {key: value for key, value in properties.items()
              if key not in {'MODEL_PROPERTIES', 'DEVICE_PROPERTIES', 'ATTENTION_BACKEND'}}
    result.update(properties.get('DEVICE_PROPERTIES', {}).get(target, {}))
    result.update(properties.get('MODEL_PROPERTIES', {}).get(role, {}))
    return result


class VLMComponentPlacementTests(unittest.TestCase):
    def test_verified_split_pins_embedding_handoff_without_changing_language_split(self):
        for pipeline in ('vlm', 'vlm-text'):
            with self.subTest(pipeline=pipeline):
                value = model()
                value['pipeline'] = pipeline
                value['plugin_config'] = {'DEVICE_PROPERTIES': {
                    'GPU.1': {'NUM_STREAMS': 2}, 'GPU.0': {'NUM_STREAMS': 3}}}
                before = copy.deepcopy(value)
                self.assertEqual(pipeline_target(value), 'HETERO')
                props = pipeline_properties(object(), value)
                language = native_role_properties(props, 'language_model', 'HETERO')
                self.assertEqual(language['MULTI_DEVICE_PRIORITIES'], 'GPU.1,GPU.0')
                self.assertEqual(language['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'], .5)
                self.assertEqual(language['DEVICE_PROPERTIES']['GPU.0']['QUERY_MODEL_RATIO'], 1)
                for role in ('text_embeddings', 'text_embeddings_per_layer'):
                    embedding = native_role_properties(props, role, 'HETERO')
                    self.assertEqual(embedding['MULTI_DEVICE_PRIORITIES'], 'GPU.1')
                    self.assertEqual(embedding['DEVICE_PROPERTIES'], {'GPU.1': {'NUM_STREAMS': 2}})
                self.assertEqual(value, before)

    def test_verified_input_device_controls_handoff_not_primary_or_open_order(self):
        value = model()
        value['_placement_preflight']['language_input_device'] = 'GPU.0'
        props = pipeline_properties(object(), value)
        embedding = native_role_properties(props, 'text_embeddings', 'HETERO')
        self.assertEqual(embedding['MULTI_DEVICE_PRIORITIES'], 'GPU.0')
        self.assertEqual(embedding['DEVICE_PROPERTIES'], {'GPU.0': {}})

    def test_role_and_exact_target_nonplacement_properties_retain_precedence(self):
        value = model()
        value['plugin_config'] = {
            'CACHE_DIR': 'global',
            'DEVICE_PROPERTIES': {
                'GPU': {'PERFORMANCE_HINT': 'LATENCY'}, 'GPU.1': {'NUM_STREAMS': 2},
                'HETERO:GPU.1,GPU.0': {'CACHE_DIR': 'target', 'DEVICE_PROPERTIES': {
                    'GPU.1': {'INFERENCE_PRECISION_HINT': 'f16'}}}},
            'MODEL_PROPERTIES': {'text_embeddings': {
                'CACHE_DIR': 'role', 'NUM_STREAMS': 4,
                'DEVICE_PROPERTIES': {'GPU.1': {'NUM_STREAMS': 6}, 'GPU.0': {'NUM_STREAMS': 7}}}}}
        props = pipeline_properties(object(), value)
        embedding = native_role_properties(props, 'text_embeddings', 'HETERO')
        self.assertEqual(embedding['CACHE_DIR'], 'role')
        self.assertEqual(embedding['DEVICE_PROPERTIES']['GPU.1'], {
            'NUM_STREAMS': 6, 'INFERENCE_PRECISION_HINT': 'f16'})
        self.assertEqual(embedding['DEVICE_PROPERTIES']['GPU'], {'PERFORMANCE_HINT': 'LATENCY'})
        self.assertNotIn('GPU.0', embedding['DEVICE_PROPERTIES'])
        self.assertNotIn('HETERO', props['DEVICE_PROPERTIES']['HETERO']['DEVICE_PROPERTIES'])
        self.assertNotIn('HETERO:GPU.1,GPU.0', props['DEVICE_PROPERTIES']['HETERO']['DEVICE_PROPERTIES'])

    def test_cpu_default_is_not_forwarded_to_gpu_embedding_role(self):
        value = model()
        value['_placement']['devices'].append('CPU')
        value['_placement']['target'] += ',CPU'
        value['_cpu_inference_threads'] = 8
        props = pipeline_properties(object(), value)
        language = native_role_properties(props, 'language_model', 'HETERO')
        self.assertEqual(language['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 8)
        self.assertNotIn('CPU', native_role_properties(props, 'text_embeddings', 'HETERO')['DEVICE_PROPERTIES'])

    def test_unverified_and_unaffected_modes_keep_existing_compile_target(self):
        changes = (
            lambda value: value.pop('_placement_preflight'),
            lambda value: value['_placement_preflight'].update(status='unknown'),
            lambda value: value['placement'].update(mode='auto'),
            lambda value: value.update(pipeline='llm'),
            lambda value: value['_placement'].update(target='HETERO:GPU.1,CPU', devices=['GPU.1', 'CPU']),
        )
        for change in changes:
            value = model()
            change(value)
            with self.subTest(value=value):
                self.assertEqual(pipeline_target(value), value['_placement']['target'])
                props = pipeline_properties(object(), value)
                self.assertNotIn('MULTI_DEVICE_PRIORITIES', props)

    def test_verified_split_without_one_language_input_device_fails_before_compile(self):
        for missing_or_ambiguous in (None, '', ['GPU.1', 'GPU.0']):
            value = model()
            value['_placement_preflight']['language_input_device'] = missing_or_ambiguous
            with self.subTest(device=missing_or_ambiguous), self.assertRaisesRegex(
                    ValueError, 'Cannot verify the language inputs_embeds device'):
                pipeline_properties(object(), value)

    def test_conflicting_explicit_priorities_fail_instead_of_rerouting(self):
        for plugin in (
            {'MULTI_DEVICE_PRIORITIES': 'CPU'},
            {'MODEL_PROPERTIES': {'text_embeddings': {'MULTI_DEVICE_PRIORITIES': 'GPU.0'}}},
            {'MODEL_PROPERTIES': {'language_model': {'MULTI_DEVICE_PRIORITIES': 'GPU.0'}}},
            {'DEVICE_PROPERTIES': {'HETERO:GPU.1,GPU.0': {'MULTI_DEVICE_PRIORITIES': 'GPU.0'}}},
        ):
            value = model()
            value['plugin_config'] = plugin
            with self.subTest(plugin=plugin), self.assertRaisesRegex(ValueError, 'Conflicting MULTI_DEVICE_PRIORITIES'):
                pipeline_properties(object(), value)
        value = model()
        value['_placement_preflight']['language_input_device'] = 'GPU.2'
        with self.assertRaisesRegex(ValueError, 'outside the selected allocation'):
            pipeline_target(value)

    @unittest.skipIf(ov is None, 'OpenVINO is not installed')
    def test_native_embedding_role_uses_only_its_verified_cpu_destination(self):
        core = ov.Core()
        if 'CPU' not in core.available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        value = model()
        value['_placement']['devices'].append('CPU')
        value['_placement']['target'] += ',CPU'
        value['_placement_preflight']['language_input_device'] = 'CPU'
        # One thread is supported even by small CI runners; larger requests may
        # be capped by OpenVINO to the processors available on that machine.
        value['_cpu_inference_threads'] = 1
        props = pipeline_properties(object(), value)
        from openvino import opset13 as ops
        parameter = ops.parameter([1, 4], ov.Type.f32)
        graph = ov.Model([ops.add(parameter, ops.constant(1, ov.Type.f32))], [parameter])
        # No GPU is available here: retaining either foreign priority would fail.
        embedding = native_role_properties(props, 'text_embeddings', pipeline_target(value))
        self.assertEqual(embedding['MULTI_DEVICE_PRIORITIES'], 'CPU')
        self.assertEqual(embedding['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 1)
        compiled = core.compile_model(graph, pipeline_target(value), embedding)
        self.assertEqual(compiled.get_property('EXECUTION_DEVICES'), ['CPU'])
        self.assertEqual(compiled.get_property('DEVICE_PROPERTIES')['CPU']['INFERENCE_NUM_THREADS'], 1)

    @unittest.skipIf(genai is None, 'OpenVINO GenAI is not installed')
    def test_native_vlm_bare_hetero_forwards_exact_target_language_properties(self):
        if 'CPU' not in ov.Core().available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        from openvino import opset13 as ops
        with tempfile.TemporaryDirectory() as folder:
            parameter = ops.parameter([1, -1, 4], ov.Type.f32, name='inputs_embeds')
            logits = ops.add(parameter, ops.constant(1, ov.Type.f32))
            logits.output(0).get_tensor().set_names({'logits'})
            ov.serialize(ov.Model([logits], [parameter]), Path(folder) / 'openvino_language_model.xml')
            with self.assertRaisesRegex(RuntimeError, 'Wrong value for property key INFERENCE_NUM_THREADS'):
                genai.VLMPipeline(folder, 'HETERO', ATTENTION_BACKEND='SDPA', MULTI_DEVICE_PRIORITIES='CPU',
                                  DEVICE_PROPERTIES={'HETERO': {'DEVICE_PROPERTIES': {
                                      'CPU': {'INFERENCE_NUM_THREADS': 'LAZYRIG_INVALID_THREADS'}}}})


if __name__ == '__main__':
    unittest.main()
