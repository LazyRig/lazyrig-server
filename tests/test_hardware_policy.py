"""Support labels are advisory; defaults and automatic allocation are conservative."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from lazyrig import device_info, placement
from lazyrig.config import load, restore_settings
from lazyrig.runtime_settings import pipeline_properties, validate_runtime
from support import Service
from test_core import wait_job
from test_device_info import DeviceCore
from test_service_placement import hardware_config
from test_server_contract import fixture_export


def hardware():
    raw = {
        'GPU.0': {'FULL_DEVICE_NAME': 'Intel(R) Graphics (iGPU)', 'DEVICE_TYPE': 'INTEGRATED'},
        'GPU.1': {'FULL_DEVICE_NAME': 'Intel(R) Arc Pro B60 (dGPU)', 'DEVICE_TYPE': 'DISCRETE'},
        'GPU.2': {'FULL_DEVICE_NAME': 'NVIDIA RTX 5060 Ti', 'DEVICE_TYPE': 'DISCRETE'},
        'CPU': {'FULL_DEVICE_NAME': 'Intel(R) Core CPU'},
        'NPU': {'FULL_DEVICE_NAME': 'Intel(R) AI Boost'},
    }
    devices = device_info.discover_devices(DeviceCore(raw))
    for d, budget in zip(devices, [8, 22, 14, 48, 8]):
        d.update(budget_gib=budget, policy='budget')
    return devices


class HardwarePolicyTests(unittest.TestCase):
    def test_enumerated_devices_are_preserved_and_support_is_not_inferred_from_gpu_id(self):
        devices = hardware()
        self.assertEqual([d['id'] for d in devices], ['GPU.0', 'GPU.1', 'GPU.2', 'CPU', 'NPU'])
        self.assertEqual([d['support_level'] for d in devices],
                         ['supported', 'supported', 'experimental', 'supported', 'experimental'])
        for name in ('AMD Ryzen CPU', 'Unknown processor'):
            self.assertEqual(device_info.support_metadata('CPU', name)['support_level'], 'supported')
        self.assertEqual(device_info.canonical_device('GPU', devices), 'GPU.0')

    def test_default_prefers_one_suitable_intel_gpu_then_cpu_without_changing_explicit_device(self):
        devices = hardware()
        self.assertEqual(device_info.preferred_device(devices), 'GPU.1')
        self.assertEqual(device_info.preferred_device(devices, weights_gib=23), 'CPU')
        devices[1]['detected'] = False
        self.assertEqual(device_info.preferred_device(devices), 'GPU.0')
        devices[0]['detected'] = False
        self.assertEqual(device_info.preferred_device(devices), 'CPU')
        self.assertEqual(placement.default_placement({'device': 'GPU.2'}, {'devices': devices}),
                         {'mode': 'auto', 'devices': ['GPU.2'], 'shares': {}, 'allow_fallback': False, 'shared_memory': 'budget'})

    def test_auto_retains_experimental_choices_and_avoids_double_counting_host_memory(self):
        config = {'devices': hardware()}
        model = {'id': 'test', 'device': 'GPU.1', 'memory_gib': 20, 'pipeline': 'llm',
                 'placement': {'mode': 'auto', 'devices': ['GPU.1', 'GPU.2', 'GPU.0', 'CPU'], 'shares': {}}}
        plans = placement.prepare(model, config)
        targets = [p['_placement']['target'] for p in plans]
        self.assertEqual(targets[:3], ['GPU.1', 'HETERO:GPU.1,GPU.2,GPU.0', 'HETERO:GPU.1,GPU.2,CPU'])
        self.assertIn('HETERO:GPU.1,GPU.2,GPU.0', targets)
        self.assertIn('HETERO:GPU.1,GPU.2,CPU', targets)
        self.assertFalse(any('GPU.0' in p['_placement']['devices'] and 'CPU' in p['_placement']['devices'] for p in plans))
        config['devices'][2]['detected'] = False
        self.assertFalse(any('GPU.2' in p['_placement']['devices'] for p in placement.prepare(model, config)))
        model['mtp'] = {'enabled': True}
        self.assertEqual([p['_placement']['target'] for p in placement.prepare(model, config)], ['GPU.1'])
        model['placement']['allow_fallback'] = True
        self.assertEqual([p['_placement']['target'] for p in placement.prepare(model, config)], ['GPU.1', 'GPU.0', 'CPU'])

    def test_saved_manual_config_and_settings_migrate_without_losing_selected_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            config = hardware_config(directory)
            config['models'][0]['placement'] = {'mode': 'manual', 'devices': ['GPU.1', 'CPU'],
                                               'shares': {'GPU.1': 50, 'CPU': 50}}
            for filename, reader in [('settings.json', lambda path: restore_settings(config)),
                                     ('config.json', lambda path: load(path))]:
                path = Path(directory) / filename
                path.write_text(json.dumps(config))
                restored = reader(path)
                self.assertEqual(restored['models'][0]['placement'],
                                 {'mode': 'auto', 'devices': ['GPU.1', 'CPU'], 'shares': {}})
                self.assertEqual(config['models'][0]['placement']['mode'], 'manual')

    def test_registration_and_experimental_mtp_selection_reach_worker_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            config = hardware_config(directory)
            config['devices'] = hardware()
            export = Path(directory) / 'export'
            fixture_export(export)
            for suffix in ('xml', 'bin'):
                (export / ('openvino_mtp_model.' + suffix)).write_text('fixture')
            service = Service(config)
            try:
                model = service.register({'id': 'default', 'path': str(export)})
                self.assertEqual(model['device'], 'GPU.1')
                self.assertEqual(model['placement']['devices'], ['GPU.1'])
                service.settings({'model': 'default', 'device': 'GPU.2',
                                  'mtp': {'enabled': True, 'device': 'GPU.2', 'num_assistant_tokens': 2}})
                result = wait_job(service, service.control('default', 'load')['id'])
                self.assertEqual(result['status'], 'completed', result.get('error'))
                saved = service.scheduler.resident['default'][0]
                self.assertEqual(saved['_placement']['target'], 'GPU.2')
                self.assertEqual(saved['mtp']['device'], 'GPU.2')
            finally:
                service.close()


class ScopedRuntimeSettingsTests(unittest.TestCase):
    def model(self, target, pipeline='llm'):
        return {'pipeline': pipeline, 'device': target.split(':')[-1].split(',')[0],
                '_placement': {'target': target, 'devices': target.split(':')[-1].split(',')},
                'plugin_config': {'EXECUTION_MODE_HINT': 'PERFORMANCE', 'COMPILATION_NUM_THREADS': 2,
                                  'SCHEDULING_CORE_TYPE': 'ANY_CORE', 'ENABLE_HYPER_THREADING': False,
                                  'ENABLE_CPU_PINNING': False}}

    def test_scoped_properties_reach_only_matching_plugins_and_preserve_custom_overrides(self):
        for pipeline in ('llm', 'vlm'):
            model = self.model('HETERO:GPU.1,CPU', pipeline)
            model['plugin_config']['DEVICE_PROPERTIES'] = {'CPU': {'ENABLE_CPU_PINNING': True}}
            original = copy.deepcopy(model)
            properties = pipeline_properties(object(), model)
            dp = properties['DEVICE_PROPERTIES']
            self.assertEqual(dp['GPU.1'], {'COMPILATION_NUM_THREADS': 2})
            self.assertEqual(dp['CPU'], {'SCHEDULING_CORE_TYPE': 'ANY_CORE',
                                        'ENABLE_HYPER_THREADING': False, 'ENABLE_CPU_PINNING': True})
            if pipeline == 'vlm':
                self.assertEqual(dp['HETERO:GPU.1,CPU']['DEVICE_PROPERTIES']['GPU.1'], dp['GPU.1'])
            self.assertEqual(model, original)
        props = pipeline_properties(object(), self.model('GPU.1'))
        self.assertNotIn('ENABLE_CPU_PINNING', props)
        self.assertEqual(props['COMPILATION_NUM_THREADS'], 2)

    def test_native_cpu_compiles_and_infers_with_scheduling_controls(self):
        try:
            import numpy as np
            import openvino as ov
            from openvino import opset13 as ops
        except ImportError:
            self.skipTest('Optional native OpenVINO runtime')
        model = self.model('CPU')
        validate_runtime(model)
        properties = pipeline_properties(object(), model)
        self.assertNotIn('COMPILATION_NUM_THREADS', properties)
        x = ops.parameter([2], ov.Type.f32)
        graph = ov.Model([ops.add(x, ops.constant(np.array([1, 2], dtype=np.float32)))], [x])
        compiled = ov.Core().compile_model(graph, 'CPU', properties)
        result = compiled([np.array([3, 4], dtype=np.float32)])[0]
        np.testing.assert_array_equal(result, [4, 6])


if __name__ == '__main__':
    unittest.main()
