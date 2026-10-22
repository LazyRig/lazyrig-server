"""Hardware identity remains usable when optional driver properties are absent."""
from enum import Enum
import unittest
from unittest.mock import patch

from lazyrig import device_info


class DeviceType(Enum):
    DISCRETE = 1
    INTEGRATED = 2


class DeviceCore:
    def __init__(self, devices):
        self.devices = devices
        self.available_devices = list(devices)

    def get_property(self, device, key):
        value = self.devices[device].get(key, RuntimeError('Property is unavailable'))
        if isinstance(value, Exception):
            raise value
        return value


class DeviceDiscoveryTests(unittest.TestCase):
    def test_mixed_devices_keep_raw_capacity_and_runtime_identity(self):
        core = DeviceCore({
            'CPU': {'FULL_DEVICE_NAME': 'Host processor'},
            'GPU.0': {'FULL_DEVICE_NAME': 'Shared graphics', 'DEVICE_TYPE': DeviceType.INTEGRATED,
                      'GPU_DEVICE_TOTAL_MEM_SIZE': 16 * 1024**3},
            'GPU.1': {'FULL_DEVICE_NAME': 'Graphics accelerator', 'DEVICE_TYPE': DeviceType.DISCRETE,
                      'GPU_DEVICE_TOTAL_MEM_SIZE': 24 * 1024**3},
            'NPU': {'FULL_DEVICE_NAME': 'Neural accelerator'},
        })
        with patch.object(device_info.memory, 'system_memory_gib', return_value=128):
            devices = device_info.discover_devices(core)
        self.assertEqual([d['device_type'] for d in devices], ['cpu', 'integrated', 'discrete', 'other'])
        self.assertEqual([d['total_memory_gib'] for d in devices], [128, 16, 24, None])
        self.assertTrue(all(d['available_memory_gib'] is None for d in devices))
        self.assertTrue(all(d['detected'] and d['canonical_id'] == d['id'] for d in devices))
        self.assertEqual(core.devices['GPU.1']['GPU_DEVICE_TOTAL_MEM_SIZE'], 24 * 1024**3)

    def test_one_plugin_property_failure_does_not_hide_devices(self):
        core = DeviceCore({'GPU.0': {}, 'GPU.1': {
            'FULL_DEVICE_NAME': 'Working graphics', 'DEVICE_TYPE': 'Type.DISCRETE',
            'GPU_DEVICE_TOTAL_MEM_SIZE': 12 * 1024**3}})
        devices = device_info.discover_devices(core)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]['name'], 'GPU.0')
        self.assertEqual(devices[0]['device_type'], 'other')
        self.assertIsNone(devices[0]['total_memory_gib'])
        self.assertEqual(devices[1]['total_memory_gib'], 12)

    def test_only_explicit_gpu_markers_are_used_when_type_is_missing(self):
        names = ['Intel Arc graphics', 'NVIDIA graphics', 'Graphics (dGPU)',
                 'Graphics (iGPU)', 'Graphics (dGPU) (iGPU)']
        core = DeviceCore({f'GPU.{i}': {'FULL_DEVICE_NAME': name} for i, name in enumerate(names)})
        devices = device_info.discover_devices(core)
        self.assertEqual([d['device_type'] for d in devices],
                         ['other', 'other', 'discrete', 'integrated', 'other'])
        self.assertEqual([d['name'] for d in devices], names)

    def test_reported_type_takes_precedence_over_name(self):
        core = DeviceCore({'GPU.0': {'FULL_DEVICE_NAME': 'Graphics (dGPU)',
                                     'DEVICE_TYPE': 'INTEGRATED'},
                           'GPU.1': {'FULL_DEVICE_NAME': 'Graphics (dGPU)',
                                     'DEVICE_TYPE': 'UNKNOWN'}})
        self.assertEqual([d['device_type'] for d in device_info.discover_devices(core)],
                         ['integrated', 'other'])

    def test_invalid_capacity_is_unknown_instead_of_an_invented_budget(self):
        invalid = [None, 0, -1, float('nan'), float('inf'), True, 'unavailable']
        core = DeviceCore({f'GPU.{i}': {'GPU_DEVICE_TOTAL_MEM_SIZE': value}
                           for i, value in enumerate(invalid)})
        self.assertTrue(all(d['total_memory_gib'] is None for d in device_info.discover_devices(core)))
        for value in (None, float('nan'), -1):
            with self.subTest(cpu_capacity=value), patch.object(
                    device_info.memory, 'system_memory_gib', return_value=value):
                self.assertIsNone(device_info.discover_devices(DeviceCore({'CPU': {}}))[0]['total_memory_gib'])

    def test_gpu_alias_resolves_only_existing_unambiguous_devices(self):
        cases = [(['CPU', 'GPU.0', 'GPU.1'], 'GPU.0'),
                 (['CPU', 'GPU.1'], 'GPU.1'),
                 (['GPU.1', 'GPU.2'], 'GPU'),
                 (['CPU'], 'GPU'), (['GPU'], 'GPU')]
        for ids, expected in cases:
            devices = [{'id': value} for value in ids]
            with self.subTest(ids=ids):
                self.assertEqual(device_info.canonical_device('GPU', devices), expected)
                self.assertEqual(device_info.canonical_device('GPU.9', devices), 'GPU.9')
                self.assertEqual(device_info.canonical_device('CPU', devices), 'CPU')


if __name__ == '__main__':
    unittest.main()
