"""Hardware-derived defaults and explicit reservations remain distinct."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lazyrig import memory
from lazyrig.service import Service as NativeService
from support import DEFAULT, Service
from test_server_contract import fixture_export


class HardwareBudgetTests(unittest.TestCase):
    def test_physical_ram_on_posix(self):
        with patch.object(memory, 'os', SimpleNamespace(name='posix', sysconf=lambda key: {
                'SC_PHYS_PAGES': 16 * 1024**2, 'SC_PAGE_SIZE': 4096}[key])):
            self.assertEqual(memory.system_memory_gib(), 64)

    def test_windows_physical_ram_and_failed_query(self):
        def query(pointer):
            self.assertEqual(pointer._obj.length, 64)
            pointer._obj.total_physical = 32 * 1024**3
            return 1
        kernel = SimpleNamespace(GlobalMemoryStatusEx=query)
        with patch.object(memory, 'os', SimpleNamespace(name='nt')), patch.object(
                memory.ctypes, 'windll', SimpleNamespace(kernel32=kernel), create=True):
            self.assertEqual(memory.system_memory_gib(), 32)
            kernel.GlobalMemoryStatusEx = lambda pointer: 0
            self.assertIsNone(memory.system_memory_gib())

    def test_discovery_uses_hardware_and_preserves_configured_budget(self):
        def prop(device, name):
            return device if name == 'FULL_DEVICE_NAME' else 24 * 1024**3
        core = SimpleNamespace(available_devices=['CPU', 'GPU.0', 'GPU.1'], get_property=prop)
        service = NativeService.__new__(NativeService)
        service.config = {'devices': [{'id': 'GPU.1', 'budget_gib': 19.8, 'policy': 'budget'}]}
        with patch.dict('sys.modules', {'openvino': SimpleNamespace(Core=lambda: core, __version__='fixture'),
                                      'openvino_genai': SimpleNamespace(__version__='fixture')}), patch.object(
                                          memory, 'system_memory_gib', return_value=64):
            result = service.discover()
        self.assertTrue(result['available'])
        self.assertEqual({d['id']: d['budget_gib'] for d in service.config['devices']},
                         {'CPU': 54.4, 'GPU.0': 20.4, 'GPU.1': 19.8})

    def test_unknown_memory_has_bounded_fallback(self):
        core = SimpleNamespace(get_property=lambda *args: float('nan'))
        with patch.object(memory, 'system_memory_gib', return_value=None):
            self.assertEqual(memory.detected_budget(core, 'CPU'), 8)
            self.assertEqual(memory.detected_budget(core, 'GPU.0'), 2)


class ReservationDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = str(self.root / 'data')
        self.service = Service(config)
        fixture_export(self.root / 'export')

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def register(self, alias, device='TEST.CPU', **extra):
        return self.service.register({'id': alias, 'path': str(self.root / 'export'),
                                      'device': device, **extra})

    def test_registration_uses_selected_budget_and_persists_number(self):
        self.assertEqual(self.register('cpu')['memory_gib'], 48)
        self.assertEqual(self.register('gpu', 'TEST.GPU')['memory_gib'], 22)
        saved = json.loads(self.service.settings_path.read_text())
        self.assertEqual(next(m['memory_gib'] for m in saved['models'] if m['id'] == 'cpu'), 48)

    def test_manual_registration_and_unrelated_edits_preserve_value(self):
        self.assertEqual(self.register('manual', memory_gib=3.5)['memory_gib'], 3.5)
        self.service.settings({'model': 'manual', 'name': 'Renamed'})
        self.service.settings({'model': 'manual', 'device': 'TEST.CPU'})
        self.assertEqual(self.service.model('manual')['memory_gib'], 3.5)
        self.assertEqual(self.service.model('qwen-chat')['memory_gib'], 20)

    def test_device_switch_refills_unless_explicit_value_is_supplied(self):
        self.register('move', memory_gib=3)
        self.service.settings({'model': 'move', 'device': 'TEST.GPU'})
        self.assertEqual(self.service.model('move')['memory_gib'], 22)
        self.service.settings({'model': 'move', 'device': 'TEST.CPU', 'memory_gib': 4})
        self.assertEqual(self.service.model('move')['memory_gib'], 4)

    def test_invalid_values_and_unknown_devices_do_not_mutate_registry(self):
        before = copy.deepcopy(self.service.config)
        for value in (0, -1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.register('invalid', memory_gib=value)
        with self.assertRaises(ValueError):
            self.register('unknown', device='GPU.missing')
        self.assertEqual(self.service.config, before)


if __name__ == '__main__':
    unittest.main()
