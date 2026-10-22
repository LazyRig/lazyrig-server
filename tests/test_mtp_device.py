"""MTP routing/admission contracts; these do not validate native GPU generation."""
import copy
from pathlib import Path
import tempfile
import threading
import types
import unittest

from lazyrig import mtp, placement
from lazyrig.config import validate
from lazyrig.runtime_settings import pipeline_properties
from lazyrig.scheduler import PlacementAdmissionError, Scheduler, load_config
from lazyrig.worker import NativeWorkerError
from support import DEFAULT
from test_scheduler_placement import Worker


class MTPDeviceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for suffix in ('xml', 'bin'):
            Path(self.folder.name, 'openvino_mtp_model.' + suffix).write_bytes(b'fixture')
        self.model = {'id': 'chat', 'path': self.folder.name, 'pipeline': 'vlm',
                      'task': 'text-generation', 'device': 'GPU.1', 'memory_gib': 5,
                      'mtp': {'enabled': True, 'device': 'CPU'}, 'generation': {'temperature': 0}}
        self.config = {'mode': 'openvino', 'devices': [
            {'id': 'GPU.1', 'policy': 'budget', 'budget_gib': 20},
            {'id': 'GPU.0', 'policy': 'budget', 'budget_gib': 8},
            {'id': 'CPU', 'policy': 'budget', 'budget_gib': 48, 'inference_threads': 6},
            {'id': 'NPU', 'policy': 'budget', 'budget_gib': 8}], 'models': [self.model]}
        self.calls, self.events = [], []
        self.failures = []
        self.scheduler = Scheduler(self.config, self.factory)
        self.cancelled = threading.Event()

    def factory(self, mode, model, timeout, cancelled):
        self.calls.append(copy.deepcopy(model))
        if self.failures:
            failure = self.failures.pop(0)
            if failure:
                raise failure
        return Worker(model)

    def emit(self, kind, **data):
        self.events.append((kind, data))

    def load(self, model=None):
        model = self.model if model is None else model
        with self.scheduler.lease_models([model], self.cancelled):
            return self.scheduler.ensure(model, self.cancelled, self.emit)

    def native_options(self, model):
        calls = []
        genai = types.SimpleNamespace(SchedulerConfig=types.SimpleNamespace,
            draft_model=lambda *a, **k: calls.append((a, k)))
        properties = pipeline_properties(genai, model)
        return calls, properties

    def test_settings_validate_known_physical_device(self):
        for device in ('CPU', 'GPU.0', 'GPU.1', ''):
            model = {**self.model, 'mtp': {'enabled': True, 'device': device}}
            mtp.validate_settings(model, self.config)
        for device in ('NPU', 'AUTO', 'HETERO:GPU.1,CPU', 'GPU.9', None, 2, {}):
            with self.subTest(device=device), self.assertRaises(ValueError):
                mtp.validate_settings({**self.model, 'mtp': {'device': device}}, self.config)
        config = copy.deepcopy(DEFAULT)
        config['models'][0]['mtp'] = {'device': 'GPU.9'}
        with self.assertRaisesRegex(ValueError, 'MTP device is not configured'):
            validate(config)

    def test_mtp_device_and_tokens_have_distinct_reload_behavior(self):
        old = load_config(self.model)
        count = copy.deepcopy(self.model)
        count['mtp']['num_assistant_tokens'] = 8
        self.assertEqual(old, load_config(count))
        count['mtp']['device'] = 'GPU.0'
        self.assertNotEqual(old, load_config(count))
        self.model['mtp']['enabled'] = count['mtp']['enabled'] = False
        self.assertEqual(load_config(self.model), load_config(count))

    def test_constructor_receives_selected_device_and_own_cpu_defaults(self):
        self.load()
        worker_model = self.calls[-1]
        self.assertEqual(worker_model['_cpu_inference_threads'], 6)
        calls, properties = self.native_options(worker_model)
        self.assertEqual(calls, [((self.folder.name, 'CPU'),
                                {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 6})])
        self.assertFalse(properties['scheduler_config'].enable_prefix_caching)
        self.assertEqual(self.model['device'], 'GPU.1')
        self.assertNotIn('_cpu_inference_threads', self.model)

    def test_draft_on_other_gpu_does_not_inherit_main_device_properties(self):
        model = copy.deepcopy(self.model)
        model['mtp']['device'] = 'GPU.0'
        model['plugin_config'] = {'PERFORMANCE_HINT': 'THROUGHPUT', 'GPU_ONLY_FIXTURE': True}
        calls, props = self.native_options(model)
        self.assertEqual(calls[0], ((self.folder.name, 'GPU.0'), {'PERFORMANCE_HINT': 'LATENCY'}))
        self.assertEqual(props['PERFORMANCE_HINT'], 'THROUGHPUT')

    def test_same_as_model_retains_native_property_inheritance(self):
        self.model['mtp'].pop('device')
        calls, _ = self.native_options(self.model)
        self.assertEqual(calls, [((self.folder.name, 'GPU.1'), {})])

    def test_cpu_model_override_takes_precedence_over_device_default(self):
        for plugin in ({'INFERENCE_NUM_THREADS': 3}, {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 4}}}):
            with self.subTest(plugin=plugin):
                self.model['plugin_config'] = plugin
                self.load()
                calls, _ = self.native_options(self.calls[-1])
                self.assertEqual(calls[0][1]['INFERENCE_NUM_THREADS'], 3 if 'INFERENCE_NUM_THREADS' in plugin else 4)
                self.assertNotIn('_cpu_inference_threads', self.calls[-1])

    def test_both_devices_leased_and_separate_draft_budget_reserved(self):
        with self.scheduler.lease_models([self.model], self.cancelled):
            self.assertEqual(self.scheduler.busy, {'@model:chat', 'CPU', 'GPU.1'})
            self.scheduler.ensure(self.model, self.cancelled, self.emit)
        resident = self.scheduler.resident['chat'][0]
        self.assertEqual(self.scheduler._reservations(resident), {'GPU.1': 5, 'CPU': 48})
        self.assertEqual(self.scheduler.busy, set())
        self.assertEqual(next(d['mtp_device'] for k, d in self.events if k == 'loading'), 'CPU')
        self.assertEqual(self.scheduler.snapshot()['workers'][0]['mtp_device'], 'CPU')

    def test_changing_draft_device_reloads_worker_but_draft_count_does_not(self):
        previous = self.load()
        self.model['mtp']['num_assistant_tokens'] = 4
        self.assertIs(self.load(), previous)
        self.model['mtp']['device'] = 'GPU.0'
        self.assertIsNot(self.load(), previous)
        self.assertFalse(previous.is_alive())
        self.assertEqual(self.calls[-1]['mtp']['device'], 'GPU.0')
        self.assertEqual(self.scheduler.snapshot()['workers'][0]['mtp_device'], 'GPU.0')

    def test_existing_draft_device_resident_is_evicted_and_main_remains_reusable(self):
        other = {**self.model, 'id': 'other', 'device': 'CPU', 'mtp': {'enabled': False}, 'memory_gib': 4}
        previous = self.load(other)
        worker = self.load()
        self.assertFalse(previous.is_alive())
        self.assertEqual(set(self.scheduler.resident), {'chat'})
        self.assertIs(self.load(), worker)

    def test_alias_does_not_double_reserve_or_lease_same_gpu(self):
        self.config['devices'] = [{'id': 'GPU', 'budget_gib': 20, 'policy': 'budget'},
                                  {'id': 'GPU.0', 'budget_gib': 20, 'policy': 'budget'}]
        self.model['device'] = 'GPU'
        self.model['mtp']['device'] = 'GPU.0'
        self.assertEqual(self.scheduler._resource_devices(self.model), {'GPU.0'})
        self.assertEqual(self.scheduler._reservations(self.model), {'GPU.0': 5})

    def test_automatic_fallback_following_and_explicit_draft_selection(self):
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'CPU'], 'shares': {}, 'allow_fallback': True}
        for draft, expected in (('', 'CPU'), ('GPU.0', 'GPU.0')):
            self.model['mtp']['device'] = draft
            self.scheduler = Scheduler(self.config, self.factory)
            self.failures = [NativeWorkerError('CL_MEM_OBJECT_ALLOCATION_FAILURE')]
            self.load()
            resolved = self.calls[-1]
            self.assertEqual(resolved['_placement']['target'], 'CPU')
            self.assertEqual(mtp.device(resolved), expected)
            self.assertEqual(placement.devices(resolved), {'CPU'})
            calls, _ = self.native_options(resolved)
            self.assertEqual(calls[0][0][1], expected)

    def test_cpu_thread_setting_reloads_cpu_draft_worker(self):
        previous = self.load()
        self.config['devices'][2]['inference_threads'] = 10
        self.assertIsNot(self.load(), previous)
        self.assertFalse(previous.is_alive())
        calls, _ = self.native_options(self.calls[-1])
        self.assertEqual(calls[0][1]['INFERENCE_NUM_THREADS'], 10)

    def test_zero_share_device_does_not_drop_an_explicit_draft_lease(self):
        self.model['placement'] = {'mode': 'manual', 'devices': ['GPU.1'], 'shares': {'GPU.1': 100}}
        self.assertEqual(placement.lease_devices(self.model, self.config), {'@model:chat', 'GPU.1', 'CPU'})

    def test_mtp_shared_memory_budget_allows_native_attempt_on_primary_gpu(self):
        with Path(self.folder.name, 'openvino_language_model.bin').open('wb') as stream:
            stream.truncate(26 * 1024**3)
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'GPU.0'], 'shares': {}}
        self.config['devices'][0].update(detected=True, total_memory_gib=24, budget_gib=36)
        for draft in ('', 'GPU.1', 'GPU.0', 'CPU'):
            with self.subTest(draft=draft):
                self.model['mtp']['device'] = draft
                self.scheduler = Scheduler(self.config, self.factory)
                self.calls, self.events = [], []
                self.load()
                self.assertEqual([m['_placement']['target'] for m in self.calls], ['GPU.1'])
                self.assertEqual(mtp.device(self.calls[0]), draft or 'GPU.1')
                self.assertFalse(any(k == 'placement_fallback' for k, _ in self.events))
                self.assertFalse(self.scheduler.busy)
                self.scheduler.close()

    def test_oversized_main_model_explains_mtp_limit_for_every_draft_device(self):
        with Path(self.folder.name, 'openvino_language_model.bin').open('wb') as stream:
            stream.truncate(24 * 1024**3)  # Sparse metadata fixture; no model allocation.
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1', 'GPU.0'], 'shares': {}}
        for draft in ('', 'GPU.1', 'GPU.0', 'CPU'):
            with self.subTest(draft=draft):
                self.model['mtp']['device'] = draft
                with self.assertRaises(PlacementAdmissionError) as caught:
                    self.load()
                error = caught.exception
                self.assertEqual(error.error_code, 'mtp_model_weight_budget_exceeded')
                self.assertEqual(error.weight_check, {'device': 'GPU.1', 'weights_gib': 24, 'budget_gib': 20})
                self.assertIn('automatic model splitting is disabled', str(error))
                self.assertIn('Moving the MTP head does not offload the main weights', str(error))
                self.assertIn('explicitly allow whole-model fallback to a selected CPU', str(error))
                self.assertIn('raise the device Planning budget', str(error))
                self.assertEqual(self.calls, [])
                self.assertFalse(self.scheduler.busy)

        # Selecting CPU for the head alone did not allow this. Selecting it for
        # the main allocation plus explicit permission does, and keeps the guard.
        self.model['placement']['devices'].append('CPU')
        self.model['placement']['allow_fallback'] = True
        self.load()
        self.assertEqual([m['_placement']['target'] for m in self.calls], ['CPU'])

    def test_non_mtp_budget_error_does_not_imply_an_mtp_restriction(self):
        with Path(self.folder.name, 'openvino_language_model.bin').open('wb') as stream:
            stream.truncate(24 * 1024**3)
        self.model['placement'] = {'mode': 'auto', 'devices': ['GPU.1'], 'shares': {}}
        self.model['mtp']['enabled'] = False
        with self.assertRaises(PlacementAdmissionError) as caught:
            self.load()
        self.assertEqual(caught.exception.error_code, 'model_weight_budget_exceeded')
        self.assertNotIn('MTP', str(caught.exception))
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
