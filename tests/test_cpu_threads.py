"""CPU device defaults apply only to workers that actually execute on CPU."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from lazyrig.runtime_settings import pipeline_properties
from lazyrig.scheduler import Scheduler
from lazyrig.worker import NativeWorkerError
from support import Service
from test_scheduler_placement import Worker
from test_service_placement import hardware_config


class CPUThreadSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = Service(hardware_config(self.temp.name))
        self.addCleanup(self.service.close)

    def cpu(self):
        return next(d for d in self.service.config['devices'] if d['id'] == 'CPU')

    def test_positive_setting_is_saved_and_auto_removes_override(self):
        self.assertNotIn('inference_threads', self.cpu())
        state = self.service.settings({'device': 'CPU', 'inference_threads': 12})
        self.assertEqual(next(d['inference_threads'] for d in state['scheduler']['devices']
                              if d['id'] == 'CPU'), 12)
        saved = json.loads(self.service.settings_path.read_text())
        self.assertEqual(next(d['inference_threads'] for d in saved['devices'] if d['id'] == 'CPU'), 12)
        self.service.settings({'device': 'CPU', 'budget_gib': 40})
        self.assertEqual(self.cpu()['inference_threads'], 12)
        self.service.settings({'device': 'CPU', 'inference_threads': None})
        self.assertNotIn('inference_threads', self.cpu())
        saved = json.loads(self.service.settings_path.read_text())
        self.assertNotIn('inference_threads', next(d for d in saved['devices'] if d['id'] == 'CPU'))

    def test_invalid_values_and_gpu_thread_count_leave_settings_unchanged(self):
        before = copy.deepcopy(self.service.config)
        for value in (0, -1, 1.5, True, '4', float('nan'), 2147483648):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.settings({'device': 'CPU', 'inference_threads': value})
            self.assertEqual(self.service.config, before)
        with self.assertRaisesRegex(ValueError, 'CPU devices only'):
            self.service.settings({'device': 'GPU.1', 'inference_threads': 4})
        self.assertEqual(self.service.config, before)


class CPUThreadWorkerTests(unittest.TestCase):
    def setUp(self):
        self.config = {'mode': 'openvino', 'terminal_metrics': False, 'load_timeout_seconds': 10,
                       'devices': [{'id': key, 'budget_gib': 40, 'policy': 'budget'}
                                   for key in ('GPU.1', 'GPU.2', 'CPU')], 'models': []}
        self.cpu = self.config['devices'][-1]
        self.cpu['inference_threads'] = 8
        self.loads = []
        self.events = []
        self.fail_gpu = False
        self.scheduler = Scheduler(self.config, self.factory)

    def factory(self, mode, model, timeout, cancelled):
        self.loads.append(copy.deepcopy(model))
        if self.fail_gpu and model.get('_placement', {}).get('target') == 'GPU.1':
            raise NativeWorkerError('out of memory')
        return Worker(model)

    def model(self, mid='text', device='CPU', **extra):
        result = {'id': mid, 'device': device, 'path': '', 'pipeline': 'llm',
                  'task': 'text-generation', 'memory_gib': 2, **extra}
        self.config['models'].append(result)
        return result

    def split(self, mid='split', **extra):
        return self.model(mid, 'GPU.1', placement={'mode': 'manual', 'devices': ['GPU.1', 'CPU'],
                         'shares': {'GPU.1': 50, 'CPU': 50}}, **extra)

    def load(self, model):
        cancel = threading.Event()
        with self.scheduler.lease_models([model], cancel):
            return self.scheduler.ensure(model, cancel, lambda kind, **data: self.events.append((kind, data)))

    def change(self, value):
        candidate = copy.deepcopy(self.config)
        if value is None:
            candidate['devices'][-1].pop('inference_threads', None)
        else:
            candidate['devices'][-1]['inference_threads'] = value
        self.scheduler.reconfigure(candidate, lambda kind, **data: self.events.append((kind, data)))
        self.config = candidate
        self.cpu = candidate['devices'][-1]

    def test_cpu_only_image_and_split_receive_default_gpu_only_does_not(self):
        for model in (self.model('cpu'), self.model('image', pipeline='image', task='text-to-image'),
                      self.split(), self.model('gpu', 'GPU.2')):
            self.load(model)
        loaded = {m['id']: m for m in self.loads}
        self.assertEqual([loaded[mid]['_cpu_inference_threads'] for mid in ('cpu', 'image', 'split')], [8, 8, 8])
        self.assertNotIn('_cpu_inference_threads', loaded['gpu'])
        self.assertNotIn('_cpu_inference_threads', self.config['models'][0])
        self.assertEqual(pipeline_properties(None, loaded['cpu'])['INFERENCE_NUM_THREADS'], 8)
        self.assertEqual(pipeline_properties(None, loaded['image'])['INFERENCE_NUM_THREADS'], 8)
        self.assertEqual(pipeline_properties(None, loaded['split'])['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 8)

    def test_auto_only_applies_default_to_successful_cpu_participating_attempt(self):
        model = self.model(device='GPU.1', placement={'mode': 'auto', 'devices': ['GPU.1', 'CPU'], 'shares': {}})
        self.fail_gpu = True
        worker = self.load(model)
        self.assertEqual([m.get('_cpu_inference_threads') for m in self.loads], [None, 8])
        self.assertIs(self.load(model), worker)
        self.assertEqual(len(self.loads), 2)

    def test_changed_default_unloads_only_actual_cpu_users(self):
        cpu, split = self.model('cpu'), self.split()
        gpu = self.model('gpu', 'GPU.2', placement={'mode': 'auto', 'devices': ['GPU.2', 'CPU'], 'shares': {}})
        cpu_worker, split_worker, gpu_worker = self.load(cpu), self.load(split), self.load(gpu)
        self.change(12)
        self.assertFalse(cpu_worker.alive)
        self.assertFalse(split_worker.alive)
        self.assertTrue(gpu_worker.alive)
        self.assertIs(self.load(gpu), gpu_worker)
        self.load(cpu)
        self.assertEqual(self.loads[-1]['_cpu_inference_threads'], 12)

    def test_same_default_retains_workers_and_auto_clears_effective_override(self):
        model = self.model()
        worker = self.load(model)
        self.change(8)
        self.assertIs(self.load(model), worker)
        self.change(None)
        self.assertFalse(worker.alive)
        self.load(model)
        self.assertNotIn('_cpu_inference_threads', self.loads[-1])
        self.assertNotIn('INFERENCE_NUM_THREADS', pipeline_properties(None, self.loads[-1]))

    def test_explicit_model_global_or_cpu_device_threads_win_and_prevent_reload(self):
        overrides = [
            {'INFERENCE_NUM_THREADS': 3},
            {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 4}}},
            {'DEVICE_PROPERTIES': {'HETERO:GPU.1,CPU': {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 5}}}}},
        ]
        for index, plugin in enumerate(overrides):
            with self.subTest(plugin=plugin):
                model = self.split('override-' + str(index), plugin_config=plugin)
                worker = self.load(model)
                self.assertNotIn('_cpu_inference_threads', self.loads[-1])
                self.change(12 + index)
                self.assertTrue(worker.alive)
                self.assertIs(self.load(model), worker)

    def test_one_role_override_keeps_default_for_other_vlm_components(self):
        model = self.split(pipeline='vlm', plugin_config={'MODEL_PROPERTIES': {
            'language_model': {'DEVICE_PROPERTIES': {'CPU': {'INFERENCE_NUM_THREADS': 3}}}}})
        self.load(model)
        loaded = self.loads[-1]
        self.assertEqual(loaded['_cpu_inference_threads'], 8)
        properties = pipeline_properties(None, loaded)
        self.assertEqual(properties['MODEL_PROPERTIES']['language_model']['DEVICE_PROPERTIES']['CPU']['INFERENCE_NUM_THREADS'], 3)
        self.change(10)
        self.assertNotIn(model['id'], self.scheduler.resident)


if __name__ == '__main__':
    unittest.main()
