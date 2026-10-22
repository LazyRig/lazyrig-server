"""Placement integration across discovery, persisted controls and image handoff."""
import copy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lazyrig import memory
from lazyrig.deadlines import DeadlineEvent
from lazyrig.service import Service as NativeService
from lazyrig.worker import NativeWorkerError
from lazyrig.worker_status import WorkerProcessError
from lazyrig.scheduler import PlacementAdmissionError
from support import DEFAULT, FixtureWorker, Service
from test_core import wait_job
from test_device_info import DeviceCore
from test_server_contract import fixture_export


def hardware_config(directory):
    config = copy.deepcopy(DEFAULT)
    config['data_dir'] = directory
    config['terminal_metrics'] = False
    config['devices'] = [
        {'id': 'GPU.1', 'name': 'Discrete graphics', 'device_type': 'discrete',
         'detected': True, 'budget_gib': 22, 'policy': 'exclusive'},
        {'id': 'CPU', 'name': 'Host CPU', 'device_type': 'cpu',
         'detected': True, 'budget_gib': 48, 'policy': 'budget'}]
    for model in config['models']:
        model['device'] = 'GPU.1'
    return config


class DiscoveryPlacementTests(unittest.TestCase):
    def test_real_discovery_updates_metadata_and_only_migrates_eligible_models(self):
        service = NativeService.__new__(NativeService)
        service.config = hardware_config('unused')
        service.config['devices'][0].update(name='Old name', budget_gib=19, policy='budget')
        service.config['devices'].append({'id': 'GPU.9', 'detected': True, 'budget_gib': 8, 'policy': 'budget'})
        original = service.config['models'][0]
        managed = {**copy.deepcopy(original), 'id': 'managed',
                   'plugin_config': {'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .5}}}}
        explicit = {**copy.deepcopy(original), 'id': 'manual',
                    'placement': {'mode': 'manual', 'devices': ['GPU.1', 'CPU'],
                                  'shares': {'GPU.1': 70, 'CPU': 30}}}
        synthetic = {**copy.deepcopy(original), 'id': 'synthetic', 'device': 'TEST.GPU'}
        service.config['models'].extend([managed, explicit, synthetic])
        core = DeviceCore({
            'CPU': {'FULL_DEVICE_NAME': 'Current CPU'},
            'GPU.0': {'FULL_DEVICE_NAME': 'Integrated graphics', 'DEVICE_TYPE': 'INTEGRATED',
                      'GPU_DEVICE_TOTAL_MEM_SIZE': 16 * 1024**3},
            'GPU.1': {'FULL_DEVICE_NAME': 'Current graphics', 'DEVICE_TYPE': 'DISCRETE',
                      'GPU_DEVICE_TOTAL_MEM_SIZE': 24 * 1024**3},
            'GPU.2': {'FULL_DEVICE_NAME': 'Second graphics', 'DEVICE_TYPE': 'DISCRETE',
                      'GPU_DEVICE_TOTAL_MEM_SIZE': 16 * 1024**3}})
        with patch.dict('sys.modules', {'openvino': SimpleNamespace(Core=lambda: core, __version__='fixture'),
                                       'openvino_genai': SimpleNamespace(__version__='fixture')}), patch.object(
                                           memory, 'system_memory_gib', return_value=128):
            result = service.discover()
        self.assertTrue(result['available'], result['note'])
        devices = {d['id']: d for d in service.config['devices']}
        self.assertEqual((devices['GPU.1']['name'], devices['GPU.1']['total_memory_gib']), ('Current graphics', 24))
        self.assertEqual((devices['GPU.1']['budget_gib'], devices['GPU.1']['policy']), (19, 'budget'))
        self.assertFalse(devices['GPU.9']['detected'])
        self.assertEqual(original['placement']['devices'], ['GPU.1'])
        self.assertNotIn('placement', managed)
        self.assertNotIn('placement', synthetic)
        self.assertNotIn('placement', service.config['models'][1])
        self.assertEqual(explicit['placement']['shares'], {'GPU.1': 70, 'CPU': 30})


class TrackingWorkers:
    def __init__(self, service, primary_failure=False):
        self.service = service
        self.primary_failure = primary_failure
        self.loads = []
        self.calls = []
        self.image_loaded = threading.Event()

    def __call__(self, mode, model, timeout, cancelled):
        selected = model.get('_placement', {})
        target = selected.get('target', model['device'])
        self.loads.append((model['id'], target, set(self.service.scheduler.busy)))
        if self.primary_failure and model['id'] == 'qwen-chat' and target == 'GPU.1':
            raise NativeWorkerError('Insufficient memory while loading model')
        worker = FixtureWorker(mode, model, timeout, cancelled)
        run = worker.run
        def checked_run(request, timeout, cancel):
            actual = set(selected.get('devices', [model['device']]))
            if not actual <= self.service.scheduler.busy:
                raise AssertionError('Inference started without its complete device lease')
            self.calls.append((model['id'], request['action'], set(self.service.scheduler.busy)))
            yield from run(request, timeout, cancel)
        worker.run = checked_run
        if model['pipeline'] == 'image':
            self.image_loaded.set()
        return worker


class ServicePlacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(hardware_config(self.temp.name))
        self.workers = TrackingWorkers(self.service)
        self.service.scheduler.factory = self.workers

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def text(self, prompt='Hello', **extra):
        return self.service.submit({'session_id': self.service.store.session(), 'model': 'qwen-chat',
                                    'prompt': prompt, 'tools_enabled': False, 'max_tokens': 128, **extra})

    def split(self, mode='single'):
        self.service.settings({'model': 'qwen-chat', 'placement': {
            'mode': 'auto', 'devices': ['GPU.1', 'CPU'] if mode == 'auto' else ['GPU.1'],
            'shares': {}}})

    def test_registration_defaults_and_device_changes_preserve_explicit_allocation(self):
        export = Path(self.temp.name) / 'export'
        fixture_export(export)
        registered = self.service.register({'id': 'new', 'path': str(export), 'device': 'GPU'})
        self.assertEqual(registered['device'], 'GPU.1')
        self.assertEqual(registered['placement'], {'mode': 'auto', 'devices': ['GPU.1'], 'shares': {}, 'allow_fallback': False, 'shared_memory': 'budget'})
        self.split()
        allocation = copy.deepcopy(self.service.model('qwen-chat')['placement'])
        self.service.settings({'model': 'qwen-chat', 'name': 'Renamed'})
        self.assertEqual(self.service.model('qwen-chat')['placement'], allocation)
        self.service.settings({'model': 'qwen-chat', 'device': 'CPU'})
        self.assertEqual(self.service.model('qwen-chat')['placement'],
                         {'mode': 'auto', 'devices': ['CPU'], 'shares': {}, 'allow_fallback': False, 'shared_memory': 'budget'})
        self.assertEqual(self.service.model('qwen-chat')['memory_gib'], 48)

    def test_saved_device_budget_above_vram_unblocks_native_load_attempt(self):
        export = Path(self.temp.name) / 'oversized-export'
        fixture_export(export)
        with (export / 'openvino_model.bin').open('wb') as stream:
            stream.truncate(26 * 1024**3)
        self.service.config['devices'][0]['total_memory_gib'] = 24
        self.service.settings({'model': 'qwen-chat', 'path': str(export),
                               'placement': {'mode': 'auto', 'devices': ['GPU.1'], 'shares': {}}})
        blocked = wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(blocked['error_code'], 'model_weight_budget_exceeded')
        self.assertEqual(blocked['weight_check']['budget_gib'], 22)
        self.assertFalse(self.workers.loads)

        state = self.service.settings({'device': 'GPU.1', 'budget_gib': 36})
        device = next(d for d in state['scheduler']['devices'] if d['id'] == 'GPU.1')
        self.assertEqual(device['budget_gib'], 36)
        self.assertEqual(device['total_memory_gib'], 24)
        loaded = wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(loaded['status'], 'completed', loaded.get('error'))
        self.assertEqual([target for _, target, _ in self.workers.loads], ['GPU.1'])
        self.assertFalse(any(e['kind'] == 'placement_fallback' for e in loaded.get('placement_events', [])))

    def test_native_load_fallback_is_reported_and_reused_on_next_turn(self):
        self.split('auto')
        self.workers.primary_failure = True
        first = wait_job(self.service, self.text()['id'])
        self.assertEqual(first['status'], 'completed', first.get('error'))
        self.assertEqual([target for _, target, _ in self.workers.loads], ['GPU.1', 'HETERO:GPU.1,CPU'])
        self.assertEqual([e['kind'] for e in first['placement_events']], ['placement_fallback', 'placement_selected'])
        events = self.service.store.events(first['id'])
        self.assertTrue(any(e['kind'] == 'placement_fallback' for e in events))
        state = next(m for m in self.service.state()['models'] if m['id'] == 'qwen-chat')
        self.assertEqual(state['effective_placement']['devices'], ['GPU.1', 'CPU'])
        self.assertEqual(state['placement_info']['effective'], state['effective_placement'])
        second = wait_job(self.service, self.text('Another turn')['id'])
        self.assertEqual(second['status'], 'completed', second.get('error'))
        self.assertEqual(len(self.workers.loads), 2)

    def test_single_device_memory_guard_failure_is_visible_without_unselected_fallback(self):
        self.split()
        def pressure(*args):
            raise memory.MemoryPressureError(64, 5, 6.4)
        self.service.scheduler.factory = pressure
        for submit in (lambda: self.service.control('qwen-chat', 'load'), self.text):
            with self.subTest(submit=submit):
                job = wait_job(self.service, submit()['id'])
                self.assertEqual(job['status'], 'failed')
                self.assertEqual(job['error_code'], 'host_memory_pressure')
                self.assertIn('5.00 GiB available', job['error'])
                self.assertFalse(job.get('placement_events'))
                self.assertFalse(self.service.scheduler.resident)

    def test_worker_exit_is_persisted_for_load_and_chat_without_fallback(self):
        self.split('auto')
        attempts = []
        def exited(*args):
            attempts.append(1)
            raise WorkerProcessError(-1073741819, pid=1234, platform='nt', stage='loading')
        self.service.scheduler.factory = exited
        for submit in (lambda: self.service.control('qwen-chat', 'load'), self.text):
            with self.subTest(submit=submit):
                before = len(attempts)
                job = wait_job(self.service, submit()['id'])
                self.assertEqual(job['status'], 'failed')
                self.assertEqual(job['error_code'], 'worker_process_exited')
                self.assertEqual(job['worker_exit']['exit_code'], -1073741819)
                self.assertEqual(job['worker_exit']['stage'], 'loading')
                self.assertIn('0xC0000005', job['error'])
                self.assertEqual(self.service.store.job(job['id'])['worker_exit'], job['worker_exit'])
                self.assertEqual(len(attempts), before + 1)
                self.assertFalse(job.get('placement_events'))
                self.assertFalse(self.service.scheduler.resident)

    def test_exit_during_generation_is_persisted_and_unloaded(self):
        calls = []
        def factory(*args):
            worker = FixtureWorker(*args)
            def run(*args):
                calls.append(1)
                raise WorkerProcessError(24, pid=1234, stage='generation')
            worker.run = run
            return worker
        self.service.scheduler.factory = factory
        job = wait_job(self.service, self.text()['id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error_code'], 'worker_process_exited')
        self.assertEqual(job['worker_exit']['exit_code'], 24)
        self.assertEqual(job['worker_exit']['stage'], 'generation')
        self.assertEqual(self.service.store.job(job['id'])['worker_exit'], job['worker_exit'])
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.service.scheduler.resident)

    def test_mtp_budget_details_are_persisted_for_load_and_chat(self):
        def rejected(*args):
            raise PlacementAdmissionError('Main model exceeds GPU budget with MTP enabled',
                error_code='mtp_model_weight_budget_exceeded',
                weight_check={'device': 'GPU.1', 'weights_gib': 23.88, 'budget_gib': 20})
        self.service.scheduler.factory = rejected
        for submit in (lambda: self.service.control('qwen-chat', 'load'), self.text):
            with self.subTest(submit=submit):
                job = wait_job(self.service, submit()['id'])
                self.assertEqual(job['status'], 'failed')
                self.assertEqual(job['error_code'], 'mtp_model_weight_budget_exceeded')
                self.assertEqual(job['phase'], 'Loading blocked by device budget')
                self.assertEqual(job['weight_check']['weights_gib'], 23.88)
                self.assertEqual(self.service.store.job(job['id'])['weight_check'], job['weight_check'])
                self.assertNotIn('worker_exit', job)

    def test_load_memory_diagnostics_are_bounded_and_persisted_for_control_and_chat(self):
        self.split()
        def measured_factory(*args):
            for index in range(40):
                memory.publish_load_memory({'stage': 'initial' if index == 0 else 'sample',
                                            'available_gib': 60 - index, 'sample_count': index + 1})
            memory.publish_load_memory({'stage': 'ready', 'available_gib': 21, 'sample_count': 41})
            return self.workers(*args)
        self.service.scheduler.factory = measured_factory
        for submit in (lambda: self.service.control('qwen-chat', 'load'), self.text):
            with self.subTest(submit=submit):
                self.service.scheduler.unload('qwen-chat', lambda *args, **kwargs: None)
                job = wait_job(self.service, submit()['id'])
                self.assertEqual(job['status'], 'completed', job.get('error'))
                reports = job['load_memory_events']
                self.assertEqual(len(reports), 32)
                self.assertEqual(reports[0]['sample_count'], 10)
                self.assertEqual(reports[-1]['stage'], 'ready')
                self.assertTrue(all(data['model'] == 'qwen-chat' and
                                    data['device'] == 'GPU.1' for data in reports))
                durable = self.service.store.job(job['id'])
                self.assertEqual(durable['load_memory_events'], reports)
                events = [event for event in self.service.store.events(job['id']) if event['kind'] == 'load_memory']
                self.assertEqual(len(events), 41)
                # Persisted diagnostics are snapshots, not aliases into the caller's result.
                reports[0]['available_gib'] = -1
                self.assertGreaterEqual(self.service.store.job(job['id'])['load_memory_events'][0]['available_gib'], 0)

    def test_failed_single_device_load_persists_last_memory_sample_for_control_and_chat(self):
        self.split()
        def pressure(*args):
            memory.publish_load_memory({'stage': 'initial', 'available_gib': 50, 'headroom_gib': 12.8})
            error = memory.MemoryPressureError(64, 10, 12.8)
            error.load_memory = {'stage': 'error', 'available_gib': 10, 'minimum_available_gib': 10,
                                 'headroom_gib': 12.8, 'sample_count': 17}
            raise error
        self.service.scheduler.factory = pressure
        for submit in (lambda: self.service.control('qwen-chat', 'load'), self.text):
            with self.subTest(submit=submit):
                job = wait_job(self.service, submit()['id'])
                self.assertEqual(job['status'], 'failed')
                self.assertEqual(job['error_code'], 'host_memory_pressure')
                self.assertEqual([data['stage'] for data in job['load_memory_events']], ['initial', 'failed'])
                self.assertEqual(job['load_memory_events'][-1]['minimum_available_gib'], 10)
                self.assertEqual(job['load_memory_events'][-1]['sample_count'], 17)
                self.assertFalse(job.get('placement_events'))
                self.assertFalse(self.service.scheduler.resident)

    def test_manual_settings_are_rejected_before_worker_or_config_changes(self):
        loaded = wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(loaded['status'], 'completed')
        original = copy.deepcopy(self.service.config)
        worker = self.service.scheduler.resident['qwen-chat']
        with self.assertRaisesRegex(ValueError, 'Use automatic allocation'):
            self.service.settings({'model': 'qwen-chat', 'placement': {
                'mode': 'manual', 'devices': ['GPU.1', 'CPU'],
                'shares': {'GPU.1': 50, 'CPU': 50}}})
        self.assertEqual(self.service.config, original)
        self.assertIs(self.service.scheduler.resident['qwen-chat'], worker)

    def test_managed_image_handoff_preserves_history_under_whole_placement_lease(self):
        self.split('auto')
        self.workers.primary_failure = True
        sid = self.service.store.session()
        job = wait_job(self.service, self.service.submit({
            'session_id': sid, 'model': 'qwen-chat', 'image_model': 'qwen-image',
            'prompt': 'Draw an image of an orange robot', 'max_tokens': 128, 'steps': 1})['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertEqual([m['role'] for m in self.service.store.messages(sid)],
                         ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(len(job['artifacts']), 1)
        self.assertEqual([target for _, target, _ in self.workers.loads],
                         ['GPU.1', 'HETERO:GPU.1,CPU', 'GPU.1', 'HETERO:GPU.1,CPU'])
        self.assertTrue(all({'GPU.1', 'CPU'} <= lanes for _, _, lanes in self.workers.calls))

    def test_direct_image_waits_for_cpu_and_restores_exact_previous_split(self):
        self.split('auto')
        self.workers.primary_failure = True
        loaded = wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(loaded['status'], 'completed', loaded.get('error'))
        self.service.settings({'model': 'qwen-image', 'restore_text_after_image': True})
        with self.service.scheduler.lease(['CPU'], DeadlineEvent(10)):
            receipt = self.service.submit_image({'model': 'qwen-image', 'prompt': 'A robot', 'steps': 1})
            end = time.monotonic() + 3
            while time.monotonic() < end:
                with self.service.scheduler.condition:
                    queued = bool(self.service.scheduler.waiting)
                if queued:
                    break
                time.sleep(.01)
            self.assertTrue(queued, 'Image request did not enter the lease queue')
            self.assertFalse(self.workers.image_loaded.is_set())
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertEqual([target for _, target, _ in self.workers.loads],
                         ['GPU.1', 'HETERO:GPU.1,CPU', 'GPU.1', 'HETERO:GPU.1,CPU'])
        self.assertTrue(all({'GPU.1', 'CPU'} <= lanes for _, _, lanes in self.workers.loads))
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.service.scheduler.resident['qwen-chat'][0]['_placement']['devices'], ['GPU.1', 'CPU'])


if __name__ == '__main__':
    unittest.main()
