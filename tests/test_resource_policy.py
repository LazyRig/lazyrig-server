"""Explicit resource policy: real admission, persistence and fixture workers.

Sparse files exercise the actual weight screen without allocating model RAM.
These checks do not claim native compatibility of any GPU/model combination.
"""
import copy
from pathlib import Path
import tempfile
import threading
import unittest

from lazyrig import placement
from lazyrig.scheduler import Scheduler, PlacementAdmissionError
from lazyrig.service import Service as NativeService, _record_failure
from lazyrig.storage import job_summary
from lazyrig.worker import NativeWorkerError
from lazyrig.worker_status import WorkerProcessError
from support import Service, sparse_file
from test_core import wait_job
from test_scheduler_placement import Worker
from test_service_placement import hardware_config
from test_server_contract import fixture_export


class ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        sparse_file(Path(self.folder.name, 'openvino_model.bin'), int(23.88 * 1024**3))
        self.model = {'id': 'qwen', 'pipeline': 'llm', 'task': 'text-generation',
                      'path': self.folder.name, 'device': 'GPU.1', 'memory_gib': 16,
                      'placement': {'mode': 'auto', 'devices': ['GPU.1', 'GPU.0'], 'shares': {}}}
        self.config = {'mode': 'openvino', 'load_timeout_seconds': 10, 'models': [self.model], 'devices': [
            {'id': 'GPU.0', 'device_type': 'integrated', 'budget_gib': 50.7, 'policy': 'budget', 'detected': True},
            {'id': 'GPU.1', 'device_type': 'discrete', 'budget_gib': 20, 'policy': 'budget', 'detected': True, 'total_memory_gib': 24},
            {'id': 'CPU', 'device_type': 'cpu', 'budget_gib': 48, 'policy': 'budget', 'detected': True}]}
        self.calls, self.events, self.failures = [], [], {}
        self.job = {}
        def factory(mode, model, timeout, cancel):
            target = model['_placement']['target']
            self.calls.append(copy.deepcopy(model))
            if target in self.failures:
                raise self.failures[target]
            return Worker(model)
        self.scheduler = Scheduler(self.config, factory)
        self.addCleanup(self.scheduler.close)

    def emit(self, kind, **data):
        self.events.append((kind, copy.deepcopy(data)))
        NativeService._placement_event(self.job, kind, data)

    def load(self):
        cancel = threading.Event()
        with self.scheduler.lease_models([self.model], cancel):
            return self.scheduler.ensure(self.model, cancel, self.emit)

    def targets(self):
        return [m['_placement']['target'] for m in self.calls]

    def test_b60_20gib_attempts_selected_split_before_whole_igpu(self):
        self.load()
        self.assertEqual(self.targets(), ['HETERO:GPU.1,GPU.0'])
        self.assertEqual([a['status'] for a in self.job['placement_attempts']], ['skipped', 'succeeded'])
        self.assertEqual(self.job['placement_attempts'][0]['weight_check']['weights_gib'], 23.88)
        self.assertFalse(self.job['placement_plan']['allow_fallback'])

    def test_whole_igpu_fallback_requires_explicit_permission(self):
        self.failures['HETERO:GPU.1,GPU.0'] = NativeWorkerError('[GPU] CL_BUILD_PROGRAM_FAILURE native compiler reason')
        with self.assertRaises(NativeWorkerError):
            self.load()
        self.assertEqual(self.targets(), ['HETERO:GPU.1,GPU.0'])
        self.assertIn('native compiler reason', self.job['placement_attempts'][-1]['reason'])
        self.model['placement']['allow_fallback'] = True
        self.load()
        self.assertEqual(self.targets()[-2:], ['HETERO:GPU.1,GPU.0', 'GPU.0'])
        self.assertEqual(self.job['placement_attempts'][-2]['reason_code'], 'gpu_kernel_build_failed')

    def test_explicit_split_is_one_exact_attempt_and_no_single_gpu_fallback(self):
        self.model['placement'].update(mode='split', allow_fallback=True)
        error = WorkerProcessError(25, stage='validation')
        error.worker_diagnostics = {'runtime': {'openvino': 'fixture'},
                                    'validation': {'reported_tokens': 0, 'expected_tokens': 2},
                                    'fault_log': {'status': 'captured', 'text': 'fault trace fixture'}}
        self.failures['HETERO:GPU.1,GPU.0'] = error
        with self.assertRaises(WorkerProcessError):
            self.load()
        self.assertEqual(self.targets(), ['HETERO:GPU.1,GPU.0'])
        attempt, = self.job['placement_attempts']
        self.assertEqual(attempt['reason_code'], 'split_inference_failed')
        self.assertEqual(attempt['worker_exit']['exit_code'], 25)
        self.assertNotIn('architecture', attempt['reason'])
        self.assertEqual(attempt['worker_diagnostics'], error.worker_diagnostics)
        _record_failure(self.job, error)
        self.assertEqual(self.job['phase'], 'Split inference check failed')
        self.assertEqual(self.job['worker_diagnostics'], error.worker_diagnostics)
        summary = job_summary({**self.job, 'id': 'fixture', 'status': 'failed'})
        self.assertNotIn('fault trace fixture', str(summary))

    def test_shared_memory_override_has_exclusive_reservation_and_no_fake_usage(self):
        self.model['placement'].update(mode='single', devices=['GPU.1'], shared_memory='allow')
        peer = {**copy.deepcopy(self.model), 'id': 'peer', 'memory_gib': 1, 'path': ''}
        peer.pop('placement')
        worker = Worker(peer)
        self.scheduler.resident['peer'] = (peer, worker)
        self.scheduler.states['peer'] = 'loaded'
        self.load()
        self.assertEqual(self.targets(), ['GPU.1'])
        self.assertFalse(worker.is_alive())
        resolved = self.calls[-1]
        self.assertTrue(resolved['_placement']['shared_memory_attempt'])
        self.assertIsNone(placement.weight_budget_check(resolved, self.config))
        self.assertEqual(placement.reservations(resolved), {'GPU.1': 20})
        device = next(d for d in self.scheduler.snapshot()['devices'] if d['id'] == 'GPU.1')
        self.assertIsNone(device['measured_free_gib'])
        self.assertEqual(self.model['memory_gib'], 16)

    def test_shared_override_follows_split_and_does_not_repeat_failed_native_primary(self):
        self.model['placement']['shared_memory'] = 'allow'
        self.failures['HETERO:GPU.1,GPU.0'] = NativeWorkerError('CL_OUT_OF_RESOURCES')
        self.load()
        self.assertEqual(self.targets(), ['HETERO:GPU.1,GPU.0', 'GPU.1'])
        self.scheduler.unload('qwen', self.emit)
        self.scheduler.preferred.clear()
        self.calls.clear()
        self.config['devices'][1]['budget_gib'] = 48
        self.failures['GPU.1'] = NativeWorkerError('CL_OUT_OF_RESOURCES')
        with self.assertRaises(NativeWorkerError):
            self.load()
        self.assertEqual(self.targets(), ['GPU.1', 'HETERO:GPU.1,GPU.0'])

    def test_mtp_conflict_missing_device_and_shared_host_stage_are_explained(self):
        self.model['placement']['mode'] = 'split'
        self.model['mtp'] = {'enabled': True}
        with self.assertRaisesRegex(ValueError, 'MTP'):
            self.load()
        self.assertTrue(self.model['mtp']['enabled'])
        self.assertEqual(self.calls, [])
        self.model['mtp']['enabled'] = False
        self.config['devices'][0]['detected'] = False
        with self.assertRaisesRegex(ValueError, 'GPU.0 is not detected'):
            self.load()
        self.config['devices'][0]['detected'] = True
        self.model['placement']['devices'].append('CPU')
        with self.assertRaisesRegex(ValueError, 'share system RAM'):
            self.load()

    def test_reservation_floor_corrects_underestimate_without_mutating_saved_value(self):
        self.config['devices'][1]['budget_gib'] = 48
        self.load()
        reserved = self.calls[-1]['_placement']['reservations']['GPU.1']
        self.assertAlmostEqual(reserved, 23.88, places=2)
        self.assertEqual(self.model['memory_gib'], 16)
        self.assertIn('below the stored', placement.info(self.model, self.config)['reservation_warning'])

    def test_unreached_candidates_are_visible_and_summary_errors_are_bounded(self):
        self.model['placement']['allow_fallback'] = True
        self.load()
        self.assertEqual(self.job['placement_attempts'][-1]['status'], 'not_attempted')
        self.job['placement_attempts'][0]['reason'] = 'X' * 4096
        self.assertEqual(len(job_summary(self.job)['placement_attempts'][0]['reason']), 600)
        self.assertEqual(len(self.job['placement_attempts'][0]['reason']), 4096)

    def test_invalid_modes_and_policy_types_fail_before_native_load(self):
        for changes in ({'mode': 'surprise'}, {'mode': 'single'}, {'allow_fallback': 'true'}, {'shared_memory': True}):
            candidate = copy.deepcopy(self.model)
            candidate['placement'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                placement.prepare(candidate, self.config)
        self.assertFalse(self.calls)

    def test_missing_primary_needs_permission_and_does_not_move_shared_override(self):
        self.config['devices'][1]['detected'] = False
        self.model['placement']['shared_memory'] = 'allow'
        with self.assertRaisesRegex(ValueError, 'primary device is not detected'):
            self.load()
        self.model['placement']['allow_fallback'] = True
        self.config['devices'][0]['budget_gib'] = 10
        with self.assertRaises(PlacementAdmissionError):
            self.load()
        self.assertFalse(self.calls)
        self.assertEqual(self.job['placement_plan']['unavailable_devices'], ['GPU.1'])

    def test_last_budget_skip_does_not_hide_the_earlier_native_split_failure(self):
        self.model['placement']['allow_fallback'] = True
        self.config['devices'][0]['budget_gib'] = 10
        self.failures['HETERO:GPU.1,GPU.0'] = NativeWorkerError('[GPU] CL_BUILD_PROGRAM_FAILURE original compiler detail')
        with self.assertRaises(PlacementAdmissionError) as caught:
            self.load()
        _record_failure(self.job, caught.exception)
        self.assertIn('see load attempts', self.job['phase'])
        self.assertEqual([a['status'] for a in self.job['placement_attempts']], ['skipped', 'failed', 'skipped'])
        self.assertIn('original compiler detail', self.job['placement_attempts'][1]['reason'])


class ResourcePolicyPersistenceTests(unittest.TestCase):
    def test_settings_roundtrip_rejection_and_durable_attempts(self):
        # Registered first, so both services close before their folder is removed.
        folder = self.enterContext(tempfile.TemporaryDirectory())
        export = Path(folder, 'export')
        fixture_export(export)
        config = hardware_config(folder)
        config['models'][0]['path'] = str(export)
        service = Service(config)
        self.addCleanup(service.close)
        policy = {'mode': 'split', 'devices': ['GPU.1', 'CPU'], 'shares': {},
                  'shared_memory': 'budget', 'allow_fallback': False}
        service.settings({'model': 'qwen-chat', 'placement': policy})
        before = copy.deepcopy(service.model('qwen-chat'))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            service.settings({'model': 'qwen-chat', 'placement': {**policy, 'mode': 'single'}})
        self.assertEqual(service.model('qwen-chat'), before)
        job = wait_job(service, service.control('qwen-chat', 'load')['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertEqual(job['placement_attempts'][0]['target'], 'HETERO:GPU.1,CPU')
        self.assertEqual(service.state(summary=True)['jobs'][0]['placement_attempts'][0]['status'], 'succeeded')
        service.close()
        restored = Service(config)
        self.addCleanup(restored.close)
        self.assertEqual(restored.model('qwen-chat')['placement'], policy)
        self.assertEqual(restored.store.job(job['id'])['placement_attempts'], job['placement_attempts'])
