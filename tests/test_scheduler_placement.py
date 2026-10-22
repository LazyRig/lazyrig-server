"""Atomic device plans, bounded fallback and resolved worker residency."""
import copy
import threading
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from lazyrig import placement
from lazyrig.deadlines import DeadlineEvent, RequestDeadline
from lazyrig.scheduler import PlacementAdmissionError, Scheduler, load_config
from lazyrig.worker import Cancelled, NativeWorkerError, WorkerCleanupError
from lazyrig.worker_status import WorkerProcessError
from lazyrig.memory import MemoryPressureError, publish_load_memory


class Worker:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.process = self
        self.alive = True
        self.closed = 0
        self.worker_pid = None

    def is_alive(self):
        return self.alive

    def close(self):
        self.closed += 1
        self.alive = False


class PlacementSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = {'mode': 'openvino', 'terminal_metrics': False, 'load_timeout_seconds': 10,
                       'devices': [{'id': key, 'budget_gib': 12, 'policy': 'budget'}
                                   for key in ('GPU.0', 'GPU.1', 'GPU.2', 'CPU')], 'models': []}
        self.model = {'id': 'text', 'device': 'GPU.0', 'path': '', 'pipeline': 'llm',
                      'task': 'text-generation', 'memory_gib': 4,
                      'placement': {'mode': 'auto', 'devices': ['GPU.0', 'GPU.1', 'CPU'], 'shares': {}}}
        self.config['models'] = [self.model]
        self.cancel = threading.Event()
        self.events = []
        self.calls = []
        self.failures = []
        self.scheduler = Scheduler(self.config, self.factory)

    def factory(self, mode, model, timeout, cancelled):
        self.calls.append((copy.deepcopy(model), timeout))
        if self.failures:
            error = self.failures.pop(0)
            if error:
                raise error
        return Worker(model)

    def emit(self, kind, **data):
        self.events.append((kind, data))

    def ensure(self, model=None):
        model = self.model if model is None else model
        with self.scheduler.lease_models([model], self.cancel):
            return self.scheduler.ensure(model, self.cancel, self.emit)

    def resolved(self, mid, devices, amount=2):
        model = {**copy.deepcopy(self.model), 'id': mid, 'device': devices[0]}
        model.pop('placement', None)
        model['_placement'] = {'target': devices[0] if len(devices) == 1 else 'HETERO:' + ','.join(devices),
                               'devices': devices, 'reservations': {d: amount for d in devices},
                               'properties': {}, 'label': 'Fixture split'}
        return model

    def resident(self, model):
        worker = Worker(model)
        with self.scheduler.condition:
            self.scheduler.resident[model['id']] = (copy.deepcopy(model), worker)
            self.scheduler.states[model['id']] = 'loaded'
        return worker

    def test_native_fallback_reuses_and_remembers_successful_placement(self):
        self.failures = [NativeWorkerError('CL_MEM_OBJECT_ALLOCATION_FAILURE private native details')]
        worker = self.ensure()
        self.assertEqual([placement.devices(model) for model, _ in self.calls], [{'GPU.0'}, {'GPU.0', 'GPU.1'}])
        self.assertIs(self.ensure(), worker)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(self.scheduler.is_loaded(self.model))
        self.scheduler.unload('text', self.emit)
        self.ensure()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(placement.devices(self.calls[-1][0]), {'GPU.0', 'GPU.1'})
        selected = next(data for kind, data in self.events if kind == 'placement_selected')
        self.assertEqual(selected['target'], 'HETERO:GPU.0,GPU.1')
        self.assertEqual(selected['placement']['devices'], ['GPU.0', 'GPU.1'])
        self.assertIn('private native details', str([data for kind, data in self.events if kind == 'placement_attempt']))
        self.assertNotIn('private native details', str([data for kind, data in self.events if kind == 'placement_fallback']))
        self.assertEqual(load_config(self.model), load_config(self.calls[-1][0]))
        changed = {**self.model, 'plugin_config': {'PERFORMANCE_HINT': 'LATENCY'}}
        self.ensure(changed)
        self.assertEqual(placement.devices(self.calls[-1][0]), {'GPU.0'})

    def test_failed_multi_gpu_validation_falls_back_to_primary_gpu_cpu_and_reuses_it(self):
        self.failures = [NativeWorkerError('CL_OUT_OF_RESOURCES'),
                         WorkerProcessError(25, stage='validation'),
                         WorkerProcessError(25, stage='validation')]
        worker = self.ensure()
        targets = [m['_placement']['target'] for m, _ in self.calls]
        self.assertEqual(targets, ['GPU.0', 'HETERO:GPU.0,GPU.1', 'HETERO:GPU.0,GPU.1,CPU', 'HETERO:GPU.0,CPU'])
        self.assertIs(self.ensure(), worker)
        self.assertEqual(len(self.calls), 4)
        fallbacks = [d for k,d in self.events if k == 'placement_fallback']
        self.assertEqual(fallbacks[-1]['reason_code'], 'split_inference_failed')
        self.assertEqual(fallbacks[-1]['worker_exit']['exit_code'], 25)
        manual = copy.deepcopy(self.model)
        manual['placement'].update(mode='manual', shares={'GPU.0':50, 'GPU.1':50, 'CPU':0})
        self.scheduler.close()
        self.failures = [WorkerProcessError(25, stage='validation')]
        before = len(self.calls)
        with self.assertRaises(WorkerProcessError):
            self.ensure(manual)
        self.assertEqual(len(self.calls), before + 1)

    def test_only_native_load_allocation_or_unsupported_device_can_retry(self):
        for error in (Cancelled(), TimeoutError('load timed out'), RequestDeadline(),
                      RuntimeError('CL_OUT_OF_RESOURCES'), NativeWorkerError('model.xml is invalid'),
                      WorkerProcessError(23, stage='loading')):
            with self.subTest(error=type(error).__name__ + str(error)):
                self.scheduler = Scheduler(self.config, self.factory)
                self.calls = []
                self.failures = [error]
                with self.assertRaises(type(error)):
                    self.ensure()
                self.assertEqual(len(self.calls), 1)
        self.scheduler = Scheduler(self.config, self.factory)
        self.calls = []
        self.failures = [NativeWorkerError('unsupported device')]
        self.ensure()
        self.assertEqual(len(self.calls), 2)

    def test_kernel_compile_failure_can_reach_selected_cpu_and_remains_explicit_in_manual(self):
        self.model['placement']['allow_fallback'] = True
        self.model['placement']['devices'] = ['GPU.0', 'CPU']
        error = NativeWorkerError('[GPU] clBuildProgram CL_BUILD_PROGRAM_FAILURE')
        error.error_code = 'gpu_kernel_build_failed'
        self.failures = [error, error]
        worker = self.ensure()
        self.assertEqual([m['_placement']['target'] for m, _ in self.calls],
                         ['GPU.0', 'HETERO:GPU.0,CPU', 'CPU'])
        self.assertIs(self.ensure(), worker)
        self.assertTrue(all(d['reason_code'] == 'gpu_kernel_build_failed'
                            for k, d in self.events if k == 'placement_fallback'))
        self.scheduler.close()
        self.model['placement'].update(mode='manual', shares={'GPU.0': 100, 'CPU': 0})
        self.calls = []
        self.failures = [error]
        with self.assertRaises(NativeWorkerError):
            self.ensure()
        self.assertEqual(len(self.calls), 1)

    def test_rejected_manual_preflight_is_reported_without_loading_or_fallback(self):
        manual = copy.deepcopy(self.model)
        manual['placement'].update(mode='manual', shares={'GPU.0': 50, 'GPU.1': 50, 'CPU': 0})
        error = NativeWorkerError('PRIVATE native allocation failure out of memory')
        error.placement_preflight = {
            'status': 'calibration_failed', 'scope': 'exported_language_graph',
            'requested_weight_shares': {'GPU.0': 50, 'GPU.1': 50},
            'devices': {'GPU.0': {'weight_share_percent': 88.2}, 'GPU.1': {'weight_share_percent': 11.8}},
            'calibration': {'query_count': 9, 'max_queries': 9, 'stop_reason': 'query budget reached'}}
        self.failures = [error]
        with self.assertRaises(NativeWorkerError):
            self.ensure(manual)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([kind for kind, _ in self.events], ['placement_plan', 'loading', 'placement_attempt', 'placement_attempt', 'placement_rejected'])
        rejected = self.events[-1][1]
        self.assertEqual(rejected['preflight'], error.placement_preflight)
        self.assertNotIn('PRIVATE', str(rejected))
        self.assertEqual(self.scheduler.states['text'], 'unloaded')
        self.assertFalse(self.scheduler.resident)
        self.assertFalse(self.scheduler.preferred)
        error.placement_preflight['calibration']['query_count'] = 99
        self.assertEqual(rejected['preflight']['calibration']['query_count'], 9)

    def test_manual_is_one_attempt_and_budget_failure_skips_only_auto_candidate(self):
        manual = copy.deepcopy(self.model)
        manual['placement'].update(mode='manual', shares={'GPU.0': 50, 'GPU.1': 50, 'CPU': 0})
        self.failures = [NativeWorkerError('CL_OUT_OF_RESOURCES')]
        with self.assertRaises(NativeWorkerError):
            self.ensure(manual)
        self.assertEqual(len(self.calls), 1)
        self.calls = []
        self.model['memory_gib'] = 15
        self.ensure()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(placement.devices(self.calls[0][0]), {'GPU.0', 'GPU.1'})
        legacy = {**self.model, 'id': 'legacy'}
        legacy.pop('placement')
        with self.assertRaisesRegex(PlacementAdmissionError, "reserves 15 GiB, exceeding GPU.0's 12 GiB budget"):
            self.ensure(legacy)

    def test_cleanup_failure_never_starts_another_candidate(self):
        failed = Worker(self.model)
        failed.close = lambda: (_ for _ in ()).throw(RuntimeError('still alive'))
        error = WorkerCleanupError(failed, NativeWorkerError('CL_OUT_OF_RESOURCES'), RuntimeError('still alive'))
        self.failures = [error]
        with self.assertRaises(WorkerCleanupError):
            self.ensure()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.scheduler.states['text'], 'cleanup-failed')
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            self.ensure()
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.scheduler.resident['text'][1], failed)

    def test_oversized_stored_weights_skip_full_gpu_before_native_load(self):
        with tempfile.TemporaryDirectory() as folder:
            # Sparse fixture: exercise metadata-only screening without allocating
            # model weights in RAM or writing gigabytes to disk.
            with (Path(folder) / 'openvino_model.bin').open('wb') as stream:
                stream.truncate(18 * 1024**3)
            self.model['path'] = folder
            self.model['placement']['devices'] = ['GPU.0', 'CPU']
            self.ensure()
            self.assertEqual([m['_placement']['target'] for m, _ in self.calls], ['HETERO:GPU.0,CPU'])
            skipped = next(data for kind, data in self.events if kind == 'placement_fallback')
            self.assertEqual(skipped['reason_code'], 'weight_budget')
            self.assertEqual(skipped['weight_check']['weights_gib'], 18)
            self.assertNotIn(('loading', {'model': 'text', 'device': 'GPU.0'}), self.events)
            # A healthy worker is reused, including when the export is removed.
            (Path(folder) / 'openvino_model.bin').unlink()
            self.ensure()
            self.assertEqual(len(self.calls), 1)

    def test_budget_edit_retries_primary_on_next_fresh_load_and_keeps_healthy_resident(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'openvino_model.bin').open('wb') as stream:
                stream.truncate(26 * 1024**3)
            self.model['path'] = folder
            self.model['placement']['devices'] = ['GPU.0', 'CPU']
            self.config['devices'][0].update(detected=True, total_memory_gib=24, budget_gib=20)
            worker = self.ensure()
            self.assertEqual(self.calls[-1][0]['_placement']['target'], 'HETERO:GPU.0,CPU')

            unrelated = copy.deepcopy(self.config)
            unrelated['devices'][2]['budget_gib'] = 36
            self.scheduler.reconfigure(unrelated, self.emit)
            self.assertIn('text', self.scheduler.preferred)

            changed = copy.deepcopy(unrelated)
            changed['devices'][0]['budget_gib'] = 36
            self.scheduler.reconfigure(changed, self.emit)
            self.assertNotIn('text', self.scheduler.preferred)
            self.assertIs(self.ensure(), worker)
            self.assertEqual(worker.closed, 0)
            self.scheduler.unload('text', self.emit)
            self.ensure()
            self.assertEqual([m['_placement']['target'] for m, _ in self.calls],
                             ['HETERO:GPU.0,CPU', 'GPU.0'])
            self.assertFalse(self.scheduler.busy)

    def test_shared_memory_budget_preserves_native_failure_and_host_pressure_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'openvino_model.bin').open('wb') as stream:
                stream.truncate(26 * 1024**3)
            self.model['path'] = folder
            self.model['placement']['devices'] = ['GPU.0', 'CPU']
            self.config['devices'][0].update(detected=True, total_memory_gib=24, budget_gib=36)
            for error, reason in ((NativeWorkerError('CL_MEM_OBJECT_ALLOCATION_FAILURE'), 'native_load_rejected'),
                                  (MemoryPressureError(64, 5, 6.4), 'host_memory_pressure')):
                with self.subTest(reason=reason):
                    self.scheduler = Scheduler(self.config, self.factory)
                    self.calls, self.events, self.failures = [], [], [error]
                    self.ensure()
                    self.assertEqual([m['_placement']['target'] for m, _ in self.calls],
                                     ['GPU.0', 'HETERO:GPU.0,CPU'])
                    fallbacks = [d for k, d in self.events if k == 'placement_fallback']
                    self.assertEqual([d['reason_code'] for d in fallbacks], [reason])
                    self.assertFalse(self.scheduler.busy)
                    self.scheduler.close()

    def test_memory_pressure_can_fallback_only_after_worker_cleanup(self):
        self.failures = [MemoryPressureError(64, 5, 6.4)]
        self.ensure()
        self.assertEqual(len(self.calls), 2)
        event = next(data for kind, data in self.events if kind == 'placement_fallback')
        self.assertEqual(event['reason_code'], 'host_memory_pressure')
        self.scheduler = Scheduler(self.config, self.factory)
        self.calls = []
        manual = copy.deepcopy(self.model)
        manual['placement'].update(mode='manual', shares={'GPU.0': 50, 'GPU.1': 50, 'CPU': 0})
        self.failures = [MemoryPressureError(64, 5, 6.4)]
        with self.assertRaises(MemoryPressureError):
            self.ensure(manual)
        self.assertEqual(len(self.calls), 1)
        failed = Worker(self.model)
        self.scheduler = Scheduler(self.config, self.factory)
        self.calls = []
        self.failures = [WorkerCleanupError(failed, MemoryPressureError(64, 5, 6.4), RuntimeError('still alive'))]
        with self.assertRaises(WorkerCleanupError):
            self.ensure()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.scheduler.states['text'], 'cleanup-failed')

    def test_load_memory_events_follow_each_attempt_and_observer_scope_ends(self):
        def factory(mode, model, timeout, cancelled):
            target = model['_placement']['target']
            self.calls.append(target)
            publish_load_memory({'stage': 'initial', 'available_gib': 50, 'headroom_gib': 12.8})
            if len(self.calls) == 1:
                error = MemoryPressureError(64, 10, 12.8)
                error.load_memory = {'stage': 'error', 'available_gib': 10,
                                     'minimum_available_gib': 10, 'headroom_gib': 12.8}
                raise error
            publish_load_memory({'stage': 'ready', 'available_gib': 30, 'headroom_gib': 12.8})
            return Worker(model)
        self.scheduler.factory = factory
        self.ensure()
        reports = [data for kind, data in self.events if kind == 'load_memory']
        self.assertEqual([data['stage'] for data in reports], ['initial', 'failed', 'initial', 'ready'])
        self.assertEqual([data['device'] for data in reports],
                         ['GPU.0', 'GPU.0', 'HETERO:GPU.0,GPU.1', 'HETERO:GPU.0,GPU.1'])
        self.assertTrue(all(data['model'] == 'text' for data in reports))
        self.assertEqual(reports[1]['minimum_available_gib'], 10)
        event_count = len(self.events)
        publish_load_memory({'stage': 'initial', 'available_gib': 99})
        self.assertEqual(len(self.events), event_count, 'Completed factory leaked its observer')

    def test_memory_reporting_failure_stops_fallback_after_cleanup_and_retains_failed_worker(self):
        def emit(kind, **data):
            if kind == 'load_memory':
                raise RuntimeError('Memory reporting storage unavailable')
        error = MemoryPressureError(64, 10, 12.8)
        error.load_memory = {'stage': 'error', 'available_gib': 10}
        self.failures = [error]
        with self.scheduler.lease_models([self.model], self.cancel):
            with self.assertRaisesRegex(RuntimeError, 'Memory reporting storage unavailable'):
                self.scheduler.ensure(self.model, self.cancel, emit)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.scheduler.resident)
        self.assertEqual(self.scheduler.states['text'], 'unloaded')

        failed = Worker(self.model)
        cleanup = WorkerCleanupError(failed, error, RuntimeError('still alive'))
        cleanup.load_memory = error.load_memory
        self.failures = [cleanup]
        self.calls.clear()
        with self.scheduler.lease_models([self.model], self.cancel):
            with self.assertRaisesRegex(RuntimeError, 'Memory reporting storage unavailable'):
                self.scheduler.ensure(self.model, self.cancel, emit)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.scheduler.resident['text'][1], failed)
        self.assertEqual(self.scheduler.states['text'], 'cleanup-failed')

    def test_initial_memory_reporting_failure_does_not_trigger_native_error_fallback(self):
        def factory(mode, model, timeout, cancelled):
            self.calls.append(model['_placement']['target'])
            publish_load_memory({'stage': 'initial', 'available_gib': 50})
            self.fail('Failed reporting should stop worker construction')
        self.scheduler.factory = factory
        def emit(kind, **data):
            if kind == 'load_memory':
                raise RuntimeError('out of memory in reporter, not native inference')
        with self.scheduler.lease_models([self.model], self.cancel):
            with self.assertRaisesRegex(RuntimeError, 'out of memory in reporter'):
                self.scheduler.ensure(self.model, self.cancel, emit)
        self.assertEqual(self.calls, ['GPU.0'])
        self.assertEqual(self.scheduler.states['text'], 'unloaded')
        publish_load_memory({'stage': 'ready'})  # observer must reset on exception too

    def test_all_attempts_share_one_load_budget_and_request_deadline(self):
        clock = [0.0]
        self.config['load_timeout_seconds'] = 5
        def factory(mode, model, timeout, cancelled):
            self.calls.append(timeout)
            if len(self.calls) == 1:
                clock[0] += 4
                raise NativeWorkerError('out of memory')
            return Worker(model)
        self.scheduler.factory = factory
        with patch('lazyrig.scheduler.time.monotonic', side_effect=lambda: clock[0]):
            self.ensure()
        self.assertEqual(self.calls, [5.0, 1.0])
        self.scheduler = Scheduler(self.config, factory)
        self.calls = []
        clock[0] = 0
        with patch('lazyrig.scheduler.time.monotonic', side_effect=lambda: clock[0]):
            self.cancel = DeadlineEvent(timeout=2)
            with self.assertRaises(RequestDeadline):
                self.ensure()
        self.assertEqual(self.calls, [2.0])

    def test_expired_load_budget_and_reporting_error_do_not_retry(self):
        clock = [0.0]
        def factory(mode, model, timeout, cancelled):
            self.calls.append(timeout)
            clock[0] += 11
            raise NativeWorkerError('out of memory')
        self.scheduler.factory = factory
        with patch('lazyrig.scheduler.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'load exceeded'):
                self.ensure()
        self.assertEqual(len(self.calls), 1)
        self.scheduler = Scheduler(self.config, self.factory)
        self.calls = []
        def emit(kind, **data):
            if kind == 'placement_selected':
                raise RuntimeError('reporting unavailable')
        with self.scheduler.lease_models([self.model], self.cancel):
            with self.assertRaisesRegex(RuntimeError, 'reporting unavailable'):
                self.scheduler.ensure(self.model, self.cancel, emit)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.scheduler.is_loaded(self.model))

    def test_secondary_device_pressure_evicts_entire_split_once(self):
        split = self.resolved('split', ['GPU.0', 'GPU.1'], 10)
        worker = self.resident(split)
        independent = self.resolved('independent', ['GPU.2'], 4)
        other = self.resident(independent)
        incoming = self.resolved('incoming', ['GPU.1'], 4)
        self.ensure(incoming)
        self.assertEqual(worker.closed, 1)
        self.assertEqual(other.closed, 0)
        snapshot = self.scheduler.snapshot()
        self.assertEqual({d['id']: d['reserved_gib'] for d in snapshot['devices']},
                         {'GPU.0': 0, 'GPU.1': 4, 'GPU.2': 4, 'CPU': 0})
        self.assertEqual(set(self.scheduler.resident), {'incoming', 'independent'})

    def test_reconfigure_preserves_resolved_plan_but_applies_secondary_budget(self):
        self.failures = [NativeWorkerError('out of memory')]
        worker = self.ensure()
        old_plan = copy.deepcopy(self.scheduler.resident['text'][0]['_placement'])
        changed = copy.deepcopy(self.config)
        changed['models'][0]['generation'] = {'max_tokens': 8192}
        self.scheduler.reconfigure(changed, self.emit)
        self.assertIs(self.scheduler.resident['text'][1], worker)
        self.assertEqual(self.scheduler.resident['text'][0]['_placement'], old_plan)
        snapshot = self.scheduler.snapshot()
        self.assertEqual(snapshot['workers'][0]['device'], 'HETERO:GPU.0,GPU.1')
        snapshot['workers'][0]['placement']['devices'].append('bad')
        self.assertEqual(self.scheduler.resident['text'][0]['_placement'], old_plan)
        changed = copy.deepcopy(changed)
        changed['devices'][1]['budget_gib'] = 11
        self.scheduler.reconfigure(changed, self.emit)
        self.assertEqual(worker.closed, 0)
        self.assertEqual(self.scheduler.resident['text'][0]['_placement']['reservations']['GPU.1'], 11)
        changed = copy.deepcopy(changed)
        changed['devices'][1]['budget_gib'] = 24
        self.scheduler.reconfigure(changed, self.emit)
        self.assertEqual(self.scheduler.resident['text'][0]['_placement']['reservations']['GPU.1'], 24)
        self.assertEqual(self.scheduler.snapshot()['devices'][1]['reserved_gib'], 24)
        self.assertEqual(worker.closed, 0)

    def test_image_retention_checks_secondary_devices_and_displaced_plans(self):
        image = self.resolved('image', ['GPU.1'], 5)
        image.update(pipeline='image', task='text-to-image')
        self.resident(image)
        text = self.resolved('text', ['GPU.0', 'GPU.1'], 8)
        self.assertFalse(self.scheduler.retain_image(image, [text]))
        text['_placement']['reservations']['GPU.1'] = 6
        self.assertTrue(self.scheduler.retain_image(image, [text]))
        self.config['devices'][1]['policy'] = 'exclusive'
        self.assertFalse(self.scheduler.retain_image(image, [text]))
        self.assertTrue(self.scheduler.retain_image({**image, 'keep_image_model_loaded': True}, [text]))

    def test_lease_closure_includes_all_overlapping_resident_devices(self):
        first = self.resolved('first', ['GPU.0', 'CPU'])
        second = self.resolved('second', ['CPU', 'GPU.1'])
        self.resident(first)
        self.resident(second)
        incoming = self.resolved('image', ['GPU.0'])
        incoming.update(pipeline='image', task='text-to-image')
        with self.scheduler.lease_models([incoming], self.cancel, restore=True) as previous:
            self.assertEqual({m['id'] for m in previous}, {'first', 'second'})
            self.assertEqual(self.scheduler.busy, {'GPU.0', 'GPU.1', 'CPU', '@model:image', '@model:first', '@model:second'})
            self.assertEqual(previous[0]['_placement'], first['_placement'])
            independent = self.resolved('other', ['GPU.2'])
            completed = threading.Event()
            def run():
                with self.scheduler.lease_models([independent], self.cancel):
                    completed.set()
            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(completed.wait(1))
            thread.join(1)
        self.assertEqual(self.scheduler.busy, set())

    def test_same_model_alias_serializes_different_device_plans(self):
        first = self.resolved('shared', ['GPU.0'])
        second = self.resolved('shared', ['GPU.2'])
        entered = threading.Event()
        def run():
            with self.scheduler.lease_models([second], self.cancel):
                entered.set()
        with self.scheduler.lease_models([first], self.cancel):
            thread = threading.Thread(target=run)
            thread.start()
            self.wait_for_queue()
            self.assertFalse(entered.is_set())
        thread.join(1)
        self.assertTrue(entered.is_set())

    def wait_for_queue(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.scheduler.condition:
                if self.scheduler.waiting:
                    return
            time.sleep(.005)
        self.fail('worker did not reach the lease queue')

    def test_queued_lease_rechecks_resident_union_before_entering(self):
        incoming = self.resolved('image', ['GPU.0'])
        incoming.update(pipeline='image', task='text-to-image')
        observed = []
        def run():
            with self.scheduler.lease_models([incoming], self.cancel, restore=True) as previous:
                observed.append((set(self.scheduler.busy), previous))
        with self.scheduler.lease(['GPU.0', 'CPU'], self.cancel):
            thread = threading.Thread(target=run)
            thread.start()
            self.wait_for_queue()
            self.resident(self.resolved('text', ['GPU.0', 'CPU']))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed[0][0], {'GPU.0', 'CPU', '@model:image', '@model:text'})
        self.assertEqual(observed[0][1][0]['id'], 'text')

    def test_manual_zero_share_devices_are_not_leased(self):
        manual = copy.deepcopy(self.model)
        manual['placement'].update(mode='manual', shares={'GPU.0': 100, 'GPU.1': 0, 'CPU': 0})
        with self.scheduler.lease_models([manual], self.cancel):
            self.assertEqual(self.scheduler.busy, {'GPU.0', '@model:text'})
            worker = self.scheduler.ensure(manual, self.cancel, self.emit)
            self.assertEqual(placement.devices(worker.model), {'GPU.0'})

    def test_gpu_alias_shares_the_physical_lane_and_reservation(self):
        old = self.resolved('old', ['GPU'], 10)
        worker = self.resident(old)
        incoming = self.resolved('incoming', ['GPU.0'], 4)
        with self.scheduler.lease_models([incoming], self.cancel):
            self.assertEqual(self.scheduler.busy, {'GPU.0', '@model:old', '@model:incoming'})
            self.scheduler.ensure(incoming, self.cancel, self.emit)
        self.assertEqual(worker.closed, 1)
        self.assertEqual(self.scheduler.snapshot()['devices'][0]['reserved_gib'], 4)

    def test_disjoint_healthy_automatic_workers_lease_only_actual_devices(self):
        first = placement.prepare(self.model, self.config)[0]
        first_worker = self.resident(first)
        second = copy.deepcopy(self.model)
        second.update(id='other', device='GPU.1')
        second['placement']['devices'] = ['GPU.1', 'GPU.0', 'CPU']
        second_worker = self.resident(placement.prepare(second, self.config)[0])
        completed = threading.Event()
        errors = []
        def run():
            try:
                with self.scheduler.lease_models([second], self.cancel):
                    self.assertIs(self.scheduler.ensure(second, self.cancel, self.emit), second_worker)
                    completed.set()
            except BaseException as error:
                errors.append(error)
        with self.scheduler.lease_models([self.model], self.cancel):
            self.assertEqual(self.scheduler.busy, {'GPU.0', '@model:text'})
            self.assertIs(self.scheduler.ensure(self.model, self.cancel, self.emit), first_worker)
            thread = threading.Thread(target=run)
            thread.start()
            parallel = completed.wait(1)
        thread.join(1)
        self.assertTrue(parallel)
        self.assertEqual(errors, [])
        self.assertEqual(self.calls, [])

    def test_dead_pinned_worker_cannot_expand_lease_and_keeps_request_options(self):
        worker = self.resident(placement.prepare(self.model, self.config)[0])
        request_model = {**self.model, 'template_context': {'test': True}, 'generation': {'max_tokens': 8192}}
        self.failures = [NativeWorkerError('out of memory'), NativeWorkerError('out of memory')]
        with self.scheduler.lease_models([request_model], self.cancel):
            self.assertEqual(self.scheduler.busy, {'GPU.0', '@model:text'})
            worker.alive = False
            with self.assertRaises(NativeWorkerError):
                self.scheduler.ensure(request_model, self.cancel, self.emit)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(placement.devices(self.calls[0][0]), {'GPU.0'})
            self.assertEqual(self.calls[0][0]['template_context'], {'test': True})
            self.assertEqual(self.calls[0][0]['generation'], {'max_tokens': 8192})
        self.ensure(request_model)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(placement.devices(self.calls[-1][0]), {'GPU.0', 'GPU.1'})
        self.assertEqual(self.scheduler._pinned.get(), {})

    def test_reconfigure_and_reuse_refresh_primary_reservation_without_reload(self):
        worker = self.ensure()
        changed = copy.deepcopy(self.config)
        changed['models'][0]['memory_gib'] = 9
        self.scheduler.reconfigure(changed, self.emit)
        self.assertEqual(self.scheduler.snapshot()['devices'][0]['reserved_gib'], 9)
        self.assertIs(self.scheduler.resident['text'][1], worker)
        next_model = {**changed['models'][0], 'memory_gib': 3}
        self.assertIs(self.ensure(next_model), worker)
        self.assertEqual(self.scheduler.snapshot()['devices'][0]['reserved_gib'], 3)
        self.assertEqual(len(self.calls), 1)

    def test_exact_physical_device_budget_wins_over_stale_generic_alias(self):
        self.config['devices'].insert(0, {'id': 'GPU', 'budget_gib': 2, 'policy': 'exclusive'})
        self.ensure()
        rows = self.scheduler.snapshot()['devices']
        self.assertEqual([d['id'] for d in rows], ['GPU.0', 'GPU.1', 'GPU.2', 'CPU'])
        self.assertEqual(rows[0]['budget_gib'], 12)
        self.assertEqual(rows[0]['reserved_gib'], 4)
        self.assertEqual(len(self.calls), 1)
        self.scheduler.reconfigure(copy.deepcopy(self.config), self.emit)
        self.assertTrue(self.scheduler.is_loaded(self.model))

    def test_restore_snapshot_loads_exact_previous_plan(self):
        self.failures = [NativeWorkerError('out of memory')]
        self.ensure()
        previous = copy.deepcopy(self.scheduler.resident['text'][0])
        self.scheduler.unload('text', self.emit)
        self.scheduler.preferred.clear()
        self.ensure(previous)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[-1][0]['_placement'], previous['_placement'])


if __name__ == '__main__':
    unittest.main()
