"""Image lifetime, cancellation, cleanup failures and idle-buffer regressions."""
import copy
import gc
import json
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import weakref
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import test_http
from test_core import wait_job
import test_image_edits
from test_image_edits import EditWorkers
from support import DEFAULT
from lazyrig.scheduler import Scheduler
from lazyrig.worker import _child, Cancelled, WorkerCleanupError


class ImageMemoryTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json
    fixture = test_image_edits.ImageEditHTTPTests.fixture
    body = test_image_edits.ImageEditHTTPTests.body

    def test_explicit_unload_releases_after_each_generation_and_edit(self):
        self.fixture()
        self.assertEqual(self.service.model('qwen-image')['keep_image_model_loaded'], 'auto')
        self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': False})
        for path, body in [('/v1/images/edits', self.body()), ('/v1/images/edits', self.body()),
                           ('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'A lake'})]:
            result = self.json(path, body)
            wait_job(self.service, result['x_job_id'])
            self.assertFalse(self.service.scheduler.resident)
            events = self.service.store.events(result['x_job_id'])
            unloaded = next(i for i, e in enumerate(events) if e['kind'] == 'unloaded')
            finished = next(i for i, e in enumerate(events) if e['kind'] == 'finished')
            self.assertLess(unloaded, finished)
            self.assertTrue(result['data'][0])
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-image'), ('close', 'qwen-image')] * 3)

    def test_automatic_default_keeps_image_on_dedicated_device_for_generations_and_edits(self):
        self.fixture()
        self.json('/api/settings', {'model': 'qwen-image', 'device': 'TEST.CPU'})
        self.json('/api/settings', {'device': 'TEST.CPU', 'policy': 'exclusive'})
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        text_worker = self.service.scheduler.resident['qwen-chat'][1]
        for endpoint, body in [('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 1}),
                               ('/v1/images/edits', self.body())]:
            for _ in range(2):
                wait_job(self.service, self.json(endpoint, body)['x_job_id'])
                self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'qwen-image'})
                self.assertIs(self.service.scheduler.resident['qwen-chat'][1], text_worker)
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('load', 'qwen-image'),
                                                 ('close', 'qwen-image'), ('load', 'qwen-image')])

    def test_automatic_keeps_shared_models_at_exact_budget_and_reuses_edit_worker(self):
        self.fixture()
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': 38})
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        for _ in range(2):
            wait_job(self.service, self.json('/v1/images/edits', self.body())['x_job_id'])
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'qwen-image'})
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('load', 'qwen-image')])

    def test_automatic_reuses_image_after_displacing_text_and_changing_resolution(self):
        self.fixture()
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': 22})
        self.json('/api/settings', {'model': 'qwen-image', 'memory_gib': 22, 'max_width': 2752, 'max_height': 2752})
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        first_worker = None
        for size in ('1024x1024', '2048x2048'):
            result = self.json('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'An orange robot', 'size': size, 'steps': 1})
            job = wait_job(self.service, result['x_job_id'])
            self.assertEqual(job['status'], 'completed')
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
            worker = self.service.scheduler.resident['qwen-image'][1]
            if first_worker is None:
                first_worker = worker
            self.assertIs(worker, first_worker)
            self.assertFalse(any(e['kind'] == 'unloaded' and e.get('model') == 'qwen-image'
                                 for e in self.service.store.events(job['id'])))
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'), ('load', 'qwen-image')])
        self.workers.responses.append(('Text requested after the images.', 8))
        result = self.json('/v1/chat/completions', {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}]})
        wait_job(self.service, result['x_job_id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.workers.lifecycle[-2:], [('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_automatic_reuses_edit_after_displacing_text_on_exclusive_device(self):
        self.fixture()
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        first_worker = None
        for _ in range(2):
            result = self.json('/v1/images/edits', self.body())
            wait_job(self.service, result['x_job_id'])
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
            worker = self.service.scheduler.resident['qwen-image'][1]
            if first_worker is None:
                first_worker = worker
            self.assertIs(worker, first_worker)
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'), ('load', 'qwen-image')])

    def test_automatic_releases_image_for_explicit_immediate_text_restoration(self):
        self.fixture()
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': 37})
        self.json('/api/settings', {'model': 'qwen-image', 'restore_text_after_image': True})
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        result = self.json('/v1/images/edits', self.body())
        wait_job(self.service, result['x_job_id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
                                                 ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])
        events = self.service.store.events(result['x_job_id'])
        self.assertTrue(any('automatic image retention' in event.get('reason', '') for event in events))
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})

    def test_automatic_warm_image_yields_memory_to_a_later_text_request(self):
        self.fixture()
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': 37})
        wait_job(self.service, self.json('/v1/images/edits', self.body())['x_job_id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-image'})
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_automatic_persists_and_existing_forced_choices_are_not_overwritten(self):
        self.fixture()
        from lazyrig.config import restore_settings
        for choice in (False, True, 'auto'):
            self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': choice})
            restored = restore_settings(self.service.config)
            self.assertEqual(restored['models'][1]['keep_image_model_loaded'], choice)

    def test_retention_setting_persists_and_can_release_a_warm_model(self):
        self.fixture()
        self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': True})
        for _ in range(2):
            wait_job(self.service, self.json('/v1/images/edits', self.body())['x_job_id'])
        self.assertEqual(self.workers.lifecycle, [('load', 'qwen-image')])
        persisted = json.loads(self.service.settings_path.read_text())
        self.assertTrue(persisted['models'][1]['keep_image_model_loaded'])
        with self.assertRaises(urllib.error.HTTPError):
            self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': 'false'})
        self.assertIn('qwen-image', self.service.scheduler.resident)
        self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': False})
        wait_job(self.service, self.json('/v1/images/edits', self.body())['x_job_id'])
        self.assertFalse(self.service.scheduler.resident)

    def test_failed_edit_releases_worker_and_restores_displaced_chat(self):
        self.fixture()
        self.service.config['models'][1]['restore_text_after_image'] = True
        original = self.workers
        def factory(*args):
            worker = original(*args)
            if args[1]['pipeline'] == 'image':
                def fail(*unused):
                    raise RuntimeError('Native edit failed')
                    yield
                worker.run = fail
            return worker
        self.service.scheduler.factory = factory
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.json('/v1/images/edits', self.body())
        result = json.loads(error.exception.read())
        job = wait_job(self.service, result['x_job_id'])
        self.assertEqual(job['status'], 'failed')
        self.assertIn('Native edit failed', job['error'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(self.workers.lifecycle[-2:], [('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_cancelled_edit_releases_worker_even_when_keep_loaded_is_enabled(self):
        self.fixture()
        self.service.config['models'][1]['keep_image_model_loaded'] = True
        started = threading.Event()
        original = self.workers
        def factory(*args):
            worker = original(*args)
            def blocked(payload, timeout, cancel):
                started.set()
                yield {'kind': 'progress', 'step': 1, 'steps': 40}
                if not cancel.wait(10): raise TimeoutError('Test did not cancel')
                raise Cancelled()
            worker.run = blocked
            return worker
        self.service.scheduler.factory = factory
        receipt = self.json('/api/image/edits', self.body())
        self.assertTrue(started.wait(3))
        self.service.cancel(receipt['id'])
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(self.service.scheduler.resident)
        self.assertEqual(self.workers.lifecycle[-1], ('close', 'qwen-image'))


class CleanupTests(unittest.TestCase):
    def test_failed_constructor_cleanup_stays_owned_and_blocks_device(self):
        config = copy.deepcopy(DEFAULT)
        workers = EditWorkers([])
        def failed_load(*args):
            worker = workers(*args)
            def stuck(): raise RuntimeError('Termination not confirmed')
            worker.retry_close = worker.close; worker.close = stuck
            raise WorkerCleanupError(worker, ValueError('Invalid native export'), RuntimeError('Termination not confirmed'))
        scheduler = Scheduler(config, failed_load)
        cancel = threading.Event(); emit = lambda *a, **kw: None
        image = config['models'][1]
        with self.assertRaises(WorkerCleanupError): scheduler.ensure(image, cancel, emit)
        self.assertEqual(scheduler.states[image['id']], 'cleanup-failed')
        with self.assertRaises(RuntimeError): scheduler.ensure(config['models'][0], cancel, emit)
        self.assertEqual(workers.lifecycle, [('load', 'qwen-image')])
        worker = scheduler.resident[image['id']][1]
        worker.close = worker.retry_close
        scheduler.close()
        self.assertFalse(scheduler.resident)

    def test_failed_unload_blocks_reuse_and_new_admission_on_the_device(self):
        config = copy.deepcopy(DEFAULT)
        config['devices'][0].update(policy='budget', budget_gib=100)
        workers = EditWorkers([])
        scheduler = Scheduler(config, workers)
        cancel = threading.Event(); emit = lambda *a, **kw: None
        model = config['models'][1]
        worker = scheduler.ensure(model, cancel, emit)
        close = worker.close
        def stuck(): raise RuntimeError('Native worker did not exit')
        worker.close = stuck
        try:
            with self.assertRaises(RuntimeError): scheduler.unload(model['id'], emit)
            self.assertEqual(scheduler.states[model['id']], 'cleanup-failed')
            for other in (model, config['models'][0]):
                with self.assertRaises(RuntimeError): scheduler.ensure(other, cancel, emit)
            self.assertEqual(workers.lifecycle, [('load', 'qwen-image')])
        finally:
            worker.close = close
            scheduler.close()
        self.assertFalse(scheduler.resident)

    def test_warm_worker_drops_request_tensors_and_generator_before_waiting(self):
        import numpy as np
        references = []; events = []
        class Tensor:
            def __init__(self, data):
                self.data = data; references.append(weakref.ref(self))
        class Generator:
            def __init__(self, seed): references.append(weakref.ref(self))
        class Pipeline:
            def __init__(self, *args, **kwargs): pass
            def generate(self, prompt, image, **kwargs):
                return Tensor(np.zeros((1, 256, 256, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.png'; Image.new('RGB', (32, 20), 'red').save(source)
            class Connection:
                sent = False
                def send(self, event): events.append(event)
                def recv(inner):
                    if inner.sent:
                        gc.collect()
                        self.assertTrue(references)
                        self.assertTrue(all(ref() is None for ref in references), 'Idle worker retained image buffers')
                        raise EOFError()
                    inner.sent = True
                    return {'action': 'image_edit', 'input_images': [{'path': str(source)}], 'prompt': 'A scarf',
                            'width': 256, 'height': 256, 'steps': 2, 'seed': 1, 'output_path': str(Path(directory) / 'out.png')}
                def close(self): pass
            fake = types.SimpleNamespace(LLMPipeline=object, VLMPipeline=object, Text2ImagePipeline=object,
                                         Image2ImagePipeline=Pipeline, TorchGenerator=Generator)
            with patch.dict(sys.modules, {'openvino_genai': fake, 'openvino': types.SimpleNamespace(Tensor=Tensor)}):
                _child(Connection(), 'openvino', {'pipeline': 'image', '_image_operation': 'edit',
                       'runtime_profile': 'qwen-image-21', 'path': directory, 'device': 'CPU'})
            self.assertEqual([e['kind'] for e in events], ['worker_diagnostics', 'ready', 'result'], events)


if __name__ == '__main__': unittest.main()
