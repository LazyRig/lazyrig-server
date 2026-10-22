"""Measured split settings must reach native compilation, not just the log."""
import copy
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyrig.worker import _child, _pipeline_setup, NativeWorkerError, ProcessWorker


class Connection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(copy.deepcopy(value))

    def recv(self):
        raise EOFError()

    def close(self):
        self.closed = True


class WorkerPlacementSetupTests(unittest.TestCase):
    def setUp(self):
        self.model = {'id': 'test', 'pipeline': 'vlm', 'device': 'GPU.1', 'path': '/model',
                      'placement': {'mode': 'manual', 'devices': ['GPU.1', 'GPU.0'],
                                    'shares': {'GPU.1': 50, 'GPU.0': 50}},
                      '_placement': {'target': 'HETERO:GPU.1,GPU.0', 'devices': ['GPU.1', 'GPU.0'],
                                     'properties': {'ATTENTION_BACKEND': 'SDPA', 'DEVICE_PROPERTIES': {
                                         'GPU.1': {'QUERY_MODEL_RATIO': .5},
                                         'GPU.0': {'QUERY_MODEL_RATIO': 1}}}}}
        self.report = {'status': 'preflight_verified', 'scope': 'exported_language_graph',
                       'language_input_device': 'GPU.1',
                       'calibrated_properties': {'DEVICE_PROPERTIES': {
                           'GPU.1': {'QUERY_MODEL_RATIO': .25}, 'GPU.0': {'QUERY_MODEL_RATIO': 1}}}}
        self.calls = []

        def pipeline(*args, **kwargs):
            self.calls.append((args, kwargs))
            return SimpleNamespace(get_generation_config=lambda: SimpleNamespace(),
                generate=lambda *a, **k: k['streamer'].write([1, 2]))

        self.genai = SimpleNamespace(LLMPipeline=pipeline, VLMPipeline=pipeline,
                                     Text2ImagePipeline=pipeline, StreamerBase=object,
                                     StreamingStatus=SimpleNamespace(RUNNING=0))
        self.ov = SimpleNamespace(Core=lambda: SimpleNamespace(
            get_property=lambda *_: ['QUERY_MODEL_RATIO']))

    def test_calibrated_ratios_and_embedding_device_reach_constructor_without_changing_model(self):
        original = copy.deepcopy(self.model)
        conn = Connection()
        with patch.dict(sys.modules, {'openvino': self.ov, 'openvino_genai': self.genai}), \
             patch('lazyrig.placement_probe.verify_manual_split', return_value=copy.deepcopy(self.report)):
            _child(conn, 'openvino', self.model)
        self.assertEqual(self.model, original)
        self.assertEqual(len(self.calls), 1)
        args, properties = self.calls[0]
        self.assertEqual(args, ('/model', 'HETERO'))
        self.assertEqual(properties['MULTI_DEVICE_PRIORITIES'], 'GPU.1,GPU.0')
        self.assertEqual(properties['DEVICE_PROPERTIES']['HETERO']['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'], .25)
        for role in ('text_embeddings', 'text_embeddings_per_layer'):
            embedding = properties['MODEL_PROPERTIES'][role]
            self.assertEqual(embedding['MULTI_DEVICE_PRIORITIES'], 'GPU.1')
            self.assertNotIn('GPU.0', embedding.get('DEVICE_PROPERTIES', {}))
            self.assertNotIn('QUERY_MODEL_RATIO', embedding.get('DEVICE_PROPERTIES', {}).get('GPU.1', {}))
        self.assertIn({'kind': 'load_stage', 'stage': 'validation'}, conn.sent)
        ready = next(item for item in conn.sent if item['kind'] == 'ready')
        self.assertEqual(ready['placement_preflight']['component_devices']['text_embeddings'], 'GPU.1')
        self.assertEqual(ready['placement_preflight']['runtime_validation']['status'], 'passed')
        checkpoints = [item['validation'] for item in conn.sent if 'validation' in item]
        self.assertEqual([item['checkpoint'] for item in checkpoints],
                         ['generate_started', 'tokens_reported', 'generate_returned'])
        self.assertEqual([item['reported_tokens'] for item in checkpoints], [0, 2, 2])
        self.assertEqual(conn.sent[0]['runtime']['openvino'], 'unknown')
        self.assertNotIn('Hello', str(conn.sent))
        self.assertTrue(conn.closed)

    def test_calibrated_gpu_cpu_ratio_reaches_existing_target(self):
        self.model['pipeline'] = 'llm'
        self.model['_placement']['target'] = 'HETERO:GPU.1,CPU'
        self.model['_placement']['devices'] = ['GPU.1', 'CPU']
        self.model['_placement']['properties']['DEVICE_PROPERTIES'].pop('GPU.0')
        self.report['calibrated_properties']['DEVICE_PROPERTIES'].pop('GPU.0')
        with patch('lazyrig.placement_probe.verify_manual_split', return_value=self.report):
            properties, target, report = _pipeline_setup(self.genai, self.ov, self.model)
        self.assertEqual(target, 'HETERO:GPU.1,CPU')
        self.assertEqual(properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'], .25)
        self.assertEqual(properties['ATTENTION_BACKEND'], 'SDPA')
        self.assertNotIn('component_devices', report)

    def test_explicit_loader_reaches_constructor_and_does_not_run_ratio_calibration(self):
        self.model['_manual_affinity'] = True
        report = {'status': 'preflight_verified', 'language_input_device': 'GPU.1',
                  'partition': {'method': 'explicit_affinity'}}
        def verify(value):
            value['partition']['native_loader_verified'] = True
        reader = SimpleNamespace(extensions=['native loader marker'], verify_loaded=verify,
                                 assignments={'input':'GPU.1', 'output':'GPU.0'})
        original = copy.deepcopy(self.model)
        conn = Connection()
        with patch.dict(sys.modules, {'openvino':self.ov, 'openvino_genai':self.genai}), \
             patch('lazyrig.manual_affinity.prepare', return_value=(reader, report)), \
             patch('lazyrig.placement_probe.verify_manual_split', side_effect=AssertionError('Legacy query used')):
            _child(conn, 'openvino', self.model)
        self.assertEqual(self.model, original)
        properties = self.calls[0][1]
        self.assertEqual(properties['extensions'], ['native loader marker'])
        self.assertNotIn('QUERY_MODEL_RATIO', str(properties))
        self.assertNotIn('_lazyrig_affinity_reader', properties)
        self.assertEqual(conn.sent[-1]['kind'], 'ready')
        self.assertTrue(conn.sent[-1]['placement_preflight']['partition']['native_loader_verified'])
        self.assertEqual(conn.sent[-1]['placement_preflight']['runtime_validation']['status'], 'passed')

    def test_rejected_calibration_never_constructs_pipeline_and_preserves_report_over_ipc(self):
        error = ValueError('Manual split could not reach requested shares')
        error.placement_preflight = {'status': 'preflight_rejected', 'calibration': {'query_count': 9}}
        conn = Connection()
        with patch.dict(sys.modules, {'openvino': self.ov, 'openvino_genai': self.genai}), \
             patch('lazyrig.placement_probe.verify_manual_split', side_effect=error):
            _child(conn, 'openvino', self.model)
        self.assertFalse(self.calls)
        errors = [item for item in conn.sent if item['kind'] == 'error']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['placement_preflight'], error.placement_preflight)
        worker = ProcessWorker.__new__(ProcessWorker)
        messages = iter(conn.sent)
        worker.conn = SimpleNamespace(poll=lambda *_: True, recv=lambda: next(messages))
        with self.assertRaises(NativeWorkerError) as caught:
            next(worker._receive(1, threading.Event()))
        self.assertEqual(caught.exception.placement_preflight, error.placement_preflight)


if __name__ == '__main__':
    unittest.main()
