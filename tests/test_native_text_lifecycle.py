"""Tiny real GenAI CPU pipeline: explicit-affinity loading and cooperative Stop.

These tests need OpenVINO/GenAI 2026.4.x, tokenizers and transformers. They
download nothing and do not establish physical GPU compatibility.
"""
import tempfile
import threading
import unittest
from pathlib import Path

try:
    import numpy as np
    import openvino as ov
    import openvino_genai as genai
    from native_text_fixture import create_fixture
except ImportError:
    ov = None

from lazyrig.manual_affinity import AffinityReader, graph_records, prepare_xml
from lazyrig.state_order import normalize_state_sink_order
from lazyrig.worker import Cancelled, ProcessWorker, _validate_split_inference


@unittest.skipIf(ov is None, 'Native text fixture dependencies unavailable')
class NativeTextLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not str(ov.__version__).startswith('2026.4.') or not str(genai.__version__).startswith('2026.4.'):
            raise unittest.SkipTest('Explicit affinity integration targets 2026.4.x')

    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.folder)
        create_fixture(self.path)

    def test_production_reader_preserves_genai_tokens_over_repeated_requests(self):
        baseline = genai.LLMPipeline(str(self.path), 'CPU', ATTENTION_BACKEND='SDPA', INFERENCE_NUM_THREADS=2)
        inputs = ov.Tensor(np.array([[1,2,1]], dtype=np.int64))
        expected = baseline.generate(inputs, max_new_tokens=4, ignore_eos=True).tokens
        del baseline
        xml, _ = normalize_state_sink_order((self.path/'openvino_model.xml').read_bytes())
        (self.path/'openvino_model.xml').write_bytes(prepare_xml(xml))
        reader = AffinityReader(ov)
        core = ov.Core()
        core.add_extension(reader.extensions)
        graph = core.read_model(self.path/'openvino_model.xml')
        # GenAI owns its Core, so this fixture checks the production loader on
        # one CPU. Separate native tests exercise real two-plugin partitioning.
        reader.assignments = {r['name']: 'CPU' for r in graph_records(graph)}
        pipeline = genai.LLMPipeline(str(self.path), 'HETERO:CPU', ATTENTION_BACKEND='SDPA',
                                    INFERENCE_NUM_THREADS=2, extensions=reader.extensions)
        report = {'partition': {}}
        reader.verify_loaded(report)
        self.assertEqual(report['partition']['prepared_projections'], 1)
        self.assertEqual(_validate_split_inference(pipeline)['status'], 'passed')
        for _ in range(3):
            self.assertEqual(pipeline.generate(inputs, max_new_tokens=4, ignore_eos=True).tokens, expected)

    def test_native_stop_returns_to_same_worker_for_next_request(self):
        worker = ProcessWorker('openvino', {'id': 'tiny', 'path': str(self.path), 'device': 'CPU',
                               'pipeline': 'llm', 'plugin_config': {'ATTENTION_BACKEND':'SDPA', 'INFERENCE_NUM_THREADS':2}},
                               30, threading.Event())
        self.addCleanup(worker.close)
        pid = worker.process.pid
        request = {'action': 'text', 'max_tokens': 1024, 'temperature': 0,
                   'prepared': {'token_ids': [1,2,1], 'context': {'input_tokens': 3}}}
        for _ in range(2):
            cancel = threading.Event()
            with self.assertRaises(Cancelled) as caught:
                for event in worker.run(request, 20, cancel):
                    if event['kind'] == 'text':
                        cancel.set()
            self.assertTrue(caught.exception.worker_reusable)
            self.assertTrue(worker.process.is_alive())
            result = list(worker.run({**request, 'max_tokens': 4}, 20, threading.Event()))[-1]
            self.assertEqual(result['kind'], 'result')
            self.assertEqual(result['generated_tokens'], 4)
            self.assertEqual(worker.process.pid, pid)


if __name__ == '__main__':
    unittest.main()
