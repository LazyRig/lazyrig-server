"""Injected native cache failures through real worker IPC, cleanup and job reporting."""
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyrig.placement import retryable_load_error, load_failure_reason
from lazyrig.runtime_settings import attention_cache_failure
from lazyrig.service import _record_failure
from lazyrig.terminal import Reporter
from lazyrig.worker import _child, NativeWorkerError, ProcessWorker


NATIVE_ERROR = ("Check 'reserved_la_bytes <= budget_in_bytes' failed at cache_orchestrator.hpp:874: "
                'Requested linear attention cache allocation exceeds the configured cache size.')


def _cache_failure_child(conn, mode, model, job_name):
    if os.name != 'nt':
        os.setsid()
    def fail(*args, **kwargs):
        raise RuntimeError(NATIVE_ERROR)
    genai = SimpleNamespace(LLMPipeline=fail, VLMPipeline=fail, Text2ImagePipeline=fail)
    properties = {'scheduler_config': SimpleNamespace(**model['cache_test_settings'])}
    with patch.dict(sys.modules, {'openvino': SimpleNamespace(), 'openvino_genai': genai}), \
         patch('lazyrig.worker._pipeline_setup', return_value=(properties, 'CPU', {})):
        _child(conn, 'openvino', model)


class AttentionCacheTests(unittest.TestCase):
    def test_native_error_crosses_process_boundary_with_actionable_settings_and_cleanup(self):
        settings = {'cache_size': 1, 'max_num_seqs': 8,
                    'num_linear_attention_blocks': 0, 'enable_prefix_caching': False}
        with patch('lazyrig.worker._spawn_child', _cache_failure_child), \
             self.assertRaises(NativeWorkerError) as caught:
            ProcessWorker('fixture', {'pipeline': 'llm', 'path': '/fixture', 'cache_test_settings': settings},
                          10, threading.Event())
        error = caught.exception
        self.assertIn(NATIVE_ERROR, str(error))
        self.assertEqual(error.error_code, 'attention_cache_budget_exceeded')
        self.assertEqual(error.cache_settings, settings)
        self.assertEqual(error.worker_stage, 'loading')
        self.assertIn(NATIVE_ERROR, str(error))
        self.assertIn('Scheduler sequence limit to 1', str(error))
        self.assertIn('Initial attention cache to 0', str(error))
        self.assertIn('max_num_seqs=8', str(error))
        self.assertIn('load_memory', error.__dict__)
        job = {'id': 'a'*32, 'model': 'test'}
        _record_failure(job, error)
        self.assertEqual(job['phase'], 'Attention-cache settings rejected')
        self.assertEqual(job['cache_settings'], settings)
        error.cache_settings['cache_size'] = 99
        self.assertEqual(job['cache_settings']['cache_size'], 1)
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            Reporter({'terminal_metrics': True}, job).finish('failed')
        output = '\n'.join(logs.output)
        self.assertIn('attention_cache_budget_exceeded', output)
        self.assertNotIn('cache_orchestrator.hpp', output)

    def test_cache_configuration_failure_does_not_trigger_device_fallback_even_during_probe(self):
        error = NativeWorkerError(NATIVE_ERROR)
        error.error_code = 'attention_cache_budget_exceeded'
        for stage in ('loading', 'validation', 'generation'):
            with self.subTest(stage=stage):
                error.worker_stage = stage
                self.assertFalse(retryable_load_error(error))
                self.assertEqual(load_failure_reason(error), 'attention_cache_budget_exceeded')
        # A different native failure in the split probe remains eligible.
        ordinary = NativeWorkerError('CL_OUT_OF_RESOURCES')
        ordinary.worker_stage = 'validation'
        self.assertTrue(retryable_load_error(ordinary))

    def test_only_exact_linear_cache_failure_is_reclassified_and_explicit_blocks_explained(self):
        for message in ('CL_OUT_OF_RESOURCES', 'out of memory', 'invalid cache precision'):
            self.assertEqual(attention_cache_failure(RuntimeError(message), {}), {})
        props = {'scheduler_config': SimpleNamespace(cache_size=1, max_num_seqs=1,
                                                     num_linear_attention_blocks=256)}
        detail = attention_cache_failure(RuntimeError(NATIVE_ERROR), props)
        self.assertIn('Clear the explicit num_linear_attention_blocks override', detail['error'])
        self.assertEqual(props['scheduler_config'].num_linear_attention_blocks, 256)
        unknown = attention_cache_failure(RuntimeError(NATIVE_ERROR), {})
        self.assertEqual(unknown['cache_settings'], {})
        self.assertNotIn('max_num_seqs=', unknown['error'])


if __name__ == '__main__':
    unittest.main()
