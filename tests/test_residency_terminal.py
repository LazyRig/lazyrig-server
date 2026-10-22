"""Device residency, terminal metrics and provider probes via real local HTTP."""
import copy
import io
import json
import threading
import unittest
import urllib.error
from contextlib import redirect_stderr
from unittest.mock import patch
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from support import DEFAULT
from lazyrig.scheduler import Scheduler
from lazyrig.terminal import Reporter


class ResidencyTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def configure(self, budget=40):
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': budget})
        self.json('/api/settings', {'model': 'qwen-image', 'keep_image_model_loaded': True})

    def test_shared_image_handoffs_reuse_both_workers(self):
        self.configure()
        workers = FixtureWorkers([(qwen_call(), 8), ('Image ready.', 8)] * 2)
        self.service.scheduler.factory = workers
        for _ in range(2):
            result = self.json('/v1/chat/completions', {'model': 'qwen-chat', 'auto_image': True,
                'image_model': 'qwen-image', 'messages': [{'role': 'user', 'content': 'Draw a lake'}], 'steps': 1})
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'qwen-image'})
            self.assertTrue(any(e['kind'] == 'resuming' and e['worker_reused'] for e in self.service.store.events(result['x_job_id'])))
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('load', 'qwen-image')])
        saved = json.loads(self.service.settings_path.read_text())
        self.assertEqual(saved['devices'][0]['policy'], 'budget')
        self.assertTrue(saved['models'][1]['keep_image_model_loaded'])

    def test_automatic_shared_handoff_reuses_both_workers(self):
        self.json('/api/settings', {'device': 'TEST.GPU', 'policy': 'budget', 'budget_gib': 40})
        self.assertEqual(self.service.model('qwen-image')['keep_image_model_loaded'], 'auto')
        workers = FixtureWorkers([(qwen_call(), 8), ('Image ready.', 8)])
        self.service.scheduler.factory = workers
        result = self.json('/v1/chat/completions', {'model': 'qwen-chat', 'auto_image': True,
            'image_model': 'qwen-image', 'messages': [{'role': 'user', 'content': 'Draw a lake'}], 'steps': 1})
        wait_job(self.service, result['x_job_id'])
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'qwen-image'})
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('load', 'qwen-image')])

    def test_budget_pressure_still_swaps_and_restores(self):
        self.configure(budget=22)
        workers = FixtureWorkers([(qwen_call(), 8), ('Image ready.', 8)])
        self.service.scheduler.factory = workers
        self.json('/v1/chat/completions', {'model': 'qwen-chat', 'auto_image': True,
            'image_model': 'qwen-image', 'messages': [{'role': 'user', 'content': 'Draw a lake'}], 'steps': 1})
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'),
            ('load', 'qwen-image'), ('close', 'qwen-image'), ('load', 'qwen-chat')])

    def test_standalone_images_retain_both_workers_when_enabled(self):
        self.configure()
        workers = FixtureWorkers([])
        self.service.scheduler.factory = workers
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        for _ in range(2):
            result = self.json('/v1/images/generations', {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 1})
            wait_job(self.service, result['x_job_id'])
            self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'qwen-image'})
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('load', 'qwen-image')])

    def test_two_llms_share_device_and_can_be_reused(self):
        self.configure()
        self.service.config['models'].append({**copy.deepcopy(self.service.config['models'][0]), 'id': 'second-llm', 'memory_gib': 10})
        workers = FixtureWorkers([('Hello', 2)] * 3)
        self.service.scheduler.factory = workers
        for mid in ('qwen-chat', 'second-llm', 'qwen-chat'):
            self.json('/v1/chat/completions', {'model': mid, 'auto_image': False,
                'messages': [{'role': 'user', 'content': 'Hello'}]})
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat', 'second-llm'})
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('load', 'second-llm')])

    def test_conversation_templates_and_sampling_reuse_one_loaded_llm(self):
        workers = FixtureWorkers([('First response', 3), ('Second response', 3)])
        self.service.scheduler.factory = workers
        for text, thinking, temperature in [('First conversation', True, 1.0), ('Separate conversation', False, .7)]:
            result = self.json('/v1/chat/completions', {'model':'qwen-chat', 'temperature':temperature,
                'chat_template_kwargs':{'enable_thinking':thinking}, 'max_tokens':32,
                'messages':[{'role':'user','content':text}]})
            wait_job(self.service, result['x_job_id'])
        self.assertEqual(workers.lifecycle, [('load','qwen-chat')])
        self.assertEqual([c['temperature'] for c in workers.calls], [1.0,.7])
        self.assertNotEqual(workers.calls[0]['prepared']['messages'],workers.calls[1]['prepared']['messages'])

    def test_request_defaults_keep_workers_and_load_settings_only_evict_changed_model(self):
        self.configure()
        second = {**copy.deepcopy(self.service.config['models'][0]), 'id':'second-llm','memory_gib':10}
        self.service.config['models'].append(second)
        workers = FixtureWorkers([('Updated defaults', 3)])
        self.service.scheduler.factory = workers
        for mid in ('qwen-chat','second-llm'):
            wait_job(self.service, self.service.control(mid,'load')['id'])
        original = self.service.scheduler.resident['qwen-chat'][1]
        self.json('/api/settings', {'model':'qwen-image','generation':{'steps':21,'seed':-1}})
        self.json('/api/settings', {'model':'qwen-chat','name':'Renamed',
            'generation':{'temperature':.8,'top_p':.83,'max_tokens':32},
            'template_context':{'enable_thinking':False}, 'idle_timeout_seconds':100})
        result = self.json('/v1/chat/completions', {'model':'qwen-chat','messages':[{'role':'user','content':'Fresh defaults'}]})
        wait_job(self.service,result['x_job_id'])
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1],original)
        self.assertEqual(workers.lifecycle,[('load','qwen-chat'),('load','second-llm')])
        self.assertEqual(workers.calls[-1]['temperature'],.8)
        self.assertEqual(workers.calls[-1]['generation']['top_p'],.83)
        self.json('/api/settings',{'model':'second-llm','plugin_config':{'INFERENCE_NUM_THREADS':2}})
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1],original)
        self.assertNotIn('second-llm',self.service.scheduler.resident)
        self.assertEqual(workers.lifecycle[-1],('close','second-llm'))

    def test_budget_change_evicts_only_older_resident_and_logs_reason(self):
        self.configure()
        self.service.config['models'].append({**copy.deepcopy(self.service.config['models'][0]), 'id':'second-llm','memory_gib':10})
        workers = FixtureWorkers([]); self.service.scheduler.factory=workers
        for mid in ('qwen-chat','second-llm'):
            wait_job(self.service,self.service.control(mid,'load')['id'])
        with self.assertLogs('lazyrig.runtime',level='INFO') as logs:
            self.json('/api/settings',{'device':'TEST.GPU','budget_gib':10})
        self.assertEqual(set(self.service.scheduler.resident),{'second-llm'})
        self.assertEqual(workers.lifecycle,[('load','qwen-chat'),('load','second-llm'),('close','qwen-chat')])
        self.assertIn('device budget or residency policy changed','\n'.join(logs.output))

    def test_least_recently_used_model_is_evicted_under_pressure(self):
        config = copy.deepcopy(DEFAULT)
        config['devices'][0].update(policy='budget', budget_gib=40)
        workers = FixtureWorkers([])
        scheduler = Scheduler(config, workers)
        first, second = config['models']
        third = {**first, 'id': 'third', 'memory_gib': 15}
        cancel = threading.Event()
        try:
            for model in (first, second, first, third):
                scheduler.ensure(model, cancel, lambda *a, **kw: None)
            self.assertEqual(set(scheduler.resident), {'qwen-chat', 'third'})
            self.assertIn(('close', 'qwen-image'), workers.lifecycle)
            self.assertNotIn(('close', 'qwen-chat'), workers.lifecycle)
        finally:
            scheduler.close()


class TerminalTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_manual_log_distinguishes_query_weights_from_requested_shares_and_vram(self):
        reporter = Reporter({'terminal_metrics': True}, {'id': 'a' * 32, 'model': 'llm'})
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            reporter.event('placement_selected', placement={'devices': ['GPU.1', 'CPU'],
                'target': 'HETERO:GPU.1,CPU', 'label': 'Manual weight split'}, preflight={
                'scope': 'exported_language_graph', 'requested_weight_shares': {'GPU.1': 50, 'CPU': 50},
                'devices': {'GPU.1': {'weight_share_percent': 53}, 'CPU': {'weight_share_percent': 47}},
                'note': 'PRIVATE DETAIL'})
        output = '\n'.join(logs.output)
        self.assertIn('preflight language weights', output)
        self.assertIn('GPU.1 53.0% (requested 50.0%)', output)
        self.assertIn('CPU 47.0% (requested 50.0%)', output)
        self.assertIn('exported graph; not measured VRAM', output)
        self.assertNotIn('PRIVATE', output)

    def test_manual_rejection_log_reports_closest_query_and_never_claims_loaded(self):
        reporter = Reporter({'terminal_metrics': True}, {'id': 'a' * 32, 'model': 'llm'})
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            reporter.event('placement_rejected', placement={'devices': ['GPU.1', 'CPU'],
                'target': 'HETERO:GPU.1,CPU', 'label': 'Manual weight split'}, preflight={
                'status': 'calibration_failed', 'scope': 'exported_language_graph',
                'requested_weight_shares': {'GPU.1': 50, 'CPU': 50},
                'devices': {'GPU.1': {'weight_share_percent': 88.2}, 'CPU': {'weight_share_percent': 11.8}},
                'calibration': {'query_count': 9, 'max_queries': 9, 'stop_reason': 'query budget reached'},
                'note': 'PRIVATE native details', 'language_input_device': 'PRIVATE DEVICE'},
                error='PRIVATE exception')
        output = '\n'.join(logs.output)
        self.assertIn('manual split rejected', output)
        self.assertIn('closest queried language weights', output)
        self.assertIn('GPU.1 88.2% (requested 50.0%)', output)
        self.assertIn('CPU 11.8% (requested 50.0%)', output)
        self.assertIn('manual split preflight checks 9/9 | query budget reached', output)
        self.assertIn('exported graph; not measured VRAM', output)
        self.assertNotIn('loaded', output)
        self.assertNotIn('PRIVATE', output)
        self.assertNotIn('language input device', output)

    def test_successful_calibration_logs_only_bounded_count_and_selected_input_device(self):
        reporter = Reporter({'terminal_metrics': True}, {'id': 'a' * 32, 'model': 'llm'})
        placement = {'devices': ['GPU.1', 'GPU.0'], 'target': 'HETERO:GPU.1,GPU.0', 'label': 'Manual weight split'}
        report = {'status': 'preflight_verified', 'scope': 'exported_language_graph',
                  'devices': {'GPU.1': {'weight_share_percent': 50}, 'GPU.0': {'weight_share_percent': 50}},
                  'state_order': {'method': 'assign_dependency_depth', 'assign_count': 128, 'reordered': True},
                  'calibration': {'query_count': 3, 'max_queries': 9, 'stop_reason': 'within tolerance'},
                  'language_input_device': 'GPU.0'}
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            reporter.event('placement_selected', placement=placement, preflight=report)
            invalid = {**report, 'calibration': {'query_count': 999, 'max_queries': 999, 'stop_reason': 'PRIVATE'},
                       'language_input_device': 'PRIVATE'}
            reporter.event('placement_selected', placement=placement, preflight=invalid)
        output = '\n'.join(logs.output)
        self.assertIn('manual split preflight checks 3/9 | within tolerance', output)
        self.assertIn('128 state updates ordered by dependency', output)
        self.assertIn('language input device GPU.0 (preflight)', output)
        self.assertEqual(output.count('manual split preflight checks'), 1)
        self.assertNotIn('PRIVATE', output)
        self.assertNotIn('999', output)

    def test_load_memory_logs_use_known_stages_and_numeric_metrics_only(self):
        reporter = Reporter({'terminal_metrics': True}, {'id': 'a' * 32, 'model': 'llm'})
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            reporter.event('load_memory', stage='initial', available_gib=48, headroom_gib=12.8,
                           job_commit_limit_bytes=35 * 1024**3, source='PRIVATE SOURCE')
            reporter.event('load_memory', stage='ready', available_gib=30, headroom_gib=12.8,
                           minimum_available_gib=25)
            reporter.event('load_memory', stage='failed', available_gib=10, headroom_gib=12.8,
                           error='PRIVATE EXCEPTION', minimum_available_gib=10)
            reporter.event('load_memory', stage='PRIVATE STAGE', available_gib='PRIVATE VALUE',
                           headroom_gib='PRIVATE HEADROOM', minimum_available_gib='PRIVATE MINIMUM',
                           job_commit_limit_bytes='PRIVATE LIMIT')
            reporter.event('load_memory', stage='initial', available_gib=None,
                           error='PRIVATE FAILURE DETAIL')
        output = '\n'.join(logs.output)
        self.assertNotIn('PRIVATE', output)
        self.assertIn('guard v2 (start)', logs.output[0])
        self.assertIn('RAM available 48.0 GiB', logs.output[0])
        self.assertIn('headroom 12.8 GiB', logs.output[0])
        self.assertIn('Windows worker commit limit 35.0 GiB', logs.output[0])
        self.assertIn('guard v2 (ready)', logs.output[1])
        self.assertIn('lowest sampled RAM 25.0 GiB', logs.output[1])
        self.assertIn('guard v2 (failed)', logs.output[2])
        self.assertIn('guard v2 (sample)', logs.output[3])
        self.assertIn('RAM available n/a', logs.output[3])
        self.assertIn('system RAM measurement unavailable', logs.output[4])

    def test_metrics_are_collected_without_client_opt_in_and_can_be_disabled(self):
        workers = FixtureWorkers([('PRIVATE OUTPUT', 8)] * 2)
        def factory(*args):
            worker = workers(*args)
            original = worker.run
            def run(*args):
                for item in original(*args):
                    yield {**item, 'timings': {'predicted_per_second': 12.5, 'prompt_ms': 125, 'prompt_per_second': 800}}
            worker.run = run
            return worker
        self.service.scheduler.factory = factory
        body = {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'PRIVATE PROMPT'}], 'auto_image': False}
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            result = self.json('/v1/chat/completions', body)
            wait_job(self.service, result['x_job_id'])
        self.assertTrue(workers.calls[0]['timings_per_token'])
        self.assertNotIn('timings', result)
        output = '\n'.join(logs.output)
        self.assertIn('12.5 t/s', output)
        self.assertIn('prompt 800.0 t/s (TTFT estimate)', output)
        self.assertIn('context ', output)
        self.assertIn('estimated-test', output)
        self.assertNotIn('PRIVATE', output)
        self.json('/api/server', {'terminal_metrics': False})
        self.assertFalse(json.loads(self.service.settings_path.read_text())['terminal_metrics'])
        with self.assertNoLogs('lazyrig.runtime', level='INFO'):
            result = self.json('/v1/chat/completions', body)
            wait_job(self.service, result['x_job_id'])
        self.assertFalse(workers.calls[-1]['timings_per_token'])

    def test_live_output_is_throttled_and_final_native_metrics_win(self):
        clock = [0.0]
        reporter = Reporter({'mode': 'openvino'}, {'id': 'a' * 32, 'model': 'llm'}, clock=lambda: clock[0])
        context = {'input_tokens': 40, 'output_budget': 20, 'generated_tokens': 10,
                   'used_tokens': 50, 'context_limit': 100, 'accuracy': 'estimated-multimodal', 'visual_tokens_included': False}
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            reporter.start_text('llm', 'initial', context)
            clock[0] = .5
            reporter.text(context, {'predicted_per_second': 1})
            clock[0] = 2
            reporter.text(context, {'predicted_per_second': 5})
            reporter.text({**context, 'accuracy': 'exact', 'visual_tokens_included': True},
                {'predicted_per_second': 5, 'x_openvino_metrics': {'tokens_per_second': 13}}, final=True)
        self.assertEqual(len(logs.output), 3)
        self.assertIn('visual tokens pending', logs.output[0])
        self.assertIn('13.0 t/s (native)', logs.output[-1])
        self.assertIn('prompt n/a (TTFT estimate)', logs.output[-1])
        self.assertIn('context 50/100 (50.0%; exact)', logs.output[-1])

    def test_provider_probes_remain_404_with_original_logs(self):
        for path in ('/api/v1/models', '/api/tags', '/v1/props', '/props', '/version'):
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, 404)
            self.assertIn(f'"GET {path} HTTP/1.1" 404 -', output.getvalue())
            self.assertNotIn('provider-discovery', output.getvalue())
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/missing-route')
        self.assertEqual(error.exception.code, 404)
        self.assertNotIn('provider-discovery', output.getvalue())
        self.assertTrue(self.json('/v1/models')['data'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
