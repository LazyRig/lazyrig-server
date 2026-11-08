"""Saving construction settings reloads live models and reports the outcome."""
import copy
import tempfile
import threading
import unittest

from support import DEFAULT, Service, FixtureWorker
from test_core import wait_job


class SettingsReloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=temp.name, terminal_metrics=False)
        self.service = Service(config)
        self.addCleanup(self.service.close)
        self.loads = []
        def factory(mode, model, *args):
            self.loads.append(copy.deepcopy(model))
            return FixtureWorker(mode, model, *args)
        self.service.scheduler.factory = factory

    def load(self):
        job = wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))

    def test_runtime_edit_automatically_reloads_with_new_config(self):
        self.load()
        old = self.service.scheduler.resident['qwen-chat'][1]
        state = self.service.settings({'model': 'qwen-chat', 'plugin_config': {'KV_CACHE_PRECISION': 'u4'}})
        jid = state['settings_effects']['reload_job_id']
        self.assertIsNotNone(jid)
        job = wait_job(self.service, jid)
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertFalse(old.alive)
        self.assertEqual(self.loads[-1]['plugin_config']['KV_CACHE_PRECISION'], 'u4')
        self.assertIn('qwen-chat', self.service.scheduler.resident)
        self.assertEqual(job['reload_models'], ['qwen-chat'])
        events = self.service.store.events(jid)
        self.assertTrue(any(e['kind'] == 'unloaded' and 'settings changed' in e.get('reason', '') for e in events))

    def test_sampling_template_context_limit_and_name_apply_without_reload(self):
        self.load()
        old = self.service.scheduler.resident['qwen-chat'][1]
        state = self.service.settings({'model': 'qwen-chat', 'name': 'Renamed', 'context_limit': 65536,
            'generation': {'temperature': .4}, 'template_context': {'enable_thinking': True}})
        self.assertIsNone(state['settings_effects']['reload_job_id'])
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1], old)
        self.assertEqual(self.service.scheduler.resident['qwen-chat'][0]['context_limit'], 65536)
        self.assertEqual(len(self.loads), 1)

    def test_unloaded_model_is_not_loaded_just_by_editing(self):
        state = self.service.settings({'model': 'qwen-chat', 'scheduler_config': {'cache_size': 2}})
        self.assertIsNone(state['settings_effects']['reload_job_id'])
        self.assertFalse(self.loads)

    def test_disabling_a_model_unloads_without_reload(self):
        self.load()
        state = self.service.settings({'model': 'qwen-chat', 'enabled': False})
        self.assertIsNone(state['settings_effects']['reload_job_id'])
        self.assertFalse(self.service.scheduler.resident)

    def test_budget_evictions_do_not_trigger_reload_loop(self):
        self.load()
        state = self.service.settings({'device': 'TEST.GPU', 'budget_gib': 10})
        self.assertIsNone(state['settings_effects']['reload_job_id'])
        self.assertFalse(self.service.scheduler.resident)
        self.assertEqual(len(self.loads), 1)

    def test_reload_failure_is_a_visible_failed_job_with_settings_saved(self):
        self.load()
        def fail(*args):
            raise RuntimeError('Injected compile failure')
        self.service.scheduler.factory = fail
        state = self.service.settings({'model': 'qwen-chat', 'plugin_config': {'KV_CACHE_PRECISION': 'u4'}})
        job = wait_job(self.service, state['settings_effects']['reload_job_id'])
        self.assertEqual(job['status'], 'failed')
        self.assertIn('Injected compile failure', job['error'])
        self.assertEqual(self.service.model('qwen-chat')['plugin_config']['KV_CACHE_PRECISION'], 'u4')
        self.assertFalse(self.service.scheduler.resident)
        self.assertFalse(self.service.running)

    def test_invalid_setting_never_unloads_or_launches_reload(self):
        self.load()
        old = self.service.scheduler.resident['qwen-chat'][1]
        with self.assertRaises(ValueError):
            self.service.settings({'model': 'qwen-chat', 'plugin_config': {'KV_CACHE_PRECISION': 'bad'}})
        self.assertIs(self.service.scheduler.resident['qwen-chat'][1], old)
        self.assertEqual(len(self.loads), 1)

    def test_new_requests_and_settings_do_not_race_a_pending_reload(self):
        self.load()
        entered, release = threading.Event(), threading.Event()
        factory = self.service.scheduler.factory
        def gated(*args):
            entered.set()
            release.wait(3)
            return factory(*args)
        self.service.scheduler.factory = gated
        try:
            state = self.service.settings({'model': 'qwen-chat', 'plugin_config': {'KV_CACHE_PRECISION': 'u4'}})
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(ValueError, 'active or queued'):
                self.service.settings({'model': 'qwen-chat', 'name': 'Other'})
            sid = self.service.store.session()
            jid = self.service.submit({'session_id': sid, 'model': 'qwen-chat', 'prompt': 'Hello',
                                       'tools_enabled': False, 'max_tokens': 128})['id']
        finally:
            release.set()
        self.assertEqual(wait_job(self.service, state['settings_effects']['reload_job_id'])['status'], 'completed')
        self.assertEqual(wait_job(self.service, jid)['status'], 'completed')
        self.assertEqual(len(self.loads), 2)
