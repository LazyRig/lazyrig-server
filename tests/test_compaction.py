"""Context safety, transcript integrity and OpenAI-compatible delivery contracts."""
import copy
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from lazyrig.compaction import (ContextCompactor, CompactionError, ContextOverflow, DEFAULTS,
                                SUMMARY_INSTRUCTION, groups, protected_groups, validate)
from lazyrig.compat import start_chat, completion, validate_chat
from lazyrig.deadlines import DeadlineEvent, deadline_context, RequestDeadline
from lazyrig.worker import Cancelled, NativeWorkerError
from support import DEFAULT, TokenizerService, Service, FixtureWorker
from test_core import wait_job
import test_http


def conversation(pairs=28, width=600):
    rows = [{'role': 'system', 'content': 'Keep exact identifiers and speak English.'},
            {'role': 'user', 'content': 'Our project is Cedar.'},
            {'role': 'assistant', 'content': 'Understood.'}]
    for i in range(pairs):
        rows.extend([{'role': 'user', 'content': f'record-{i:03d}: ' + 'x' * width},
                     {'role': 'assistant', 'content': f'answer-{i:03d}: ' + 'y' * width}])
    rows.append({'role': 'user', 'content': 'What is the next step? Keep ID CEDAR-42.'})
    return rows


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.tokens = TokenizerService('openvino')
        self.compactor = ContextCompactor(self.tokens)
        self.model = {**copy.deepcopy(DEFAULT['models'][0]), 'context_limit': 4096}
        self.options = dict(DEFAULTS)
        self.calls = []

    def summarize(self, prepared, budget, remaining):
        self.assertTrue(prepared['context']['fits'])
        self.assertLessEqual(prepared['context']['input_tokens'] + budget, self.model['context_limit'])
        self.assertFalse(prepared['template_context']['enable_thinking'])
        self.assertGreater(remaining, 0)
        self.calls.append(copy.deepcopy(prepared))
        return 'Project Cedar. Preserve CEDAR-42. Work remains in progress.', {
            'prompt_tokens': prepared['context']['input_tokens'], 'completion_tokens': 17}

    def compact(self, rows, **extra):
        return self.compactor.prepare(self.model, rows, extra.pop('budget', 256),
            tools=extra.pop('tools', None), add_system=False, options=self.options,
            summarize=extra.pop('summarize', self.summarize), **extra)

    def test_overflow_is_summarized_in_fitting_chunks_without_mutating_input(self):
        rows = conversation()
        before = copy.deepcopy(rows)
        compacted, prepared, stats = self.compact(rows)
        self.assertEqual(rows, before)
        self.assertTrue(stats['applied'])
        self.assertGreater(stats['summary_calls'], 1)
        self.assertTrue(prepared['context']['fits'])
        self.assertEqual(compacted[:3], rows[:3])
        self.assertEqual(compacted[-1], rows[-1])
        self.assertLess(stats['input_tokens_after'], stats['input_tokens_before'])
        # Every removed record is sent to the summarizer, without sampling it.
        seen = ''.join(c['messages'][-1]['content'].split('Next conversation records (may continue across chunks):\n')[1] for c in self.calls)
        for row in rows:
            if row not in compacted:
                self.assertIn(row['content'], seen)

    def test_threshold_and_output_reservation_each_trigger_compaction(self):
        rows = conversation(pairs=10, width=450)
        self.options['threshold'] = .50
        compacted, prepared, stats = self.compact(rows)
        self.assertTrue(stats['applied'])
        self.compactor.cache.clear()
        self.options['threshold'] = .95
        compacted, prepared, stats = self.compact(rows, budget=2500)
        self.assertTrue(stats['applied'])
        self.assertTrue(prepared['context']['fits'])

    def test_small_request_does_not_load_summarizer_or_change_prompt(self):
        rows = [{'role': 'user', 'content': 'Hello'}]
        expected = self.tokens.prepare(self.model, rows, 256, False, add_system=False)
        actual, prepared, stats = self.compact(rows)
        self.assertEqual(actual, rows)
        self.assertEqual(prepared, expected)
        self.assertFalse(stats['applied'])
        self.assertFalse(self.calls)

    def test_opt_out_keeps_exact_prompt_and_rejects_overflow(self):
        self.options['enabled'] = False
        with self.assertRaises(ContextOverflow):
            self.compact(conversation())
        self.assertFalse(self.calls)

    def test_identical_prefix_reuses_summary_but_changed_history_does_not(self):
        rows = conversation()
        self.compact(rows)
        calls = len(self.calls)
        extended = rows + [{'role': 'assistant', 'content': 'Next step is testing.'}, {'role': 'user', 'content': 'Proceed.'}]
        _, _, stats = self.compact(extended)
        self.assertTrue(stats['reused'])
        self.assertEqual(len(self.calls), calls)
        changed = copy.deepcopy(extended)
        changed[4]['content'] = 'Changed historical answer.'
        _, _, stats = self.compact(changed)
        self.assertFalse(stats['reused'])
        self.assertGreater(len(self.calls), calls)

    def test_model_template_tools_and_budget_invalidate_cached_summary(self):
        rows = conversation()
        self.compact(rows)
        self.model['template_context']['enable_thinking'] = True
        _, _, stats = self.compact(rows)
        self.assertFalse(stats['reused'])
        _, _, stats = self.compact(rows, tools=[{'type': 'function', 'function': {'name': 'look'}}])
        self.assertFalse(stats['reused'])
        _, _, stats = self.compact(rows, budget=512)
        self.assertFalse(stats['reused'])

    def test_latest_user_and_all_its_tool_results_are_hard_protected(self):
        rows = conversation() + [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'Result A'},
            {'role': 'tool', 'tool_call_id': 'b', 'content': 'Result B'}]
        compacted, _, _ = self.compact(rows)
        self.assertEqual(compacted[-4:], rows[-4:])

    def test_all_system_rows_images_orphans_and_unfinished_calls_are_pinned(self):
        rows = conversation(pairs=4, width=10)
        rows.insert(5, {'role': 'system', 'content': 'Never change this instruction.'})
        rows.insert(6, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]})
        rows.insert(7, {'role': 'tool', 'tool_call_id': 'unknown', 'content': 'An orphan result'})
        rows.insert(8, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'pending', 'function': {'name': 'read'}}]})
        atomic, pinned = protected_groups(rows, self.options, False)
        indices = {i for n in pinned for i in range(*atomic[n])}
        self.assertTrue({5, 6, 7, 8, len(rows)-1} <= indices)

    def test_nonadjacent_parallel_tool_groups_cannot_be_split(self):
        rows = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'one'}, {'id': 'two'}]},
                {'role': 'tool', 'tool_call_id': 'one', 'content': '1'},
                {'role': 'assistant', 'content': 'Intermediate record'},
                {'role': 'tool', 'tool_call_id': 'two', 'content': '2'}]
        atomic, _ = groups(rows)
        self.assertEqual(atomic, [(0, 4)])

    def test_oversized_latest_request_fails_before_summary_or_cache_write(self):
        rows = [{'role': 'user', 'content': 'x' * 20000}]
        with self.assertRaisesRegex(ContextOverflow, 'protected'):
            self.compact(rows)
        self.assertFalse(self.calls)
        self.assertFalse(self.compactor.cache)

    def test_summary_failures_preserve_input_and_do_not_poison_cache(self):
        rows = conversation()
        before = copy.deepcopy(rows)
        def fail(*args):
            raise CompactionError('Summary failed')
        with self.assertRaises(CompactionError):
            self.compact(rows, summarize=fail)
        with self.assertRaisesRegex(CompactionError, 'empty'):
            self.compact(rows, summarize=lambda *args: ('', {}))
        with self.assertRaises(CompactionError):
            self.compact(conversation(), summarize=lambda *args: ('z' * 20000, {}))
        self.assertEqual(rows, before)
        self.assertFalse(self.compactor.cache)

    def test_request_deadline_and_summary_pass_cap_are_bounded(self):
        with deadline_context(DeadlineEvent(timeout=.001)):
            time.sleep(.003)
            with self.assertRaises(RequestDeadline):
                self.compact(conversation())
        self.model['context_limit'] = 2048
        with self.assertRaisesRegex(CompactionError, '16-pass'):
            self.compact(conversation(pairs=150, width=900))
        self.assertEqual(len(self.calls), 16)

    def test_settings_validation_rejects_wrong_types_and_nonfinite_numbers(self):
        for options in ({'enabled': 1}, {'threshold': float('nan')}, {'target_ratio': .9},
                        {'protect_last_n': True}, {'timeout_seconds': 0}, {'unknown': 1}, []):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate(options)


class CompactionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=self.temp.name, terminal_metrics=False)
        config['models'][0]['context_limit'] = 4096
        self.service = Service(config)
        self.addCleanup(self.service.close)

    def body(self):
        return {'model': 'qwen-chat', 'messages': conversation(pairs=18, width=450), 'max_tokens': 256}

    def test_sync_admission_compacts_and_saved_original_history_survives(self):
        body = self.body()
        job = wait_job(self.service, start_chat(self.service, body)['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        result = completion(job, body)
        self.assertTrue(result['x_compaction']['applied'])
        self.assertTrue(result['x_context']['fits'])
        self.assertEqual(self.service.store.messages(job['session_id'])[:-1], body['messages'])
        second = wait_job(self.service, start_chat(self.service, {**body, 'messages': body['messages'] +
            [result['choices'][0]['message'], {'role': 'user', 'content': 'Continue'}]})['id'])
        self.assertTrue(second['compaction']['reused'])

    def test_native_session_also_compacts_without_replacing_saved_messages(self):
        rows = conversation(pairs=18, width=450)
        sid = self.service.store.session()
        for message in rows[:-1]:
            self.service.store.append(sid, message)
        job = wait_job(self.service, self.service.submit({'session_id': sid, 'model': 'qwen-chat',
            'prompt': rows[-1]['content'], 'tools_enabled': False, 'max_tokens': 256})['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertTrue(job['compaction']['applied'])
        self.assertEqual(self.service.store.messages(sid)[:-1], rows)

    def test_compaction_opt_out_and_invalid_types_reject_before_load(self):
        with self.assertRaises(ContextOverflow):
            start_chat(self.service, {**self.body(), 'auto_compact': False})
        for value in ['false', 0, None]:
            with self.assertRaisesRegex(ValueError, 'boolean'):
                start_chat(self.service, {**self.body(), 'auto_compact': value})
        self.assertFalse(self.service.scheduler.resident)

    def test_inspection_reports_pending_compaction_without_loading(self):
        info = self.service.inspect_context(self.body())
        self.assertFalse(info['fits'])
        self.assertTrue(info['compaction_needed'])
        self.assertFalse(self.service.scheduler.resident)

    def test_cancellation_during_summary_never_generates_a_final_answer(self):
        entered = threading.Event()
        class BlockingWorker(FixtureWorker):
            def run(self, request, timeout, cancelled):
                entered.set()
                while not cancelled.is_set():
                    time.sleep(.01)
                raise Cancelled('Cancelled during summary')
                yield
        self.service.scheduler.factory = BlockingWorker
        jid = start_chat(self.service, self.body(), preflight=False)['id']
        self.assertTrue(entered.wait(3))
        self.service.cancel(jid)
        job = wait_job(self.service, jid)
        self.assertEqual(job['status'], 'cancelled')
        self.assertFalse(self.service.compactor.cache)
        self.assertFalse(job['text'])

    def test_capped_summary_is_rejected_without_losing_history(self):
        class CappedWorker(FixtureWorker):
            def run(self, request, timeout, cancelled):
                yield {'kind': 'result', 'text': 'Incomplete summary', 'generated_tokens': request['max_tokens']}
        self.service.scheduler.factory = CappedWorker
        job = wait_job(self.service, start_chat(self.service, self.body(), preflight=False)['id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error_code'], 'context_compaction_failed')
        self.assertFalse(self.service.compactor.cache)
        self.assertFalse(job['text'])

    def test_native_overflow_gets_one_retry_before_output_only(self):
        calls = []
        class RetryWorker(FixtureWorker):
            def run(self, request, timeout, cancelled):
                summary = request['prepared']['messages'][0].get('content') == SUMMARY_INSTRUCTION
                calls.append('summary' if summary else 'answer')
                if not summary and calls.count('answer') == 1:
                    raise NativeWorkerError('maximum context length exceeded')
                yield from super().run(request, timeout, cancelled)
        self.service.scheduler.factory = RetryWorker
        body = {**self.body(), 'messages': conversation(pairs=9, width=300)}
        job = wait_job(self.service, start_chat(self.service, body)['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertTrue(job['context_overflow_retry'])
        self.assertEqual(calls.count('answer'), 2)
        self.assertGreater(calls.count('summary'), 0)


class CompactionHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_sync_stream_and_async_report_compaction(self):
        self.service.config['models'][0]['context_limit'] = 4096
        body = {'model': 'qwen-chat', 'messages': conversation(pairs=18, width=450), 'max_tokens': 256}
        result = self.json('/v1/chat/completions', body)
        self.assertTrue(result['x_compaction']['applied'])
        with self.request('/v1/chat/completions', {**body, 'stream': True}) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith('data: {')]
        final = next(e for e in events if e.get('x_compaction'))
        self.assertTrue(final['x_compaction']['reused'])
        receipt = self.json('/api/chat/jobs', body)
        wait_job(self.service, receipt['id'])
        result = self.json(receipt['result_url'])
        self.assertTrue(result['x_compaction']['applied'])

    def test_settings_round_trip_and_request_override(self):
        state = self.json('/api/server', {'context_compaction': {'enabled': False}})
        self.assertFalse(state['server']['context_compaction']['enabled'])
        self.assertEqual(state['server']['context_compaction']['threshold'], .75)
        restored = json.loads(self.service.settings_path.read_text())
        self.assertFalse(restored['context_compaction']['enabled'])
