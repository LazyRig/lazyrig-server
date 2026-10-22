"""Client tool turns keep completed workers; native failures still retire them."""
import json
import time
import unittest
import urllib.error
from unittest.mock import patch

import test_http
from test_handoff import FixtureWorkers, qwen_call
from lazyrig.tool_calls import ToolCallError
from lazyrig.worker import NativeWorkerError


TOOL = {'type': 'function', 'function': {'name': 'lookup', 'parameters': {
    'type': 'object', 'properties': {'query': {'type': 'string'}}}}}


def call(query='private lookup content'):
    return '<tool_call>' + json.dumps({'name': 'lookup', 'arguments': {'query': query}}) + '</tool_call>'


class ToolWorkerRetentionTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **options):
        return {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Private request'}],
                'tools': [TOOL], 'max_tokens': 8192, **options}

    def fixture(self, outputs):
        workers = FixtureWorkers([(text, 40) for text in outputs])
        self.service.scheduler.factory = workers
        return workers

    def failed(self, body):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/v1/chat/completions', body)
        error = json.load(caught.exception)
        return error, self.json('/api/jobs/' + error['x_job_id'])

    def test_client_tool_loop_reuses_worker_and_honors_each_client_budget(self):
        model = self.service.config['models'][0]
        model.update(context_limit=262144, generation={'max_tokens': 128})
        workers = self.fixture([call('first'), call('second'), call('third'), 'Complete.'])
        messages = self.body()['messages']
        options = [{'max_tokens': 8192}, {'max_tokens': 32768},
                   {'max_tokens': 16384, 'max_completion_tokens': 4096}, {'max_tokens': 512}]
        for index, values in enumerate(options):
            result = self.json('/v1/chat/completions', self.body(messages=messages, **values))
            expected = values.get('max_completion_tokens', values['max_tokens'])
            job = self.json('/api/jobs/' + result['x_job_id'])
            self.assertEqual(job['request']['max_tokens'], expected)
            self.assertEqual(workers.calls[-1]['max_tokens'], expected)
            self.assertEqual(result['x_context']['output_budget'], expected)
            self.assertEqual(job['status'], 'completed')
            self.assertEqual(result['x_artifacts'], [])
            if index < 3:
                self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
                message = result['choices'][0]['message']
                messages += [message, {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
                                       'content': json.dumps({'result': index})}]
            else:
                self.assertEqual(result['choices'][0]['message']['content'], 'Complete.')
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])
        self.assertEqual(workers.calls[-1]['prepared']['messages'], messages)
        self.assertEqual(len(workers.calls), 4)

    def test_rejected_client_call_keeps_worker_and_reports_category_without_content(self):
        raw = call() + 'PRIVATE_GENERATED_SUFFIX'
        workers = self.fixture([raw, call()])
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            error, job = self.failed(self.body())
        self.assertEqual(error['error']['code'], 'invalid_tool_call')
        self.assertFalse(error['x_retryable'])
        self.assertEqual(job['text'], raw)
        self.assertEqual(job['status'], 'failed')
        self.assertNotIn('tool_calls', error)
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])
        self.assertTrue(self.service.scheduler.is_loaded(self.service.model('qwen-chat')))
        events = self.service.store.events(job['id'])
        self.assertTrue(any(e['kind'] == 'response_rejected' and e['worker_retained'] for e in events))
        output = '\n'.join(logs.output)
        self.assertIn('reason: invalid_tool_call', output)
        self.assertIn('model remains loaded', output)
        self.assertNotIn('PRIVATE_GENERATED_SUFFIX', output)
        self.assertNotIn('private lookup content', output)
        self.assertNotIn('Private request', output)
        result = self.json('/v1/chat/completions', self.body())
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])
        self.assertEqual(len(workers.calls), 2)

    def test_streaming_rejection_never_dispatches_partial_calls_or_reloads(self):
        workers = self.fixture([call() + '<tool_call>{broken}</tool_call>', call()])
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            jid = response.headers['X-Job-Id']
            rows = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                    if line.startswith('data: {')]
        self.assertTrue(any('error' in row for row in rows))
        self.assertFalse(any(choice.get('delta', {}).get('tool_calls')
                             for row in rows for choice in row.get('choices', [])))
        self.assertEqual(self.json('/api/jobs/' + jid)['status'], 'failed')
        result = self.json('/v1/chat/completions', self.body())
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])

    def test_invalid_json_keeps_worker_for_corrected_client_request(self):
        workers = self.fixture(['{"private":"unfinished', '{"ok":true}'])
        body = self.body(tools=[], response_format={'type': 'json_object'})
        error, job = self.failed(body)
        self.assertIn('Model did not produce valid JSON', error['error']['message'])
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(self.service.scheduler.is_loaded(self.service.model('qwen-chat')))
        result = self.json('/v1/chat/completions', body)
        self.assertTrue(result['x_json_valid'])
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])

    def test_rejected_managed_image_call_keeps_llm_without_dispatching_image(self):
        workers = self.fixture([qwen_call() + 'Trailing text', 'No image created.'])
        error, job = self.failed(self.body(tools=[], auto_image=True, image_model='qwen-image'))
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['artifacts'], [])
        self.assertEqual(job['tool_attempts'][0]['status'], 'rejected')
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])
        self.assertEqual(len(workers.calls), 1)
        self.json('/v1/chat/completions', self.body())
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat')])

    def test_native_error_even_after_result_unloads_before_next_client_request(self):
        for after_result in [False, True]:
            with self.subTest(after_result=after_result):
                self.service.scheduler.close()
                workers = self.fixture([call(), call()])
                def factory(*args):
                    worker = workers(*args)
                    original = worker.run
                    if len(workers.lifecycle) == 1:
                        def broken(*run_args):
                            if after_result:
                                yield from original(*run_args)
                            raise NativeWorkerError('Fixture native GPU failure')
                        worker.run = broken
                    return worker
                self.service.scheduler.factory = factory
                error, job = self.failed(self.body())
                self.assertEqual(job['status'], 'failed')
                self.assertNotIn('qwen-chat', self.service.scheduler.resident)
                self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat')])
                self.json('/v1/chat/completions', self.body())
                self.assertEqual(workers.lifecycle[-1], ('load', 'qwen-chat'))
                self.assertEqual(len(workers.lifecycle), 3)

    def test_stop_and_expiry_during_validation_still_unload(self):
        for mode in ['stop', 'expiry']:
            with self.subTest(mode=mode):
                workers = self.fixture([call()])
                def reject(*args):
                    signal = next(iter(self.service.running.values()))[0]
                    if mode == 'stop':
                        signal.set()
                    else:
                        signal.deadline = time.monotonic() - 1
                    raise ToolCallError('Rejected during ' + mode)
                with patch('lazyrig.service.parse_tool_calls', side_effect=reject):
                    error, job = self.failed(self.body())
                self.assertEqual(job['status'], 'failed')
                self.assertNotIn('qwen-chat', self.service.scheduler.resident)
                self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat')])

    def test_exited_worker_is_not_retained_after_rejected_call(self):
        workers = self.fixture([call() + 'Trailing text', call()])
        def factory(*args):
            worker = workers(*args)
            original = worker.run
            if len(workers.lifecycle) == 1:
                def exited(*run_args):
                    yield from original(*run_args)
                    worker.alive = False
                worker.run = exited
            return worker
        self.service.scheduler.factory = factory
        self.failed(self.body())
        self.assertNotIn('qwen-chat', self.service.scheduler.resident)
        self.json('/v1/chat/completions', self.body())
        self.assertEqual(workers.lifecycle, [('load', 'qwen-chat'), ('close', 'qwen-chat'), ('load', 'qwen-chat')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
