"""Structured client-tool parameters; complete model responses are fixtures."""
import copy
import json
import unittest
import urllib.error

import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers
from lazyrig.tool_calls import ToolCallError, parse_tool_calls


# Relevant fields documented in Hermes's tools reference, checked 2026-10-01.
# This is a local schema fixture, not a live Hermes session or tool executor.
CLARIFY = {'type': 'function', 'function': {'name': 'clarify', 'parameters': {
    'type': 'object', 'properties': {'questions': {
        'type': 'array', 'minItems': 1, 'maxItems': 5, 'items': {
            'type': 'object', 'properties': {
                'question': {'type': 'string'}, 'choices': {'type': 'array', 'items': {'type': 'string'}},
                'multi_select': {'type': 'boolean'},
            }, 'required': ['question']},
    }}, 'required': ['questions']}}}
QUESTIONS = [{'question': 'Keep the original framing?', 'choices': ['Keep', 'Crop'], 'multi_select': False},
             {'question': 'Which background?', 'choices': ['Blue', 'White'], 'multi_select': True}]


def xml_call(value, tool=CLARIFY, key='questions'):
    return ('<tool_call><function=' + tool['function']['name'] + '><parameter=' + key + '>' + value +
            '</parameter></function></tool_call>')


def arguments(value, schema=None):
    tool = copy.deepcopy(CLARIFY)
    if schema is not None:
        tool['function']['parameters']['properties']['questions'] = schema
    return parse_tool_calls(xml_call(value, tool), [tool])['calls'][0]['arguments']['questions']


class StructuredValueTests(unittest.TestCase):
    def test_nested_questions_accept_json_and_data_literals(self):
        variants = [json.dumps(QUESTIONS), repr(QUESTIONS),
                    repr(QUESTIONS).replace('False', 'false').replace('True', 'true'),
                    json.dumps(QUESTIONS).replace('false', 'False').replace('true', 'True'),
                    repr(QUESTIONS).replace('}', ',}').replace(']', ',]')]
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(arguments(value), QUESTIONS)
                self.assertEqual(json.loads(json.dumps(arguments(value))), QUESTIONS)

    def test_complete_fences_and_one_json_string_wrapper(self):
        for value in (json.dumps(QUESTIONS), repr(QUESTIONS)):
            for wrapped in (value, json.dumps(value)):
                self.assertEqual(arguments(wrapped), QUESTIONS)
                for language in ('', 'json', 'python', 'JSON'):
                    with self.subTest(language=language, wrapped=wrapped[:30]):
                        self.assertEqual(arguments('```' + language + '\n' + wrapped + '\n```'), QUESTIONS)

    def test_objects_and_schema_unions_use_same_decoder(self):
        data = {'nested': [False, True, None, -2, 1.5, -2.5e-10], 'empty': {}}
        for schema in ({'type': 'object'}, {'type': ['object', 'null']},
                       {'anyOf': [{'type': 'object'}, {'type': 'array'}]},
                       {'oneOf': [{'type': 'null'}, {'type': 'object'}]}):
            with self.subTest(schema=schema):
                self.assertEqual(arguments(repr(data), schema), data)
        for schema in ({'type': ['array', 'boolean']}, {'type': ['array', 'null']}):
            self.assertEqual(arguments(repr(QUESTIONS), schema), QUESTIONS)
        self.assertIsNone(arguments('null', {'type': ['array', 'null']}))
        self.assertIs(arguments('false', {'type': ['array', 'boolean']}), False)
        with self.assertRaises(ToolCallError):
            arguments('"null"', {'type': ['array', 'null']})

    def test_literal_text_is_not_replaced_inside_structures(self):
        strings = ["Don't replace False, True or None.", 'Quotes: "blue"; café 🌄',
                   'Line one\nLine two\tTabbed', r'C:\photos\new\image.png',
                   "Don't lose the control character: \x1b", 'A slash / and a literal \\/',
                   'A literal [,] and {,} and # comment', 'false, true, null',
                   '__import__("os").system("never execute this")']
        self.assertEqual(arguments(repr(strings)), strings)
        self.assertEqual(arguments(json.dumps(strings)), strings)
        # Mixed JSON/Python spellings leave double-quoted JSON string escapes intact.
        mixed = '[' + json.dumps(strings) + ', False, None]'
        self.assertEqual(arguments(mixed), [strings, False, None])
        self.assertEqual(arguments('["https:\\/\\/example.test\\/image.png", False]'),
                         ['https://example.test/image.png', False])

    def test_strings_unknown_types_and_ambiguous_unions_are_preserved(self):
        for schema in ({'type': 'string'}, {}, {'type': ['array', 'string']},
                       {'anyOf': [{'type': 'object'}, {'type': 'string'}]},
                       {'oneOf': [{'type': 'array'}, {'type': 'string'}]}):
            with self.subTest(schema=schema):
                self.assertEqual(arguments(repr(QUESTIONS), schema), repr(QUESTIONS))
        encoded = json.dumps(json.dumps(QUESTIONS))
        self.assertEqual(arguments(encoded, {'type': ['array', 'string']}), json.dumps(QUESTIONS))

    def test_incomplete_ambiguous_or_executable_structures_are_rejected(self):
        invalid = ["[{'question': 'Which edit?'}", "[{'question': 'Which edit?'}] extra",
                   "[{'question': 'A' 'B'}]", '[1 2]', '[,]', '{,}', '[False,,]',
                   "[{'question':,}]", "[{question: 'Which edit?'}]", '[undefined]',
                   '[NaN]', '[Infinity]', '[1e999]', '[0x20]', '[1_000]',
                   '[False # comment\n]', '[False // comment\n]',
                   "[('Which edit?',)]", "[{'Blue', 'White'}]", "[b'bytes']", "[r'raw']",
                   "[f'{False}']", "['''multiline''']", '["""multiline"""]',
                   '[False or True]', '[1 + 2]', '[x for x in []]', '[*[]]', '[{**{}}]',
                   '[__import__("os").system("never execute this")]', '[False.__class__]',
                   "[{'question': 'A', 'question': 'B'}]", '[{"question":"A","question":"B"}]',
                   "[{'question': 'A', '\\u0071uestion': 'B'}]", '[]\n[]',
                   '```json\n[]', '```json\n[]\n``` suffix', '```json\nfalse\n```',
                   'Which edit?', '{"question":"Which edit?"}',
                   json.dumps(json.dumps(json.dumps(QUESTIONS)))]
        for value in invalid:
            with self.subTest(value=value[:80]), self.assertRaises(ToolCallError):
                arguments(value)
        with self.assertRaises(ToolCallError):
            arguments('[False]', {'type': 'object'})

    def test_fallback_is_bounded(self):
        self.assertEqual(arguments('[' * 64 + 'False' + ']' * 64), self.nested(64))
        for value in ('[' * 65 + 'False' + ']' * 65, '[' + 'False,' * 50001 + ']',
                      "['" + 'x' * (1024 * 1024) + "']"):
            with self.subTest(length=len(value)), self.assertRaises(ToolCallError):
                arguments(value)

    @staticmethod
    def nested(depth):
        value = False
        for _ in range(depth):
            value = [value]
        return value

    def test_normalization_preserves_call_boundaries_and_json_call_rules(self):
        call = xml_call(repr(QUESTIONS))
        self.assertFalse(parse_tool_calls('```xml\n' + call + '\n```', [CLARIFY])['calls'])
        for value in (call[:-12], call + 'Extra text', '<think>Thinking\n' + call,
                      call.replace('clarify', 'unknown'), call + xml_call('[False please]')):
            with self.subTest(value=value[:80]), self.assertRaises(ToolCallError):
                parse_tool_calls(value, [CLARIFY])
        invalid = '<tool_call>{"name":"clarify","arguments":{"questions":' + repr(QUESTIONS) + '}}</tool_call>'
        with self.assertRaises(ToolCallError):
            parse_tool_calls(invalid, [CLARIFY])


class StructuredValueHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **extra):
        return {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Edit the image background.'}],
                'tools': [CLARIFY], 'max_tokens': 512, **extra}

    def check_call(self, choice):
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        message = choice['message']
        self.assertEqual(message['reasoning_content'], 'Check the requested edit.')
        self.assertEqual(message['content'], 'Which changes would you like?')
        self.assertEqual(len(message['tool_calls']), 1)
        call = message['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'clarify')
        self.assertEqual(json.loads(call['function']['arguments']), {'questions': QUESTIONS})
        return call

    def test_questions_survive_sync_stream_saved_result_and_followup(self):
        raw = '<think>Check the requested edit.</think>\nWhich changes would you like?\n' + xml_call(repr(QUESTIONS))
        workers = FixtureWorkers([(raw, 80)] * 3 + [('I will keep the framing and use blue.', 12)])
        self.service.scheduler.factory = workers
        self.check_call(self.json('/v1/chat/completions', self.body())['choices'][0])
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            wire = response.read().decode()
        self.assertIn('data: [DONE]', wire)
        records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        self.assertFalse(any('error' in row for row in records), records)
        choices = [row['choices'][0] for row in records if row.get('choices')]
        message = {'content': '', 'reasoning_content': '', 'tool_calls': []}
        for choice in choices:
            for key in ('content', 'reasoning_content'):
                message[key] += choice['delta'].get(key) or ''
            message['tool_calls'].extend(choice['delta'].get('tool_calls', []))
        self.check_call({'message': message, 'finish_reason': choices[-1]['finish_reason']})
        receipt = self.json('/api/chat/jobs', self.body())
        wait_job(self.service, receipt['id'])
        saved = self.json(receipt['result_url'])
        call = self.check_call(saved['choices'][0])
        self.assertEqual(self.json(receipt['result_url']), saved)
        self.assertEqual(self.json('/api/jobs/' + receipt['id'])['text'], raw)
        messages = self.body()['messages'] + [saved['choices'][0]['message'],
            {'role': 'tool', 'tool_call_id': call['id'], 'name': 'clarify',
             'content': '{"responses":[{"user_response":"Keep"},{"user_response":"Blue"}],"outcome":"submitted"}'}]
        followup = self.json('/v1/chat/completions', self.body(messages=messages))
        self.assertEqual(followup['choices'][0]['message']['content'], 'I will keep the framing and use blue.')
        self.assertEqual(len(workers.calls), 4)
        self.assertTrue(all(payload['action'] == 'text' for payload in workers.calls))

    def test_malformed_second_call_never_returns_a_partial_tool_batch(self):
        raw = xml_call(repr(QUESTIONS)) + xml_call("[{'question': 'Which edit?'}")
        self.service.scheduler.factory = FixtureWorkers([(raw, 80)] * 2)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/v1/chat/completions', self.body())
        self.assertEqual(caught.exception.code, 502)
        self.assertEqual(caught.exception.headers['X-Should-Retry'], 'false')
        jid = caught.exception.headers['X-Job-Id']
        error = json.load(caught.exception)
        self.assertIn('parameter questions (expected array)', str(error))
        self.assertIn('no tool was executed', str(error))
        self.assertNotIn('tool_calls', error)
        self.assertEqual(self.json('/api/jobs/' + jid)['text'], raw)
        with self.assertRaises(urllib.error.HTTPError):
            self.json('/api/jobs/' + jid + '/result')
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            wire = response.read().decode()
        records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        self.assertTrue(any('error' in row for row in records), records)
        self.assertFalse(any(choice.get('delta', {}).get('tool_calls')
                             for row in records for choice in row.get('choices', [])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
