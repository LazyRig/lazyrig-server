"""Client tool argument regressions; model output is a controlled fixture."""
import copy
import json
import unittest
import urllib.error

import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers
from lazyrig.tool_calls import ToolCallError, parse_tool_calls


# Relevant field types from Hermes terminal_tool.py at 375930d0894819fd2707ba885459d3f81d9e3057.
# No Hermes commands are executed by these tests or by LazyRig's client-tool route.
TERMINAL = {'type': 'function', 'function': {'name': 'terminal', 'parameters': {
    'type': 'object', 'properties': {
        'command': {'type': 'string'}, 'background': {'type': 'boolean', 'default': False},
        'timeout': {'type': 'integer'}, 'pty': {'type': 'boolean'},
        'notify': {'anyOf': [{'type': 'boolean'}, {'type': 'array', 'items': {'type': 'string'}}]},
    }, 'required': ['command']}}}
COMMAND = 'python edit_image.py --prompt "Change the background to blue"'


def xml_call(value, key='background'):
    return ('<tool_call><function=terminal><parameter=command>' + COMMAND + '</parameter>'
            '<parameter=' + key + '>' + value + '</parameter></function></tool_call>')


class ParameterValueTests(unittest.TestCase):
    def arguments(self, value, key='background', tool=TERMINAL):
        return parse_tool_calls(xml_call(value, key), [tool])['calls'][0]['arguments']

    def test_boolean_spelling_and_balanced_quotes_become_real_booleans(self):
        for flag in (False, True):
            for spelling in (str(flag), str(flag).lower(), str(flag).upper(), str(flag).swapcase()):
                for literal in (spelling, '"' + spelling + '"', "'" + spelling + "'"):
                    for key in ('background', 'pty'):
                        with self.subTest(literal=literal, key=key):
                            args = self.arguments(' \n' + literal + '\n ', key)
                            self.assertIs(args[key], flag)
                            self.assertEqual(args['command'], COMMAND)
                            self.assertIs(json.loads(json.dumps(args))[key], flag)

    def test_boolean_unions_and_hermes_notify_array(self):
        for variant in ({'type': ['boolean', 'null']},
                        {'anyOf': [{'type': 'boolean'}, {'type': 'null'}]},
                        {'oneOf': [{'type': 'boolean'}, {'type': 'null'}]}):
            tool = copy.deepcopy(TERMINAL)
            tool['function']['parameters']['properties']['background'] = variant
            with self.subTest(variant=variant):
                self.assertIs(self.arguments('False', tool=tool)['background'], False)
                self.assertIsNone(self.arguments('null', tool=tool)['background'])
        self.assertIs(self.arguments('False', 'notify')['notify'], False)
        self.assertEqual(self.arguments('["ready", "False"]', 'notify')['notify'], ['ready', 'False'])
        for literal in ('yes', '0', '"perhaps"', '{}'):
            with self.subTest(literal=literal), self.assertRaises(ToolCallError):
                self.arguments(literal, 'notify')

    def test_strings_ambiguous_unions_and_unknown_parameters_are_not_coerced(self):
        for definition in ({'type': 'string'}, {'type': ['string', 'boolean']},
                           {'anyOf': [{'type': 'boolean'}, {'type': 'string'}]}, {}):
            tool = copy.deepcopy(TERMINAL)
            tool['function']['parameters']['properties']['background'] = definition
            self.assertEqual(self.arguments('False', tool=tool)['background'], 'False')
        tool['function']['parameters']['properties']['background'] = {'type': 'string'}
        for value in ('true', 'False', 'opaque', '"false"', '0'):
            with self.subTest(value=value):
                self.assertEqual(self.arguments(value, tool=tool)['background'], value)
        self.assertEqual(self.arguments('False', 'unknown')['unknown'], 'False')

    def test_ambiguous_invalid_and_executable_boolean_values_are_rejected(self):
        for value in ('', 'yes', 'no', 'on', 'off', '1', '0', 'null', 'None', '[]', '{}',
                      'false true', 'False because this is an edit', 'false,', 'True # comment',
                      '"False', "'False\"", '" false "', '"\\u0066alse"', '`False`',
                      '```json\nfalse\n```', 'Fa', 'NaN', 'Infinity', '1e999',
                      '__import__("os").system("echo unsafe")',
                      '<parameter=pty>False</parameter>'):
            with self.subTest(value=value), self.assertRaises(ToolCallError):
                self.arguments(value)

    def test_json_blocks_are_not_repaired_or_recursively_rewritten(self):
        for flag in (False, True, 'False'):
            args = {'command': COMMAND, 'background': flag}
            raw = '<tool_call>' + json.dumps({'name': 'terminal', 'arguments': args}) + '</tool_call>'
            self.assertEqual(parse_tool_calls(raw, [TERMINAL])['calls'][0]['arguments'], args)
        for arguments in ('{"background":False}', '{"background":false,"background":true}'):
            raw = '<tool_call>{"name":"terminal","arguments":' + arguments + '}</tool_call>'
            with self.subTest(arguments=arguments), self.assertRaises(ToolCallError):
                parse_tool_calls(raw, [TERMINAL])

    def test_typed_json_values_keep_validation(self):
        tool = copy.deepcopy(TERMINAL)
        for kind, good, expected, bad in (
            ('integer', '3', 3, 'True'), ('number', '3.5', 3.5, 'True'),
            ('object', '{"enabled":false,"label":"False"}', {'enabled': False, 'label': 'False'}, '{"enabled":False please}'),
            ('array', '[false,"False"]', [False, 'False'], '[False please]'),
            ('null', 'null', None, 'None'),
        ):
            tool['function']['parameters']['properties']['background'] = {'type': kind}
            with self.subTest(kind=kind):
                self.assertEqual(self.arguments(good, tool=tool)['background'], expected)
                with self.assertRaises(ToolCallError):
                    self.arguments(bad, tool=tool)

    def test_normalization_does_not_bypass_call_boundaries(self):
        call = xml_call('False')
        for text in ('```xml\n' + call + '\n```', 'Example: ' + call,
                     '`' + call + '`', '<think>' + call + '</think>Explanation.'):
            self.assertFalse(parse_tool_calls(text, [TERMINAL])['calls'])
        for text in (call[:-12], '<think>Still deciding\n' + call,
                     call + 'Unfinished explanation', call.replace('terminal', 'unknown'),
                     call.replace('</function>', '<parameter=background>True</parameter></function>')):
            with self.subTest(text=text[:50]), self.assertRaises(ToolCallError):
                parse_tool_calls(text, [TERMINAL])


class ParameterHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def body(self, **extra):
        return {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Edit the image background.'}],
                'tools': [TERMINAL], 'max_tokens': 256, **extra}

    def check_call(self, choice):
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        message = choice['message']
        self.assertEqual(message['reasoning_content'], 'Use the edit script.')
        self.assertEqual(message['content'], 'I will edit the image.')
        self.assertEqual(len(message['tool_calls']), 1)
        call = message['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'terminal')
        args = json.loads(call['function']['arguments'])
        self.assertIs(args['background'], False)
        self.assertEqual(args['command'], COMMAND)
        return call

    def test_boolean_survives_sync_stream_saved_result_and_client_followup(self):
        raw = '<think>Use the edit script.</think>\nI will edit the image.\n' + xml_call('False')
        workers = FixtureWorkers([(raw, 40)] * 3 + [('The edit finished.', 5)])
        self.service.scheduler.factory = workers
        plain = self.json('/v1/chat/completions', self.body())
        self.check_call(plain['choices'][0])
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            wire = response.read().decode()
        self.assertIn('data: [DONE]', wire)
        records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        self.assertFalse(any('error' in row for row in records), records)
        choices = [row['choices'][0] for row in records if row.get('choices')]
        message = {'content': '', 'reasoning_content': '', 'tool_calls': []}
        for choice in choices:
            delta = choice['delta']
            for key in ('content', 'reasoning_content'):
                message[key] += delta.get(key) or ''
            message['tool_calls'].extend(delta.get('tool_calls', []))
        self.check_call({'message': message, 'finish_reason': choices[-1]['finish_reason']})
        receipt = self.json('/api/chat/jobs', self.body())
        wait_job(self.service, receipt['id'])
        saved = self.json(receipt['result_url'])
        call = self.check_call(saved['choices'][0])
        self.assertEqual(self.json(receipt['result_url']), saved)
        messages = self.body()['messages'] + [saved['choices'][0]['message'],
            {'role': 'tool', 'tool_call_id': call['id'], 'name': 'terminal',
             'content': '{"output":"edited.png","exit_code":0}'}]
        followup = self.json('/v1/chat/completions', self.body(messages=messages))
        self.assertEqual(followup['choices'][0]['message']['content'], 'The edit finished.')
        self.assertEqual(len(workers.calls), 4)
        self.assertTrue(all(payload['action'] == 'text' for payload in workers.calls))

    def test_malformed_boolean_returns_error_without_a_partial_call(self):
        self.service.scheduler.factory = FixtureWorkers([(xml_call('False please'), 40)] * 2)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/v1/chat/completions', self.body())
        error = json.load(caught.exception)
        self.assertIn('no tool was executed', str(error))
        self.assertNotIn('tool_calls', error)
        with self.request('/v1/chat/completions', self.body(stream=True)) as response:
            wire = response.read().decode()
        records = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        self.assertTrue(any('error' in row for row in records), records)
        self.assertFalse(any(choice.get('delta', {}).get('tool_calls')
                             for row in records for choice in row.get('choices', [])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
