"""Hermes-style image tool results: role/call pairing, pixels, and replay."""
import json
import unittest
import urllib.error
from PIL import Image
import test_http
from test_core import wait_job
from test_handoff import qwen_call
import test_vision_mtp as vision_fixture
from test_vision_mtp import image_url, parts


VISION_TOOL = {'type': 'function', 'function': {
    'name': 'vision_analyze', 'description': 'Load an image for inspection.',
    'parameters': {'type': 'object', 'properties': {
        'image_url': {'type': 'string'}, 'question': {'type': 'string'}}}}}


def tool_history():
    calls = [{'id': 'vision-' + color, 'type': 'function', 'function': {
        'name': 'vision_analyze', 'arguments': json.dumps({
            'image_url': '/client/' + color + '.png', 'question': 'What is depicted?'})}}
        for color in ('red', 'blue')]
    return [
        {'role': 'user', 'content': 'Inspect both images.'},
        {'role': 'assistant', 'content': None, 'tool_calls': calls},
        {'role': 'tool', 'name': 'vision_analyze', 'tool_call_id': 'vision-red',
         'content': parts(image_url('red'))},
        {'role': 'tool', 'name': 'vision_analyze', 'tool_call_id': 'vision-blue',
         'content': parts(image_url('blue'))[1:]},
    ]


class VisionToolTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json
    vlm = vision_fixture.VisionHTTPTests.vlm
    body = vision_fixture.VisionHTTPTests.body

    def test_nine_user_and_tool_images_in_context_sync_stream_and_async_preserve_pairing_and_pixels(self):
        self.vlm(('All nine images arrived.',) * 3)
        history = tool_history()
        user_colors = [(index * 24, 128, 255 - index * 24) for index in range(7)]
        history[0]['content'] = parts(*(image_url(color) for color in user_colors))
        body = self.body(messages=history, tools=[VISION_TOOL], auto_image=False)
        context = self.json('/api/context', body)
        self.assertEqual(context['image_count'], 9)
        self.assertIsNone(context['fits'])
        self.assertFalse(context['visual_tokens_included'])

        result = self.json('/v1/chat/completions', body)
        self.assertEqual(result['choices'][0]['message']['content'], 'All nine images arrived.')
        with self.request('/v1/chat/completions', {**body, 'stream': True}) as response:
            raw = response.read().decode()
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        self.assertFalse(any('error' in chunk for chunk in chunks), chunks)
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices')),
                         'All nine images arrived.')
        receipt = self.json('/api/chat/jobs', body)
        job = wait_job(self.service, receipt['id'])
        self.assertEqual(job['status'], 'completed', job.get('error'))
        saved = self.json(receipt['result_url'])
        self.assertEqual(saved['choices'][0]['message']['content'], 'All nine images arrived.')
        self.assertEqual(self.json(receipt['result_url']), saved)

        for call in self.workers.calls:
            prepared = call['prepared']
            messages = prepared['messages']
            self.assertEqual(len(prepared['images']), 9)
            self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool', 'tool'])
            self.assertEqual(messages[1]['tool_calls'], history[1]['tool_calls'])
            for index, rgb in enumerate(user_colors):
                self.assertIn(f'<ov_genai_image_{index}>', messages[0]['content'])
                with Image.open(prepared['images'][index]['path']) as image:
                    self.assertEqual(image.getpixel((0, 0)), rgb)
            for offset, (color, rgb) in enumerate((('red', (255, 0, 0)), ('blue', (0, 0, 255)))):
                index = offset + len(user_colors)
                tool = messages[offset + 2]
                self.assertEqual(tool['tool_call_id'], 'vision-' + color)
                self.assertEqual(tool['name'], 'vision_analyze')
                self.assertIn(f'<ov_genai_image_{index}>', tool['content'])
                with Image.open(prepared['images'][index]['path']) as image:
                    self.assertEqual(image.getpixel((0, 0)), rgb)
            self.assertEqual(prepared['context']['prompt_sha256'], context['prompt_sha256'])

        stored = self.service.store.messages(job['session_id'])
        self.assertEqual(stored[2]['tool_call_id'], 'vision-red')
        self.assertEqual(stored[3]['tool_call_id'], 'vision-blue')
        self.assertTrue(stored[3]['content'][0]['image_url']['url'].startswith('lazyrig-input:'))
        self.assertNotIn('base64', json.dumps(job))
        self.assertTrue(history[3]['content'][0]['image_url']['url'].startswith('data:'))

    def test_tool_image_history_survives_image_generation_worker_swap(self):
        self.vlm((qwen_call(), 'Generated after inspecting the references.'))
        history = tool_history() + [{'role': 'user', 'content': 'Generate a new image from those references.'}]
        body = self.body(messages=history, auto_image=True, image_model='qwen-image', steps=1)
        result = self.json('/v1/chat/completions', body)
        calls = [call for call in self.workers.calls if call['action'] == 'text']
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['prepared']['images'], calls[1]['prepared']['images'])
        for call in calls:
            tool_rows = [m for m in call['prepared']['messages'] if m.get('name') == 'vision_analyze']
            self.assertEqual([m['tool_call_id'] for m in tool_rows], ['vision-red', 'vision-blue'])
            self.assertTrue(all('<ov_genai_image_' in m['content'] for m in tool_rows))
        self.assertEqual(len(result['x_artifacts']), 1)
        self.assertIn(('close', 'qwen-chat'), self.workers.lifecycle)
        self.assertEqual(self.workers.lifecycle.count(('load', 'qwen-chat')), 2)

    def test_invalid_tool_images_and_text_only_pipeline_still_fail_before_loading(self):
        body = self.body(messages=tool_history(), auto_image=False)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/v1/chat/completions', body)
        self.assertEqual(error.exception.code, 400)
        self.assertIn('VLM', error.exception.read().decode())
        self.vlm()
        invalid = []
        for call_id in (None, '', 7):
            messages = tool_history()
            messages[-1]['tool_call_id'] = call_id
            invalid.append(messages)
        for role in ('system', 'assistant'):
            messages = tool_history()
            messages[-1]['role'] = role
            invalid.append(messages)
        for url in ('data:image/png;base64,aGVsbG8=', 'http://127.0.0.1/a.png', 'lazyrig-input:' + '0' * 64):
            messages = tool_history()
            messages[-1]['content'] = parts(url)
            invalid.append(messages)
        for messages in invalid:
            with self.subTest(last=messages[-1].get('tool_call_id')), self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/chat/jobs', {**body, 'messages': messages})
            self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.workers.calls)
        self.assertFalse(self.workers.lifecycle)


if __name__ == '__main__':
    unittest.main(verbosity=2)
