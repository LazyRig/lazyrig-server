"""HTTP disconnect recovery and image readiness independent of LLM restoration."""
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
import urllib.error
from unittest.mock import patch
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers
import test_image_edits
from lazyrig.server import handler
from support import Service
from lazyrig.idempotency import image_identity


class ImageRecoveryTests(unittest.TestCase):
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def setUp(self):
        test_http.HTTPTests.setUp(self)
        self.gates = []
        self.workers = FixtureWorkers([('The image is ready.', 5)])
        self.service.scheduler.factory = self.workers
        self.body = {'model': 'qwen-image', 'prompt': 'A lake', 'steps': 1}

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        test_http.HTTPTests.tearDown(self)

    def post(self, body=None, key='image-request-one', path='/api/image/jobs', headers=None):
        with self.request(path, self.body if body is None else body,
                          {'Idempotency-Key': key, **(headers or {})}) as response:
            return json.load(response)

    def restore_gate(self, fail=False):
        self.service.config['models'][1]['restore_text_after_image'] = True
        wait_job(self.service, self.service.control('qwen-chat', 'load')['id'])
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)
        def factory(mode, model, timeout, cancelled):
            if model['id'] == 'qwen-chat':
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('Test did not release restoration')
                if fail:
                    raise RuntimeError('Restore fixture failed')
            return self.workers(mode, model, timeout, cancelled)
        self.service.scheduler.factory = factory
        return entered, release

    def test_saved_image_returns_before_restoration_and_chat_waits_for_device(self):
        entered, release = self.restore_gate()
        result = self.post(path='/v1/images/generations')
        self.assertTrue(entered.wait(2))
        self.assertTrue(result['x_result_ready'])
        self.assertEqual(result['x_status'], 'running')
        self.assertNotIn('b64_json', result['data'][0])
        self.assertEqual(result['data'][0]['url'], self.base + result['x_artifacts'][0]['url'])
        with self.request(result['x_artifacts'][0]['url']) as response:
            self.assertIn(b'HANDOFF TEST FIXTURE', response.read())
        self.assertEqual(self.json(result['result_url'])['data'], result['data'])
        self.assertTrue(self.json('/api/health')['ok'])
        self.assertEqual(self.json('/api/image/jobs')['data'][0]['id'], result['x_job_id'])
        chat = self.json('/api/chat/jobs', {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Is it ready?'}], 'max_tokens': 64})
        self.assertFalse(any(call['action'] != 'image' for call in self.workers.calls))
        release.set()
        self.assertEqual(wait_job(self.service, result['x_job_id'])['status'], 'completed')
        self.assertEqual(wait_job(self.service, chat['id'])['status'], 'completed')
        self.assertEqual(set(self.service.scheduler.resident), {'qwen-chat'})
        self.assertEqual([call['action'] for call in self.workers.calls], ['image', 'text'])

    def test_restoration_error_keeps_image_downloadable_with_warning(self):
        entered, release = self.restore_gate(fail=True)
        result = self.post(path='/v1/images/generations')
        self.assertTrue(entered.wait(2))
        release.set()
        self.assertEqual(wait_job(self.service, result['x_job_id'])['status'], 'failed')
        recovered = self.json(result['result_url'])
        self.assertEqual(recovered['data'], result['data'])
        self.assertIn('Restore fixture failed', recovered['x_warning'])
        self.assertTrue(recovered['x_result_ready'])
        self.assertEqual(self.post()['id'], result['x_job_id'])
        self.assertEqual(len(self.workers.calls), 1)

    def test_concurrent_retries_only_generate_once_and_new_requests_remain_independent(self):
        release = threading.Event(); self.gates.append(release)
        def factory(*args):
            worker = self.workers(*args); run = worker.run
            def blocked(*params):
                if not release.wait(10):
                    raise TimeoutError('Test did not release generation')
                yield from run(*params)
            worker.run = blocked
            return worker
        self.service.scheduler.factory = factory
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.post(), range(6)))
        self.assertEqual(len({row['id'] for row in results}), 1)
        jid = results[0]['id']
        with self.request(results[0]['result_url']) as response:
            self.assertEqual(response.status, 202)
        release.set(); wait_job(self.service, jid)
        self.assertEqual(self.post()['id'], jid)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({**self.body, 'prompt': 'A different image'})
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(json.loads(error.exception.read())['x_job_id'], jid)
        self.assertEqual(len(self.workers.calls), 1)
        a = self.json('/api/image/jobs', self.body)
        b = self.json('/api/image/jobs', self.body)
        self.assertNotEqual(a['id'], b['id'])
        wait_job(self.service, a['id']); wait_job(self.service, b['id'])
        self.assertEqual(len(self.workers.calls), 3)

    def test_completed_retry_survives_restart_and_model_removal(self):
        key = '123456789abcdef0123456789abcdef0'
        original = self.post(key=key); wait_job(self.service, original['id'])
        self.assertEqual(original['id'], key)
        config = copy.deepcopy(self.service.config)
        self.service.close()
        self.service = Service(config)
        self.service.config['models'] = []
        self.server.RequestHandlerClass = handler(self.service)
        replay = self.post(key=key)
        self.assertEqual(replay['id'], original['id'])
        self.assertEqual(self.json(replay['result_url'])['data'][0]['url'], self.base + replay['x_artifacts'][0]['url'])
        self.assertFalse(self.service.running)

    def test_known_url_before_submission_pending_file_and_same_key_recovery(self):
        key = '0123456789abcdef0123456789abcdef'
        path = '/artifacts/' + key + '.png'
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(path)
        self.assertEqual(error.exception.code, 404)
        entered, release = threading.Event(), threading.Event(); self.gates.append(release)
        def factory(*args):
            worker = self.workers(*args); run = worker.run
            def writing(payload, timeout, cancel):
                Path(payload['output_path']).write_text('PARTIAL IMAGE')
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('Test did not release image writing')
                yield from run(payload, timeout, cancel)
            worker.run = writing
            return worker
        self.service.scheduler.factory = factory
        receipt = self.post(key=key, path='/v1/images/generations', headers={'Prefer': 'respond-async'})
        self.assertTrue(entered.wait(2))
        self.assertEqual(receipt['id'], key)
        self.assertEqual(receipt['artifact_url'], self.base + path)
        self.assertEqual(receipt['artifacts_url'], self.base + '/api/artifacts')
        with self.request(path) as response:
            self.assertEqual(response.status, 202)
            self.assertEqual(response.headers['Retry-After'], '1')
            self.assertFalse(json.load(response)['x_result_ready'])
        self.assertEqual(self.json('/api/artifacts')['data'], [])
        self.assertEqual(self.post(key=key)['id'], key)
        release.set(); wait_job(self.service, key)
        with self.request(path) as response:
            self.assertEqual(response.status, 200)
            self.assertIn('image/png', response.headers['Content-Type'])
            self.assertIn(b'HANDOFF TEST FIXTURE', response.read())
        result = self.post(key=key, path='/v1/images/generations')
        self.assertEqual(result['data'][0]['url'], self.base + path)
        self.assertEqual(self.json('/api/artifacts')['data'][0]['url'], self.base + path)
        self.assertEqual(len(self.workers.calls), 1)

    def test_chosen_request_id_cannot_overwrite_jobs_or_existing_files(self):
        existing = self.service.control('qwen-chat', 'load')
        original = wait_job(self.service, existing['id'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post(key=existing['id'])
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(self.service.store.job(existing['id']), original)
        self.assertIsNone(self.service.store.image_request(image_identity(existing['id'], self.body, False)))
        orphan_id = 'a' * 32
        orphan = self.service.store.directory / 'artifacts' / (orphan_id + '.png')
        orphan.write_text('Existing saved image')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post(key=orphan_id)
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(orphan.read_text(), 'Existing saved image')
        self.assertFalse(self.workers.calls)

    def test_artifact_catalog_discovers_old_files_without_a_job_and_authenticates(self):
        from PIL import Image
        result = self.post(); wait_job(self.service, result['id'])
        orphan = self.service.store.directory / 'artifacts' / ('b' * 32 + '.png')
        Image.new('RGB', (32, 32), 'red').save(orphan)
        (orphan.parent / 'private.txt').write_text('Not an image')
        self.service.config.update(api_key='recovery-test-key-123456', browser_access='api-key')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.json('/api/artifacts')
        self.assertEqual(error.exception.code, 401)
        headers = {'Authorization': 'Bearer recovery-test-key-123456', 'Origin': 'https://frontend.example'}
        with self.request('/api/artifacts', headers=headers) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], headers['Origin'])
            rows = json.load(response)['data']
        self.assertEqual(len(rows), 2)
        old = next(row for row in rows if row['name'] == orphan.name)
        self.assertEqual(old['url'], self.base + '/artifacts/' + orphan.name)
        self.assertIsNone(old['job_id'])
        with self.request('/artifacts/' + orphan.name, headers=headers) as response:
            self.assertEqual(response.read(), orphan.read_bytes())
        self.assertEqual(self.json('/api/health')['links']['artifacts'], '/api/artifacts')
        self.assertEqual(len(self.workers.calls), 1)

    def test_dropped_receipt_recovered_by_same_key(self):
        original_reply = self.server.RequestHandlerClass.reply
        def disconnect(http, data, status=200):
            if status == 202:
                raise BrokenPipeError('Client lost its acknowledgement')
            return original_reply(http, data, status)
        with patch.object(self.server.RequestHandlerClass, 'reply', disconnect):
            with self.assertRaises(OSError):
                self.post()
        replay = self.post()
        self.assertEqual(wait_job(self.service, replay['id'])['status'], 'completed')
        self.assertEqual(len(self.service.store.image_jobs()), 1)
        self.assertEqual(len(self.workers.calls), 1)

    def test_restart_marks_old_unfinished_image_interrupted_and_preserves_result(self):
        result = self.post(); job = wait_job(self.service, result['id'])
        job.update(status='running', phase='loading qwen-chat')
        self.service.store.save_job(job)
        for i in range(205):
            later = self.service._new_job(None, 'qwen-chat', str(i), {})
            later.update(kind='chat', status='completed'); self.service.store.save_job(later)
        config = copy.deepcopy(self.service.config)
        self.service.close()
        self.service = Service(config)
        self.server.RequestHandlerClass = handler(self.service)
        replay = self.post()
        self.assertEqual(replay['id'], result['id'])
        self.assertEqual(replay['status'], 'interrupted')
        recovered = self.json(replay['result_url'])
        self.assertTrue(recovered['x_result_ready'])
        self.assertIn('Server restarted', recovered['x_warning'])
        self.assertFalse(self.service.scheduler.resident)

    def test_invalid_request_does_not_reserve_key_but_failed_job_does(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({**self.body, 'n': 2})
        self.assertEqual(error.exception.code, 400)
        def failed(*unused):
            raise RuntimeError('Image load fixture failed')
        self.service.scheduler.factory = failed
        result = self.post(); wait_job(self.service, result['id'])
        replay = self.post()
        self.assertEqual(replay['id'], result['id'])
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('Image load fixture failed', replay['x_warning'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/artifacts/' + result['id'] + '.png')
        self.assertEqual(error.exception.code, 500)
        self.assertEqual(json.loads(error.exception.read())['x_job_id'], result['id'])
        self.assertEqual(len(self.service.store.image_jobs()), 1)
        for key in ('', 'has spaces', 'a' * 129):
            with self.subTest(key=key), self.assertRaises(urllib.error.HTTPError) as error:
                self.post(key=key)
            self.assertEqual(error.exception.code, 400)

    def test_listing_finds_images_behind_chat_history_and_requires_auth(self):
        result = self.post(); wait_job(self.service, result['id'])
        # A migrated older image job without kind also remains discoverable.
        old = self.service.store.job(result['id']); old.pop('kind'); self.service.store.save_job(old)
        for i in range(205):
            job = self.service._new_job(None, 'qwen-chat', str(i), {})
            job.update(kind='chat', status='completed'); self.service.store.save_job(job)
        self.service.config.update(api_key='recovery-test-key-123456', browser_access='api-key')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.json('/api/image/jobs')
        self.assertEqual(error.exception.code, 401)
        headers = {'Authorization': 'Bearer recovery-test-key-123456', 'Origin': 'https://frontend.example'}
        with self.request('/api/image/jobs?model=qwen-image', headers=headers) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], headers['Origin'])
            listing = json.load(response)['data']
        self.assertEqual([row['id'] for row in listing], [result['id']])
        self.assertTrue(listing[0]['x_result_ready'])
        self.assertEqual(listing[0]['data'][0]['url'], self.base + listing[0]['x_artifacts'][0]['url'])
        self.assertNotIn('b64_json', listing[0]['data'][0])
        with self.request(listing[0]['x_artifacts'][0]['url'], headers=headers) as response:
            self.assertIn(b'HANDOFF TEST FIXTURE', response.read())
        with self.request('/api/image/jobs?model=qwen-chat', headers=headers) as response:
            self.assertEqual(json.load(response)['data'], [])
        self.assertEqual(len(self.workers.calls), 1)

    def test_explicit_base64_keeps_downloadable_url_and_artifact_authentication(self):
        self.service.config['api_key'] = 'recovery-test-key-123456'
        headers = {'Authorization': 'Bearer recovery-test-key-123456'}
        result = self.post({**self.body, 'response_format': 'b64_json'},
                           path='/v1/images/generations', headers=headers)
        item = result['data'][0]
        self.assertEqual(item['url'], self.base + result['x_artifacts'][0]['url'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(result['x_artifacts'][0]['url'])
        self.assertEqual(error.exception.code, 401)
        with self.request(result['x_artifacts'][0]['url'], headers=headers) as response:
            self.assertEqual(response.read(), base64.b64decode(item['b64_json']))
        with self.request(result['result_url'], headers=headers) as response:
            self.assertEqual(json.load(response)['data'], result['data'])
        self.assertEqual(len(self.workers.calls), 1)

    def test_multipart_edit_retries_hash_parsed_image_bytes(self):
        test_image_edits.ImageEditHTTPTests.fixture(self)
        headers = {'Idempotency-Key': 'edited-image'}
        receipts = []
        for _ in range(2):
            with test_image_edits.ImageEditHTTPTests.upload(self, path='/api/image/edits', headers=headers) as response:
                receipts.append(json.load(response))
        self.assertEqual(receipts[0]['id'], receipts[1]['id'])
        wait_job(self.service, receipts[0]['id'])
        self.assertEqual(len(self.workers.calls), 1)
        with self.assertRaises(urllib.error.HTTPError) as error:
            test_image_edits.ImageEditHTTPTests.upload(self, path='/api/image/edits', headers=headers, files=[('image', b'different bytes')])
        self.assertEqual(error.exception.code, 409)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post(key='edited-image')
        self.assertEqual(error.exception.code, 409)

    def test_cli_recovers_lost_receipt_and_downloads_authenticated_saved_url(self):
        self.service.config['api_key'] = 'recovery-test-key-123456'
        helper = Path(__file__).resolve().parents[1] / 'tools/image_job.py'
        state = Path(self.temp.name) / 'client-receipt.json'
        request = Path(self.temp.name) / 'request.json'
        request.write_text(json.dumps({**self.body, 'response_format': 'url'}))
        output = Path(self.temp.name) / 'recovered.png'
        env = {**os.environ, 'LAZYRIG_API_KEY': self.service.config['api_key']}
        def cli(*args, success=True):
            process = subprocess.run([sys.executable, str(helper), '--url', self.base, *map(str, args)],
                                     capture_output=True, text=True, env=env, timeout=15)
            self.assertEqual(process.returncode, 0 if success else 1, process.stderr)
            return json.loads(process.stdout if success else process.stderr)
        receipt = cli('submit', '--request', request, '--state', state)
        wait_job(self.service, receipt['job_id'])
        saved = json.loads(state.read_text()); saved.pop('job_id'); state.write_text(json.dumps(saved))
        self.assertNotIn(env['LAZYRIG_API_KEY'], state.read_text())
        recovered = cli('submit', '--state', state)
        self.assertEqual(recovered['job_id'], receipt['job_id'])
        downloaded = cli('status', '--state', state, '--output', output)
        self.assertEqual(downloaded['image_status'], 'completed')
        self.assertIn(b'HANDOFF TEST FIXTURE', output.read_bytes())
        self.assertEqual(cli('list')['jobs'][0]['job_id'], receipt['job_id'])
        self.assertEqual(cli('status', '--job', receipt['job_id'], '--output', output)['sha256'], downloaded['sha256'])
        request.write_text(json.dumps({**self.body, 'prompt': 'Changed request'}))
        self.assertIn('different request', cli('submit', '--state', state, '--request', request, success=False)['error'])
        self.assertEqual(len(self.workers.calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
