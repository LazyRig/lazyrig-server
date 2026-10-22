"""Real socket tests for transport bounds and the production-control surface."""
import copy
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import ssl
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from support import DEFAULT, Service, fixture_png
from lazyrig.admission import AdmissionError
from lazyrig.config import validate
from lazyrig.deadlines import DeadlineEvent, RequestDeadline
from lazyrig.http_transport import BoundedHTTPServer
from lazyrig.server import handler, tls_context


class HTTPHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=self.temp.name, http_timeout_seconds=.25)
        self.service = Service(config)
        self.server = None

    def start(self, *, cap=8, tls=None):
        self.server = BoundedHTTPServer(('127.0.0.1', 0), handler(self.service),
                                       max_connections=cap, http_timeout=.25, tls=tls)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.close_connections()
            self.server.server_close()
            self.thread.join(2)
        self.service.close()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None, *, tls=False):
        if tls:
            connection = http.client.HTTPSConnection('127.0.0.1', self.server.server_port, timeout=3,
                                                     context=ssl._create_unverified_context())
        else:
            connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            connection.request('POST' if body is not None else 'GET', path,
                               body=json.dumps(body) if body is not None else None,
                               headers={'Content-Type': 'application/json', **(headers or {})})
            result = connection.getresponse()
            payload = result.read()
            return result.status, dict(result.headers), json.loads(payload) if payload else None
        finally:
            connection.close()

    def wait_connections(self, count):
        for _ in range(100):
            with self.server._connection_lock:
                if len(self.server._connections) == count:
                    return
            time.sleep(.01)
        self.fail('Transport did not reach the expected active connection count')

    def test_partial_body_times_out_while_other_requests_respond(self):
        self.start()
        with socket.create_connection(self.server.server_address, timeout=2) as slow:
            slow.sendall(b'POST /api/sessions HTTP/1.1\r\nHost: localhost\r\n'
                         b'Content-Type: application/json\r\nContent-Length: 200\r\n\r\n{')
            self.assertEqual(self.request('/api/health')[0], 200)
            reply = slow.recv(8192)
            self.assertIn(b'408', reply)
        for _ in range(100):
            if self.service.runtime_requests == 0:
                break
            time.sleep(.01)
        self.assertEqual(self.service.runtime_requests, 0)
        self.assertEqual(self.service.store.sessions(), [])

    def test_unused_connection_expires_silently(self):
        self.start()
        with patch('lazyrig.server.BaseHTTPRequestHandler.log_message') as logged, \
                patch.object(self.server, 'handle_error') as handler_error:
            with socket.create_connection(self.server.server_address, timeout=2) as idle:
                self.wait_connections(1)
                self.assertEqual(idle.recv(1), b'')
            self.wait_connections(0)
            handler_error.assert_not_called()
            logged.assert_not_called()
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_keep_alive_expiry_is_silent_after_completed_request(self):
        self.start()
        with patch('lazyrig.server.BaseHTTPRequestHandler.log_message') as logged, \
                patch.object(self.server, 'handle_error') as handler_error:
            connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
            try:
                connection.request('GET', '/api/health')
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())['ok'])
                self.assertEqual(connection.sock.recv(1), b'')
                self.wait_connections(0)
            finally:
                connection.close()
            handler_error.assert_not_called()
            logged.assert_not_called()
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_incomplete_request_line_is_logged_including_buffered_next_request(self):
        self.start()
        for preceding in (b'', b'GET /api/health HTTP/1.1\r\nHost: localhost\r\n\r\n'):
            with self.subTest(keep_alive=bool(preceding)), \
                    patch('lazyrig.server.BaseHTTPRequestHandler.log_message') as logged, \
                    patch.object(self.server, 'handle_error') as handler_error:
                with socket.create_connection(self.server.server_address, timeout=2) as slow:
                    slow.sendall(preceding + b'GET /api/hea')
                    response = b''
                    while data := slow.recv(8192):
                        response += data
                self.wait_connections(0)
                if preceding:
                    self.assertIn(b'200 OK', response)
                handler_error.assert_not_called()
                messages = [call.args[0] % call.args[1:] for call in logged.call_args_list]
                self.assertEqual(messages, [
                    'Incomplete HTTP request: receive deadline exceeded; connection closed'])
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_header_deadline_is_absolute_even_when_client_trickles_bytes(self):
        self.start()
        with patch('lazyrig.server.BaseHTTPRequestHandler.log_message') as logged, \
                socket.create_connection(self.server.server_address, timeout=2) as slow:
            slow.sendall(b'GET /api/health HTTP/1.1\r\nHost: localhost\r\nX-Slow: ')
            for _ in range(10):
                try:
                    slow.sendall(b'a')
                except OSError:
                    break
                time.sleep(.05)
            self.wait_connections(0)
            messages = [call.args[0] % call.args[1:] for call in logged.call_args_list]
            self.assertEqual(messages, [
                'Incomplete HTTP request: receive deadline exceeded; connection closed'])
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_connection_cap_is_retryable_and_releases_after_disconnect(self):
        self.start(cap=1)
        with socket.create_connection(self.server.server_address, timeout=2) as slow:
            slow.sendall(b'GET /api/health HTTP/1.1\r\n')
            self.wait_connections(1)
            status, headers, _ = self.request('/api/health')
            self.assertEqual(status, 503)
            self.assertEqual(headers['Retry-After'], '1')
            self.assertEqual(headers['X-Should-Retry'], 'true')
        self.wait_connections(0)
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_inference_compute_can_outlast_http_receive_timeout(self):
        self.start()
        self.service.config['http_timeout_seconds'] = .1
        start = time.monotonic()
        status, _, result = self.request('/v1/chat/completions',
                                         {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}]})
        self.assertEqual(status, 200, result)
        self.assertGreater(time.monotonic() - start, .1)
        self.assertTrue(result['choices'][0]['message']['content'])

    def test_readiness_is_separate_from_public_liveness(self):
        self.start()
        self.assertEqual(self.request('/api/ready')[0], 200)
        self.assertEqual(self.request('/api/ready?model=unknown')[0], 503)
        self.service.detected['available'] = False
        status, headers, result = self.request('/api/ready')
        self.assertEqual(status, 503)
        self.assertEqual(headers['Retry-After'], '1')
        self.assertIn('runtime_unavailable', result['reasons'])
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_typed_overload_and_deadline_errors_have_retry_semantics(self):
        self.start()
        with patch('lazyrig.compat.serve_chat', side_effect=AdmissionError('Busy')):
            status, headers, result = self.request('/v1/chat/completions', {})
        self.assertEqual(status, 503)
        self.assertEqual(result['error']['code'], 'queue_full')
        self.assertEqual(headers['Retry-After'], '1')
        self.assertEqual(headers['X-Should-Retry'], 'true')
        with patch('lazyrig.compat.serve_chat', side_effect=RequestDeadline()):
            status, headers, result = self.request('/v1/chat/completions', {})
        self.assertEqual(status, 504)
        self.assertEqual(headers['X-Should-Retry'], 'false')
        self.assertEqual(result['error']['code'], 'request_timeout')

    def test_image_discovery_uses_terminal_fallback_after_final_storage_failure(self):
        self.start()
        job = self.service._new_job(None, 'qwen-image', 'Saved before storage failure', {'width': 256, 'steps': 1})
        job.update(kind='image', status='running', phase='Saving result')
        self.service.store.save_job(job)
        filename = job['id'] + '.png'
        fixture_png(self.service.store.directory / 'artifacts' / filename)
        job['artifacts'] = [{'url': '/artifacts/' + filename, 'name': filename, 'prompt': job['prompt'], 'simulated': False}]
        job['result_ready'] = True
        orphan = 'f' * 32 + '.png'
        fixture_png(self.service.store.directory / 'artifacts' / orphan)
        with patch.object(self.service.store, 'save_job', side_effect=OSError('simulated disk full')):
            with self.assertLogs('lazyrig.service', level='ERROR'):
                self.service._finish_job(job, DeadlineEvent(timeout=60), Mock())
            status = self.request('/api/jobs/' + job['id'])[2]
            result = self.request('/api/jobs/' + job['id'] + '/result')[2]
            listed = next(item for item in self.request('/api/image/jobs')[2]['data'] if item['id'] == job['id'])
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(listed['status'], status['status'])
            self.assertFalse(status['durable'])
            self.assertFalse(result['x_durable'])
            self.assertFalse(listed['x_durable'])
            self.assertTrue(listed['x_result_ready'])
            self.assertEqual(listed['data'], result['data'])
            artifacts = self.request('/api/artifacts')[2]['data']
            recovered = next(item for item in artifacts if item['job_id'] == job['id'])
            self.assertEqual(recovered['job_status'], 'completed')
            self.assertFalse(recovered['x_durable'])
            self.assertTrue(recovered['x_storage_warning'])
            self.assertTrue(any(item['name'] == orphan and item['job_id'] is None for item in artifacts))

    def test_canonical_origin_generates_urls_without_trusting_forwarded_headers(self):
        self.start()
        self.service.config['public_base_url'] = 'https://rig.example'
        status, _, receipt = self.request('/api/image/jobs', {'model': 'qwen-image', 'prompt': 'Test'},
                                          {'Host': 'rig.example', 'Origin': 'https://rig.example', 'X-Forwarded-Host': 'evil.example'})
        self.assertEqual(status, 202, receipt)
        self.assertTrue(receipt['result_url'].startswith('/api/jobs/'))
        self.assertTrue(receipt['artifact_url'].startswith('https://rig.example/'))
        self.assertEqual(self.request('/api/health', headers={'Host': 'evil.example'})[0], 400)

    def test_inference_credential_cannot_use_admin_routes(self):
        self.start()
        self.service.config.update(api_key='owner-key-123456789', inference_api_key='inference-key-123456789')
        client = {'Authorization': 'Bearer inference-key-123456789'}
        self.assertEqual(self.request('/v1/models', headers=client)[0], 200)
        self.assertEqual(self.request('/api/ready', headers=client)[0], 200)
        for path in ('/api/state', '/api/folders', '/api/runtime/update', '/api/downloads'):
            self.assertEqual(self.request(path, headers=client)[0], 403, path)
        for path in ('/api/server', '/api/models', '/api/settings', '/api/downloads', '/api/runtime/update'):
            self.assertEqual(self.request(path, {}, headers=client)[0], 403, path)
        self.assertEqual(self.request('/api/state', headers={'Authorization': 'Bearer owner-key-123456789'})[0], 200)
        self.assertEqual(self.request('/api/ready')[0], 401)
        self.assertEqual(self.request('/api/health')[0], 200)

    def test_ambiguous_body_framing_is_rejected_and_connection_closed(self):
        self.start()
        with socket.create_connection(self.server.server_address, timeout=2) as connection:
            connection.sendall(b'POST /api/sessions HTTP/1.1\r\nHost: localhost\r\n'
                               b'Content-Type: application/json\r\nContent-Length: 2\r\n'
                               b'Transfer-Encoding: chunked\r\n\r\n{}')
            reply = connection.recv(8192)
            self.assertIn(b'400', reply)
            self.assertIn(b'Connection: close', reply)

    @unittest.skipUnless(shutil.which('openssl'), 'TLS fixture requires openssl')
    def test_stalled_tls_handshake_does_not_block_other_clients(self):
        cert, key = Path(self.temp.name) / 'cert.pem', Path(self.temp.name) / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                        '-subj', '/CN=localhost', '-keyout', str(key), '-out', str(cert)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
        tls = tls_context({'tls_cert': str(cert), 'tls_key': str(key)})
        self.start(tls=tls)
        with socket.create_connection(self.server.server_address, timeout=2):
            self.assertEqual(self.request('/api/health', tls=True)[0], 200)
        self.wait_connections(0)

    @unittest.skipIf(os.name == 'nt', 'POSIX SIGTERM lifecycle; Windows console close remains a native acceptance test')
    def test_sigterm_cancels_accepted_job_and_exits_with_durable_terminal_state(self):
        directory = Path(self.temp.name)
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = str(directory / 'launch-data')
        config_path, ready_path = directory / 'launch.json', directory / 'port.txt'
        config_path.write_text(json.dumps(config))
        script = '''import os
from pathlib import Path
from support import Service
from lazyrig import server
server.Service = Service
original = server.BoundedHTTPServer
def create(address, handler, **kwargs):
    instance = original((address[0], 0), handler, **kwargs)
    Path(os.environ['LAZYRIG_TEST_PORT_FILE']).write_text(str(instance.server_port))
    return instance
server.BoundedHTTPServer = create
server.run_server()
'''
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root / 'tests'), str(root)]),
               'LAZYRIG_TEST_PORT_FILE': str(ready_path)}
        log_path = directory / 'launch.log'
        with log_path.open('w') as log:
            process = subprocess.Popen([sys.executable, '-c', script, '--config', str(config_path)],
                                       cwd=root, env=env, stdout=log, stderr=log)
            try:
                port = None
                for _ in range(200):
                    if ready_path.exists():
                        port = int(ready_path.read_text())
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=.2)
                        try:
                            connection.request('GET', '/api/health')
                            if connection.getresponse().status == 200:
                                break
                        except OSError:
                            pass
                        finally:
                            connection.close()
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    time.sleep(.025)
                else:
                    self.fail('Fixture server did not become healthy')
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                try:
                    connection.request('POST', '/api/image/jobs',
                                       body=json.dumps({'model': 'qwen-image', 'prompt': 'Shutdown test', 'steps': 100}),
                                       headers={'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    receipt = json.loads(response.read())
                    self.assertEqual(response.status, 202, receipt)
                finally:
                    connection.close()
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=10), 0, log_path.read_text())
                with sqlite3.connect(Path(config['data_dir']) / 'state.sqlite3') as db:
                    job = json.loads(db.execute('SELECT body FROM jobs WHERE id=?', (receipt['id'],)).fetchone()[0])
                self.assertEqual(job['status'], 'cancelled', job)
                self.assertNotIn('Traceback', log_path.read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)


class HTTPConfigurationTests(unittest.TestCase):
    def test_optional_scoped_key_requires_a_separate_owner_and_canonical_origin(self):
        config = copy.deepcopy(DEFAULT)
        config.update(inference_api_key='client-1234567890')
        with self.assertRaises(ValueError):
            validate(config)
        config.update(api_key='owner-12345678901', public_base_url='https://Rig.Example:443/')
        self.assertEqual(validate(config)['public_base_url'], 'https://rig.example')
        for change in ({'public_base_url': 'https://rig.example/path'}, {'max_connections': 0},
                       {'http_timeout_seconds': float('nan')}, {'max_jobs': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate({**config, **change})


if __name__ == '__main__':
    unittest.main()
