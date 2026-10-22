"""Real HTTP/TLS transport checks; any model responses use labelled fixtures."""
import copy
import json
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from http.server import ThreadingHTTPServer
import test_http
from test_handoff import FixtureWorkers
from support import DEFAULT
from lazyrig.config import normalize_origin, validate
from lazyrig.server import handler, tls_context
from support import Service


class ConnectionTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def preflight(self, origin, **headers):
        return urllib.request.urlopen(urllib.request.Request(self.base+'/v1/chat/completions', method='OPTIONS',
            headers={'Origin': origin, 'Access-Control-Request-Method': 'POST',
                     'Access-Control-Request-Headers': 'Content-Type, Authorization, X-Stainless-Lang, X-Client-Name, Last-Event-ID',
                     **headers}), timeout=5)

    def test_preflight_sdk_headers_without_auth_and_private_network(self):
        key = 'a-test-key-over-16-characters'
        self.json('/api/server', {'allowed_origins': ['https://FRONTEND.example:443/chat/'], 'api_key': key})
        self.assertEqual(self.service.config['allowed_origins'], ['https://frontend.example'])
        with self.preflight('https://frontend.example', **{'Access-Control-Request-Private-Network': 'true'}) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://frontend.example')
            self.assertEqual(response.headers['Access-Control-Allow-Credentials'], 'true')
            self.assertEqual(response.headers['Access-Control-Allow-Private-Network'], 'true')
            self.assertIn('x-stainless-lang', response.headers['Access-Control-Allow-Headers'])
            self.assertIn('last-event-id', response.headers['Access-Control-Allow-Headers'])
        # Authentication errors remain readable by an allowed browser origin.
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.request('/v1/models', headers={'Origin': 'https://frontend.example'})
        self.assertEqual(err.exception.code, 401)
        self.assertEqual(err.exception.headers['Access-Control-Allow-Origin'], 'https://frontend.example')
        with self.request('/v1/models', headers={'Origin': 'https://frontend.example', 'Authorization': 'Bearer '+key}) as response:
            self.assertEqual(response.status, 200)

    def test_unlisted_origin_rejected_and_visible_in_admin_diagnostics(self):
        for origin in ('https://unlisted.example', 'null'):
            with self.assertRaises(urllib.error.HTTPError) as err:
                self.preflight(origin)
            self.assertEqual(err.exception.code, 400)
            self.assertIsNone(err.exception.headers.get('Access-Control-Allow-Origin'))
            self.assertIsNone(err.exception.headers.get('Access-Control-Allow-Private-Network'))
        errors = self.json('/api/state')['server']['connection_errors']
        self.assertEqual([e['origin'] for e in errors], ['https://unlisted.example', 'null'])

    def test_cors_applies_to_stream_and_saved_artifact(self):
        self.json('/api/server', {'allowed_origins': ['http://localhost:5173']})
        self.service.scheduler.factory = FixtureWorkers([('Hello.', 4)])
        headers = {'Origin': 'http://localhost:5173'}
        with self.request('/v1/chat/completions', {'model': 'qwen-chat', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True}, headers) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], headers['Origin'])
            self.assertEqual(response.headers['Access-Control-Expose-Headers'], 'X-Job-Id, Location, Retry-After, X-Should-Retry')
            self.assertIn('Hello.', response.read().decode())
        path = self.service.store.directory/'artifacts'/('a'*32+'.png')
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        with self.request('/artifacts/'+path.name, headers=headers) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], headers['Origin'])

    def test_origin_normalization_and_invalid_values(self):
        self.assertEqual(normalize_origin(' https://Example.com:443/app?x=1 '), 'https://example.com')
        self.assertEqual(normalize_origin('http://[::1]:5173'), 'http://[::1]:5173')
        for origin in ('*', 'null', 'file:///page.html', 'https://user:pass@host', 'https://*.example', 'https://', 'http://host:wrong', 'https://bad\nname'):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                validate({**copy.deepcopy(DEFAULT), 'allowed_origins': [origin]})
        with self.assertRaises(ValueError):
            validate({**copy.deepcopy(DEFAULT), 'tls_cert': 'one-file-only.pem'})

    def test_devtools_discovery_is_quiet_without_bypassing_api_auth(self):
        key = 'a-test-key-over-16-characters'
        self.json('/api/server', {'api_key': key})
        with patch.object(BaseHTTPRequestHandler, 'log_message') as log:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/.well-known/appspecific/com.chrome.devtools.json')
            self.assertEqual(error.exception.code, 404)
            self.assertEqual(json.load(error.exception), {'error': 'Not found'})
            log.assert_not_called()
            for path in ('/api/state', '/.well-known/appspecific/other.json'):
                with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                    self.request(path)
                self.assertEqual(error.exception.code, 401)
            self.assertEqual(log.call_count, 2)
        with self.request('/api/state', headers={'Authorization': 'Bearer '+key}) as response:
            self.assertEqual(response.status, 200)


@unittest.skipUnless(shutil.which('openssl'), 'OpenSSL executable required to create an ephemeral test certificate')
class TLSConnectionTests(unittest.TestCase):
    tearDown = test_http.HTTPTests.tearDown

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        cert, key = [str(Path(self.temp.name)/name) for name in ('cert.pem', 'key.pem')]
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key,
                        '-out', cert, '-days', '1', '-subj', '/CN=localhost',
                        '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1'], check=True, capture_output=True)
        config = {**copy.deepcopy(DEFAULT), 'data_dir': self.temp.name, 'tls_cert': cert, 'tls_key': key,
                  'allowed_origins': ['https://frontend.example']}
        self.service = Service(config)
        self.service.scheduler.factory = FixtureWorkers([])
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.service))
        self.server.socket = tls_context(config).wrap_socket(self.server.socket, server_side=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'https://127.0.0.1:'+str(self.server.server_port)
        self.client = ssl.create_default_context(cafile=cert)

    def test_trusted_https_same_origin_and_correct_image_url_scheme(self):
        with urllib.request.urlopen(urllib.request.Request(self.base+'/api/health', headers={'Origin': self.base}), context=self.client) as response:
            self.assertTrue(json.load(response)['ok'])
        request = urllib.request.Request(self.base+'/v1/images/generations', data=json.dumps({
            'model': 'qwen-image', 'prompt': 'transport test', 'steps': 1, 'response_format': 'url'}).encode(),
            headers={'Content-Type': 'application/json', 'Origin': 'https://frontend.example'})
        with urllib.request.urlopen(request, context=self.client) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://frontend.example')
            result = json.load(response)
        self.assertTrue(result['data'][0]['url'].startswith(self.base+'/artifacts/'))
        with urllib.request.urlopen(result['data'][0]['url'], context=self.client) as response:
            self.assertIn(b'HANDOFF TEST FIXTURE', response.read())


if __name__ == '__main__':
    unittest.main(verbosity=2)
