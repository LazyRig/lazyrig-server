"""Runtime lifecycle tests; no real package is installed or downloaded."""
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import venv

from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from lazyrig import runtime_update as update
import test_http

CURRENT = {'openvino': '2026.4.0', 'openvino-genai': '2026.4.0.0',
           'openvino-tokenizers': '2026.4.0.0'}
TAG = Tag('cp313', 'cp313', 'win_amd64')


def release(version, **overrides):
    return [{'filename': f'openvino_genai-{version}-cp313-cp313-win_amd64.whl',
             'packagetype': 'bdist_wheel', 'requires_python': '>=3.12', 'yanked': False,
             **overrides}]


def ready():
    return {'status': 'ready', 'installed': dict(CURRENT), 'target': '2026.4.1.0',
            'latest_release': '2026.4.1.0', 'available': True, 'can_update': True,
            'check_id': 'abc123', 'checked_at': time.time(), 'message': 'Update available'}


class LegacyEnvironmentTests(unittest.TestCase):
    """A real existing environment without packaging or pip must still start."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='lazyrig-legacy-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        shutil.copytree(update.ROOT/'lazyrig', cls.root/'lazyrig',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy(update.ROOT/'requirements-openvino.txt', cls.root/'requirements-openvino.txt')
        venv.EnvBuilder(with_pip=False).create(cls.root/'.venv')
        cls.python = cls.root/'.venv'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    def test_parser_works_without_installing_packaging_or_pip(self):
        code = '''import importlib.util, sys
assert importlib.util.find_spec('packaging') is None
assert importlib.util.find_spec('pip') is None
from lazyrig import runtime_update as update
assert not any(name.startswith(('packaging', 'pip.')) for name in sys.modules)
parser = update._packaging_api()
assert parser.Version('2026.4.1.0') > parser.Version('2026.4.0.0')
assert parser.Version('2026.5.0.dev1').is_prerelease
assert update.PathFinder.find_spec('pip') is None
assert importlib.util.find_spec('packaging') is None
assert parser.Version('2026.4.1.0') in update.runtime_requirement()
'''
        result = subprocess.run([str(self.python), '-c', code], cwd=self.root,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_managed_server_health_dashboard_and_update_check_without_packaging(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        code = '''import importlib.util
assert importlib.util.find_spec('packaging') is None
assert importlib.util.find_spec('pip') is None
from lazyrig import runtime_update as update
assert update.managed_environment()
update.fetch_releases = lambda: {'releases': {'2026.4.1.0': [
    {'filename': 'openvino_genai-2026.4.1.0-py3-none-any.whl', 'packagetype': 'bdist_wheel',
     'requires_python': '>=3.12', 'yanked': False}]}}
from lazyrig.server import main
main()
'''
        with (self.root/'server.log').open('w+') as log:
            process = subprocess.Popen([str(self.python), '-c', code, '--port', str(port),
                                        '--data-dir', str(self.root/'data')], cwd=self.root,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + 15
                while True:
                    try:
                        with urllib.request.urlopen(base+'/api/health', timeout=1) as response:
                            self.assertTrue(json.load(response)['ok'])
                        break
                    except OSError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0); self.fail(log.read())
                        time.sleep(.05)
                for route in ('/', '/api/state', '/api/runtime/update'):
                    with urllib.request.urlopen(base+route, timeout=3) as response:
                        self.assertEqual(response.status, 200)
                request = urllib.request.Request(base+'/api/runtime/check', data=b'{}',
                                                  headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 202)
                deadline = time.monotonic() + 5
                while True:
                    with urllib.request.urlopen(base+'/api/runtime/update', timeout=3) as response:
                        result = json.load(response)
                    if result['status'] != 'checking' or time.monotonic() > deadline:
                        break
                    time.sleep(.02)
                self.assertEqual(result['status'], 'ready', result)
                self.assertEqual(result['target'], '2026.4.1.0')
                # There is deliberately no OpenVINO in this upgrade fixture.
                self.assertFalse(result['can_update'])
                self.assertIn('runtime installation is incomplete', result['message'])
                self.assertIsNone(process.poll())
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)


class SelectionTests(unittest.TestCase):
    def test_stable_non_yanked_platform_python_and_application_range(self):
        data = {'releases': {
            '2026.4.0.0': release('2026.4.0.0'),
            '2026.4.1.0': release('2026.4.1.0'),
            '2026.4.2.0': release('2026.4.2.0', yanked=True),
            '2026.5.0.0.dev1': release('2026.5.0.0.dev1'),
            '2026.5.0.0rc1': release('2026.5.0.0rc1'),
            '2026.5.0.0': release('2026.5.0.0', requires_python='>=3.14'),
            '2026.6.0.0': release('2026.6.0.0', filename='openvino_genai-2026.6.0.0-cp313-cp313-manylinux_2_28_x86_64.whl'),
            '2027.0.0.0': release('2027.0.0.0'),
        }}
        self.assertEqual(update.select_release(data, SpecifierSet('>=2026.4,<2027'), {TAG}, '3.13.9'),
                         ('2027.0.0.0', '2026.4.1.0'))

    def check(self, current=None, target='2026.4.1.0', managed=True, custom=False):
        with patch.object(update, 'installed_versions', return_value=current or CURRENT), \
             patch.object(update, 'managed_environment', return_value=managed), \
             patch.object(update, 'custom_runtime', return_value=custom), \
             patch.object(update, 'runtime_requirement', return_value=SpecifierSet('>=2026.4,<2027')), \
             patch.object(update, 'fetch_releases', return_value={}), \
             patch.object(update, 'select_release', return_value=('2026.4.1.0', target)):
            return update.check_update()

    def test_new_stable_available(self):
        self.assertTrue(self.check()['can_update'])

    def test_existing_or_newer_version_never_downgraded(self):
        self.assertFalse(self.check(target='2026.4.0.0')['available'])
        self.assertIn('not be downgraded', self.check(target='2026.3.0.0')['message'])

    def test_nightly_custom_missing_and_unmanaged_environments_are_preserved(self):
        cases = [dict(current={**CURRENT, 'openvino': '2026.5.0.dev1'}),
                 dict(current={**CURRENT, 'openvino-genai': None}),
                 dict(managed=False), dict(custom=True), dict(target=None)]
        for values in cases:
            with self.subTest(values=values):
                self.assertFalse(self.check(**values)['can_update'])

    def test_pip_configuration_cannot_redirect_the_installer(self):
        with patch.dict(os.environ, {'PIP_INDEX_URL': 'https://unrelated.invalid', 'PIP_PRE': '1'}):
            env = update.clean_pip_environment()
        self.assertNotIn('PIP_INDEX_URL', env)
        self.assertNotIn('PIP_PRE', env)
        self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)

    def test_constraints_preserve_installed_packages_and_runtime_minimums(self):
        dist = SimpleNamespace(metadata={'Name': 'optimum-intel'}, version='1.0',
                               requires=['openvino<2026.5', 'openvino>=2030; python_version<"2"'])
        with patch.object(update.metadata, 'distributions', return_value=[dist]), \
             patch.object(update, 'installed_versions', return_value=CURRENT):
            value = update.constraints_text()
        self.assertIn('optimum-intel==1.0', value)
        self.assertIn('openvino<2026.5', value)
        self.assertIn('openvino>=2026.4.0', value)
        self.assertNotIn('2030', value)


class InstallTests(unittest.TestCase):
    def run_install(self, fail_at=None):
        calls = []
        def run(command, log, timeout=1800):
            calls.append(command)
            if fail_at == len(calls):
                raise subprocess.CalledProcessError(1, command)
        with patch.object(update, 'installed_versions', return_value=CURRENT), \
             patch.object(update, 'custom_runtime', return_value=False), \
             patch.object(update, 'constraints_text', return_value=''), \
             patch.object(update, 'run_command', side_effect=run):
            result = update.install_runtime('2026.4.1.0', io.StringIO())
        return result, calls

    def test_stage_both_versions_then_install_and_verify(self):
        result, calls = self.run_install()
        self.assertEqual(result[0], 'succeeded')
        self.assertIn('download', calls[0])
        self.assertIn('download', calls[1])
        self.assertIn('install', calls[2])
        self.assertIn('--no-index', calls[2])
        self.assertIn('check', calls[3])
        self.assertIn('lazyrig.probe', calls[4])

    def test_install_and_verification_failure_restore_previous_runtime(self):
        for failure in (3, 4, 5):
            with self.subTest(failure=failure):
                result, calls = self.run_install(failure)
                self.assertEqual(result[0], 'rolled_back')
                restoration = next(call for call in calls if '--force-reinstall' in call)
                self.assertIn('openvino-genai==2026.4.0.0', restoration)
                self.assertIn('--no-deps', restoration)
                self.assertIn('check', calls[-1])

    def test_download_failure_does_not_attempt_installation(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_install(1)

    def test_environment_lock_blocks_another_process_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            script = ('import sys; sys.prefix=sys.argv[1]; '
                      'from lazyrig.runtime_update import environment_lock; '
                      'lock=environment_lock(exclusive=True); lock.__enter__(); lock.__exit__(None,None,None)')
            command = [sys.executable, '-c', script, directory]
            with patch.object(sys, 'prefix', directory), update.environment_lock():
                busy = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(busy.returncode, 0)
            self.assertIn('runtime is in use', busy.stderr)
            free = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(free.returncode, 0, free.stderr)

    def test_helper_waits_for_parent_then_restarts_saved_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = subprocess.Popen([sys.executable, '-c', 'import sys;sys.stdin.read()'], stdin=subprocess.PIPE)
            restart = [sys.executable, '-c', 'from pathlib import Path;Path("restarted").touch()']
            update.write_json(root/'plan.json', {'id': 'process-check', 'parent_pid': parent.pid,
                                                'target': '2026.4.1.0', 'cwd': directory, 'restart': restart})
            script = '''import sys
from pathlib import Path
from lazyrig import runtime_update as update
root = Path(sys.argv[1]); sys.prefix = str(root)
update.update_paths = lambda: (root/'status.json', root/'update.log')
def simulated_install(target, log):
    (root/'installed').write_text(target)
    return 'succeeded', 'Simulated runtime installation'
update.install_runtime = simulated_install
print('READY', flush=True)
raise SystemExit(update.resume_update(root/'plan.json'))
'''
            helper = subprocess.Popen([sys.executable, '-c', script, directory], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(helper.stdout.readline().strip(), 'READY')
                self.assertFalse((root/'installed').exists())
                parent.stdin.close(); parent.wait(timeout=5)
                output, error = helper.communicate(timeout=10)
                self.assertEqual(helper.returncode, 0, error)
                self.assertTrue((root/'installed').exists())
                self.assertTrue((root/'restarted').exists())
                self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'succeeded')
            finally:
                for process in (parent, helper):
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=5)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = SimpleNamespace(lock=threading.RLock(), running={}, runtime_requests=0,
                                       downloads=SimpleNamespace(running={}), runtime_maintenance=False)
        self.updater = update.RuntimeUpdater(self.service, self.directory.name)
        self.updater.result = ready()
        self.updater.shutdown = lambda: None
        self.updater.restart = [sys.executable, '-m', 'lazyrig', '--port', '8133']
        self.updater.cwd = str(update.ROOT)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(update, 'managed_environment', return_value=True))
        self.stack.enter_context(patch.object(update, 'installed_versions', return_value=CURRENT))
        self.launch = self.stack.enter_context(patch.object(update, 'launch_helper'))

    def test_busy_context_jobs_and_imports_block_update(self):
        for attribute, value in [('runtime_requests', 1), ('running', {'job': 1})]:
            old = getattr(self.service, attribute)
            setattr(self.service, attribute, value)
            with self.assertRaisesRegex(ValueError, 'Wait for'):
                self.updater.apply('abc123')
            setattr(self.service, attribute, old)
        self.service.downloads.running['import'] = 1
        with self.assertRaisesRegex(ValueError, 'Wait for'):
            self.updater.apply('abc123')
        self.launch.assert_not_called()
        self.assertFalse(self.service.runtime_maintenance)

    def test_direct_service_preparation_blocks_update_without_http_request(self):
        self.service._preparing = 1
        self.assertEqual(self.service.runtime_requests, 0)
        with self.assertRaisesRegex(ValueError, 'preparing'):
            self.updater.apply('abc123')
        self.launch.assert_not_called()
        self.assertFalse(self.service.runtime_maintenance)
        self.assertFalse((Path(self.directory.name) / '.runtime').exists())

    def test_expired_wrong_or_changed_check_cannot_install(self):
        with self.assertRaises(ValueError):
            self.updater.apply('other')
        self.updater.result['checked_at'] -= 1000
        with self.assertRaises(ValueError):
            self.updater.apply('abc123')
        self.updater.result = ready()
        self.updater.result['installed']['openvino'] = '2026.3.0'
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            self.updater.apply('abc123')
        self.launch.assert_not_called()

    def test_plan_uses_original_command_and_prevents_second_update(self):
        receipt = self.updater.apply('abc123')
        self.assertEqual(receipt['status'], 'updating')
        self.assertTrue(self.service.runtime_maintenance)
        plan = json.loads(self.launch.call_args.args[0].read_text())
        self.assertEqual(plan['restart'], self.updater.restart)
        self.assertEqual(plan['target'], '2026.4.1.0')
        with self.assertRaises(ValueError):
            self.updater.apply('abc123')
        self.launch.assert_called_once()

    def test_helper_failure_leaves_server_available(self):
        self.launch.side_effect = OSError('launch failed')
        with self.assertRaises(OSError):
            self.updater.apply('abc123')
        self.assertFalse(self.service.runtime_maintenance)
        self.assertEqual(self.updater.snapshot()['last_update']['status'], 'failed')

    def test_failed_parent_wait_does_not_modify_runtime_or_start_a_second_server(self):
        root = Path(self.directory.name)
        plan = root / 'plan.json'
        update.write_json(plan, {'id': 'abc123', 'parent_pid': 42})
        with patch.object(update, 'update_paths', return_value=(root/'status.json', root/'log.txt')), \
             patch.object(update, 'wait_for_parent', side_effect=RuntimeError('Still running')), \
             patch.object(update, 'install_runtime') as install:
            with self.assertRaisesRegex(RuntimeError, 'Still running'):
                update.resume_update(plan)
        install.assert_not_called()
        self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'failed')


class UpdateHTTPTests(unittest.TestCase):
    setUp, tearDown = test_http.HTTPTests.setUp, test_http.HTTPTests.tearDown
    request, json = test_http.HTTPTests.request, test_http.HTTPTests.json

    def test_missing_all_parser_sources_only_disables_update_check(self):
        self.service.runtime_updater.shutdown = lambda: None
        with patch.object(update, '_packaging_api', side_effect=RuntimeError('Optional parser unavailable')):
            self.json('/api/runtime/check', {})
            deadline = time.monotonic() + 3
            while True:
                result = self.json('/api/runtime/update')
                if result['status'] != 'checking' or time.monotonic() > deadline:
                    break
                time.sleep(.01)
        self.assertEqual(result['status'], 'error')
        self.assertIn('Optional parser unavailable', result['message'])
        self.assertTrue(self.json('/api/health')['ok'])

    def test_runtime_routes_require_api_key_and_admin_origin(self):
        self.service.config['api_key'] = 'test-key'
        for path, body in [('/api/runtime/update', None), ('/api/runtime/check', {}),
                           ('/api/runtime/update', {'check_id': 'x'})]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path, body)
            self.assertEqual(error.exception.code, 401)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/runtime/check', {}, {'Authorization': 'Bearer test-key',
                                                  'Origin': 'https://unrelated.example'})
        self.assertEqual(error.exception.code, 400)

    def test_maintenance_rejects_new_work_but_keeps_health_and_status_readable(self):
        self.service.runtime_maintenance = True
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/sessions', {})
        self.assertEqual(error.exception.code, 503)
        self.assertTrue(self.json('/api/health')['ok'])
        self.assertEqual(self.json('/api/runtime/update')['status'], 'idle')
        self.assertEqual(self.service.runtime_requests, 0)

    def test_http_check_is_nonblocking_and_invalid_requests_release_guard(self):
        self.service.runtime_updater.shutdown = lambda: None
        with patch.object(update, 'check_update', return_value=ready()):
            result = self.json('/api/runtime/check', {})
            self.assertIn(result['status'], {'checking', 'ready'})
            for _ in range(100):
                result = self.json('/api/runtime/update')
                if result['status'] != 'checking':
                    break
                time.sleep(.01)
        self.assertEqual(result['status'], 'ready')
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/settings', {})
        # The client may receive the error before the handler's finally block
        # runs. Require bounded cleanup, not an ordering guarantee of TCP.
        deadline = time.monotonic() + 1
        while self.service.runtime_requests and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.service.runtime_requests, 0)


if __name__ == '__main__':
    unittest.main()
