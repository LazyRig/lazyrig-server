"""Stable OpenVINO updates for LazyRig's private environment; never import the runtime here."""
import argparse
import ast
from contextlib import contextmanager
import copy
from functools import lru_cache
from importlib import import_module
from importlib.machinery import PathFinder
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('openvino', 'openvino-genai', 'openvino-tokenizers')
INDEX = 'https://pypi.org/simple'
JSON_URL = 'https://pypi.org/pypi/openvino-genai/json'
_PACKAGING_LOCK = threading.RLock()


def _import_packaging(prefix):
    """Load one consistent parser implementation, only on an update request."""
    requirements = import_module(prefix + '.requirements')
    specifiers = import_module(prefix + '.specifiers')
    tags = import_module(prefix + '.tags')
    utils = import_module(prefix + '.utils')
    version = import_module(prefix + '.version')
    return SimpleNamespace(Requirement=requirements.Requirement, SpecifierSet=specifiers.SpecifierSet,
                           sys_tags=tags.sys_tags, canonicalize_name=utils.canonicalize_name,
                           parse_wheel_filename=utils.parse_wheel_filename,
                           InvalidVersion=version.InvalidVersion, Version=version.Version)


@lru_cache(maxsize=1)
def _packaging_api():
    # Existing uv-created environments may contain neither packaging nor pip.
    # Never make their absence prevent server startup, and never install anything
    # during a read-only update check. Reuse an available parser when possible.
    with _PACKAGING_LOCK:
        for prefix in ('packaging', 'pip._vendor.packaging'):
            try:
                return _import_packaging(prefix)
            except ImportError:
                continue
        bundled = importlib.util.find_spec('ensurepip')
        if bundled and bundled.origin:
            directory = Path(bundled.origin).parent / '_bundled'
            for wheel in sorted(directory.glob('pip-*.whl'), reverse=True):
                # Standard CPython includes this local wheel for ensurepip.
                # Import its parser read-only; do not bootstrap/install pip here.
                location = str(wheel)
                sys.path.insert(0, location)
                try:
                    return _import_packaging('pip._vendor.packaging')
                except ImportError:
                    continue
                finally:
                    sys.path.remove(location)
    raise RuntimeError('The optional packaging helper is unavailable. LazyRig can keep running. '
                       'Install packaging in this environment to enable runtime update checks.')


def installed_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def installed_build_versions():
    """Display build IDs without importing runtime libraries or changing pip versions."""
    result = dict(installed_versions())
    for name in PACKAGES:
        module = sys.modules.get(name.replace('-', '_'))
        result[name] = getattr(module, '__version__', None) or result[name]
    # Tokenizers exposes its build ID in a small static file. Reading the literal
    # avoids loading its native extension just to populate the update dialog.
    if 'openvino_tokenizers' not in sys.modules:
        try:
            path = metadata.distribution('openvino-tokenizers').locate_file('openvino_tokenizers/__version__.py')
            for node in ast.parse(path.read_text(encoding='utf-8')).body:
                if (isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)
                        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                    result['openvino-tokenizers'] = node.value.value
                    break
        except (metadata.PackageNotFoundError, OSError, SyntaxError, UnicodeError):
            pass
    return result


def custom_runtime():
    """Local/VCS/direct-URL builds need their original installation workflow."""
    for name in PACKAGES:
        try:
            if metadata.distribution(name).read_text('direct_url.json'):
                return True
        except metadata.PackageNotFoundError:
            pass
    return False


def managed_environment(root=ROOT):
    return (Path(sys.prefix).resolve() == (Path(root) / '.venv').resolve()
            and sys.prefix != sys.base_prefix)


def runtime_requirement(root=ROOT):
    Requirement = _packaging_api().Requirement
    for line in (Path(root) / 'requirements-openvino.txt').read_text().splitlines():
        if line.strip().startswith('openvino-genai'):
            return Requirement(line.strip()).specifier
    raise ValueError('OpenVINO GenAI requirement is missing. Restore requirements-openvino.txt.')


def fetch_releases():
    request = Request(JSON_URL, headers={'User-Agent': 'LazyRig-runtime-updater', 'Accept': 'application/json'})
    with urlopen(request, timeout=20) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise ValueError('The package-index response was too large.')
    return json.loads(content)


def select_release(data, specifier, tags=None, python_version=None):
    """Choose a non-yanked stable wheel for this Python/OS and application range."""
    parser = _packaging_api()
    Version, InvalidVersion = parser.Version, parser.InvalidVersion
    SpecifierSet, parse_wheel_filename, sys_tags = parser.SpecifierSet, parser.parse_wheel_filename, parser.sys_tags
    tags = set(sys_tags()) if tags is None else set(tags)
    python_version = python_version or '.'.join(map(str, sys.version_info[:3]))
    releases, compatible = [], []
    for raw, files in data['releases'].items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or not any(not f.get('yanked') for f in files):
            continue
        releases.append(version)
        if version not in specifier:
            continue
        for file in files:
            if file.get('yanked') or file.get('packagetype') != 'bdist_wheel':
                continue
            if python_version not in SpecifierSet(file.get('requires_python') or ''):
                continue
            try:
                _, _, _, wheel_tags = parse_wheel_filename(file['filename'])
            except (ValueError, KeyError):
                continue
            if tags & wheel_tags:
                compatible.append(version)
                break
    return (str(max(releases)) if releases else None,
            str(max(compatible)) if compatible else None)


def check_update(root=ROOT):
    Version = _packaging_api().Version
    current = installed_versions()
    specifier = runtime_requirement(root)
    latest, target = select_release(fetch_releases(), specifier)
    blocked = None
    if not managed_environment(root):
        blocked = "Automatic installation requires this LazyRig folder's .venv. Update custom environments manually."
    elif any(not value for value in current.values()):
        blocked = 'The runtime installation is incomplete. Run setup to repair it first.'
    elif any(Version(value).is_prerelease or Version(value).is_devrelease for value in current.values()):
        blocked = 'Experimental/nightly runtime detected. Keep using its model-specific installation profile.'
    elif custom_runtime():
        blocked = 'Custom runtime installation detected. Update it with its original installation workflow.'
    elif not target:
        blocked = 'No compatible stable wheel is available for this Python, platform and LazyRig version.'
    installed = Version(current['openvino-genai']) if current['openvino-genai'] else None
    available = bool(target and installed and Version(target) > installed)
    if blocked:
        message = blocked
    elif available:
        message = 'A stable runtime update is available. Updating unloads models and restarts LazyRig.'
    elif target and installed and installed > Version(target):
        message = 'Your installed runtime is newer than the compatible stable release. It will not be downgraded.'
    else:
        message = 'Your compatible stable runtime is up to date.'
    return {'installed': current, 'latest_release': latest, 'target': target,
            'requirement': str(specifier), 'available': available,
            'can_update': available and not blocked, 'message': message,
            'checked_at': time.time(), 'source': 'PyPI · stable releases'}


@contextmanager
def environment_lock(exclusive=False):
    """OS-owned locks disappear on exit, including after a crash."""
    path = Path(sys.prefix) / 'lazyrig-runtime.lock'
    stream = path.open('a+b')
    try:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('The runtime is in use. Close other LazyRig servers and runtime tools, then retry.') from exc
        yield
    finally:
        # Closing the descriptor releases its lock on Windows and POSIX.
        stream.close()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2); file.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_paths(root=ROOT):
    return Path(root) / 'logs/runtime-update.json', Path(root) / 'logs/runtime-update.log'


def constraints_text():
    """Preserve other installed packages and their declared runtime requirements."""
    parser = _packaging_api()
    Requirement, canonicalize_name = parser.Requirement, parser.canonicalize_name
    # Do not resolve a newer GenAI against older Runtime or Tokenizers packages.
    lines = [f'{name}>={version}' for name, version in installed_versions().items() if version]
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata['Name'])
        if name in PACKAGES:
            continue
        lines.append(f'{name}=={dist.version}')
        for raw in dist.requires or ():
            req = Requirement(raw)
            if canonicalize_name(req.name) in PACKAGES and (not req.marker or req.marker.evaluate()):
                if req.url:
                    raise RuntimeError('An installed package requires a custom runtime URL. Manage that profile manually.')
                lines.append(canonicalize_name(req.name) + str(req.specifier))
    return '\n'.join(sorted(set(lines))) + '\n'


def clean_pip_environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
    env['PIP_CONFIG_FILE'] = os.devnull
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def run_command(command, log, timeout=1800):
    print('Running:', ' '.join(map(str, command)), flush=True)
    log.write('\n$ ' + ' '.join(map(str, command)) + '\n'); log.flush()
    subprocess.run(command, cwd=ROOT, env=clean_pip_environment(), stdin=subprocess.DEVNULL,
                   stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def install_runtime(target, log, root=ROOT):
    """Stage wheels before changing packages; keep old runtime wheels for recovery."""
    parser = _packaging_api()
    Version, parse_wheel_filename, canonicalize_name = parser.Version, parser.parse_wheel_filename, parser.canonicalize_name
    version = Version(target)
    if version.is_prerelease or version.is_devrelease or version not in runtime_requirement(root):
        raise ValueError('The selected release is outside the supported stable runtime range.')
    before = installed_versions()
    if any(not v for v in before.values()) or any(Version(v).is_prerelease for v in before.values()):
        raise RuntimeError('Use setup or the experimental profile to repair this runtime.')
    if custom_runtime():
        raise RuntimeError('Custom runtime detected. Use its original installation workflow.')
    if version <= Version(before['openvino-genai']):
        return 'unchanged', 'The compatible runtime is already installed; no downgrade was performed.'
    # A parser imported from ensurepip's wheel is not an installed pip module
    # that a fresh subprocess could execute. Inspect normal paths, not sys.modules.
    if PathFinder.find_spec('pip') is None:
        run_command([sys.executable, '-m', 'ensurepip', '--upgrade'], log, 180)
    pip = [sys.executable, '-m', 'pip', '--disable-pip-version-check', '--no-input']
    probe = [sys.executable, '-m', 'lazyrig.probe']
    with tempfile.TemporaryDirectory(prefix='lazyrig-runtime-') as folder:
        folder = Path(folder)
        constraints = folder / 'constraints.txt'
        constraints.write_text(constraints_text())
        fresh, previous = folder / 'new', folder / 'previous'
        fresh.mkdir(); previous.mkdir()
        requirement = f'openvino-genai=={version}'
        # Download/resolve before touching installed packages. No source builds or nightlies.
        run_command(pip + ['download', '--index-url', INDEX, '--only-binary=:all:',
                          '--dest', str(fresh), '-c', str(constraints), requirement], log)
        for wheel in fresh.glob('*.whl'):
            name, candidate, _, _ = parse_wheel_filename(wheel.name)
            if canonicalize_name(name) in PACKAGES and candidate.is_prerelease:
                raise RuntimeError('The stable release resolved to an experimental dependency; update was stopped.')
        old_requirements = [f'{name}=={before[name]}' for name in PACKAGES]
        run_command(pip + ['download', '--index-url', INDEX, '--only-binary=:all:', '--no-deps',
                          '--dest', str(previous), *old_requirements], log)
        try:
            # Only the three runtime packages change; all other installed
            # dependencies were constrained to their existing versions above.
            run_command(pip + ['install', '--no-index', '--find-links', str(fresh), '--only-binary=:all:',
                              '--upgrade', '-c', str(constraints), requirement], log)
            run_command(pip + ['check'], log, 120)
            run_command(probe, log, 120)
        except Exception as exc:
            log.write('\nUpdate verification failed; restoring previous runtime wheels.\n'); log.flush()
            try:
                run_command(pip + ['install', '--no-index', '--find-links', str(previous), '--no-deps',
                                  '--force-reinstall', *old_requirements], log)
                run_command(probe, log, 120)
                run_command(pip + ['check'], log, 120)
            except Exception as rollback:
                raise RuntimeError('Update and recovery failed. Run setup-windows.bat (or setup.sh) before restarting.') from rollback
            return 'rolled_back', f'Update failed; previous runtime restored. See logs/runtime-update.log ({type(exc).__name__}).'
    return 'succeeded', 'Stable runtime updated and verified. Models will load on their next request.'


def wait_for_parent(pid, timeout=120):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # Already exited.
                return
            raise RuntimeError('Cannot verify that the server has stopped. Close it and use update-runtime.bat.')
        try:
            if kernel.WaitForSingleObject(handle, timeout * 1000) != 0:
                raise RuntimeError('The server did not stop in time. No runtime packages were changed.')
        finally:
            kernel.CloseHandle(handle)
    else:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)
        raise RuntimeError('The server did not stop in time. No runtime packages were changed.')


def launch_helper(plan_path):
    command = [sys.executable, '-m', 'lazyrig.runtime_update', '--resume', str(plan_path)]
    if os.name == 'nt':
        return subprocess.Popen(command, cwd=ROOT,
                                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP)
    _, log_path = update_paths()
    with log_path.open('a') as output:
        return subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


class RuntimeUpdater:
    def __init__(self, service, root=ROOT):
        self.service, self.root = service, Path(root)
        self.lock = threading.RLock()
        self.result = {'status': 'idle', 'installed': installed_versions(), 'can_update': False}
        self.shutdown = None
        self.restart = None
        self.cwd = None

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.result)
        result['installed_builds'] = installed_build_versions()
        status_path, _ = update_paths(self.root)
        try:
            result['last_update'] = json.loads(status_path.read_text())
        except (OSError, ValueError):
            result['last_update'] = None
        if not self.shutdown:
            result['can_update'] = False
            result['message'] = 'Use update-runtime.bat, or start LazyRig with its standard launcher for dashboard updates.'
        return result

    def check(self):
        with self.lock:
            if self.result['status'] in {'checking', 'updating'}:
                return self.snapshot()
            check_id = uuid.uuid4().hex
            self.result = {'status': 'checking', 'installed': installed_versions(),
                           'can_update': False, 'check_id': check_id, 'message': 'Checking stable releases on PyPI…'}
        def work():
            try:
                result = check_update(self.root)
                result.update(status='ready', check_id=check_id)
            except Exception as exc:
                result = {'status': 'error', 'can_update': False, 'check_id': check_id,
                          'installed': installed_versions(), 'message': f'Could not check for updates: {exc}'}
            with self.lock:
                self.result = result
        threading.Thread(target=work, daemon=True).start()
        return self.snapshot()

    def apply(self, check_id):
        with self.service.lock, self.lock:
            result = self.result
            if (not self.shutdown or not self.restart or not self.cwd or not managed_environment(self.root)
                    or result['status'] != 'ready' or not result.get('can_update')
                    or check_id != result.get('check_id') or time.time() - result['checked_at'] > 900):
                raise ValueError('Check for updates again before installing a compatible stable release.')
            if (self.service.runtime_requests or getattr(self.service, '_preparing', 0)
                    or self.service.running or self.service.downloads.running):
                raise ValueError('Wait for preparing/active/queued requests and model imports to finish before updating.')
            if getattr(self.service, 'restart_required', False):
                raise ValueError('Restart once to apply pending server-address settings before updating the runtime.')
            if installed_versions() != result['installed']:
                raise ValueError('The environment changed since the check. Check for updates again.')
            self.service.runtime_maintenance = True
            plan = {'id': check_id, 'target': result['target'], 'parent_pid': os.getpid(),
                    'restart': self.restart, 'cwd': self.cwd, 'created': time.time()}
            plan_path = self.root / '.runtime' / ('update-' + check_id + '.json')
            status_path, _ = update_paths(self.root)
            try:
                write_json(plan_path, plan)
                write_json(status_path, {'id': check_id, 'status': 'updating', 'message': 'Stopping LazyRig to update its runtime.'})
                launch_helper(plan_path)
            except Exception:
                self.service.runtime_maintenance = False
                plan_path.unlink(missing_ok=True)
                write_json(status_path, {'id': check_id, 'status': 'failed', 'message': 'Could not start the updater. The server is still running.'})
                raise
            result.update(status='updating', can_update=False, message='Updating runtime; this dashboard will reconnect after restart.')
            return self.snapshot()

    def stop_server(self):
        # Called after the update receipt is sent. Keep shutdown out of the HTTP thread.
        threading.Thread(target=self.shutdown, daemon=True).start()


def resume_update(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    plan_path.unlink()
    status_path, log_path = update_paths()
    try:
        wait_for_parent(int(plan['parent_pid']))
    except Exception as exc:
        write_json(status_path, {'id': plan['id'], 'status': 'failed', 'message': str(exc),
                                'finished_at': time.time()})
        raise
    ready_to_restart = True
    try:
        with environment_lock(exclusive=True), log_path.open('a', encoding='utf-8') as log:
            status, message = install_runtime(plan['target'], log)
    except Exception as exc:
        status, message = 'failed', str(exc)
        # Verify the actual environment before restarting after any failed operation.
        try:
            with environment_lock(exclusive=True), log_path.open('a', encoding='utf-8') as log:
                run_command([sys.executable, '-m', 'lazyrig.probe'], log, 120)
        except Exception:
            ready_to_restart = False
    write_json(status_path, {'id': plan['id'], 'status': status, 'message': message,
                            'installed': installed_versions(), 'finished_at': time.time()})
    print(message, flush=True)
    if ready_to_restart:
        print('Restarting LazyRig. Keep this console open.', flush=True)
        # No arbitrary HTTP-supplied command: this is the original server command.
        return subprocess.call(plan['restart'], cwd=plan['cwd'])
    print('See logs/runtime-update.log. Run setup to repair the environment, then start LazyRig.', flush=True)
    if os.name == 'nt':
        input('Press Enter to close…')
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check only; do not install')
    parser.add_argument('--yes', action='store_true', help='Install an available stable update without prompting')
    parser.add_argument('--resume', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.resume:
            if not managed_environment():
                raise RuntimeError('The updater must run in this LazyRig folder’s .venv.')
            return resume_update(args.resume)
        result = check_update()
        print(json.dumps(result['installed'], indent=2))
        print('Latest stable GenAI:', result['latest_release'] or 'unavailable')
        print('Latest compatible:', result['target'] or 'unavailable')
        print(result['message'])
        if args.check or not result['can_update']:
            return 0
        if not args.yes and input('Close LazyRig and other runtime tools. Install this update? [y/N] ').strip().lower() != 'y':
            return 0
        status_path, log_path = update_paths()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with environment_lock(exclusive=True), log_path.open('a', encoding='utf-8') as log:
            status, message = install_runtime(result['target'], log)
        write_json(status_path, {'id': uuid.uuid4().hex, 'status': status, 'message': message,
                                'installed': installed_versions(), 'finished_at': time.time()})
        print(message)
        print('Start LazyRig with your usual launcher. Log:', log_path)
        return 0 if status in {'succeeded', 'unchanged'} else 1
    except Exception as exc:
        print('Runtime update stopped:', exc, file=sys.stderr)
        print('See logs/runtime-update.log. Settings and model files are not modified.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
