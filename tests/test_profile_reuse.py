"""Runtime reuse contracts; no model packages or network access are required."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from lazyrig import conversion, profile_check
from lazyrig.worker import Cancelled


# Preserve the exact historical profile bytes so future dependency updates do
# not erase the regression reproducer for the two hashes from this incident.
REQUIREMENTS = """# Based on the official Qwen-Image-2.1 notebook; upstream code pinned on 2026-09-26.
--index-url https://pypi.org/simple
--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
openvino==2026.5.0.dev20260919
openvino-tokenizers==2026.5.0.0.dev20260919
openvino-genai==2026.5.0.0.dev20260919
optimum-intel @ https://github.com/huggingface/optimum-intel/archive/615f4b1d4e4a7100e961e737c1d08588349b101a.zip
diffusers @ https://github.com/huggingface/diffusers/archive/e0abab83b5df05de9e7abd788643c1a7c1e42e28.zip
accelerate>=1.0,<2
sentencepiece>=0.2,<1
protobuf>=4,<7
Jinja2>=3.1.6,<4
packaging>=24,<27
Pillow>=10,<13
numpy>=1.24,<2.5
"""


class RuntimeReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requirements = self.root / 'requirements.txt'
        self.requirements.write_text(REQUIREMENTS)
        self.profile = 'qwen-image-21'
        for name, value in [('ROOT', self.root), ('PROFILES', {self.profile: self.requirements})]:
            patcher = patch.object(conversion, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        self.cancel = threading.Event()
        self.messages = []
        self.log = self.root / 'setup.log'

    def runtime(self, text, ready=True, profile=None):
        profile = profile or self.profile
        folder = self.root / '.runtime' / 'converters' / (profile + '-' + hashlib.sha256(text.encode()).hexdigest()[:12])
        python = conversion._runtime_python(folder)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (folder / 'installed-package.fixture').write_text('Retain installed packages in place.')
        if ready:
            (folder / 'lazyrig-ready.json').write_text(json.dumps({'profile': profile, 'python': str(python)}))
        return folder, python

    def ensure(self):
        return conversion.ensure_profile(self.profile, self.cancel, self.messages.append, self.log)

    def create_environment(self, command, *args, **kwargs):
        if '-m' in command and 'venv' in command:
            python = conversion._runtime_python(Path(command[-1]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()

    def test_packaging_addition_reuses_previous_hash_without_installs_or_moves(self):
        earlier = REQUIREMENTS.replace('packaging>=24,<27\n', '')
        folder, python = self.runtime(earlier)
        self.assertEqual(folder.name, 'qwen-image-21-9a21f9af350e')
        self.assertEqual(conversion.profile_dir(self.profile).name, 'qwen-image-21-8bd5cc23a827')
        with patch.object(conversion, 'run_logged') as run:
            self.assertEqual(self.ensure(), python)
        self.assertEqual(len(run.call_args_list), 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][1:3], ['-m', 'lazyrig.profile_check'])
        self.assertEqual(commands[1][-1], 'check')
        self.assertFalse(any('install' in command or 'venv' in command for command in commands))
        self.assertFalse(conversion.profile_dir(self.profile).exists())
        self.assertEqual((folder / 'installed-package.fixture').read_text(), 'Retain installed packages in place.')
        marker = json.loads((folder / 'lazyrig-ready.json').read_text())
        self.assertEqual(marker['requirements_sha256'], hashlib.sha256(REQUIREMENTS.encode()).hexdigest())
        # Reuse across calls/restarts is recorded on disk, without re-probing.
        with patch.object(conversion, 'run_logged', side_effect=AssertionError('Unexpected command')):
            self.assertEqual(self.ensure(), python)

    def test_current_legacy_ready_environment_never_runs_setup(self):
        self.runtime(REQUIREMENTS.replace('packaging>=24,<27\n', ''))
        _, python = self.runtime(REQUIREMENTS)
        with patch.object(conversion, 'run_logged', side_effect=AssertionError('Unexpected command')):
            self.assertEqual(self.ensure(), python)

    def test_partial_new_environment_does_not_hide_compatible_older_runtime(self):
        _, old_python = self.runtime(REQUIREMENTS.replace('packaging>=24,<27\n', ''))
        partial, partial_python = self.runtime(REQUIREMENTS, ready=False)
        def execute(command, *args, **kwargs):
            if command[0] == partial_python:
                raise RuntimeError('Incomplete new environment')
        with patch.object(conversion, 'run_logged', side_effect=execute) as run:
            self.assertEqual(self.ensure(), old_python)
        self.assertFalse(any('install' in call.args[0] for call in run.call_args_list))
        self.assertFalse((partial / 'lazyrig-ready.json').exists())
        # On later loads (including after restart), check saved readiness before
        # probing the incomplete current-hash directory again.
        self.messages.clear()
        with patch.object(conversion, 'run_logged', side_effect=AssertionError('Repeated runtime probe')):
            self.assertEqual(self.ensure(), old_python)
        self.assertEqual(self.messages, [])

    def test_missing_or_corrupt_marker_is_recovered_after_read_only_checks(self):
        folder, python = self.runtime(REQUIREMENTS, ready=False)
        marker = folder / 'lazyrig-ready.json'
        for content in (None, '{invalid', '[]'):
            with self.subTest(marker=content):
                marker.unlink(missing_ok=True)
                if content is not None:
                    marker.write_text(content)
                with patch.object(conversion, 'run_logged') as run:
                    self.assertEqual(self.ensure(), python)
                self.assertEqual(run.call_count, 2)
                self.assertTrue(json.loads(marker.read_text())['requirements_sha256'])

    def test_incompatible_versions_or_dependency_conflicts_preserve_old_environment(self):
        folder, old_python = self.runtime(REQUIREMENTS.replace('packaging>=24,<27\n', ''))
        marker_before = (folder / 'lazyrig-ready.json').read_bytes()
        for failed_stage in ('lazyrig.profile_check', 'check'):
            with self.subTest(stage=failed_stage):
                target = conversion.profile_dir(self.profile)
                (target / 'lazyrig-ready.json').unlink(missing_ok=True)
                conversion._runtime_python(target).unlink(missing_ok=True)
                def execute(command, *args, **kwargs):
                    if command[0] == old_python and failed_stage in command:
                        raise RuntimeError('Incompatible installed runtime')
                    self.create_environment(command, *args, **kwargs)
                with patch.object(conversion, 'run_logged', side_effect=execute) as run:
                    result = self.ensure()
                self.assertEqual(result, conversion.profile_python(self.profile))
                self.assertTrue(any('install' in call.args[0] for call in run.call_args_list))
                self.assertEqual((folder / 'lazyrig-ready.json').read_bytes(), marker_before)
                self.assertTrue(old_python.is_file())
                self.assertEqual((folder / 'installed-package.fixture').read_text(), 'Retain installed packages in place.')

    def test_cancelling_validation_does_not_start_installation_or_mark_ready(self):
        folder, _ = self.runtime(REQUIREMENTS.replace('packaging>=24,<27\n', ''))
        marker_before = (folder / 'lazyrig-ready.json').read_bytes()
        with patch.object(conversion, 'run_logged', side_effect=Cancelled()) as run:
            with self.assertRaises(Cancelled):
                self.ensure()
        self.assertEqual(run.call_count, 1)
        self.assertEqual((folder / 'lazyrig-ready.json').read_bytes(), marker_before)
        self.assertFalse(conversion.profile_dir(self.profile).exists())

    def test_another_profile_is_not_adopted(self):
        _, other_python = self.runtime(REQUIREMENTS, profile='standard')
        with patch.object(conversion, 'run_logged', side_effect=self.create_environment) as run:
            self.assertEqual(self.ensure(), conversion.profile_python(self.profile))
        self.assertFalse(any(call.args[0][0] == other_python for call in run.call_args_list))

    def test_failed_setup_does_not_create_ready_marker(self):
        def execute(command, *args, **kwargs):
            self.create_environment(command, *args, **kwargs)
            if 'install' in command:
                raise RuntimeError('Installation interrupted')
        with patch.object(conversion, 'run_logged', side_effect=execute):
            with self.assertRaises(RuntimeError):
                self.ensure()
        self.assertFalse((conversion.profile_dir(self.profile) / 'lazyrig-ready.json').exists())


class InstalledRequirementTests(unittest.TestCase):
    def setUp(self):
        self.installed = {name: ('1.0', None) for name in ('torch', 'torchvision')}
        self.installed.update({'numpy': ('2.4.0', None), 'packaging': ('26.0', None),
                               'openvino': ('2026.5.0.dev20260919', None),
                               'optimum-intel': ('2.2.0', 'https://example.test/pinned-commit.zip')})
        def distribution(name):
            if name not in self.installed:
                raise metadata.PackageNotFoundError(name)
            version, url = self.installed[name]
            return types.SimpleNamespace(version=version, read_text=lambda file: json.dumps({'url': url}) if url else None)
        patcher = patch.object(profile_check.metadata, 'distribution', side_effect=distribution)
        patcher.start(); self.addCleanup(patcher.stop)
        self.requirements = ('# A profile comment\n--index-url https://example.test/simple\n'
                             'openvino==2026.5.0.dev20260919\npackaging>=24,<27\n'
                             'optimum-intel @ https://example.test/pinned-commit.zip\n')

    def test_compatible_versions_and_exact_pinned_source_pass(self):
        profile_check.check_requirements(self.requirements.replace('\n', '\r\n'))

    def test_wrong_runtime_version_is_rejected(self):
        self.installed['openvino'] = ('2026.4.1', None)
        with self.assertRaisesRegex(RuntimeError, 'Installed version'):
            profile_check.check_requirements(self.requirements)

    def test_missing_new_requirement_is_rejected(self):
        del self.installed['packaging']
        with self.assertRaises(metadata.PackageNotFoundError):
            profile_check.check_requirements(self.requirements)

    def test_missing_or_different_pinned_source_is_rejected(self):
        for url in (None, 'https://example.test/different-commit.zip'):
            with self.subTest(url=url):
                self.installed['optimum-intel'] = ('2.2.0', url)
                with self.assertRaisesRegex(RuntimeError, 'Installed source'):
                    profile_check.check_requirements(self.requirements)

    def test_inactive_environment_marker_does_not_require_absent_package(self):
        profile_check.check_requirements(self.requirements + 'not-installed; python_version < "0"\n')


if __name__ == '__main__':
    unittest.main()
