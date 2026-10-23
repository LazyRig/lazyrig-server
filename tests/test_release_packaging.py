"""Keep the user release complete and separate from creator/private material."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import posixpath
import re
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from tools.build_release import ROOT, build, build_source_archive, release_files, repository_files
from lazyrig import __version__


class PromptPackagingTests(unittest.TestCase):
    def test_only_current_lazyrig_prompts_are_selected_from_an_old_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / 'lazyrig/prompts'
            prompts.mkdir(parents=True)
            current = {'image_generation.txt', 'image_editing.txt'}
            stale = {'system_prompt_t2i.txt', 'system_prompt_edit.txt', 'private-prompt.txt'}
            for name in current | stale:
                (prompts / name).write_text('Packaging fixture, not a model prompt.\n')
            for selector in (release_files, repository_files):
                selected = {p.relative_to(prompts).as_posix() for p in selector(root)}
                self.assertEqual(selected, current)

    def test_creator_documents_and_old_launchers_do_not_enter_user_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = {'README.md', 'docs/GETTING-STARTED.md', 'tools/image_job.py'}
            excluded = {
                'CREATOR-START-HERE.md', 'CONTRIBUTING.md', 'requirements-dev.txt',
                'maintainer/PUBLISHING.md', 'docs/PUBLISHING.md', 'docs/VALIDATION.md',
                'docs/new-private-plan.md', 'tools/build_release.py', 'tests/support.py',
                'start-demo.bat', 'start-server-old.bat', 'requirements-old.txt',
                'server-data/settings.json', '.env', 'models/weights.bin',
            }
            for name in current | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Packaging fixture.\n')
            self.assertEqual({p.relative_to(root).as_posix() for p in release_files(root)}, current)

    def test_private_handoff_cannot_enter_public_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = {'README.md', 'CONTRIBUTING.md', 'tests/test_example.py',
                      'tools/build_release.py', '.github/workflows/tests.yml'}
            private = {'CREATOR-START-HERE.md', 'maintainer/PUBLISHING.md',
                       'maintainer/PRIVATE-NOTES.md', 'maintainer/publication/DRAFT.md'}
            for name in public | private:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Packaging fixture.\n')
            self.assertEqual({p.relative_to(root).as_posix() for p in repository_files(root)},
                             public)

    def test_symlinked_content_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            outside = Path(directory) / 'private'
            outside.mkdir()
            (outside / 'secrets.py').write_text('PRIVATE = True\n')
            try:
                (root / 'lazyrig').symlink_to(outside, target_is_directory=True)
                (root / 'README.md').symlink_to(outside / 'secrets.py')
            except OSError as error:
                self.skipTest(f'Symlink creation unavailable: {error}')
            self.assertEqual(release_files(root), [])


class SplitArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        with redirect_stdout(io.StringIO()):
            cls.user_path = build(cls.output)
            cls.creator_path = build_source_archive(cls.output)
        cls.archives = {}
        for role, path in (('user', cls.user_path), ('creator', cls.creator_path)):
            with zipfile.ZipFile(path) as archive:
                cls.archives[role] = {
                    name.removeprefix('lazyrig-server/'): archive.read(name)
                    for name in archive.namelist()
                }

    def test_user_archive_has_runtime_installers_help_and_notices(self):
        files = self.archives['user']
        needed = {
            'lazyrig/__main__.py', 'lazyrig/server.py', 'lazyrig/worker.py',
            'lazyrig/web/index.html', 'lazyrig/web/app.js', 'lazyrig/web/chat.js',
            'lazyrig/prompts/image_generation.txt', 'lazyrig/prompts/image_editing.txt',
            'start-server.bat', 'start-server.sh', 'setup-windows.bat', 'update-runtime.bat',
            'lazyrig/runtime_update.py', 'docs/RUNTIME-UPDATES.md',
            'setup-windows.ps1', 'setup-portable.bat', 'setup.sh',
            'requirements-openvino.txt', 'requirements-converter.txt',
            'requirements-qwen-image-21.txt', 'configs/openvino.example.json',
            'README.md', 'CHANGELOG.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
            'docs/GETTING-STARTED.md', 'docs/WINDOWS-SETUP.md', 'docs/API.md',
            'docs/MODEL-LICENSES.md', 'tools/image_job.py',
            'docs/BETA-TEST.md',
        }
        self.assertFalse(needed - files.keys())
        referenced_images = set()
        for doc in ('README.md', 'docs/GETTING-STARTED.md'):
            for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', files[doc].decode()):
                referenced_images.add(posixpath.normpath(posixpath.join(posixpath.dirname(doc), target)))
        self.assertTrue(referenced_images, 'The public guide must include its screenshots')
        self.assertFalse(referenced_images - files.keys(), 'A public screenshot is missing from the ZIP')
        for name in referenced_images:
            self.assertTrue(files[name].startswith(b'\x89PNG\r\n\x1a\n'), name)
        self.assertFalse(any(name.startswith(('maintainer/', 'tests/', '.github/')) for name in files))
        for name in ('CONTRIBUTING.md', 'CREATOR-START-HERE.md', 'requirements-dev.txt',
                     'tools/build_release.py', 'docs/VALIDATION.md', 'docs/PUBLISHING.md'):
            self.assertNotIn(name, files)

    def test_source_archive_contains_development_files_without_private_handoff(self):
        files = self.archives['creator']
        needed = {
            'CONTRIBUTING.md', 'requirements-dev.txt',
            '.gitignore', '.gitattributes', '.github/workflows/tests.yml',
            'tools/build_release.py', 'tests/test_release_packaging.py', 'tests/support.py',
        }
        self.assertFalse(needed - files.keys())
        self.assertNotIn('CREATOR-START-HERE.md', files)
        self.assertFalse(any(name.startswith('maintainer/') for name in files))
        self.assertFalse(any(name.endswith('.zip') for name in files))
        self.assertFalse(any(name.startswith(('server-data/', '.venv/', '.runtime/', 'test-results/'))
                             for name in files))

    def test_shared_application_and_user_documentation_are_identical(self):
        for name, data in self.archives['user'].items():
            if name != 'MANIFEST.sha256.json':
                self.assertEqual(self.archives['creator'].get(name), data, name)

    def test_prerelease_version_is_not_truncated_in_either_archive_name(self):
        self.assertIn('beta', __version__)
        for label, path in (('Server', self.user_path), ('Source', self.creator_path)):
            self.assertEqual(path.name, f'LazyRig_{label}_v{__version__}.zip')
            self.assertIn(f'__version__ = "{__version__}"'.encode(),
                          self.archives['user']['lazyrig/__init__.py'])

    def test_each_archive_has_valid_manifest_crc_and_shell_permissions(self):
        for path in (self.user_path, self.creator_path):
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read('lazyrig-server/MANIFEST.sha256.json'))
                self.assertEqual(set(archive.namelist()),
                                 {'lazyrig-server/' + name for name in manifest}
                                 | {'lazyrig-server/MANIFEST.sha256.json'})
                for name, expected in manifest.items():
                    info = archive.getinfo('lazyrig-server/' + name)
                    self.assertEqual(hashlib.sha256(archive.read(info)).hexdigest(), expected, name)
                    if name.endswith('.sh'):
                        self.assertEqual((info.external_attr >> 16) & 0o777, 0o755)

    def test_identical_source_produces_repeatable_archives(self):
        with redirect_stdout(io.StringIO()):
            rebuilt = (build(self.output / 'repeat'), build_source_archive(self.output / 'repeat'))
        for original, repeated in zip((self.user_path, self.creator_path), rebuilt):
            self.assertEqual(original.read_bytes(), repeated.read_bytes())

    def test_missing_installer_dependency_blocks_release_build(self):
        root = self.output / 'incomplete-source'
        for path in release_files():
            relative = path.relative_to(ROOT)
            if relative.as_posix() == 'setup-windows.ps1':
                continue
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, root / relative)
        with self.assertRaisesRegex(ValueError, 'setup-windows.ps1'):
            build(self.output / 'incomplete-output', root=root)


if __name__ == '__main__':
    unittest.main()
