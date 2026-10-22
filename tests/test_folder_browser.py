"""Folder permission failures must not disable the rest of the picker."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse

from lazyrig import catalog
import test_http


@contextmanager
def unavailable_windows_drives():
    """Inject Windows drive errors without changing the host's pathlib flavor."""
    original = Path.exists

    def exists(path):
        drive = str(path).replace('\\', '/').rstrip('/')
        if len(drive) == 2 and drive[1] == ':':
            if drive == 'G:':
                raise PermissionError(13, 'Access is denied', 'G:\\')
            if drive == 'D:':
                raise OSError(5, 'Device is not ready', 'D:\\')
            return drive == 'C:'
        return original(path)

    with patch.object(catalog, 'os', SimpleNamespace(name='nt')), \
            patch.object(Path, 'exists', exists):
        yield


class FolderBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'model').mkdir()

    def test_unrelated_denied_and_unavailable_drives_do_not_break_listing(self):
        with unavailable_windows_drives():
            result = catalog.browse(str(self.root))
        self.assertEqual(result['path'], str(self.root))
        self.assertIn('model', [item['name'] for item in result['folders']])
        self.assertIn('C:/', result['roots'])
        self.assertNotIn('G:/', result['roots'])
        self.assertNotIn('D:/', result['roots'])

    def test_selected_folder_permission_errors_are_actionable(self):
        for operation in ('resolve', 'is_dir', 'iterdir'):
            with self.subTest(operation=operation), \
                    patch.object(Path, operation, side_effect=PermissionError(13, 'Access denied')):
                with self.assertRaisesRegex(ValueError, 'account running LazyRig'):
                    catalog.browse(str(self.root))

    def test_unavailable_selected_folder_is_a_request_error(self):
        with patch.object(Path, 'iterdir', side_effect=OSError(5, 'Device is not ready')):
            with self.assertRaisesRegex(ValueError, 'drive is connected'):
                catalog.browse(str(self.root))

    def test_iterator_failure_during_enumeration_is_handled(self):
        def interrupted(_):
            yield self.root / 'model'
            raise PermissionError(13, 'Access denied')

        with patch.object(Path, 'iterdir', interrupted):
            with self.assertRaisesRegex(ValueError, 'account running LazyRig'):
                catalog.browse(str(self.root))

    def test_unreadable_child_does_not_hide_accessible_siblings(self):
        blocked = self.root / 'blocked'
        blocked.mkdir()
        (self.root / '.hidden').mkdir()
        (self.root / 'regular-file').write_text('file')
        original = Path.is_dir

        def is_dir(path):
            if path == blocked:
                raise PermissionError(13, 'Access denied')
            return original(path)

        with patch.object(Path, 'is_dir', is_dir):
            result = catalog.browse(str(self.root))
        self.assertEqual([item['name'] for item in result['folders']], ['model'])

    def test_missing_directory_remains_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'Directory not found'):
            catalog.browse(str(self.root / 'missing'))


class FolderBrowserHTTPTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_missing_models_folder_can_fall_back_despite_unreadable_drive(self):
        # The dashboard first tries models_dir, then falls back to the home
        # folder on a fresh installation where models_dir does not exist yet.
        root = Path(self.temp.name)
        (root / 'export').mkdir()
        with unavailable_windows_drives(), patch.object(Path, 'home', return_value=root):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/folders?path=' + urllib.parse.quote(str(root / 'models')))
            self.assertEqual(error.exception.code, 400)
            result = self.json('/api/folders')
            self.assertEqual(result['path'], str(root.resolve()))
            self.assertIn('export', [item['name'] for item in result['folders']])
            self.assertNotIn('G:/', result['roots'])
            self.assertNotIn('D:/', result['roots'])

    def test_explicit_denied_folder_returns_400_and_server_stays_healthy(self):
        with patch.object(Path, 'iterdir', side_effect=PermissionError(13, 'Access denied')):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/folders?path=' + urllib.parse.quote(self.temp.name))
        self.assertEqual(error.exception.code, 400)
        self.assertIn('account running LazyRig', json.load(error.exception)['error'])
        self.assertTrue(self.json('/api/health')['ok'])
        self.assertEqual(self.json('/api/folders?path=' + urllib.parse.quote(self.temp.name))['path'],
                         str(Path(self.temp.name).resolve()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
