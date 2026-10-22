"""Large metadata fixtures must stay sparse on every CI platform."""
import os
from pathlib import Path
import stat
import tempfile
import unittest

from support import sparse_file


class SparseFixtureTests(unittest.TestCase):
    def test_logical_size_zero_reads_sparse_storage_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            size = 64 * 1024**2
            path = sparse_file(Path(folder, 'weights.bin'), size)
            metadata = path.stat()
            self.assertEqual(metadata.st_size, size)
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes

                self.assertTrue(metadata.st_file_attributes & stat.FILE_ATTRIBUTE_SPARSE_FILE)
                allocated = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
                allocated.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
                allocated.restype = wintypes.DWORD
                high = wintypes.DWORD()
                ctypes.set_last_error(0)
                low = allocated(str(path), ctypes.byref(high))
                if low == 0xFFFFFFFF and ctypes.get_last_error():
                    raise ctypes.WinError(ctypes.get_last_error())
                self.assertLess((high.value << 32) | low, size)
            else:
                self.assertLess(metadata.st_blocks * 512, size)
            with path.open('rb') as stream:
                self.assertEqual(stream.read(16), bytes(16))
                stream.seek(size - 16)
                self.assertEqual(stream.read(), bytes(16))
            path.unlink()
            self.assertFalse(path.exists())
