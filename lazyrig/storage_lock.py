"""One live server owns a data directory, including its recovery procedure."""
import os


class DataDirectoryInUse(RuntimeError):
    pass


class DirectoryLock:
    """An OS-held lock: process exit releases it, stale files do not block startup.

    Never unlink this file on close. Replacing a locked inode would allow a new
    process to acquire a different lock while the old owner is still running.
    """
    def __init__(self, directory):
        self.file = None
        path = directory / '.lazyrig-owner.lock'
        handle = path.open('a+b')
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise DataDirectoryInUse(
                f'Cannot acquire the LazyRig data-directory lock: {directory}. '
                'Another server may already own it. Stop that server or choose '
                'a different data_dir; do not delete its lock file while it is running.'
            ) from exc
        except BaseException:
            handle.close()
            raise
        self.file = handle

    def close(self):
        handle, self.file = self.file, None
        if handle is not None:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
