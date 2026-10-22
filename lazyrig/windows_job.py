"""Own a native Windows worker and its descendants, including venv children.

The actual Python worker joins before importing model libraries. Only the server
keeps a handle, so an unexpected server exit also releases the worker tree.
"""
import ctypes as c
import time
import uuid


class BasicLimits(c.Structure):
    _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', c.c_uint32),
                ('minimum_working_set', c.c_size_t), ('maximum_working_set', c.c_size_t),
                ('active_process_limit', c.c_uint32), ('affinity', c.c_size_t),
                ('priority_class', c.c_uint32), ('scheduling_class', c.c_uint32)]


class IOCounters(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class ExtendedLimits(c.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IOCounters), ('process_memory', c.c_size_t),
                ('job_memory', c.c_size_t), ('peak_process_memory', c.c_size_t), ('peak_job_memory', c.c_size_t)]


class Accounting(c.Structure):
    _fields_ = [(name, c.c_int64) for name in ('user_time', 'kernel_time', 'period_user', 'period_kernel')] + [
        (name, c.c_uint32) for name in ('page_faults', 'total_processes', 'active_processes', 'terminated_processes')]


def _api():
    api = c.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([c.c_void_p, c.c_wchar_p], c.c_void_p),
        'OpenJobObjectW': ([c.c_uint32, c.c_int, c.c_wchar_p], c.c_void_p),
        'SetInformationJobObject': ([c.c_void_p, c.c_int, c.c_void_p, c.c_uint32], c.c_int),
        'QueryInformationJobObject': ([c.c_void_p, c.c_int, c.c_void_p, c.c_uint32, c.c_void_p], c.c_int),
        'AssignProcessToJobObject': ([c.c_void_p, c.c_void_p], c.c_int),
        'GetCurrentProcess': ([], c.c_void_p),
        'TerminateJobObject': ([c.c_void_p, c.c_uint32], c.c_int),
        'CloseHandle': ([c.c_void_p], c.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name); function.argtypes = arguments; function.restype = result
    return api


def _check(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


def join_worker_job(name):
    api = _api()
    handle = _check(api.OpenJobObjectW(0x0001, False, name))  # JOB_OBJECT_ASSIGN_PROCESS
    try:
        _check(api.AssignProcessToJobObject(handle, api.GetCurrentProcess()))
    finally:
        _check(api.CloseHandle(handle))


class WindowsJob:
    def __init__(self, memory_limit_bytes=None):
        if (memory_limit_bytes is not None and
                (type(memory_limit_bytes) is not int or
                 not 0 < memory_limit_bytes <= (1 << (8 * c.sizeof(c.c_size_t))) - 1)):
            raise ValueError('Windows worker commit limit must be a positive size_t byte count')
        self.memory_limit_bytes = memory_limit_bytes
        self.api = _api()
        self.name = 'Local\\LazyRigWorker_' + uuid.uuid4().hex
        self.handle = _check(self.api.CreateJobObjectW(None, self.name))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            # A kernel-enforced limit on the job's committed virtual memory.
            # This is not a cap on VRAM or every driver-managed allocation.
            limits.basic.flags |= 0x0200  # JOB_OBJECT_LIMIT_JOB_MEMORY
            limits.job_memory = memory_limit_bytes
        try:
            _check(self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.api.CloseHandle(self.handle); self.handle = None
            raise

    def active_processes(self):
        if self.handle is None:
            return 0
        info = Accounting()
        _check(self.api.QueryInformationJobObject(self.handle, 1, c.byref(info), c.sizeof(info), None))
        return info.active_processes

    def peak_commit_bytes(self):
        if self.handle is None:
            return None
        info = ExtendedLimits()
        _check(self.api.QueryInformationJobObject(self.handle, 9, c.byref(info), c.sizeof(info), None))
        return int(info.peak_job_memory)

    def terminate(self, timeout=5):
        if self.handle is None:
            return
        _check(self.api.TerminateJobObject(self.handle, 1))
        deadline = time.monotonic() + timeout
        while self.active_processes():
            if time.monotonic() >= deadline:
                raise RuntimeError('Windows model processes did not exit; device must not be reused')
            time.sleep(.02)

    def close(self):
        if self.handle is not None:
            self.terminate()
            _check(self.api.CloseHandle(self.handle))
            self.handle = None
