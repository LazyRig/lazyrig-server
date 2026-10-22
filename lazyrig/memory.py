"""Conservative device budgets and defaults, not model peak-memory estimates."""
import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
from pathlib import Path, PurePosixPath
import time


_LOAD_MEMORY_OBSERVER = ContextVar('lazyrig_load_memory_observer', default=None)


@contextmanager
def observe_load_memory(callback):
    """Receive bounded worker load diagnostics without changing factory arguments."""
    token = _LOAD_MEMORY_OBSERVER.set(callback)
    try:
        yield
    finally:
        _LOAD_MEMORY_OBSERVER.reset(token)


def publish_load_memory(data):
    callback = _LOAD_MEMORY_OBSERVER.get()
    if callback is not None:
        callback(dict(data))


class MemoryPressureError(RuntimeError):
    """A sampled host-memory safety margin was reached during model loading."""

    def __init__(self, total_gib, available_gib, headroom_gib, source='physical RAM'):
        self.total_gib = total_gib
        self.available_gib = available_gib
        self.headroom_gib = headroom_gib
        self.source = source
        super().__init__(
            f'Model loading stopped to preserve host-memory headroom: '
            f'{self.available_gib:.1f} GiB available; at least '
            f'{headroom_gib:.1f} GiB reserved for the system '
            f'({self.total_gib:.1f} GiB total, {self.source}). '
            'This is a sampled safety guard, not a hard memory-allocation limit.')


def _windows_memory_status():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in (
                'total_physical', 'available_physical', 'total_pagefile',
                'available_pagefile', 'total_virtual', 'available_virtual',
                'available_extended_virtual')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status


def _cgroup_v2_limits():
    """Effective cgroup v2 bounds, including parents when visible in this namespace.

    Inactive file pages can be reclaimed, so they do not count as unavailable.
    Missing controllers/metrics leave the physical-memory sample unchanged.
    """
    root = Path('/sys/fs/cgroup')
    folders = [root]
    try:
        line = next(line for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
        parts = PurePosixPath(line[3:]).parts
        if '..' not in parts:
            current = root.joinpath(*(part for part in parts if part != '/'))
            while current != root:
                folders.append(current)
                current = current.parent
    except (OSError, StopIteration, ValueError):
        pass
    for folder in folders:
        try:
            maximum = (folder / 'memory.max').read_text().strip()
            if maximum == 'max':
                continue
            maximum = int(maximum)
            current = int((folder / 'memory.current').read_text().strip())
            if maximum <= 0 or current < 0:
                continue
            reclaimable = 0
            try:
                stats = dict(line.split() for line in (folder / 'memory.stat').read_text().splitlines())
                reclaimable = max(0, min(current, int(stats.get('inactive_file', '0'))))
            except (OSError, ValueError):
                pass
            yield maximum, max(0, min(maximum, maximum - current + reclaimable))
        except (OSError, ValueError):
            continue


def host_memory_snapshot():
    """Read live available physical memory without importing a native runtime.

    Returns GiB values or None if the OS cannot provide trustworthy readings.
    Linux additionally honors visible cgroup v2 limits. This is not GPU telemetry.
    """
    try:
        if os.name == 'nt':
            status = _windows_memory_status()
            if status is None:
                return None
            total, available = status.total_physical, status.available_physical
            source = 'Windows physical RAM'
        else:
            fields = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, value = line.split(':', 1)
                parts = value.split()
                if key in {'MemTotal', 'MemAvailable'} and len(parts) == 2 and parts[1] == 'kB':
                    fields[key] = int(parts[0]) * 1024
            total, available = fields['MemTotal'], fields['MemAvailable']
            source = 'Linux physical RAM'
            for limit, remaining in _cgroup_v2_limits():
                if limit < total or remaining < available:
                    source = 'Linux physical RAM / cgroup v2'
                total, available = min(total, limit), min(available, remaining)
        if total <= 0 or available < 0 or available > total:
            return None
        return {'total_gib': total / 1024**3, 'available_gib': available / 1024**3,
                'source': source}
    except (AttributeError, KeyError, OSError, ValueError):
        return None


class LoadMemoryGuard:
    """Stop a loading worker before sampled RAM availability consumes headroom.

    Sampling cannot prevent a sudden native allocation from overshooting. It is
    deliberately scoped to loading; generation settings and tokens are unchanged.
    """

    def __init__(self):
        self._next_check = 0
        self.last_snapshot = None
        self.headroom_gib = None
        self.allocation_limit_bytes = None
        self.minimum_available_gib = None
        self.samples = 0

    def diagnostics(self):
        snapshot = self.last_snapshot
        return {'status': 'available' if snapshot is not None else 'unavailable',
                'source': snapshot['source'] if snapshot else None,
                'total_gib': snapshot['total_gib'] if snapshot else None,
                'available_gib': snapshot['available_gib'] if snapshot else None,
                'headroom_gib': self.headroom_gib,
                'minimum_available_gib': self.minimum_available_gib,
                'sample_count': self.samples}

    def check(self, force=False):
        now = time.monotonic()
        if not force and now < self._next_check:
            return
        self._next_check = now + .1
        snapshot = host_memory_snapshot()
        self.samples += 1
        self.last_snapshot = snapshot
        self.headroom_gib = None
        self.allocation_limit_bytes = None
        if snapshot is None:
            return
        headroom = max(1.0, min(8.0, snapshot['total_gib'] * .1))
        self.headroom_gib = headroom
        available = snapshot['available_gib']
        self.minimum_available_gib = (available if self.minimum_available_gib is None else
                                      min(self.minimum_available_gib, available))
        self.allocation_limit_bytes = max(0, int((available - headroom) * 1024**3))
        if available <= headroom:
            raise MemoryPressureError(snapshot['total_gib'], snapshot['available_gib'],
                                      headroom, snapshot['source'])


def system_memory_gib():
    """Read physical RAM using the operating system, without a new dependency."""
    try:
        if os.name == 'nt':
            status = _windows_memory_status()
            if status is None:
                return None
            total = status.total_physical
        else:
            total = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        return total / 1024**3 if total > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def detected_budget(core, device_id):
    """Leave 15% headroom when discovering a device for the first time."""
    if device_id == 'CPU':
        total = system_memory_gib()
    else:
        try:
            total = float(core.get_property(device_id, 'GPU_DEVICE_TOTAL_MEM_SIZE')) / 1024**3
        except Exception:
            total = None
    if total and math.isfinite(total) and total > 0:
        return max(.1, min(9999, math.floor(total * .85 * 10) / 10))
    return 8 if device_id == 'CPU' else 2


def device_reservation(devices, device_id):
    """Use the configured usable budget, including any operator adjustment."""
    device = next((d for d in devices if d['id'] == device_id), None)
    if device is None:
        raise ValueError('Select a configured device for this model')
    return float(device['budget_gib'])
