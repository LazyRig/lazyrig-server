"""One monotonic request budget shared by admission, queueing and native work."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
import time


class RequestDeadline(TimeoutError):
    code = 'request_timeout'
    status = 504
    retryable = False

    def __init__(self, message='Request deadline exceeded. Saved artifacts are retained.'):
        super().__init__(message)


class DeadlineEvent(threading.Event):
    """Cancellation event whose polling also enforces an absolute deadline.

    Use ``cancelled`` in finalization: observing the final cancel intent must
    never raise another timeout while releasing a job or a worker.
    """
    def __init__(self, timeout=None, *, deadline=None):
        super().__init__()
        if timeout is not None:
            if not math.isfinite(float(timeout)) or float(timeout) <= 0:
                raise ValueError('Request timeout must be a positive finite number')
            calculated = time.monotonic() + float(timeout)
            deadline = calculated if deadline is None else min(float(deadline), calculated)
        if deadline is not None and not math.isfinite(float(deadline)):
            raise ValueError('Request deadline must be finite')
        self.deadline = float(deadline) if deadline is not None else None

    @property
    def cancelled(self):
        return super().is_set()

    @property
    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def check(self):
        if not self.cancelled and self.expired:
            raise RequestDeadline()

    def is_set(self):
        self.check()
        return self.cancelled

    def remaining(self, seconds=None):
        self.check()
        remaining = max(0.001, self.deadline - time.monotonic()) if self.deadline is not None else None
        if seconds is None:
            return remaining
        return float(seconds) if remaining is None else min(float(seconds), remaining)

    def wait(self, timeout=None):
        if self.cancelled:
            return True
        result = super().wait(self.remaining(timeout))
        self.check()
        return result


_current = ContextVar('lazyrig_request_deadline', default=None)


def current_deadline():
    return _current.get()


@contextmanager
def deadline_context(signal):
    token = _current.set(signal)
    try:
        check_deadline(signal)
        yield signal
    finally:
        _current.reset(token)


def check_deadline(signal=None):
    signal = signal if signal is not None else current_deadline()
    if isinstance(signal, DeadlineEvent):
        signal.check()


def time_budget(seconds, signal=None):
    signal = signal if signal is not None else current_deadline()
    return signal.remaining(seconds) if isinstance(signal, DeadlineEvent) else seconds


def cancellation_requested(signal):
    return signal.cancelled if isinstance(signal, DeadlineEvent) else signal.is_set()
