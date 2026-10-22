"""Bounded preparation, job reservations and request-local deadline propagation."""
import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from .deadlines import DeadlineEvent, deadline_context


class AdmissionError(ValueError):
    def __init__(self, message, code='queue_full'):
        super().__init__(message)
        self.code = code
        self.status = self.status_code = 503
        self.error_type = 'server_overloaded' if code in {'queue_full', 'preparation_full'} else 'server_unavailable'
        self.retryable = True
        self.retry_after = 1


@dataclass
class Reservation:
    service: object
    signal: DeadlineEvent
    committed: bool = False


_current = contextvars.ContextVar('lazyrig_admission', default=None)


@contextmanager
def admission(service, timeout=None, signal=None):
    """Reserve before downloads/tokenization and reuse it across nested APIs."""
    prior = _current.get()
    if prior is not None and prior.service is service:
        prior.signal.check()
        yield prior
        return
    signal = signal or DeadlineEvent(timeout=timeout if timeout is not None else service.config.get('request_timeout_seconds', 1800))
    signal.check()
    with service.lock:
        service._check_admission()
        if len(service.running) + service._reservations >= service.config.get('max_jobs', 8):
            raise AdmissionError('Server queue is full; retry after an active job finishes')
        if service._preparing >= service.config.get('max_preparing', 2):
            raise AdmissionError('Request preparation is busy; retry shortly', 'preparation_full')
        service._preparing += 1
        service._reservations += 1
        service._preparation_signals.add(signal)
    reservation = Reservation(service, signal)
    token = _current.set(reservation)
    try:
        with deadline_context(signal):
            yield reservation
    finally:
        _current.reset(token)
        with service.lock:
            service._preparing -= 1
            service._preparation_signals.discard(signal)
            if not reservation.committed:
                service._reservations -= 1


def commit(reservation):
    """Called under service.lock when a persisted job occupies its running slot."""
    if reservation.committed:
        raise RuntimeError('An admission reservation cannot launch two jobs')
    reservation.committed = True
    reservation.service._reservations -= 1


@contextmanager
def keyed(service, identity, signal):
    """Serialize only matching retry keys; slow preparation never owns service.lock."""
    key = identity[0]
    with service.lock:
        if service.closed:
            raise AdmissionError('Server is shutting down', 'server_unavailable')
        entry = service._key_locks.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    acquired = False
    try:
        while not acquired:
            signal.check()
            acquired = entry[0].acquire(timeout=min(.05, signal.remaining(.05)))
        signal.check()
        with service.lock:
            if service.closed:
                raise AdmissionError('Server is shutting down', 'server_unavailable')
        yield
    finally:
        if acquired:
            entry[0].release()
        with service.lock:
            entry[1] -= 1
            if entry[1] == 0:
                service._key_locks.pop(key, None)
