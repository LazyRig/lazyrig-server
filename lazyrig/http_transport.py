"""Bounded HTTP connections; slow clients cannot create unlimited threads.

These limits protect a trusted workstation/LAN service. They do not replace an
internet-facing reverse proxy's TLS, rate limiting and access policy.
"""
import io
import socket
import ssl
import threading
import time
from http.server import ThreadingHTTPServer


class ReceiveTimeout(TimeoutError):
    pass


class DeadlineReader(io.RawIOBase):
    """Apply an absolute deadline even when a peer trickles bytes continuously."""
    def __init__(self, connection, timeout):
        self.connection = connection
        self.timeout = timeout
        self.deadline = None

    def readable(self):
        return True

    def start(self):
        self.deadline = time.monotonic() + self.timeout

    def finish(self):
        self.deadline = None
        self.connection.settimeout(self.timeout)

    def readinto(self, buffer):
        remaining = self.timeout if self.deadline is None else self.deadline - time.monotonic()
        if remaining <= 0:
            raise ReceiveTimeout("HTTP headers or body exceeded their receive deadline")
        self.connection.settimeout(min(self.timeout, remaining))
        try:
            return self.connection.recv_into(buffer)
        except TimeoutError as exc:
            raise ReceiveTimeout("HTTP headers or body exceeded their receive deadline") from exc


class BoundedHTTPServer(ThreadingHTTPServer):
    """Acquire capacity before creating a thread, including TLS handshakes."""
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler, *, max_connections=64, http_timeout=30, tls=None):
        self.http_timeout = float(http_timeout)
        self.tls_context = tls
        self._connection_slots = threading.BoundedSemaphore(int(max_connections))
        self._connection_lock = threading.Lock()
        self._connections = set()
        self.closing = threading.Event()
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if self.closing.is_set():
            self.shutdown_request(request)
            return
        if not self._connection_slots.acquire(blocking=False):
            try:
                # At capacity TLS peers are closed without allocating a handshake
                # thread. Plain HTTP can receive an explicit retryable response.
                if self.tls_context is None:
                    request.settimeout(min(self.http_timeout, .1))
                    request.sendall(b'HTTP/1.1 503 Service Unavailable\r\n'
                                    b'Connection: close\r\nRetry-After: 1\r\n'
                                    b'X-Should-Retry: true\r\nContent-Length: 0\r\n\r\n')
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        with self._connection_lock:
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._connection_lock:
                self._connections.discard(request)
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        original = request
        try:
            request.settimeout(self.http_timeout)
            if self.closing.is_set():
                self.shutdown_request(request)
                return
            if self.tls_context is not None:
                request = self.tls_context.wrap_socket(request, server_side=True, do_handshake_on_connect=False)
                with self._connection_lock:
                    self._connections.discard(original)
                    self._connections.add(request)
                if self.closing.is_set():
                    self.shutdown_request(request)
                    return
                # SSL's handshake timeout bounds the full operation. It runs in
                # this already-counted thread, never in the accepting thread.
                request.do_handshake()
            super().process_request_thread(request, client_address)
        except (OSError, ssl.SSLError):
            self.shutdown_request(request)
        finally:
            with self._connection_lock:
                self._connections.discard(original)
                self._connections.discard(request)
            self._connection_slots.release()

    def close_connections(self):
        """Wake idle readers during graceful server shutdown."""
        self.closing.set()
        with self._connection_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def wait_for_connections(self, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._connection_lock:
                if not self._connections:
                    return True
            time.sleep(.01)
        return False

