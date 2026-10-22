import argparse
import io
import os
import sys
from contextlib import nullcontext
import json
import mimetypes
import re
import secrets
import select
import signal
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from .config import load, restore_settings, normalize_origin
from .service import Service, TERMINAL
from . import compat
from .catalog import browse, inspect_export
from .mtp import model_capabilities
from . import __version__
from .image_edit import parse_multipart
from .idempotency import IdempotencyConflict
from .prompt_enhancement import PROMPT_DIRECTORY, PROMPT_ROUTES
from .http_transport import BoundedHTTPServer, DeadlineReader, ReceiveTimeout
from .admission import AdmissionError
from .deadlines import RequestDeadline

WEB = Path(__file__).parent / "web"
DEVTOOLS_DISCOVERY_PATH = "/.well-known/appspecific/com.chrome.devtools.json"

def inference_path(path):
    return (path in {"/api/health", "/api/ready", "/v1/models", "/v1/chat/completions", "/v1/images/generations",
                     "/v1/images/edits", "/api/image/edits", "/api/context", "/api/chat/jobs", "/api/image/jobs", "/api/artifacts"} or
            path in PROMPT_ROUTES or
            re.fullmatch(r"/api/jobs/[a-f0-9]{32}(?:/(?:result|events|cancel))?", path) is not None or
            re.fullmatch(r"/artifacts/[a-f0-9]{32}\.(?:svg|png)", path) is not None)


def tls_context(config):
    if not config.get("tls_cert"):
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(config["tls_cert"], config["tls_key"], password=lambda: "")
    return context


def handler(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        server_version = "LazyRig/" + __version__

        def setup(self):
            super().setup()
            timeout = float(service.config.get("http_timeout_seconds", 30))
            self.connection.settimeout(timeout)
            self.rfile.close()
            self._deadline_reader = DeadlineReader(self.connection, timeout)
            self.rfile = io.BufferedReader(self._deadline_reader)

        def handle_one_request(self):
            self.response_started = False
            self.job_id = None
            self.cancel_on_disconnect = False
            self._deadline_reader.start()
            try:
                # Browsers open spare connections and leave keep-alive sockets
                # idle. Expire these silently before a request has started.
                # Peek also sees a partial next request already buffered by the
                # preceding read, without consuming it or resetting the deadline.
                try:
                    if not self.rfile.peek(1):
                        self.close_connection = True
                        return
                except ReceiveTimeout:
                    self.close_connection = True
                    return
                super().handle_one_request()
            except (ConnectionError, ssl.SSLEOFError) as exc:
                self.close_connection = True
                jid = self.job_id
                if jid:
                    if self.cancel_on_disconnect:
                        service.cancel(jid, reason="client_disconnect")
                    service.store.event(jid, "client_disconnected", transport_error=type(exc).__name__,
                                        cancel_requested=self.cancel_on_disconnect,
                                        message="HTTP connection closed. Job status and saved artifacts remain available.")
                super().log_message("Connection closed (%s); job=%s; %s", type(exc).__name__, jid or "none",
                                    "retrieve /api/jobs/" + jid + "/result" if jid else "request ended")

        def check_disconnect(self):
            """Notice an aborted synchronous request even before response headers.

            Poll in the HTTP handler, never in the model worker. Event/result
            observers and asynchronous acknowledgements do not own job lifetime.
            """
            if not self.cancel_on_disconnect:
                return
            if getattr(self.server, 'closing', None) is not None and self.server.closing.is_set():
                raise ConnectionAbortedError("Server is shutting down")
            if not select.select([self.connection], [], [], 0)[0]:
                return
            if isinstance(self.connection, ssl.SSLSocket):
                # SSL does not support MSG_PEEK. Nonblocking read processes TLS
                # close_notify without waiting for another record. An unexpected
                # pipelined application byte means this connection cannot be reused.
                timeout = self.connection.gettimeout()
                try:
                    self.connection.settimeout(0)
                    data = self.connection.recv(1)
                    if data:
                        self.close_connection = True
                except (ssl.SSLWantReadError, ssl.SSLWantWriteError, BlockingIOError):
                    return
                except ssl.SSLZeroReturnError as exc:
                    raise ConnectionAbortedError("Client closed the TLS inference request") from exc
                finally:
                    self.connection.settimeout(timeout)
            else:
                data = self.connection.recv(1, socket.MSG_PEEK)
            if not data:
                raise ConnectionAbortedError("Client closed the inference request")

        def send_response_only(self, code, message=None):
            self.response_code = code
            super().send_response_only(code, message)

        def end_headers(self):
            if getattr(self, "response_code", 200) >= 200:
                self.response_started = True
            super().end_headers()

        @property
        def scheme(self):
            return "https" if isinstance(self.connection, ssl.SSLSocket) else "http"

        @property
        def base_url(self):
            return service.config.get("public_base_url") or self.scheme + "://" + self.headers["Host"]

        def origin_allowed(self):
            origin = self.headers.get("Origin")
            if not origin:
                return True
            try:
                own = normalize_origin(self.scheme + "://" + self.headers.get("Host", ""))
            except ValueError:
                own = ""
            if origin == own or origin == service.config.get("public_base_url") or origin in service.config.get("allowed_origins", []):
                return True
            # Browser apps, including file/sandbox origins, may use inference
            # with an explicit API key. Admin routes keep the exact allowlist.
            return (service.config.get("browser_access") == "api-key" and
                    len(service.config.get("api_key", "")) >= 16 and inference_path(urlparse(self.path).path))

        def log_error(self, fmt, *args):
            if len(args) == 1 and isinstance(args[0], ReceiveTimeout):
                # Idle expiry is handled before entering the request parser.
                # Here the peer sent data but did not finish a request in time.
                super().log_message("Incomplete HTTP request: receive deadline exceeded; connection closed")
                return
            super().log_error(fmt, *args)

        def log_message(self, fmt, *args):
            # Chrome probes localhost when DevTools opens. We do not expose a
            # workspace; this exact, expected 404 does not need a server warning.
            if (getattr(self, "command", None) == "GET" and
                    urlparse(getattr(self, "path", "")).path == DEVTOOLS_DISCOVERY_PATH
                    and len(args) > 1 and str(args[1]) == "404"):
                return
            if args and str(args[1] if len(args) > 1 else "") not in {"200", "201", "202"}:
                super().log_message(fmt, *args)

        def valid_host(self):
            host = self.headers.get("Host", "")
            parsed = urlparse("http://" + host)
            canonical_host = urlparse(service.config.get("public_base_url", "")).netloc.lower()
            if (self.server.server_address[0] != "0.0.0.0" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                    and not (canonical_host and host.lower() == canonical_host)):
                self.reject_connection("Host rejected. Use localhost/127.0.0.1, or select listen address 0.0.0.0 and restart for network access.")
            origin = self.headers.get("Origin")
            if not self.origin_allowed():
                self.reject_connection("CORS origin rejected. Add this frontend origin under API & server, or select 'Any frontend with API key' for inference clients. "
                                       "Origin null comes from a local file or sandboxed page.")

        def reject_connection(self, reason):
            origin, host, path = self.headers.get("Origin", ""), self.headers.get("Host", ""), urlparse(self.path).path
            detail = reason + " Origin=" + json.dumps(origin) + "; Host=" + json.dumps(host)
            service.connection_error(origin, path, detail)
            self.close_connection = True
            super().log_message("Request rejected: %s %s | %s", self.command, path, detail)
            raise ValueError(detail)

        def authorized(self):
            path = urlparse(self.path).path
            if path in {"/", "/app.js", "/style.css", "/ui.js", "/redesign.css", "/favicon.svg", "/help.js", "/chat.js", "/client.js", "/api/health"}:
                return True
            key = service.config.get("api_key", "")
            authorization = self.headers.get("Authorization", "")
            if not key or secrets.compare_digest(authorization, "Bearer " + key):
                return True
            inference_key = service.config.get("inference_api_key", "")
            if inference_key and secrets.compare_digest(authorization, "Bearer " + inference_key):
                if inference_path(path):
                    return True
                self.reply({"error": {"message": "This route requires the owner API key", "type": "permission_error"}}, 403)
                return False
            self.reply({"error": {"message": "An API key is required", "type": "authentication_error"}}, 401)
            return False

        def cors_headers(self):
            origin = self.headers.get("Origin")
            self.send_header("Vary", "Origin, Access-Control-Request-Method, Access-Control-Request-Headers")
            if origin and self.origin_allowed():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Credentials", "true")
                self.send_header("Access-Control-Expose-Headers", "X-Job-Id, Location, Retry-After, X-Should-Retry")

        def do_OPTIONS(self):
            self._deadline_reader.finish()
            try:
                self.valid_host()
                if self.headers.get("Access-Control-Request-Method", "GET") not in {"GET", "POST", "OPTIONS"}:
                    raise ValueError("This API supports GET, POST and OPTIONS")
                requested = self.headers.get("Access-Control-Request-Headers", "")
                headers = [h.strip().lower() for h in requested.split(",") if h.strip()]
                if len(requested) > 8192 or any(not re.fullmatch(r"[!#$%&'*+.^_`|~0-9a-z-]+", h) for h in headers):
                    raise ValueError("Invalid preflight header names")
                self.send_response(204)
                self.cors_headers()
                self.send_header("Access-Control-Allow-Headers", ", ".join(sorted(set(headers + ["content-type", "authorization", "last-event-id"]))))
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Max-Age", "60")
                if self.headers.get("Origin") and self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except ValueError as exc:
                self.reply({"error": str(exc)}, 400)

        def reply(self, value, code=200, *, retry_after=None):
            if getattr(self, "response_started", False):
                # Sending a second HTTP status line after a partial response
                # corrupts the connection, including when error delivery fails.
                self.close_connection = True
                return
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            if isinstance(value, dict) and isinstance(value.get('x_retryable'), bool):
                self.send_header("X-Should-Retry", str(value['x_retryable']).lower())
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            if self.close_connection:
                self.send_header("Connection", "close")
            if getattr(self, "job_id", None):
                self.send_header("X-Job-Id", self.job_id)
            if code == 202 and isinstance(value, dict) and value.get("result_url"):
                self.send_header("Location", value["result_url"])
                self.send_header("Retry-After", "1")
            self.cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path):
            if not path.is_file():
                self.reply({"error": "Not found"}, 404)
                return
            data = path.read_bytes()
            self.send_response(200)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._deadline_reader.finish()
            try:
                self.valid_host()
                if urlparse(self.path).path == DEVTOOLS_DISCOVERY_PATH:
                    self.reply({"error": "Not found"}, 404)
                    return
                if not self.authorized():
                    return
                url = urlparse(self.path)
                path = url.path
                if path == "/api/state":
                    self.reply(service.state(summary=parse_qs(url.query).get("summary") == ["1"]))
                elif path == "/api/runtime/update":
                    self.reply(service.runtime_updater.snapshot())
                elif path == "/api/health":
                    self.reply({"ok": True, "mode": service.config["mode"], "version": __version__, "pid": os.getpid(),
                                "features": {"chat_job_idempotency": True, "image_job_idempotency": True, "image_result_ready": True,
                                             "predictable_image_urls": True, "client_disconnect_cancellation": True},
                                "links": {"artifacts": "/api/artifacts", "image_jobs": "/api/image/jobs"}})
                elif path == "/api/ready":
                    state = service.readiness(parse_qs(url.query).get("model", [None])[0])
                    self.reply(state, 200 if state["ready"] else 503, retry_after=None if state["ready"] else 1)
                elif path == '/api/artifacts':
                    origin = self.base_url
                    self.reply({"object": "list", "data": [
                        {**artifact, 'url': origin + artifact['url']} for artifact in service.store.artifacts(job_lookup=service.job)]})
                elif path == '/api/image/jobs':
                    model = parse_qs(url.query).get('model', [None])[0]
                    # Final state may be available only in memory while storage
                    # recovers. Discovery must agree with status/result routes.
                    jobs = [service.job(job['id']) for job in service.store.image_jobs(model)]
                    self.reply({"object": "list", "data": [
                        {**compat.job_receipt(job, self.base_url), 'created': job['created'], 'prompt': job['prompt'],
                         'x_result_ready': bool(job.get('artifacts')),
                         'data': compat.image_completion(self, service, job, 'url')['data'] if job.get('artifacts') else []}
                        for job in jobs]})
                elif path == "/v1/models":
                    self.reply({"object": "list", "data": [{"id": m["id"], "object": "model", "created": 0,
                        "owned_by": "openvino", "task": m["task"], "context_length": m.get("context_limit"),
                        "device": m["device"], "pipeline": m["pipeline"], **model_capabilities(m)} for m in service.config["models"] if m.get("enabled", True)]})
                elif path == "/api/folders":
                    query = parse_qs(url.query)
                    self.reply(browse(query.get("path", [None])[0]))
                elif path == "/api/downloads":
                    self.reply({"downloads": service.downloads.snapshot()})
                elif match := re.fullmatch(r"/api/sessions/([a-f0-9]{32})", path):
                    self.reply({"messages": service.store.messages(match[1])})
                elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path):
                    self.reply(service.job(match[1]))
                elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})/result", path):
                    compat.serve_job_result(self, service, match[1])
                elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})/events", path):
                    jid = match[1]
                    service.job(jid)
                    query = parse_qs(url.query)
                    after = int(self.headers.get("Last-Event-ID", query.get("after", ["0"])[0]))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.cors_headers()
                    self.end_headers()
                    self.close_connection = True
                    heartbeat = time.monotonic()
                    while True:
                        if getattr(self.server, 'closing', None) is not None and self.server.closing.is_set():
                            return
                        events = service.store.events(jid, after)
                        for event in events:
                            after = event["seq"]
                            self.wfile.write(f"id: {after}\ndata: {json.dumps(event)}\n\n".encode())
                        if events:
                            self.wfile.flush()
                        job = service.job(jid)
                        if job["status"] in TERMINAL and not service.store.events(jid, after):
                            self.wfile.write(b"event: end\ndata: {}\n\n")
                            self.wfile.flush()
                            return
                        if time.monotonic() - heartbeat >= 10:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                            heartbeat = time.monotonic()
                        time.sleep(0.02 if job.get("kind") == "chat" else 0.1)
                elif re.fullmatch(r"/artifacts/[a-f0-9]{32}\.(svg|png)", path):
                    jid = Path(path).stem
                    try:
                        job = service.job(jid)
                    except KeyError:
                        job = None
                    if job and job.get('kind') == 'image':
                        self.job_id = jid
                        receipt = compat.job_receipt(job, self.base_url)
                        expected = '/artifacts/' + jid + '.png'
                        if path != expected:
                            self.reply({'error': 'Not found'}, 404)
                            return
                        if not job.get('artifacts'):
                            # Do not expose a file while the native worker is
                            # still writing it, or label a failed job as ready.
                            if job['status'] not in TERMINAL:
                                self.reply(receipt, 202)
                            else:
                                self.reply(compat.job_error(job), compat.job_error_status(job))
                            return
                    self.send_file(service.store.directory / "artifacts" / Path(path).name)
                elif path in PROMPT_ROUTES:
                    self.send_file(PROMPT_DIRECTORY / PROMPT_ROUTES[path])
                elif path in {"/", "/app.js", "/style.css", "/ui.js", "/redesign.css", "/favicon.svg", "/help.js", "/chat.js", "/client.js"}:
                    self.send_file(WEB / ("index.html" if path == "/" else path[1:]))
                else:
                    self.reply({"error": "Not found"}, 404)
            except (ConnectionError, ssl.SSLEOFError):
                raise
            except AdmissionError as exc:
                self.reply({"error": {"message": str(exc), "type": exc.error_type, "code": exc.code},
                            "x_retryable": exc.retryable}, exc.status, retry_after=exc.retry_after)
            except KeyError as exc:
                self.reply({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc:
                self.reply({"error": str(exc)}, 400)
            except Exception as exc:
                self.reply({"error": str(exc)}, 500)

        def do_POST(self):
            runtime_request = False
            try:
                self.valid_host()
                if not self.authorized():
                    self.close_connection = True
                    return
                path = urlparse(self.path).path
                with service.lock:
                    if service.draining or service.closed:
                        self.close_connection = True
                        raise AdmissionError("Server is shutting down; retry when it is ready", "server_unavailable")
                # Count requests before parsing their body: updates must not race
                # with context preparation, model loads, settings or imports.
                if path not in {"/api/runtime/check", "/api/runtime/update"}:
                    with service.lock:
                        if service.runtime_maintenance:
                            self.close_connection = True
                            self.reply({"error": {"message": "Runtime update in progress. Retry after LazyRig restarts.",
                                                  "type": "runtime_maintenance"}, "x_retryable": True}, 503, retry_after=1)
                            return
                        service.runtime_requests += 1
                        runtime_request = True
                content_type = self.headers.get('Content-Type', '')
                media_type = content_type.split(';', 1)[0].strip().lower()
                multipart = media_type == 'multipart/form-data' and path in {'/v1/images/edits', '/api/image/edits'}
                if media_type != 'application/json' and not multipart:
                    self.close_connection = True
                    raise ValueError('Use application/json, or multipart/form-data for image edits')
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                    self.close_connection = True
                    raise ValueError("Send one Content-Length header; transfer encoding is not supported")
                if not 0 < length <= 32 * 1024 * 1024:
                    self.close_connection = True
                    raise ValueError("Request body must be 1 byte–32 MiB")
                def invalid_constant(value):
                    raise ValueError("Non-finite JSON numbers are not allowed")
                self._deadline_reader.start()
                payload = self.rfile.read(length)
                self._deadline_reader.finish()
                if len(payload) != length:
                    self.close_connection = True
                    raise ValueError("Request body ended before Content-Length bytes were received")
                body = parse_multipart(content_type, payload) if multipart else json.loads(payload, parse_constant=invalid_constant)
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                if path == "/api/runtime/check":
                    self.reply(service.runtime_updater.check(), 202)
                elif path == "/api/runtime/update":
                    receipt = service.runtime_updater.apply(body.get("check_id"))
                    try:
                        self.reply(receipt, 202)
                    finally:
                        # Even a lost HTTP receipt must let the waiting helper proceed.
                        service.runtime_updater.stop_server()
                elif path == "/v1/chat/completions":
                    compat.serve_chat(self, service, body, asynchronous=self.wants_async())
                elif path == "/v1/images/generations":
                    compat.serve_image(self, service, body, asynchronous=self.wants_async())
                elif path == '/v1/images/edits':
                    compat.serve_image(self, service, body, asynchronous=self.wants_async(), editing=True)
                elif path == '/api/image/edits':
                    compat.serve_image(self, service, body, asynchronous=True, editing=True)
                elif path == "/api/chat/jobs":
                    compat.serve_chat(self, service, body, asynchronous=True)
                elif path == "/api/image/jobs":
                    compat.serve_image(self, service, body, asynchronous=True)
                elif path == "/api/models":
                    self.reply(service.register(body), 201)
                elif path == "/api/models/inspect":
                    self.reply(inspect_export(body["path"]))
                elif path == "/api/devices/refresh":
                    self.reply(service.refresh_devices())
                elif path == "/api/server/api-key":
                    # Admin authorization/origin checks above apply. Save settings
                    # separately so generating a draft never rotates the live key.
                    self.reply({"api_key": "lr_" + secrets.token_urlsafe(32), "saved": False})
                elif match := re.fullmatch(r"/api/models/([A-Za-z0-9_.-]+)/(load|unload|remove)", path):
                    if match[2] == "remove":
                        self.reply(service.remove(match[1]))
                    else:
                        self.reply(service.control(match[1], match[2]), 202)
                elif path == "/api/downloads":
                    self.reply(service.downloads.start(body), 202)
                elif match := re.fullmatch(r"/api/downloads/([a-f0-9]{32})/cancel", path):
                    self.reply(service.downloads.cancel(match[1]))
                elif path == "/api/server":
                    self.reply(service.server_settings(body))
                elif path == "/api/sessions":
                    self.reply({"id": service.store.session(str(body.get("title", "New conversation")))}, 201)
                elif path == "/api/context":
                    self.reply(service.inspect_context(body, origin=self.base_url))
                elif path == "/api/jobs":
                    self.reply(service.submit(body), 202)
                elif path == "/api/settings":
                    self.reply(service.settings(body))
                elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path):
                    self.reply(service.cancel(match[1]))
                else:
                    self.reply({"error": "Not found"}, 404)
            except (ConnectionError, ssl.SSLEOFError):
                raise
            except AdmissionError as exc:
                self.reply({"error": {"message": str(exc), "type": exc.error_type, "code": exc.code},
                            "x_retryable": exc.retryable}, exc.status, retry_after=exc.retry_after)
            except RequestDeadline as exc:
                self.reply({"error": {"message": str(exc), "type": "request_timeout", "code": exc.code},
                            "x_retryable": exc.retryable}, exc.status)
            except ReceiveTimeout:
                self.close_connection = True
                self._deadline_reader.finish()
                self.reply({"error": {"message": "HTTP request body receive deadline exceeded", "type": "request_timeout"},
                            "x_retryable": False}, 408)
            except IdempotencyConflict as exc:
                self.job_id = exc.job_id
                self.reply({"error": {"message": str(exc), "type": "idempotency_conflict"},
                            "x_job_id": exc.job_id, "status_url": "/api/jobs/" + exc.job_id,
                            "result_url": "/api/jobs/" + exc.job_id + "/result"}, 409)
            except (KeyError, ValueError, TypeError) as exc:
                self.reply({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
            except Exception as exc:
                self.reply({"error": str(exc)}, 500)
            finally:
                if runtime_request:
                    with service.lock:
                        service.runtime_requests -= 1

        def wants_async(self):
            return "respond-async" in {p.strip().lower() for p in self.headers.get("Prefer", "").split(",")}
    return Handler

def main():
    from .terminal import configure_logging, LOGGER
    configure_logging()
    LOGGER.info('Startup: checking the runtime environment lock...')
    from .runtime_update import environment_lock, managed_environment
    # Hold a shared environment lock until all native runtime objects are gone.
    with environment_lock() if managed_environment() else nullcontext():
        run_server()


def run_server():
    from .terminal import configure_logging
    configure_logging()
    from .terminal import LOGGER
    LOGGER.info('Startup: reading configuration...')
    parser = argparse.ArgumentParser(description="LazyRig local multimodal server")
    parser.add_argument("--config", help="JSON configuration (default: OpenVINO, empty registry)")
    parser.add_argument("--data-dir", help="Directory for conversations and outputs")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0", "localhost"])
    parser.add_argument("--tls-cert", help="PEM certificate trusted by the frontend browser")
    parser.add_argument("--tls-key", help="PEM private key for the HTTPS certificate")
    parser.add_argument("--allow-origin", action="append", help="Frontend URL to allow through CORS; repeat for multiple frontends")
    parser.add_argument("--mode", choices=["openvino"], help="Require this backend; reject a conflicting configuration")
    args = parser.parse_args()
    config = load(args.config, args.data_dir)
    if args.mode and config["mode"] != args.mode:
        parser.error(f"Configuration uses {config['mode']}; this launcher requires {args.mode}.")
    config = restore_settings(config)
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host
    for key in ("tls_cert", "tls_key"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.allow_origin:
        config["allowed_origins"] = list(config.get("allowed_origins", [])) + args.allow_origin
    from .config import validate
    validate(config)
    try:
        tls = tls_context(config)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot configure HTTPS: {exc}\nCheck --tls-cert and --tls-key (PEM files).\n")
    try:
        # Reserve the port before opening persistent state or recovering old jobs.
        server = BoundedHTTPServer((config.get("host", "127.0.0.1"), config.get("port", 8765)), handler(None),
                                   max_connections=config["max_connections"], http_timeout=config["http_timeout_seconds"], tls=tls)
    except OSError as exc:
        parser.exit(1, f"Cannot open port {config.get('port', 8765)}: {exc}\nClose the old server window, then run start-server.bat again, or use --port 8766 with a different --data-dir.\n")
    try:
        service = Service(config, restore=False)
        service.runtime_updater.shutdown = server.shutdown
        service.runtime_updater.restart = [sys.executable, '-m', 'lazyrig', *sys.argv[1:]]
        service.runtime_updater.cwd = os.getcwd()
        server.RequestHandlerClass = handler(service)
    except BaseException:
        server.server_close()
        raise
    print(f"LazyRig {config['mode']} mode: {'https' if tls else 'http'}://127.0.0.1:{server.server_port}", flush=True)
    stopping = threading.Event()
    def stop(signum, frame):
        if stopping.is_set():
            return
        stopping.set()
        service.begin_drain()
        # HTTPServer.shutdown must run outside the serve_forever thread.
        threading.Thread(target=server.shutdown, name="lazyrig-shutdown", daemon=True).start()
    previous_term = signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        service.begin_drain()
        server.server_close()
        server.close_connections()
        server.wait_for_connections()
        service.close()

if __name__ == "__main__":
    main()
