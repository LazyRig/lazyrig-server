import copy
import json
import secrets
import logging
import re
import threading
import time
from pathlib import Path
from . import __version__
from .config import validate, restore_settings
from .context import DEFAULT_OUTPUT_TOKENS, TokenizerService, context_info
from .chat_images import capture_sources, image_tools, parse_action
from .prompt_enhancement import enabled as enhancement_enabled, run as enhance_image_prompt
from .scheduler import Scheduler, PlacementAdmissionError
from .storage import Store, new_id, STREAM_JOB_EVENTS
from .worker import Cancelled, NativeWorkerError
from .worker_status import WorkerProcessError
from .terminal import Reporter
from .catalog import inspect_export
from .memory import detected_budget, device_reservation, MemoryPressureError
from .downloads import DownloadManager
from .tool_calls import ToolCallError, parse_tool_calls
from .request_options import template_model
from .reasoning import split_reasoning, assistant_message
from .response_format import ResponseFormatError, check_json_output, is_json
from .runtime_settings import CONTROLS
from .vision import ImageStore, is_vlm
from .image_edit import prepare as prepare_image_edit
from .idempotency import IdempotencyConflict
from .image_sizes import validate_dimensions, is_qwen_image_21
from .image_options import NATIVE_OPTIONS, validate_options as validate_image_options
from .mtp import model_capabilities, request_options as mtp_request, enabled as mtp_enabled, ensure_export as ensure_mtp_export
from .sampling import QWEN_PRESETS, WORKER_FIELDS as SAMPLING_WORKER_FIELDS, validate as validate_sampling
from .admission import admission, commit as commit_admission, AdmissionError
from .deadlines import DeadlineEvent, RequestDeadline, deadline_context
from .device_info import discover_devices, canonical_device, preferred_device
from . import placement

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _record_failure(job, exc):
    """Use the same saved diagnostics for explicit Load and generation jobs."""
    job.update(status='failed', phase='Failed', error=str(exc) or type(exc).__name__)
    if getattr(exc, 'worker_diagnostics', None):
        job['worker_diagnostics'] = copy.deepcopy(exc.worker_diagnostics)
    if isinstance(exc, WorkerProcessError):
        job.update(phase='Model worker exited' if exc.worker_exit['exit_observed'] else 'Worker connection closed',
                   error_code=exc.error_code, worker_exit=copy.deepcopy(exc.worker_exit))
        if exc.worker_stage == 'validation':
            job['phase'] = 'Split inference check failed'
    elif isinstance(exc, PlacementAdmissionError):
        job.update(phase='Loading blocked by device budget', error_code=exc.error_code)
        if exc.weight_check:
            job['weight_check'] = copy.deepcopy(exc.weight_check)
    elif isinstance(exc, MemoryPressureError):
        job.update(phase='Loading stopped to preserve system RAM', error_code='host_memory_pressure')
    elif isinstance(exc, NativeWorkerError) and getattr(exc, 'error_code', None) == 'attention_cache_budget_exceeded':
        job.update(phase='Attention-cache settings rejected', error_code=exc.error_code,
                   cache_settings=copy.deepcopy(exc.cache_settings))
    elif isinstance(exc, NativeWorkerError) and getattr(exc, 'error_code', None) == 'gpu_kernel_build_failed':
        job.update(phase='GPU kernel compilation failed', error_code=exc.error_code)
    elif isinstance(exc, NativeWorkerError) and getattr(exc, 'worker_stage', None) == 'validation':
        job.update(phase='Split inference check failed', error_code='split_inference_failed')
    elif (isinstance(exc, NativeWorkerError) and getattr(exc, 'placement_preflight', None)
          and (job.get('placement_events') or [{}])[-1].get('kind') == 'placement_rejected'):
        job.update(phase='Manual device split rejected', error_code='manual_split_rejected')
    if (isinstance(exc, PlacementAdmissionError) and len(job.get('placement_attempts', [])) > 1
            and any(a.get('status') == 'failed' for a in job.get('placement_attempts', []))):
        job['phase'] = 'No permitted placement could load; see load attempts'


class Service:
    def __init__(self, config, restore=True, *, worker_factory=None, tokenizer_factory=TokenizerService):
        self.config = restore_settings(config) if restore else copy.deepcopy(config)
        self.store = Store(config["data_dir"])
        self.lock = threading.RLock()
        self.running = {}
        self.active_jobs = {}
        self._reservations = 0
        self._preparing = 0
        self._key_locks = {}
        self._preparation_signals = set()
        self._storage_error = None
        self._pending_terminal = {}
        self._last_compaction = time.monotonic()
        self.draining = False
        self.closed = False
        self.runtime_maintenance = False
        self.runtime_requests = 0
        self.settings_path = self.store.directory / "settings.json"
        try:
            self.scheduler = Scheduler(self.config, **({"worker_factory": worker_factory} if worker_factory else {}))
            self.images = ImageStore(self.store.directory)
            self.tokens = tokenizer_factory(self.config["mode"], self.images)
            logging.getLogger('lazyrig.runtime').info('Startup: detecting OpenVINO runtime and devices...')
            self.detected = self.discover()
            self.downloads = DownloadManager(self)
            from .runtime_update import RuntimeUpdater
            self.runtime_updater = RuntimeUpdater(self)
            self.maintenance = threading.Thread(target=self._idle_loop, daemon=True)
            self.maintenance.start()
        except BaseException:
            self.store.close()
            raise

    def admission(self, timeout=None, signal=None):
        return admission(self, timeout=timeout, signal=signal)

    def _check_admission(self):
        if self.closed or self.draining or self.runtime_maintenance:
            raise AdmissionError('Server is draining, shutting down or under runtime maintenance', 'server_unavailable')
        health = self.store.health() if hasattr(self.store, 'health') else {'ok': True}
        if self._storage_error or not health['ok']:
            raise AdmissionError('Durable storage is unavailable; new jobs are paused until a write probe succeeds', 'storage_unhealthy')
        minimum = self.config.get('min_free_disk_mib', 256) * 1024 * 1024
        free = health.get('free_disk_bytes')
        if minimum and (free is None or free < minimum):
            raise AdmissionError('Insufficient available disk space for safe job admission; free space or adjust min_free_disk_mib', 'disk_pressure')

    def begin_drain(self):
        with self.lock:
            self.draining = True

    def job(self, jid):
        # Accepted synchronous clients must receive a terminal result even if
        # the final database write fails. These snapshots are flushed on recovery.
        with self.lock:
            pending = self._pending_terminal.get(jid)
            return copy.deepcopy(pending) if pending is not None else self.store.job(jid)

    def readiness(self, model_id=None):
        with self.lock:
            storage = self.store.health() if hasattr(self.store, 'health') else {'ok': self._storage_error is None}
            reasons = []
            if self.closed or self.draining:
                reasons.append('server_draining')
            if self.runtime_maintenance:
                reasons.append('runtime_maintenance')
            if not self.detected.get('available'):
                reasons.append('runtime_unavailable')
            if self._storage_error or not storage['ok']:
                reasons.append('storage_unhealthy')
            minimum = self.config.get('min_free_disk_mib', 256) * 1024 * 1024
            if minimum and (storage.get('free_disk_bytes') is None or storage['free_disk_bytes'] < minimum):
                reasons.append('disk_pressure')
            with self.scheduler.condition:
                failed = [mid for mid, state in self.scheduler.states.items() if state == 'cleanup-failed']
            if failed:
                reasons.append('worker_cleanup_failed')
            if model_id:
                try:
                    self.model(model_id)
                except ValueError:
                    reasons.append('model_unavailable')
            full = len(self.running) + self._reservations >= self.config.get('max_jobs', 8)
            if full:
                reasons.append('queue_full')
            if self._preparing >= self.config.get('max_preparing', 2):
                reasons.append('preparation_full')
            return {'ready': not reasons, 'reasons': reasons, 'runtime_available': bool(self.detected.get('available')),
                    'storage': storage, 'cleanup_failed_models': failed,
                    'admission': {'active_and_queued': len(self.running), 'preparing': self._preparing,
                                  'reserved': self._reservations, 'max_jobs': self.config.get('max_jobs', 8),
                                  'max_preparing': self.config.get('max_preparing', 2)}}

    def discover(self):
        try:
            import openvino as ov
            core = ov.Core()
            devices = discover_devices(core)
            import openvino_genai as genai
            known = {d['id']: d for d in self.config['devices']}
            for configured in known.values():
                configured['detected'] = False
            for d in devices:
                if d['id'] in known:
                    # Detected capacity is metadata, not permission to overwrite
                    # the operator's usable budget or residency policy.
                    known[d['id']].update(d)
                else:
                    budget = detected_budget(core, d['id'])
                    self.config["devices"].append({**d, "budget_gib": budget, "policy": "budget"})
            detected = {d['id'] for d in devices}
            for model in self.config.get('models', []):
                primary = canonical_device(model['device'], self.config['devices'])
                if (model.get('task') == 'text-generation' and 'placement' not in model
                        and (primary == 'GPU' or primary.startswith('GPU.'))
                        and primary in detected and not placement.has_managed_properties(model)):
                    model['device'] = primary
                    model['placement'] = placement.default_placement(model, self.config)
            return {"devices": devices, "available": True, "recommended_device": preferred_device(self.config['devices']), "note": "Detected by OpenVINO Core", "openvino_version": ov.__version__, "genai_version": getattr(genai, "__version__", "unknown")}
        except Exception as exc:
            return {"devices": [], "available": False, "note": "OpenVINO runtime unavailable: " + str(exc)}

    def state(self, summary=False):
        with self.lock:
            jobs = self.store.jobs(summary=summary)
            jobs = [copy.deepcopy(self._pending_terminal.get(job['id'], job)) for job in jobs]
            if summary:
                # Dashboard polling needs status, not entire prompts, prepared
                # token IDs or conversation histories. Full records stay at /jobs/ID.
                fields = {'id', 'kind', 'model', 'status', 'phase', 'created', 'updated',
                          'context', 'timings', 'progress', 'error', 'artifacts', 'durable', 'storage_error',
                          'placement_plan', 'placement_attempts'}
                jobs = [{**{k: v for k, v in j.items() if k in fields},
                         'prompt': j.get('prompt', '')[:160],
                         'request': {k: v for k, v in j.get('request', {}).items()
                                     if k in {'max_tokens', 'width', 'height'}}} for j in jobs]
            with self.scheduler.condition:
                effective = {mid: copy.deepcopy(model.get('_placement'))
                             for mid, (model, _) in self.scheduler.resident.items()}
            models = [{**copy.deepcopy(m), **model_capabilities(m),
                       'placement_info': {**placement.info(m, self.config), 'effective': effective.get(m['id'])},
                       'effective_placement': effective.get(m['id'])} for m in self.config['models']]
            return {"version": __version__, "mode": self.config["mode"],
                    "models": models, "hardware": self.detected,
                    "runtime_controls": copy.deepcopy(CONTROLS),
                    "sampling_presets": copy.deepcopy(QWEN_PRESETS),
                    "server": {**{k: self.config.get(k) for k in ("host", "port", "models_dir", "load_timeout_seconds", "request_timeout_seconds", "download_max_gib", "allowed_origins", "tls_cert", "tls_key", "browser_access", "terminal_metrics", "max_jobs", "max_preparing", "max_connections", "http_timeout_seconds", "public_base_url", "min_free_disk_mib", "event_retention_days")},
                               "connection_errors": copy.deepcopy(getattr(self, "connection_errors", [])),
                               "api_key_set": bool(self.config.get("api_key")), "inference_api_key_set": bool(self.config.get("inference_api_key")), "restart_required": getattr(self, "restart_required", False)},
                    "downloads": self.downloads.snapshot(),
                    "scheduler": self.scheduler.snapshot(), "sessions": self.store.sessions(),
                    "jobs": jobs, "readiness": self.readiness(), "capabilities": {"text-generation": True, "text-to-image": True,
                    "image-input": True, "image-editing": True}}

    def refresh_devices(self):
        with self.lock:
            if self.running or self._preparing:
                raise ValueError("Wait for active jobs before refreshing devices")
            self.detected = self.discover()
            self._persist()
            return self.state()

    def model(self, mid, task=None):
        m = next((m for m in self.config["models"] if m["id"] == mid), None)
        if not m or not m.get("enabled", True) or (task and m["task"] != task):
            raise ValueError("Unknown model or incompatible task")
        return copy.deepcopy(m)

    def settings(self, body):
        with self.lock:
            if self.running or self._preparing:
                raise ValueError("Wait for active or queued jobs before changing model settings")
            candidate = copy.deepcopy(self.config)
            mid = body.get("model")
            if mid:
                m = next((m for m in candidate["models"] if m["id"] == mid), None)
                if m is None:
                    raise ValueError("Unknown model")
                previous_device = m['device']
                for key in ("name", "device", "placement", "path", "pipeline", "memory_gib", "context_limit", "generation", "enabled", "default_image_tool", "plugin_config", "scheduler_config", "template_context", "mtp", "idle_timeout_seconds", "max_width", "max_height", "keep_image_model_loaded", "restore_text_after_image", "diffusion_scheduler"):
                    if key in body:
                        m[key] = body[key]
                m['device'] = canonical_device(m['device'], candidate['devices'])
                if body.get('default_image_tool') is True:
                    for other in candidate['models']:
                        if other['id'] != mid and other.get('default_image_tool'):
                            other['default_image_tool'] = False
                if m['device'] != previous_device and 'memory_gib' not in body:
                    m['memory_gib'] = device_reservation(candidate['devices'], m['device'])
                if m['device'] != previous_device and 'placement' not in body:
                    if placement.has_managed_properties(m):
                        m.pop('placement', None)
                    else:
                        m['placement'] = placement.default_placement(m, candidate)
                placement.require_automatic_allocation(m)
                if "path" in body and self.config["mode"] == "openvino":
                    m["inspection"] = inspect_export(m["path"])
                    m["runtime_profile"] = m["inspection"].get("runtime_profile", "default")
            else:
                d = next((d for d in candidate["devices"] if d["id"] == body.get("device")), None)
                if d is None:
                    raise ValueError("Unknown device")
                for key in ("budget_gib", "policy", "inference_threads"):
                    if key in body:
                        d[key] = body[key]
            validate(candidate)
            if mid and candidate["mode"] == "openvino":
                ensure_mtp_export(m)
            self.scheduler.reconfigure(candidate, lambda *args, **kwargs: None)
            self.config = candidate
            self.scheduler.config = candidate
            self.tokens.tokenizers.clear()
            self._persist()
        return self.state()

    def _persist(self):
        temp = self.settings_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        temp.replace(self.settings_path)

    def server_settings(self, body):
        if {"default_auto_image", "default_image_model"} & body.keys():
            raise ValueError("Global image routing settings were removed. Select an image tool per chat or use the image endpoints.")
        with self.lock:
            candidate = copy.deepcopy(self.config)
            for key in ("host", "port", "models_dir", "load_timeout_seconds", "request_timeout_seconds", "download_max_gib", "api_key", "inference_api_key", "allowed_origins", "tls_cert", "tls_key", "browser_access", "terminal_metrics", "max_jobs", "max_preparing", "max_connections", "http_timeout_seconds", "public_base_url", "min_free_disk_mib", "event_retention_days"):
                if key in body:
                    candidate[key] = body[key]
            validate(candidate)
            self.restart_required = any(candidate.get(k, "") != self.config.get(k, "") for k in ("host", "port", "tls_cert", "tls_key", "max_connections", "http_timeout_seconds")) or getattr(self, "restart_required", False)
            self.config.update(candidate)
            self._persist()
        return self.state()

    def connection_error(self, origin, path, reason):
        with self.lock:
            self.connection_errors = (getattr(self, "connection_errors", []) + [{
                "origin": origin[:512], "path": path[:256], "reason": reason, "time": time.time()}])[-20:]

    def register(self, body, from_download=False):
        with self.lock:
            if (self.running or self._preparing) and not from_download:
                raise ValueError("Wait for current jobs before editing the model registry")
            report = inspect_export(body["path"])
            device = canonical_device(body.get('device') or preferred_device(
                self.config['devices'], report.get('size_bytes', 0) / 1024**3), self.config['devices'])
            if not device:
                raise ValueError('No available device. Refresh devices before registering a model.')
            model = {"id": body["id"], "name": body.get("name") or body["id"], "path": report["path"],
                     "task": report["task"], "pipeline": body.get("pipeline") or report["pipeline"],
                     "device": device, "context_limit": int(body.get("context_limit", min(32768, int(report["declared_context_limit"] or 32768)))),
                     "memory_gib": float(body['memory_gib']) if 'memory_gib' in body else device_reservation(self.config['devices'], device),
                     "max_width": 1024, "max_height": 1024,
                     "generation": {}, "plugin_config": {}, "template_context": {}, "scheduler_config": {},
                     "enabled": True, "idle_timeout_seconds": 0, "inspection": report}
            model["runtime_profile"] = report.get("runtime_profile", "default")
            if is_qwen_image_21(model):
                model.update(max_width=2752, max_height=2752)
                model["generation"] = {"width": 2048, "height": 2048, "steps": 40, "seed": -1}
            for key in ("placement", "generation", "plugin_config", "template_context", "scheduler_config", "mtp", "idle_timeout_seconds", "keep_image_model_loaded", "restore_text_after_image", "max_width", "max_height", "diffusion_scheduler"):
                if key in body:
                    model[key] = body[key]
            if 'placement' not in model and not placement.has_managed_properties(model):
                model['placement'] = placement.default_placement(model, self.config)
            placement.require_automatic_allocation(model)
            candidate = copy.deepcopy(self.config)
            candidate["models"].append(model)
            validate(candidate)
            if candidate["mode"] == "openvino":
                ensure_mtp_export(model)
            self.config["models"] = candidate["models"]
            self._persist()
            return model

    def remove(self, mid):
        with self.lock:
            if self.running or self._preparing:
                raise ValueError("Wait for current jobs before removing a model")
            if not any(m["id"] == mid for m in self.config["models"]):
                raise ValueError("Unknown model")
            self.scheduler.unload(mid, lambda *a, **kw: None)
            self.config["models"] = [m for m in self.config["models"] if m["id"] != mid]
            self._persist()
            return {"ok": True, "files_deleted": False}

    def _idle_loop(self):
        while not self.closed:
            time.sleep(1)
            with self.lock:
                if not self.closed and (self._storage_error or (hasattr(self.store, 'health') and not self.store.health()['ok'])):
                    try:
                        if self.store.health(probe=True)['ok']:
                            for jid, job in list(self._pending_terminal.items()):
                                durable = {**job, 'durable': True}
                                durable.pop('storage_error', None)
                                if durable.get('error_code') == 'storage_failure' and durable['status'] == 'completed':
                                    durable.pop('error_code', None)
                                self.store.save_job(durable)
                                self.store.event(jid, 'finished', status=job['status'], error=job['error'])
                                self._pending_terminal.pop(jid, None)
                            self._storage_error = None
                    except Exception:
                        pass  # Keep admissions closed until an actual write succeeds.
                if self.closed or self.running or self._preparing or self.runtime_maintenance:
                    continue
                if time.monotonic() - self._last_compaction >= 60:
                    self._last_compaction = time.monotonic()
                    self._compact_old_events()
                for mid, (model, worker) in list(self.scheduler.resident.items()):
                    if self.scheduler.states.get(mid) == 'cleanup-failed':
                        continue  # Explicit unload / next request retries it.
                    timeout = model.get("idle_timeout_seconds", 0)
                    if timeout and time.monotonic() - self.scheduler.last_used.get(mid, time.monotonic()) >= timeout:
                        try:
                            self.scheduler.unload(mid, lambda *a, **kw: None, reason=f'idle timeout ({timeout}s)')
                        except Exception as exc:
                            logging.getLogger(__name__).warning('Idle unload failed for %s: %s', mid, exc)

    def _compact_old_events(self):
        days = self.config.get('event_retention_days', 0)
        if not days:
            return {'jobs': 0, 'text_events_removed': 0}
        try:
            return self.store.compact_events(time.time() - days * 86400, limit=20)
        except Exception as exc:
            self._storage_error = str(exc)
            logging.getLogger(__name__).warning('Event compaction failed; admissions paused: %s', exc)
            return {'jobs': 0, 'text_events_removed': 0, 'error': str(exc)}

    def control(self, mid, action):
        if action not in {"load", "unload"}:
            raise ValueError("Unknown model action")
        model = self.model(mid)
        job = self._new_job(None, mid, action, {"action": action})
        def run(cancel, emit):
            with self.scheduler.lease_models([model], cancel):
                if action == "load":
                    self.scheduler.ensure(model, cancel, emit)
                else:
                    self.scheduler.unload(mid, emit, reason='explicit unload request')
        return self._launch(job, run)

    def _new_job(self, sid, mid, prompt, request):
        return {"id": new_id(), "session_id": sid, "model": mid, "prompt": prompt, "request": request,
                "status": "queued", "phase": "Waiting for device", "created": time.time(), "updated": time.time(),
                "text": "", "artifacts": [], "context": None, "error": None, "mode": self.config["mode"]}

    def _launch(self, job, operation, retry_identity=None):
        with self.admission() as reservation:
            return self._launch_reserved(job, operation, reservation, retry_identity)

    def _launch_reserved(self, job, operation, reservation, retry_identity=None):
        reporter = Reporter(self.config, job)
        with self.lock:
            self._check_admission()
            cancel = reservation.signal
            cancel.check()
            job['deadline_seconds'] = self.config.get('request_timeout_seconds', 1800)
            def emit(kind, **data):
                reporter.event(kind, **data)
                if kind in {"loading", "loaded", "unloading", "unloaded"}:
                    job["phase"] = kind + " " + data.get("model", "")
                self._placement_event(job, kind, data)
                if kind == "progress":
                    job["progress"] = data
                job["updated"] = time.time()
                if kind in STREAM_JOB_EVENTS:
                    self.store.event(job["id"], kind, job_update=job, **data)
                else:
                    self.store.event(job["id"], kind, **data)
                    self.store.save_job(job)
            def target():
                try:
                    with deadline_context(cancel):
                        cancel.check()
                        job["status"] = "running"
                        operation(cancel, emit)
                        cancel.check()
                except Cancelled:
                    job.update(status="cancelled", phase="Cancelled", error="Stopped by client. Saved artifacts are retained.")
                except RequestDeadline as exc:
                    job.update(status="failed", phase="Request deadline exceeded", error=str(exc), error_code='request_timeout')
                except Exception as exc:
                    _record_failure(job, exc)
                finally:
                    self._finish_job(job, cancel, reporter)
            thread = threading.Thread(target=target, daemon=True)
            # Persist the job and retry identity together before inference can start.
            try:
                self.store.save_job(job, retry_identity=retry_identity)
            except IdempotencyConflict:
                raise
            except Exception as exc:
                self._storage_error = str(exc)
                raise AdmissionError('Could not persist the job; durable storage is unavailable', 'storage_unhealthy') from exc
            self.running[job["id"]] = (cancel, thread)
            self.active_jobs[job["id"]] = job
            commit_admission(reservation)
            try:
                thread.start()
            except BaseException as exc:
                job.update(status='failed', phase='Failed to start job', error=str(exc))
                self._finish_job(job, cancel, reporter)
                raise
            return copy.deepcopy(job)

    def image_parameters(self, model, body):
        generation = {**model.get("generation", {}), **body}
        width, height = validate_dimensions(model, generation)
        steps, seed = int(generation.get("steps", 20)), int(generation.get("seed", -1))
        if not 1 <= steps <= 100 or not -1 <= seed <= 2147483647:
            raise ValueError("Invalid image steps or seed")
        # Resolve random seeds once, before persisting the accepted job. Polls,
        # reconnects and idempotent retries reuse this job and its actual seed.
        if seed == -1:
            seed = secrets.randbelow(2147483648)
        params = {"width": width, "height": height, "steps": steps, "seed": seed}
        for key in NATIVE_OPTIONS:
            if key in generation:
                params[key] = generation[key]
        validate_image_options(model, params)
        return params

    @staticmethod
    def _placement_event(job, kind, data):
        if kind == 'placement_plan':
            job['placement_plan'] = {k: copy.deepcopy(v) for k, v in data.items() if k != 'attempts'}
            job['placement_attempts'] = copy.deepcopy(data.get('attempts', []))[:32]
            return True
        if kind == 'placement_attempt':
            if job.get('placement_plan', {}).get('plan_id') == data.get('plan_id'):
                attempt = next((a for a in job.get('placement_attempts', []) if a['attempt'] == data['attempt']), None)
                if attempt is not None:
                    attempt.update(copy.deepcopy(data))
            return True
        if kind == 'load_memory':
            events = job.setdefault('load_memory_events', [])
            events.append(copy.deepcopy(data))
            del events[:-32]
            return True
        if kind not in {'placement_selected', 'placement_fallback', 'placement_rejected'}:
            return False
        label = data.get('label') or data.get('target') or data.get('to_target') or data.get('model', '')
        prefix = {'placement_selected': 'Using device placement: ',
                  'placement_fallback': 'Trying another device placement: ',
                  'placement_rejected': 'Manual device split rejected: '}[kind]
        job['phase'] = prefix + str(label)
        job.setdefault('placement_events', []).append({'kind': kind, **copy.deepcopy(data)})
        return True

    def _image_worker(self, job, model, request, cancel, emit):
        """Run one image operation, with one fresh-worker GPU resource retry.

        The caller holds the device lease, resolves the seed/enhancement once,
        and publishes the artifact only after this returns. Never replay those
        surrounding operations, or extend the accepted request's deadline.
        """
        output = Path(request['output_path'])
        recovery = None
        try:
            for attempt in range(2):
                if cancel.is_set():
                    raise Cancelled()
                worker = self.scheduler.ensure(model, cancel, emit)
                with self.scheduler.condition:
                    loaded = self.scheduler.resident.get(model['id'])
                    actual = loaded[0] if loaded else model
                    gpu_worker = any(device == 'GPU' or device.startswith('GPU.')
                                     for device in placement.devices(actual))
                job['phase'] = 'Editing image' if request['action'] == 'image_edit' else 'Generating image'
                job['progress'] = None
                emit('image_started', model=model['id'])
                received_result = False
                try:
                    for item in worker.run(copy.deepcopy(request), cancel.remaining(self.config.get('request_timeout_seconds', 3600)), cancel):
                        if item['kind'] == 'progress':
                            job['progress'] = {'step': item['step'], 'steps': item['steps']}
                            emit('progress', **job['progress'])
                        elif item['kind'] == 'result':
                            received_result = True
                            if item.get('inference'):
                                job['image_inference'] = item['inference']
                except NativeWorkerError as exc:
                    code = re.search(r'\b(CL_OUT_OF_RESOURCES|CL_MEM_OBJECT_ALLOCATION_FAILURE)\b', str(exc))
                    if attempt and code and gpu_worker:
                        job['error_code'] = 'gpu_resource_error'
                    if (attempt or not code or not gpu_worker
                            or received_result or output.exists() or job.get('artifacts') or job.get('result_ready')):
                        raise
                    if cancel.is_set():
                        raise Cancelled()
                    recovery = {'retry_count': 1, 'max_retries': 1, 'reason': code.group(1),
                                'first_error': str(exc), 'status': 'retrying'}
                    job['image_recovery'] = recovery
                    job['progress'] = None
                    job.pop('image_inference', None)
                    job['phase'] = 'GPU resource error · restarting image worker · retry 1/1'
                    emit('image_retry', model=model['id'], recovery=copy.deepcopy(recovery), message=job['phase'])
                    # ProcessWorker closes on native errors; unload additionally
                    # verifies cleanup and removes the resident before replacement.
                    # Any failure here aborts recovery, retaining cleanup ownership.
                    self.scheduler.unload(model['id'], emit, reason='GPU resource error: automatic image retry 1/1')
                    continue
                if not output.is_file():
                    raise RuntimeError('Image worker returned no file')
                if recovery:
                    recovery['status'] = 'succeeded'
                    emit('image_recovered', model=model['id'], recovery=copy.deepcopy(recovery))
                return
        except BaseException:
            if recovery and recovery['status'] != 'succeeded':
                recovery['status'] = 'cancelled' if cancel.cancelled else 'failed'
            raise

    def submit_image(self, body, editing=False, origin='', retry_identity=None, request_id=None, background=True):
        with self.admission():
            return self._submit_image(body, editing, origin, retry_identity, request_id, background)

    def _submit_image(self, body, editing=False, origin='', retry_identity=None, request_id=None, background=True):
        model = self.model(body["model"], "text-to-image")
        enhance = enhancement_enabled(body)
        enhancer = None
        if enhance:
            if not body.get('enhancement_model'):
                raise ValueError('Select an enhancement_model before enabling prompt enhancement')
            enhancer = self.model(body['enhancement_model'], 'text-generation')
            if editing and not is_vlm(enhancer):
                raise ValueError('Edit prompt enhancement requires a vision-language model to inspect the source image')
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 8000:
            raise ValueError("Image prompt must contain 1–8000 characters")
        params = self.image_parameters(model, body)
        notes = []
        if editing:
            inputs, notes = prepare_image_edit(self.images, model, body, origin)
            params.update(inputs)
            # Same registry alias, but a distinct native pipeline. Scheduler
            # comparison closes the old worker before loading the new mode.
            model['_image_operation'] = 'edit'
        job = self._new_job(None, model["id"], prompt, params)
        if request_id is not None:
            job['id'] = request_id
            # A client-selected ID must never overwrite an older artifact,
            # including one whose original database record is unavailable.
            if any((self.store.directory / 'artifacts' / (request_id + ext)).exists() for ext in ('.png', '.svg')):
                raise IdempotencyConflict(request_id)
        job["kind"] = "image"
        job['background'] = background
        job['operation'] = 'edit' if editing else 'generate'
        job['parameter_notes'] = notes
        job["response_format"] = body.get("response_format", "url")
        job['enhance_prompt'] = enhance
        def run(cancel, emit):
            models = [model] + ([enhancer] if enhancer else [])
            with self.scheduler.lease_models(models, cancel, restore=model.get('restore_text_after_image', False)) as previous:
                # The scheduler snapshots residents and leases every device in
                # their placement before any image displacement/restoration.
                if enhancer and not any(set(placement.devices(m)) & set(placement.devices(enhancer)) for m in previous):
                    previous.append(enhancer)
                succeeded = False
                try:
                    effective_prompt, effective_params = prompt, params
                    if enhance:
                        effective_prompt, effective_params = enhance_image_prompt(self, job, model, enhancer, prompt, params,
                            editing, body.get('image_size_explicit', False), cancel, emit)
                        job['request'].update(effective_params)
                    if cancel.is_set():
                        raise Cancelled()
                    ext = ".png"
                    output = self.store.directory / "artifacts" / (job["id"] + ext)
                    self._image_worker(job, model, {"action": "image_edit" if editing else "image", "prompt": effective_prompt,
                        "output_path": str(output), **effective_params}, cancel, emit)
                    artifact = {"url": "/artifacts/" + output.name, "name": output.name, "prompt": effective_prompt,
                                "simulated": False, 'operation': job['operation'],
                                **{key: value for key, value in effective_params.items() if key != 'input_images'}}
                    if editing:
                        artifact['source_image_sha256'] = params['input_images'][0]['sha256']
                    if job.get('image_inference'):
                        artifact['inference'] = job['image_inference']
                    job["artifacts"].append(artifact)
                    job['result_ready'] = True
                    job['phase'] = 'Image saved; finishing model cleanup'
                    emit("artifact", artifact=artifact)
                    succeeded = True
                finally:
                    restore_previous = previous if model.get('restore_text_after_image', False) else []
                    keep_image = self.scheduler.retain_image(model, restore_previous)
                    if cancel.cancelled or cancel.expired or not succeeded or not keep_image:
                        reason = ('image job stopped or failed' if cancel.cancelled or cancel.expired or not succeeded else
                                  'image retention disabled' if model.get('keep_image_model_loaded') is False else
                                  'automatic image retention: required models do not fit together on the device')
                        self.scheduler.unload(model["id"], emit, reason=reason)
                    if model.get('restore_text_after_image', False) and previous and not (cancel.cancelled or cancel.expired):
                        for old in previous:
                            self.scheduler.ensure(old, cancel, emit)
                    if succeeded and keep_image and model['id'] in self.scheduler.resident:
                        self.scheduler.last_used[model["id"]] = time.monotonic()
        return self._launch(job, run, retry_identity=retry_identity)

    def inspect_context(self, body, origin=''):
        with self.admission():
            return self._inspect_context(body, origin)

    def _inspect_context(self, body, origin=''):
        model = template_model(self.model(body["model"], "text-generation"), body)
        budget = int(body.get("max_completion_tokens", body.get("max_tokens", model.get("generation", {}).get("max_tokens", DEFAULT_OUTPUT_TOKENS))))
        if not 1 <= budget <= model["context_limit"]:
            raise ValueError("Invalid output token budget")
        if "messages" in body:
            from .compat import validate_chat
            return validate_chat(self, body, origin=origin)[-1]['context']
        history = self.store.messages(body["session_id"])
        draft = body.get("draft", "")
        if draft:
            history.append({"role": "user", "content": draft})
        return self.tokens.prepare(model, history, budget, body.get("tools_enabled", True))["context"]

    def submit(self, body, captured=False, background=True, retry_identity=None, request_id=None):
        with self.admission() as reservation:
            return self._submit_reserved(body, reservation, captured, background, retry_identity, request_id)

    def _submit_reserved(self, body, reservation, captured=False, background=True, retry_identity=None, request_id=None):
        sid = body["session_id"]
        history = self.store.messages(sid)
        with self.lock:
            if any(job.get('session_id') == sid for job in self.active_jobs.values()):
                raise ValueError("This conversation already has an active job")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500000:
            raise ValueError("Enter a prompt of 1–500000 characters")
        llm = self.model(body["model"], "text-generation")
        body = {**llm.get("generation", {}), **body}
        validate_sampling(body)
        prompt_model = template_model(llm, body)
        if is_json(prompt_model["response_format"]) and (body.get("tools_enabled", True) or body.get("tools")):
            raise ValueError("JSON response_format requires tools_enabled=false and no active tools")
        image = self.model(body["image_model"], "text-to-image") if body.get("tools_enabled", True) else None
        enhance = enhancement_enabled(body)
        if enhance and image is None:
            raise ValueError('Prompt enhancement requires an enabled image tool')
        budget = int(body.get("max_tokens", DEFAULT_OUTPUT_TOKENS))
        if not 1 <= budget <= int(llm["context_limit"]):
            raise ValueError("Invalid output token budget")
        image_params = self.image_parameters(image, body) if image else {}
        applied_mtp = mtp_request(llm, body)
        temperature = float(body.get("temperature", 0 if mtp_enabled(llm) else 0.7))
        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be 0–2")
        # Preflight runs in the queued job so tokenizer compilation never blocks HTTP.
        jid = request_id or new_id()
        job = {"id": jid, "session_id": sid, "status": "queued", "created": time.time(), "updated": time.time(),
               "prompt": prompt, "model": llm["id"], "image_model": image["id"] if image else None,
               "context": None, "text": "", "artifacts": [], "error": None, "phase": "Waiting for device",
               "request": {"max_tokens": budget, "temperature": temperature, **image_params},
               "image_request": image_params, "mode": self.config["mode"], "kind": "chat", "background": background}
        job["generation"] = {key: body[key] for key in SAMPLING_WORKER_FIELDS if key in body}
        job["mtp"] = applied_mtp
        if applied_mtp["enabled"]:
            job["generation"]["num_assistant_tokens"] = applied_mtp["num_assistant_tokens"]
        job["external"] = body.get("external", False)
        job["tools"] = body.get("tools")
        job["template_context"] = prompt_model["template_context"]
        job["response_format"] = prompt_model["response_format"]
        job["timings_per_token"] = body.get("timings_per_token", False)
        job["input_message"] = body.get("input_message", {"role": "user", "content": prompt})
        if not captured:
            job["input_message"] = self.images.capture([job["input_message"]], llm)[0]
        sources = (body.get('image_sources', []) if captured else
                   capture_sources(self.images, body.get('image_sources', []), history+[job['input_message']])) if image else []
        job['image_sources'] = sources
        if image:
            job['tools'] = image_tools(image, sources, enhance)
        job['enhance_prompt'] = enhance
        job['image_size_explicit'] = body.get('image_size_explicit', 'width' in body or 'height' in body)
        reservation.signal.check()
        with self.lock:
            self._check_admission()
            if any(active.get('session_id') == sid for active in self.active_jobs.values()):
                raise ValueError("This conversation already has an active job")
            cancelled = reservation.signal
            job['deadline_seconds'] = self.config.get('request_timeout_seconds', 1800)
            def target():
                try:
                    with deadline_context(cancelled):
                        self._run(job, llm, image, history, cancelled, body.get("prepared"))
                except BaseException as exc:
                    # Deadline can expire before the worker thread even enters
                    # _run; it still owns a persisted job and an admission slot.
                    job.update(status='failed', phase='Failed', error=str(exc) or type(exc).__name__)
                    if isinstance(exc, RequestDeadline):
                        job.update(phase='Request deadline exceeded', error_code='request_timeout')
                    self._finish_job(job, cancelled, Reporter(self.config, job))
            thread = threading.Thread(target=target, daemon=True)
            persisted = False
            try:
                self.store.save_job(job, retry_identity=retry_identity)
                persisted = True
                self.store.event(jid, "queued", message="Waiting for device")
            except IdempotencyConflict:
                raise
            except Exception as exc:
                self._storage_error = str(exc)
                if persisted:
                    # The retry identity already owns this job. It must become
                    # terminal even when the initial event cannot be written;
                    # otherwise retries would recover a queued job with no worker.
                    job.update(status='failed', phase='Failed to persist admission',
                               error='Inference was not started because the admission event could not be saved.',
                               error_code='storage_failure')
                    self._finish_job(job, cancelled, Reporter(self.config, job))
                raise AdmissionError('Could not persist the job; durable storage is unavailable', 'storage_unhealthy') from exc
            self.running[jid] = (cancelled, thread)
            self.active_jobs[jid] = job
            commit_admission(reservation)
            try:
                thread.start()
            except BaseException as exc:
                job.update(status='failed', phase='Failed to start job', error=str(exc))
                self._finish_job(job, cancelled, Reporter(self.config, job))
                raise
            return job

    def cancel(self, jid, reason="client_request"):
        with self.lock:
            job = self.job(jid)
            if jid in self.running and job['status'] not in TERMINAL:
                cancel = self.running[jid][0]
                if not getattr(cancel, 'cancelled', threading.Event.is_set(cancel)):
                    cancel.set()
                    job = self.active_jobs[jid]
                    job.update(cancel_requested=True, cancel_reason=reason,
                               phase="Cancelling", updated=time.time())
                    self.store.save_job(job)
                    self.store.event(jid, "cancelling", reason=reason, message="Stopping active work")
            return {"ok": True, "id": jid, "status": job['status'],
                    "cancel_requested": bool(job.get('cancel_requested')),
                    "status_url": "/api/jobs/" + jid}

    def _finish_job(self, job, cancelled, reporter):
        # Serialize the final state with cancellation. A stop accepted during
        # image saving/unloading must not subsequently become "completed".
        with self.lock:
            try:
                if job['status'] not in TERMINAL:
                    if getattr(cancelled, 'cancelled', threading.Event.is_set(cancelled)):
                        job.update(status="cancelled", phase="Cancelled",
                                   error="Stopped by client. Saved artifacts are retained.")
                    elif getattr(cancelled, 'expired', False):
                        job.update(status='failed', phase='Request deadline exceeded', error='Request exceeded its end-to-end deadline', error_code='request_timeout')
                    else:
                        job.update(status="completed", phase="Complete")
                job['updated'] = time.time()
                try:
                    reporter.finish(job['status'])
                except Exception:
                    logging.getLogger(__name__).exception('Terminal reporting failed for job %s', job['id'])
                try:
                    self.store.save_job(job)
                    self.store.event(job['id'], "finished", status=job['status'], error=job['error'])
                except Exception as exc:
                    self._storage_error = str(exc)
                    self._pending_terminal[job['id']] = {**copy.deepcopy(job), 'durable': False,
                        'storage_error': 'Terminal state has not been saved to durable storage',
                        'error_code': job.get('error_code') or 'storage_failure'}
                    logging.getLogger(__name__).exception('Job %s finished but its terminal state could not be persisted; admissions paused', job['id'])
            finally:
                # No storage or terminal-output failure may retain an admission slot.
                self.running.pop(job['id'], None)
                self.active_jobs.pop(job['id'], None)

    def _run(self, job, llm, image, history, cancelled, initial_prepared=None):
        reporter = Reporter(self.config, job)
        jid = job["id"]
        timeout = self.config.get("request_timeout_seconds", 1800)
        budget = job["request"]["max_tokens"]
        active_model = None
        text_completed = False
        # Template overrides change the prompt, never the resident worker identity.
        prompt_model = copy.deepcopy(llm)
        prompt_model["template_context"] = job["template_context"]
        prompt_model["response_format"] = job["response_format"]

        def emit(kind, **data):
            reporter.event(kind, **data)
            if kind in STREAM_JOB_EVENTS:
                job['updated'] = time.time()
                self.store.event(jid, kind, job_update=job, **data)
            else:
                self.store.event(jid, kind, **data)
            if kind in {"loading", "unloading", "loaded", "unloaded"}:
                job["phase"] = kind.capitalize() + " " + data.get("model", "")
                save()
            elif self._placement_event(job, kind, data):
                save()
            elif kind in {'image_started', 'image_retry', 'image_recovered'}:
                save()

        def save():
            job["updated"] = time.time()
            self.store.save_job(job)

        def text_pass(prepared, tools_enabled, force_image=False, resuming=False, stage="initial"):
            nonlocal active_model, text_completed
            text_completed = False
            active_model = llm["id"]
            worker = self.scheduler.ensure(llm, cancelled, emit)
            job["status"], job["phase"], job["text"] = "running", "Preparing final response" if resuming else "Processing conversation", ""
            job["context"] = prepared["context"]
            job["thinking_open"] = prepared.get("thinking_open", False)
            job.update(content="", reasoning_content="", reasoning_incomplete=False)
            job["current_template_context"] = prepared.get("template_context", {})
            job["template_notes"] = prepared.get("template_notes", [])
            job.pop("timings", None)
            job.pop('sampling', None)
            save()
            emit("context", **job["context"])
            reporter.start_text(llm['id'], stage, job['context'])
            result = None
            for item in worker.run({"action": "text", "prepared": prepared, "tools_enabled": tools_enabled,
                                    "force_image": force_image, "max_tokens": budget,
                                    "timings_per_token": job["timings_per_token"] or reporter.enabled,
                                    "response_format": job["response_format"],
                                    "temperature": job["request"]["temperature"], "generation": job.get("generation", {})}, cancelled.remaining(timeout), cancelled):
                if item['kind'] == 'sampling':
                    job['sampling'] = item['settings']
                    emit('sampling', settings=item['settings'], stage=stage)
                    save()
                if item["kind"] in {"text", "result"}:
                    job["text"] = item["text"]
                    job.update(split_reasoning(job["text"], job["thinking_open"], final=item["kind"] == "result"))
                    job["phase"] = "Generating final response" if resuming else "Generating response"
                    ci = prepared["context"]
                    count = item.get("input_tokens")
                    if count is None:
                        count = ci["input_tokens"]
                    accuracy = ci["accuracy"] if item.get("input_tokens") is None else item["accuracy"]
                    job["context"] = context_info(int(count), int(llm["context_limit"]), budget,
                        item["generated_tokens"], accuracy, prompt_sha256=ci["prompt_sha256"],
                        template_context=job["current_template_context"], template_notes=job["template_notes"])
                    if prepared.get("images"):
                        job["context"].update(image_count=len(prepared["images"]), visual_tokens_included=item.get("input_tokens") is not None)
                        if item.get("input_tokens") is None:
                            job["context"].update(fits=None if job["context"]["fits"] else False, note=ci.get("note"))
                    if job["timings_per_token"] and item.get("timings") is not None:
                        job["timings"] = item["timings"]
                    reporter.text(job['context'], item.get('timings'), final=item['kind'] == 'result')
                    emit("text", text=job["text"], context=job["context"], stage=stage,
                         content=job["content"], reasoning_content=job["reasoning_content"],
                         thinking_open=job["thinking_open"], final=item["kind"] == "result",
                         **({"timings": job.get("timings")} if job["timings_per_token"] else {}))
                    if item["kind"] == "result":
                        result = item
            if result is None:
                raise RuntimeError("Worker returned no result")
            # Only a fully drained, successful worker exchange is reusable after
            # output validation fails. Seeing a result alone is not sufficient.
            text_completed = True
            job["json_valid"] = check_json_output(job["response_format"], job["content"],
                                                   job["context"]["generated_tokens"] >= budget)
            if job["timings_per_token"]:
                job.setdefault("timing_passes", []).append({"stage": stage, "timings": job.get("timings")})
            usage = job.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            usage["prompt_tokens"] += job["context"]["input_tokens"]
            usage["completion_tokens"] += job["context"]["generated_tokens"]
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            save()
            if cancelled.is_set():
                raise Cancelled()
            return result["text"]

        try:
            models = [llm] + ([image] if image else [])
            with self.scheduler.lease_models(models, cancelled):
                try:
                    history = history + [job.get("input_message", {"role": "user", "content": job["prompt"]})]
                    prepared = initial_prepared or self.tokens.prepare(prompt_model, history, budget, image is not None,
                        tools=job.get("tools"), add_system=not job.get("external", False))
                    job["context"] = prepared["context"]
                    save()
                    if prepared["context"]["fits"] is False:
                        raise ValueError("Context overflow: input plus output budget exceeds the configured limit. Nothing was truncated.")
                    self.store.append(job["session_id"], history[-1])
                    first = text_pass(prepared, image is not None)
                    if not image and job.get('tools'):
                        try:
                            parse_tool_calls(first, job['tools'], job.get('thinking_open', False))
                        except ToolCallError as exc:
                            limited = job['context']['generated_tokens'] >= budget
                            if not (exc.code == 'incomplete_reasoning' or (exc.incomplete and limited)):
                                job['error_code'] = exc.code
                                raise
                    tool = None
                    if image:
                        for attempt in range(2):
                            diagnostic = {"attempt": attempt + 1, "raw_response": first,
                                          "thinking_open": prepared.get("thinking_open", False),
                                          "context": copy.deepcopy(job["context"])}
                            job.setdefault("tool_attempts", []).append(diagnostic)
                            try:
                                tool = parse_action(first, job['tools'], prepared.get("thinking_open", False))
                                diagnostic["status"] = "image_call" if tool else "text_response"
                                save()
                                break
                            except ToolCallError as exc:
                                diagnostic.update(status="rejected", error=str(exc), code=exc.code)
                                limited = job["context"]["generated_tokens"] >= budget
                                diagnostic["output_limit_reached"] = limited
                                save()
                                if not (exc.incomplete and limited):
                                    raise
                                if attempt:
                                    raise ValueError(f"The LLM response is still incomplete at the {budget}-token output limit. "
                                                     "Increase Output tokens (try 2048), or disable thinking in model template settings. No image was generated.") from exc
                                # Retry only an interrupted response, once, at the same
                                # requested budget. Never execute or invent a partial call.
                                retry_model = copy.deepcopy(prompt_model)
                                retry_model.setdefault("template_context", {})["enable_thinking"] = False
                                retry_history = copy.deepcopy(history)
                                instruction = ("Your previous response reached the output limit before it was complete. "
                                               "Reply concisely. For a new image call generate_image; to change an available image call edit_image with its image_id. Use one short complete prompt "
                                               "using the tool format above. Otherwise answer normally. Do not include a preamble.")
                                if retry_history and retry_history[0]["role"] == "system":
                                    retry_history[0]["content"] += "\n\n" + instruction
                                else:
                                    retry_history.insert(0, {"role": "system", "content": instruction})
                                prepared = self.tokens.prepare(retry_model, retry_history, budget, True, tools=job['tools'], add_system=False)
                                if prepared["context"]["fits"] is False:
                                    raise ValueError("The incomplete tool response could not be retried within the context limit. No image was generated.") from exc
                                emit("tool_retry", attempt=2, reason=exc.code, max_tokens=budget, enable_thinking=False)
                                first = text_pass(prepared, True, stage="retry")
                    if not tool:
                        self.store.append(job["session_id"], assistant_message(first, prepared.get("thinking_open", False)))
                    else:
                        tool_name, tool = tool['name'], tool['arguments']
                        editing = tool_name == 'edit_image'
                        call_id = new_id()
                        tool_message = {"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                            "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool)}}]}
                        if job.get("reasoning_content"):
                            tool_message["reasoning_content"] = job["reasoning_content"]
                        self.store.append(job["session_id"], tool_message)
                        history.append(tool_message)
                        emit("tool_call", tool=tool_name, arguments=tool)
                        job["text"] = ""
                        save()
                        result_body = None
                        try:
                            image_model = copy.deepcopy(image)
                            params = copy.deepcopy(job['image_request'])
                            if editing:
                                source = next((s for s in job['image_sources'] if s['id'] == tool['image_id']), None)
                                if source is None:
                                    raise ValueError('edit_image selected an unknown image_id; no image was generated')
                                info = self.images.describe(source['ref'])
                                inputs, notes = prepare_image_edit(self.images, image_model, {'image': Path(info['path']).read_bytes()})
                                params.update(inputs)
                                job['parameter_notes'] = notes
                                image_model['_image_operation'] = 'edit'
                            effective_prompt = tool['prompt']
                            if job['enhance_prompt']:
                                effective_prompt, params = enhance_image_prompt(self, job, image_model, llm, effective_prompt,
                                    params, editing, job['image_size_explicit'], cancelled, emit)
                            if cancelled.is_set():
                                raise Cancelled()
                            job['image_request'] = params
                            job['request'].update({k: v for k, v in params.items() if k != 'input_images'})
                            active_model = image['id']
                            ext = ".png"
                            filename = jid + ext
                            output = self.store.directory / "artifacts" / filename
                            self._image_worker(job, image_model, {"action": "image_edit" if editing else "image", "prompt": effective_prompt,
                                "output_path": str(output), **params}, cancelled, emit)
                            artifact = {"url": "/artifacts/" + filename, "name": filename, "prompt": effective_prompt,
                                        "simulated": False, 'operation': 'edit' if editing else 'generate',
                                        **{k: v for k, v in params.items() if k != 'input_images'}}
                            if editing:
                                artifact['source_image_sha256'] = params['input_images'][0]['sha256']
                            if job.get('image_inference'):
                                artifact['inference'] = job['image_inference']
                            job["artifacts"].append(artifact)
                            save()
                            emit("artifact", artifact=artifact)
                            result_body = {"ok": True, "image_url": artifact["url"], "simulated": artifact["simulated"],
                                           "note": "Image saved. Its pixels were not passed to the LLM."}
                        except BaseException as exc:
                            result_body = {"ok": False, "error": str(exc) or type(exc).__name__}
                            raise
                        finally:
                            if result_body is not None:
                                message = {"role": "tool", "tool_call_id": call_id, "name": tool_name,
                                           "content": json.dumps(result_body), "artifacts": list(job["artifacts"])}
                                history.append(message)
                                self.store.append(job["session_id"], message)
                            # Managed tools must resume this LLM immediately.
                            keep_image = self.scheduler.retain_image(image, [llm])
                            if cancelled.cancelled or cancelled.expired or not result_body or not result_body.get('ok') or not keep_image:
                                reason = ('image job stopped or failed' if cancelled.cancelled or cancelled.expired or not result_body or not result_body.get('ok') else
                                          'image retention disabled' if image.get('keep_image_model_loaded') is False else
                                          'automatic image retention: required models do not fit together on the device')
                                self.scheduler.unload(image["id"], emit, reason=reason)
                            else:
                                self.scheduler.last_used[image['id']] = time.monotonic()
                        reused = self.scheduler.is_loaded(llm)
                        emit('resuming' if reused else 'restoring', model=llm['id'], worker_reused=reused,
                             **({} if reused else {'cache_preserved': False}))
                        job["phase"] = "Image saved; resuming chat model" if reused else "Image saved; restoring chat model"
                        save()
                        prepared = self.tokens.prepare(prompt_model, history, budget, False, add_system=not job.get("external", False))
                        job["context"] = prepared["context"]
                        save()
                        if prepared["context"]["fits"] is False:
                            raise ValueError("Image was saved, but its tool result exceeds the remaining context budget")
                        response = text_pass(prepared, False, resuming=True, stage="continuation")
                        self.store.append(job["session_id"], assistant_message(response, prepared.get("thinking_open", False)))
                except BaseException as exc:
                    # Rejected model output is a request failure, not a native
                    # worker failure. Client tools are never executed here; a
                    # corrected request can reuse the completed text worker.
                    validation_error = isinstance(exc, (ToolCallError, ResponseFormatError))
                    if validation_error:
                        job['error_code'] = exc.code
                    retain = (validation_error and text_completed and active_model == llm['id']
                              and not cancelled.cancelled and not cancelled.expired
                              and self.scheduler.is_loaded(llm))
                    clean_stop = (isinstance(exc, Cancelled) and
                                  (getattr(exc, 'worker_reusable', False) or text_completed) and
                                  active_model == llm['id'] and self.scheduler.is_loaded(llm))
                    if retain:
                        emit('response_rejected', model=llm['id'], code=exc.code, worker_retained=True)
                    elif clean_stop:
                        emit('generation_stopped', model=llm['id'], worker_retained=True)
                    elif active_model:
                        # Native failure, cancellation, expiry and incomplete
                        # exchanges still require cleanup before device reuse.
                        self.scheduler.unload(active_model, emit,
                            reason='generation stop required worker shutdown' if isinstance(exc, Cancelled)
                            else 'active request failed')
                    raise
        except Cancelled:
            job["status"], job["phase"] = "cancelled", "Cancelled"
            job["error"] = "Stopped by client." + (" Saved artifacts are retained." if job.get('artifacts') else "")
        except RequestDeadline as exc:
            job.update(status='failed', phase='Request deadline exceeded', error=str(exc), error_code='request_timeout')
        except Exception as exc:
            _record_failure(job, exc)
        finally:
            self.scheduler.last_used[llm["id"]] = time.monotonic()
            self._finish_job(job, cancelled, reporter)

    def close(self):
        with self.lock:
            self.draining = True
            self.closed = True
            running = list(self.running.values())
            for event, _ in running:
                event.set()
            for event in self._preparation_signals:
                event.set()
        self.downloads.close()
        finish_by = time.monotonic() + 10
        for _, thread in running:
            thread.join(max(0, finish_by - time.monotonic()))
        while self._preparing and time.monotonic() < finish_by:
            time.sleep(.02)
        if any(t.is_alive() for _, t in running) or self._preparing:
            return  # Do not close storage underneath unfinished jobs or preparation.
        self.scheduler.close()
        self.store.close()
