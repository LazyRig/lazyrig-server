"""Fair admission across device lanes, with conservative memory reservations."""
import contextlib
from contextvars import ContextVar
import copy
import logging
import threading
import time
from .worker import Cancelled, NativeWorkerError, ProcessWorker, WorkerCleanupError
from .deadlines import time_budget
from .memory import observe_load_memory
from .runtime_settings import effective_cpu_thread_default
from . import placement
from .mtp import devices as mtp_devices, device as mtp_device, enabled as mtp_enabled


class PlacementAdmissionError(ValueError):
    """A planned reservation cannot fit the configured device budget."""
    def __init__(self, message, *, error_code='device_budget_exceeded', weight_check=None):
        super().__init__(message)
        self.error_code = error_code
        self.weight_check = copy.deepcopy(weight_check)


def load_config(model):
    """Only options consumed when constructing a native pipeline identify it.

    History, templates, sampling, output limits and UI metadata are per request
    or admission policy. They must not invalidate a healthy resident worker.
    """
    pipeline = 'vlm' if model['pipeline'] == 'vlm-text' else model['pipeline']
    result = {'path': model.get('path', ''), 'device': model['device'], 'pipeline': pipeline,
              'runtime_profile': model.get('runtime_profile', 'default'),
              'plugin_config': model.get('plugin_config', {}),
              'placement': model.get('placement', {})}
    if pipeline == 'image':
        scheduler = model.get('diffusion_scheduler', {})
        result.update(operation=model.get('_image_operation', 'generate'),
                      diffusion_scheduler={'type': scheduler.get('type', 'AUTO'), 'config': scheduler.get('config', {})})
    else:
        result.update(scheduler_config=model.get('scheduler_config', {}),
                      mtp_enabled=model.get('mtp', {}).get('enabled', False))
        if mtp_enabled(model):
            result['mtp_device'] = model.get('mtp', {}).get('device') or ''
    return copy.deepcopy(result)

class Scheduler:
    def __init__(self, config, worker_factory=ProcessWorker):
        self.config = config
        self.factory = worker_factory
        self.condition = threading.Condition(threading.RLock())
        self.waiting = []
        self.busy = set()
        self.resident = {}
        self.states = {}
        self.last_used = {}
        self.preferred = {}
        self._pinned = ContextVar('lazyrig_pinned_placements', default={})

    @contextlib.contextmanager
    def lease(self, devices, cancelled):
        devices = set(devices)
        ticket = (object(), devices)
        with self.condition:
            self.waiting.append(ticket)
            try:
                while True:
                    if cancelled.is_set():
                        raise Cancelled("Cancelled while queued")
                    earlier = self.waiting[:self.waiting.index(ticket)]
                    if not devices & self.busy and not any(devices & d for _, d in earlier):
                        self.waiting.remove(ticket)
                        self.busy.update(devices)
                        break
                    self.condition.wait(time_budget(0.1, cancelled))
            except BaseException:
                self.waiting.remove(ticket)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.busy.difference_update(devices)
                self.condition.notify_all()

    def _resource_devices(self, model):
        return {placement.canonical_device(device, self.config['devices'])
                for device in placement.devices(model) | mtp_devices(model)}

    def _reservations(self, model):
        result = {}
        for device, amount in placement.reservations(model).items():
            key = placement.canonical_device(device, self.config['devices'])
            result[key] = result.get(key, 0.0) + float(amount)
        for device in mtp_devices(model):
            key = placement.canonical_device(device, self.config['devices'])
            # The head also uses a grafted LM output layer and runtime cache.
            # Its BIN size is not its peak usage. A separate draft device gets
            # its whole configured budget until native usage is measurable.
            if key not in result:
                result[key] = float(self._device_config(key)['budget_gib'])
        return result

    @staticmethod
    def _configured_devices(config):
        physical = {}
        for device in config['devices']:
            key = placement.canonical_device(device['id'], config['devices'])
            if key not in physical or device['id'] == key:
                physical[key] = device
        return list(physical.values())

    def _device_config(self, device):
        return next(d for d in self._configured_devices(self.config)
                    if placement.canonical_device(d['id'], self.config['devices']) == device)

    @staticmethod
    def _target(model):
        return model.get('_placement', {}).get('target', model['device'])

    @classmethod
    def _placement_identity(cls, model):
        plan = model.get('_placement', {})
        return (cls._target(model), placement.devices(model), plan.get('properties', {}),
                plan.get('shared_memory_attempt', False))

    def _resolved(self, model, loaded, config=None):
        """Keep the actual target, refreshing admission-only reservation values."""
        resolved = copy.deepcopy(model)
        resolved.pop('_placement', None)
        if '_placement' not in loaded:
            return self._with_cpu_threads(resolved, config)
        plans = placement.prepare(copy.deepcopy(resolved), self.config if config is None else config)
        matched = next((plan for plan in plans if self._placement_identity(plan) == self._placement_identity(loaded)), None)
        resolved['_placement'] = copy.deepcopy((matched or loaded)['_placement'])
        return self._with_cpu_threads(resolved, config)

    def _with_cpu_threads(self, model, config=None):
        """Attach effective device defaults to a resolved worker, never registry data.

        The actual placement matters: a healthy GPU-only Auto worker must not be
        invalidated because CPU is merely one of its possible fallback devices.
        """
        config = self.config if config is None else config
        result = copy.deepcopy(model)
        result.pop('_cpu_inference_threads', None)
        actual = set(result.get('_placement', {}).get('devices', [result['device']])) | mtp_devices(result)
        cpus = {device for device in actual if device.split('.')[0] == 'CPU'}
        default = next((d.get('inference_threads') for d in config['devices'] if d['id'] in cpus), None)
        effective = effective_cpu_thread_default(result, default)
        if effective is not None:
            result['_cpu_inference_threads'] = effective
        return result

    def _lease_closure(self, models, restore):
        resources, pinned = set(), {}
        for model in models:
            effective = model
            record = self.resident.get(model['id'])
            if ('_placement' not in model and record and self.states.get(model['id']) == 'loaded'
                    and load_config(record[0]) == load_config(model) and record[1].process.is_alive()):
                effective = self._resolved(model, record[0])
                pinned[model['id']] = effective
            resources.update(placement.lease_devices(effective, self.config))
        previous = {}
        while True:
            before = set(resources)
            for mid, (model, _) in self.resident.items():
                occupied = self._resource_devices(model) | {'@model:' + mid}
                if not occupied & resources:
                    continue
                resources.update(occupied)
                if restore and model.get('task') == 'text-generation':
                    # Snapshots preserve the exact loaded placement on restore.
                    resources.update(placement.lease_devices(model, self.config))
                    previous[mid] = copy.deepcopy(model)
            if resources == before:
                return resources, list(previous.values()), pinned

    @contextlib.contextmanager
    def lease_models(self, models, cancelled, restore=False):
        """Lease complete placement/cleanup plans atomically.

        When requested, yield the overlapping resident text-model snapshots for
        restoration before leaving the lease. Recompute after queueing because
        another admitted job may have changed the resident placement meanwhile.
        Never acquire another lane while retaining a partial lease. Healthy
        residents pin their actual target for this workflow, allowing independent
        workers to run together. If a pinned worker exits, reconstruction uses
        that same target; failure ends the job instead of expanding its lease.
        A later fresh request can obtain the full fallback plan again.
        """
        models = [copy.deepcopy(model) for model in models]
        with self.condition:
            resources, _, _ = self._lease_closure(models, restore)
        while True:
            with self.lease(resources, cancelled):
                with self.condition:
                    required, previous, pinned = self._lease_closure(models, restore)
                if required <= resources:
                    requested = {model['id'] for model in models}
                    bindings = {mid: plan for mid, plan in self._pinned.get().items() if mid not in requested}
                    bindings.update(pinned)
                    token = self._pinned.set(bindings)
                    try:
                        yield previous
                    finally:
                        self._pinned.reset(token)
                    return
                resources.update(required)

    def unload(self, mid, emit, reason='requested'):
        with self.condition:
            record = self.resident.get(mid)
            if not record:
                return
            self.states[mid] = "unloading"
        if self.config.get('terminal_metrics', True):
            logging.getLogger('lazyrig.runtime').info('%s | unload reason: %s', mid, reason)
        reporting_error = None
        try:
            emit("unloading", model=mid, device=self._target(record[0]), reason=reason)
        except Exception as exc:
            # Persistence/reporting failures must never prevent native cleanup.
            reporting_error = exc
        try:
            record[1].close()
        except BaseException as exc:
            with self.condition:
                self.states[mid] = 'cleanup-failed'
            try:
                emit('unload_failed', model=mid, error=str(exc), worker_pid=getattr(record[1], 'worker_pid', None))
            except Exception:
                logging.getLogger('lazyrig.runtime').exception('Could not report failed cleanup for %s', mid)
            raise
        with self.condition:
            self.resident.pop(mid, None)
            self.states[mid] = "unloaded"
            self.last_used.pop(mid, None)
        try:
            emit("unloaded", model=mid, cache_preserved=False, worker_pid=getattr(record[1], 'worker_pid', None), reason=reason)
        except Exception as exc:
            reporting_error = reporting_error or exc
        if reporting_error is not None:
            raise reporting_error

    def reconfigure(self, candidate, emit):
        """Apply validated settings while idle, retaining compatible workers."""
        models = {m['id']: m for m in candidate['models']}
        old_budgets = {d['id']: float(d['budget_gib']) for d in self._configured_devices(self.config)}
        new_budgets = {d['id']: float(d['budget_gib']) for d in self._configured_devices(candidate)}
        for mid, preferred in list(self.preferred.items()):
            desired = models.get(mid)
            # A budget edit can make a previously skipped primary GPU viable.
            # Keep healthy residents, but reconsider the ordered candidates on
            # the next fresh load instead of remembering the old fallback.
            budget_changed = desired and any(old_budgets.get(d) != new_budgets.get(d)
                                              for d in placement.lease_devices(desired, candidate))
            if (not desired or not desired.get('enabled', True) or budget_changed
                    or load_config(preferred) != load_config(desired)):
                self.preferred.pop(mid, None)
        available = {placement.canonical_device(d['id'], candidate['devices']) for d in candidate['devices']}
        for mid, (old, worker) in list(self.resident.items()):
            current = models.get(mid)
            # Retained image workers keep their currently loaded operation.
            desired = {**current, '_image_operation': old['_image_operation']} if current and '_image_operation' in old else current
            if not current or not current.get('enabled', True):
                self.unload(mid, emit, reason='model removed or disabled')
            elif load_config(old) != load_config(desired):
                old_settings, new_settings = load_config(old), load_config(desired)
                changed = sorted(k for k in old_settings.keys() | new_settings.keys()
                                 if old_settings.get(k) != new_settings.get(k))
                self.unload(mid, emit, reason='load-time settings changed: ' + ', '.join(changed))
            elif not self._resource_devices(old) <= available:
                self.unload(mid, emit, reason='placement device removed')
            else:
                desired = self._resolved(desired, old, candidate)
                if old.get('_cpu_inference_threads') != desired.get('_cpu_inference_threads'):
                    self.unload(mid, emit, reason='CPU device inference threads changed')
                    continue
                with self.condition:
                    self.resident[mid] = (desired, worker)
        for device in self._configured_devices(candidate):
            key = placement.canonical_device(device['id'], candidate['devices'])
            occupants = sorted(((mid, m) for mid, (m, _) in self.resident.items()
                                if key in self._resource_devices(m)),
                               key=lambda item: self.last_used.get(item[0], 0))
            used = sum(self._reservations(m).get(key, 0) for _, m in occupants)
            while occupants and (used > float(device['budget_gib']) or (device['policy'] == 'exclusive' and len(occupants) > 1)):
                mid, model = occupants.pop(0)
                self.unload(mid, emit, reason='device budget or residency policy changed')
                used -= self._reservations(model).get(key, 0)
        self.config = candidate

    def _admit(self, model, emit):
        mid = model['id']
        target_devices = self._resource_devices(model)
        need = self._reservations(model)
        policies = {device: self._device_config(device) for device in target_devices}
        for device, amount in need.items():
            budget = float(policies[device]['budget_gib'])
            if amount > budget:
                raise PlacementAdmissionError(f"{mid} reserves {amount:.1f} GiB, exceeding {device}'s {budget:.1f} GiB budget")
        with self.condition:
            pending_cleanup = [key for key, (old, _) in self.resident.items()
                               if self._resource_devices(old) & target_devices
                               and self.states.get(key) == 'cleanup-failed']
        for key in pending_cleanup:
            self.unload(key, emit, reason='retrying failed cleanup')
        with self.condition:
            occupants = sorted(((key, old) for key, (old, _) in self.resident.items()
                                if key != mid and self._resource_devices(old) & target_devices),
                               key=lambda item: self.last_used.get(item[0], 0))
        used = {device: sum(self._reservations(old).get(device, 0) for _, old in occupants)
                for device in target_devices}
        for key, old in occupants:
            overlap = self._resource_devices(old) & target_devices
            exclusive = any(policies[device]['policy'] == 'exclusive' for device in overlap)
            insufficient = any(used[device] + need.get(device, 0) > float(policies[device]['budget_gib'])
                               for device in overlap)
            if exclusive or insufficient:
                self.unload(key, emit, reason=('exclusive device switch to ' if exclusive else 'memory reservation needed for ') + mid)
                old_reservations = self._reservations(old)
                for device in target_devices:
                    used[device] -= old_reservations.get(device, 0)

    def ensure(self, model, cancelled, emit):
        mid = model['id']
        pinned = self._pinned.get().get(mid)
        if '_placement' not in model and pinned and load_config(pinned) == load_config(model) and '_placement' in pinned:
            model = {**copy.deepcopy(model), '_placement': copy.deepcopy(pinned['_placement'])}
        if cancelled.is_set():
            raise Cancelled()
        with self.condition:
            existing = self.resident.get(mid)
            reusable = bool(existing and self.states.get(mid) == 'loaded'
                            and load_config(existing[0]) == load_config(model) and existing[1].process.is_alive())
        resolved = self._resolved(model, existing[0]) if reusable else None
        if reusable and existing[0].get('_cpu_inference_threads') != resolved.get('_cpu_inference_threads'):
            reusable = False
        if reusable:
            self._admit(resolved, emit)
            if cancelled.is_set():
                raise Cancelled()
            with self.condition:
                self.resident[mid] = (resolved, existing[1])
                self.last_used[mid] = time.monotonic()
            return existing[1]
        if existing:
            if self.states.get(mid) == 'cleanup-failed':
                reason = 'retrying failed cleanup'
            elif resolved is not None and existing[0].get('_cpu_inference_threads') != resolved.get('_cpu_inference_threads'):
                reason = 'CPU device inference threads changed'
            elif load_config(existing[0]) == load_config(model):
                reason = 'worker process exited'
            else:
                changed = sorted(k for k in load_config(model) if load_config(existing[0]).get(k) != load_config(model)[k])
                reason = 'load-time settings changed: ' + ', '.join(changed)
            self.unload(mid, emit, reason=reason)
        load_deadline = time.monotonic() + float(self.config.get('load_timeout_seconds', 600))
        candidates = [copy.deepcopy(model)] if '_placement' in model else placement.prepare(copy.deepcopy(model), self.config)
        candidates = [self._with_cpu_threads(candidate) for candidate in candidates]
        preferred = self.preferred.get(mid)
        if preferred and load_config(preferred) != load_config(model):
            self.preferred.pop(mid, None)
            preferred = None
        if preferred and '_placement' not in model:
            matched = next((candidate for candidate in candidates
                            if self._placement_identity(candidate) == self._placement_identity(preferred)), None)
            if matched is not None:
                candidates = [matched, *(candidate for candidate in candidates if candidate is not matched)]
        plan_id = f'{mid}:{time.monotonic_ns()}'
        emit('placement_plan', model=mid, plan_id=plan_id,
             mode=model.get('placement', {}).get('mode', 'single'),
             split=placement.split_status(model, self.config),
             allow_fallback=model.get('placement', {}).get('allow_fallback', False),
             shared_memory=model.get('placement', {}).get('shared_memory', 'budget'),
             unavailable_devices=[d['id'] for d in self.config['devices']
                                  if d['id'] in placement.devices(model) and d.get('detected') is False],
             preferred_reused=bool(preferred and self._placement_identity(candidates[0]) == self._placement_identity(preferred)),
             attempts=[{'attempt': i + 1, 'target': self._target(candidate),
                        'label': candidate.get('_placement', {}).get('label', 'Single device'),
                        'status': 'not_attempted',
                        'reservations': self._reservations(candidate),
                        'notes': candidate.get('_placement', {}).get('notes', [])}
                       for i, candidate in enumerate(candidates)])

        def report_attempt(index, resolved, status, **details):
            emit('placement_attempt', model=mid, plan_id=plan_id, attempt=index + 1,
                 target=self._target(resolved), status=status, **details)

        for index, resolved in enumerate(candidates):
            if cancelled.is_set():
                raise Cancelled()
            weight_check = placement.weight_budget_check(resolved, self.config)
            try:
                if weight_check:
                    mtp = mtp_enabled(resolved)
                    guidance = (
                        'With MTP enabled, LazyRig loads the main language model on one device; '
                        'automatic model splitting is disabled. Moving the MTP head does not offload '
                        'the main weights. Use a smaller model, choose a primary device with enough '
                        "memory, or explicitly allow whole-model fallback to a selected CPU. "
                        'Disable MTP to try automatic model splitting.' if mtp else
                        'Select additional devices for automatic offload, use a smaller model, '
                        'or choose a primary device with enough memory.')
                    raise PlacementAdmissionError(
                        f"Stored language weights ({weight_check['weights_gib']:.1f} GiB) exceed "
                        f"{weight_check['device']}'s {weight_check['budget_gib']:.1f} GiB configured budget. "
                        'To permit a shared-memory loading attempt, enable the primary GPU shared-memory '
                        'attempt in Devices for this model, or raise the device Planning budget. This allows the native loader to try; it does not '
                        'allocate extra VRAM or guarantee driver support. '
                        + guidance + ' Runtime and cache memory also need headroom.',
                        error_code='mtp_model_weight_budget_exceeded' if mtp else 'model_weight_budget_exceeded',
                        weight_check=weight_check)
                self._admit(resolved, emit)
            except PlacementAdmissionError as exc:
                report_attempt(index, resolved, 'skipped',
                               reason_code='weight_budget' if weight_check else 'reservation_budget',
                               reason=str(exc)[:4096], **({'weight_check': weight_check} if weight_check else {}))
                if index + 1 >= len(candidates):
                    raise
                emit('placement_fallback', model=mid, from_target=self._target(resolved),
                     to_target=self._target(candidates[index + 1]),
                     reason='stored weights exceed GPU budget' if weight_check else 'configured reservation exceeds device budget',
                     reason_code='weight_budget' if weight_check else 'reservation_budget',
                     **({'weight_check': weight_check} if weight_check else {}), attempt=index + 2)
                continue
            remaining = load_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Model load exceeded its configured deadline')
            target = self._target(resolved)
            with self.condition:
                self.states[mid] = 'loading'
            emit('loading', model=mid, device=target,
                 **({'mtp_device': mtp_device(resolved)} if mtp_enabled(resolved) else {}))
            report_attempt(index, resolved, 'loading')
            try:
                remaining = load_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Model load exceeded its configured deadline')
                timeout = time_budget(remaining, cancelled)
                # ProcessWorker confirms native-process cleanup before raising.
                # WorkerCleanupError retains the worker and forbids fallback.
                with observe_load_memory(lambda data: emit('load_memory', model=mid, device=target, **data)):
                    worker = self.factory(self.config['mode'], resolved, timeout, cancelled)
            except WorkerCleanupError as exc:
                with self.condition:
                    self.resident[mid] = (copy.deepcopy(resolved), exc.worker)
                    self.states[mid] = 'cleanup-failed'
                if getattr(exc, 'load_memory', None):
                    emit('load_memory', model=mid, device=target, **{**exc.load_memory, 'stage': 'failed'})
                emit('unload_failed', model=mid, error=str(exc), worker_pid=getattr(exc.worker, 'worker_pid', None))
                report_attempt(index, resolved, 'failed', reason_code='cleanup_failed', reason=str(exc)[:4096],
                               **({'worker_diagnostics': copy.deepcopy(exc.worker_diagnostics)} if getattr(exc, 'worker_diagnostics', None) else {}))
                raise
            except BaseException as exc:
                with self.condition:
                    self.states[mid] = 'unloaded'
                if getattr(exc, 'load_memory', None):
                    emit('load_memory', model=mid, device=target, **{**exc.load_memory, 'stage': 'failed'})
                report_attempt(index, resolved, 'cancelled' if isinstance(exc, Cancelled) else 'failed',
                               reason_code='cancelled' if isinstance(exc, Cancelled) else 'load_timeout' if isinstance(exc, TimeoutError) else placement.load_failure_reason(exc),
                               reason=(str(exc) or type(exc).__name__)[:4096],
                               stage=getattr(exc, 'worker_stage', 'loading'),
                               **({'worker_diagnostics': copy.deepcopy(exc.worker_diagnostics)} if getattr(exc, 'worker_diagnostics', None) else {}),
                               **({'worker_exit': copy.deepcopy(exc.worker_exit)} if getattr(exc, 'worker_exit', None) else {}))
                preflight = getattr(exc, 'placement_preflight', None)
                if (isinstance(exc, NativeWorkerError) and isinstance(preflight, dict) and preflight
                        and resolved.get('placement', {}).get('mode') == 'manual'):
                    emit('placement_rejected', model=mid, device=target, target=target,
                         placement=copy.deepcopy(resolved.get('_placement', {})),
                         preflight=copy.deepcopy(preflight))
                    raise  # A rejected manual target must never become an automatic fallback.
                if index + 1 >= len(candidates) or not placement.retryable_load_error(exc):
                    raise
                if cancelled.is_set():
                    raise Cancelled()
                if time.monotonic() >= load_deadline:
                    raise TimeoutError('Model load exceeded its configured deadline') from exc
                emit('placement_fallback', model=mid, from_target=target,
                     to_target=self._target(candidates[index + 1]),
                     reason='host memory headroom reached' if placement.load_failure_reason(exc) == 'host_memory_pressure' else 'native load rejected',
                     reason_code=placement.load_failure_reason(exc), attempt=index + 2,
                     **({'worker_exit': copy.deepcopy(exc.worker_exit)} if getattr(exc, 'worker_exit', None) else {}))
                continue
            with self.condition:
                self.resident[mid] = (copy.deepcopy(resolved), worker)
                self.states[mid] = 'loaded'
                self.last_used[mid] = time.monotonic()
                if '_placement' in resolved:
                    self.preferred[mid] = copy.deepcopy(resolved)
            report_attempt(index, resolved, 'succeeded')
            if '_placement' in resolved:
                selected = resolved['_placement']
                emit('placement_selected', model=mid, device=target, target=target,
                     devices=sorted(placement.devices(resolved)), label=selected.get('label', target),
                     notes=copy.deepcopy(selected.get('notes', [])),
                     placement=copy.deepcopy(selected),
                     preflight=copy.deepcopy(getattr(worker, 'placement_preflight', None)))
            emit('loaded', model=mid, device=target, worker_pid=getattr(worker, 'worker_pid', None))
            return worker
        raise ValueError(f'No valid placement is available for {mid}')

    def is_loaded(self, model):
        with self.condition:
            record = self.resident.get(model['id'])
            return bool(record and load_config(record[0]) == load_config(model) and self.states.get(model['id']) == 'loaded'
                        and record[1].process.is_alive())

    def retain_image(self, model, required=()):
        """Keep an image only when all affected devices admit required models."""
        choice = model.get('keep_image_model_loaded', 'auto')
        if isinstance(choice, bool):
            return choice
        with self.condition:
            resident = {mid: m for mid, (m, _) in self.resident.items()}
            resolved = resident.get(model['id'], model)
            peers = {m['id']: resident.get(m['id'], m) for m in [*required, *resident.values()]
                     if m['id'] != model['id']}
        need = self._reservations(resolved)
        for device in self._resource_devices(resolved):
            policy = self._device_config(device)
            occupants = [m for m in peers.values() if device in self._resource_devices(m)]
            if occupants and policy['policy'] == 'exclusive':
                return False
            if need.get(device, 0) + sum(self._reservations(m).get(device, 0) for m in occupants) > float(policy['budget_gib']):
                return False
        return True

    def snapshot(self):
        with self.condition:
            return {'states': dict(self.states), 'busy': sorted(self.busy), 'queued': len(self.waiting),
                    'workers': [{'model': mid, 'device': self._target(m), 'devices': sorted(placement.devices(m)),
                                 **({'mtp_device': mtp_device(m)} if mtp_enabled(m) else {}),
                                 'placement': copy.deepcopy(m.get('_placement', {})),
                                 'pid': getattr(worker, 'worker_pid', None),
                                 'operation': m.get('_image_operation', 'generate') if m['pipeline'] == 'image' else 'text'}
                                for mid, (m, worker) in self.resident.items()],
                    'devices': [{**d, 'reserved_gib': sum(self._reservations(m).get(placement.canonical_device(d['id'], self.config['devices']), 0)
                                                        for m, _ in self.resident.values()),
                                 'memory_source': 'configured reservations', 'measured_free_gib': None}
                                for d in self._configured_devices(self.config)]}

    def close(self):
        errors = []
        for mid in list(self.resident):
            try:
                self.unload(mid, lambda *args, **kwargs: None, reason='server shutdown')
            except Exception as exc:
                errors.append(mid + ': ' + str(exc))
        if errors:
            raise RuntimeError('Worker cleanup failed: ' + '; '.join(errors))
