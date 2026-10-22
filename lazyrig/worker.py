"""One disposable native process per loaded model."""
import copy
import json
import multiprocessing as mp
import os
import re
import socket
import signal
import subprocess
import sys
import time
from pathlib import Path
from .metrics import GenerationTimer
from .response_format import configure_output, is_json
from .runtime_settings import pipeline_properties, pipeline_target, attention_cache_failure
from .mtp import configure_generation
from .image_options import NATIVE_OPTIONS
from .diffusion_settings import configure_pipeline as configure_diffusion
from .sampling import apply as apply_sampling, snapshot as sampling_snapshot
from .deadlines import check_deadline, time_budget
from .memory import LoadMemoryGuard, publish_load_memory
from .worker_status import WorkerProcessError
from .worker_diagnostics import create_fault_log, enable_fault_log, read_fault_log, runtime_versions
from .gpu_errors import kernel_build_failure

class Cancelled(Exception):
    def __init__(self, message='Job cancelled', *, worker_reusable=False):
        super().__init__(message)
        self.worker_reusable = worker_reusable


TEXT_STOP_GRACE_SECONDS = 5.0
WORKER_EXIT_GRACE_SECONDS = 2.0


def _text_stop_requested(conn, state):
    if not state[0] and conn.poll(0):
        control = conn.recv()
        if control != {'action': 'cancel_text'}:
            raise ValueError('Unexpected control message during text generation')
        state[0] = True
    return state[0]

class NativeWorkerError(RuntimeError):
    """An error reported by the native child, rather than IPC or cleanup."""
    pass

class WorkerCleanupError(RuntimeError):
    def __init__(self, worker, operation_error, cleanup_error):
        self.worker = worker
        super().__init__('Model load failed (' + str(operation_error) + '); worker cleanup is unconfirmed (' + str(cleanup_error) + ')')

def _native_image(conn, pipeline, model, request):
    # Keep tensors, the generator and PIL buffers out of the long-lived receive
    # loop. Warm workers must not hold the last request's output while idle.
    import openvino_genai as genai
    from PIL import Image
    progress = {'step': 0, 'steps': None, 'callbacks': 0}
    def callback(step, total, latent):
        progress.update(step=int(step) + 1, steps=int(total), callbacks=progress['callbacks'] + 1)
        conn.send({'kind': 'progress', 'step': int(step) + 1, 'steps': int(total), 'simulated': False})
        return False
    effective = {key: request[key] for key in NATIVE_OPTIONS if key in request}
    if hasattr(pipeline, 'get_generation_config'):
        native = pipeline.get_generation_config()
        for key in NATIVE_OPTIONS:
            if key not in effective and hasattr(native, key):
                effective[key] = getattr(native, key)
    random_options = ({'generator': genai.TorchGenerator(request['seed'])}
                      if model.get('runtime_profile') == 'qwen-image-21' else {'rng_seed': request['seed']})
    image_args, edit_options = (), {}
    if request['action'] == 'image_edit':
        import openvino as ov
        import numpy as np
        with Image.open(request['input_images'][0]['path']) as source, source.convert('RGB') as rgb:
            image_args = (ov.Tensor(np.array(rgb, dtype=np.uint8)[None]),)
        if 'strength' in request:
            edit_options['strength'] = request['strength']
    result = pipeline.generate(request['prompt'], *image_args, width=request['width'], height=request['height'],
        num_inference_steps=request['steps'], num_images_per_prompt=1, callback=callback,
        **random_options, **edit_options,
        **{key: request[key] for key in NATIVE_OPTIONS if key in request})
    # The callback's total is the runtime's actual schedule. Classic img2img
    # may use fewer steps than requested because of strength; compare against
    # that reported total, not num_inference_steps.
    if progress['steps'] and progress['step'] < progress['steps']:
        raise RuntimeError(f"Image pipeline returned after {progress['step']}/{progress['steps']} denoising steps; no artifact was published")
    inference = {'runtime': 'openvino', 'runtime_version': getattr(genai, '__version__', 'unknown'),
                 'pipeline': 'Image2ImagePipeline' if image_args else 'Text2ImagePipeline',
                 'seed': request['seed'], 'requested_steps': request['steps'], 'denoising': dict(progress),
                 'returned_normally': True, 'visual_verification': 'not_performed', **effective}
    if model.get('runtime_profile') == 'qwen-image-21':
        inference['classifier_free_guidance'] = effective.get('guidance_scale', 4) > 1 and effective.get('negative_prompt') is not None
    # A retrievable artifact must always be a fully encoded image. A killed
    # worker can leave a .part file, never a truncated public PNG.
    target = Path(request['output_path'])
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        with Image.fromarray(result.data[0]) as output, temporary.open('wb') as stream:
            inference.update(width=output.width, height=output.height)
            output.save(stream, format='PNG')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return inference


def _pipeline_setup(genai, ov, model):
    """Apply explicit placement only to this worker's native configuration."""
    runtime_model = copy.deepcopy(model)
    report = {}
    affinity_reader = None
    if runtime_model.get('_placement'):
        from .placement import runtime_properties
        from .placement_probe import verify_manual_split
        core = ov.Core()
        try:
            if runtime_model.get('_manual_affinity'):
                from .manual_affinity import prepare
                affinity_reader, report = prepare(core, ov, genai, runtime_model)
                for values in runtime_model['_placement'].get('properties', {}).get('DEVICE_PROPERTIES', {}).values():
                    values.pop('QUERY_MODEL_RATIO', None)
            else:
                placement_properties = runtime_properties(runtime_model, core)
                report = verify_manual_split(core, runtime_model, placement_properties)
        finally:
            del core
        overrides = report.get('calibrated_properties', {})
        if overrides:
            if report.get('status') != 'preflight_verified':
                raise ValueError('Cannot apply unverified manual split calibration')
            properties = runtime_model['_placement'].setdefault('properties', {})
            for device, values in overrides.get('DEVICE_PROPERTIES', {}).items():
                properties.setdefault('DEVICE_PROPERTIES', {}).setdefault(device, {}).update(values)
        runtime_model['_placement_preflight'] = report
    properties = pipeline_properties(genai, runtime_model)
    if affinity_reader is not None:
        from .manual_affinity import scope_compiled_cache
        scope_compiled_cache(properties, affinity_reader.assignments)
        properties['extensions'] = affinity_reader.extensions
        properties['_lazyrig_affinity_reader'] = affinity_reader
    target = pipeline_target(runtime_model)
    if target == 'HETERO' and report.get('language_input_device'):
        report['component_devices'] = {
            role: report['language_input_device']
            for role in ('text_embeddings', 'text_embeddings_per_layer')}
    return properties, target, report


def _validate_split_inference(pipeline, progress=None):
    """Exercise both prefill and decode before advertising a split as ready.

    This stateless text probe has no user input, tools or persistent history.
    It cannot certify images, long contexts or every later kernel shape.
    """
    import openvino_genai as genai
    counts = [0]
    class ProbeStreamer(genai.StreamerBase):
        def write(self, tokens):
            counts[0] += len(tokens) if isinstance(tokens, list) else 1
            if progress is not None:
                progress('tokens_reported', counts[0])
            return genai.StreamingStatus.RUNNING
        def end(self):
            pass
    config = pipeline.get_generation_config()
    config.max_new_tokens = 2
    config.do_sample = False
    config.num_beams = 1
    config.num_return_sequences = 1
    config.apply_chat_template = False
    config.ignore_eos = True
    config.stop_strings = set()
    config.stop_token_ids = set()
    started = time.monotonic()
    if progress is not None:
        progress('generate_started', 0)
    pipeline.generate('Hello', generation_config=config, streamer=ProbeStreamer())
    generated = counts[0]
    if generated != 2:
        raise RuntimeError('Split inference validation did not complete prefill and decode')
    if progress is not None:
        progress('generate_returned', generated)
    return {'status': 'passed', 'generated_tokens': generated,
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'scope': 'short text prefill and decode; not image or long-context validation'}


def _child(conn, mode, model, job_name=None):
    placement_report = {}
    properties = {}
    fault_stream = enable_fault_log(model.get('_worker_fault_log'))
    try:
        if job_name:
            from .windows_job import join_worker_job
            join_worker_job(job_name)
        if mode != "openvino":
            raise ValueError("Only the OpenVINO backend is supported")
        import openvino_genai as genai
        import openvino as ov
        conn.send({'kind': 'worker_diagnostics', 'runtime': runtime_versions(ov, genai),
                   'fault_handler_enabled': fault_stream is not None})
        cls = {"llm": genai.LLMPipeline, "vlm-text": genai.VLMPipeline, "vlm": genai.VLMPipeline,
               "image": genai.Text2ImagePipeline}[model["pipeline"]]
        if model['pipeline'] == 'image' and model.get('_image_operation') == 'edit':
            cls = genai.Image2ImagePipeline
        properties, target, placement_report = _pipeline_setup(genai, ov, model)
        affinity_reader = properties.pop('_lazyrig_affinity_reader', None)
        pipeline = cls(model["path"], target, **properties)
        if affinity_reader is not None:
            affinity_reader.verify_loaded(placement_report)
        selected = model.get('_placement', {}).get('devices', [])
        if model['pipeline'] in {'llm', 'vlm', 'vlm-text'} and len(selected) > 1:
            conn.send({'kind': 'load_stage', 'stage': 'validation'})
            def probe_progress(checkpoint, generated):
                conn.send({'kind': 'worker_diagnostics', 'validation': {
                    'checkpoint': checkpoint, 'reported_tokens': generated, 'expected_tokens': 2}})
            placement_report['runtime_validation'] = _validate_split_inference(pipeline, probe_progress)
        if model['pipeline'] == 'image':
            configure_diffusion(genai, pipeline, model)
        conn.send({"kind": "ready", "pid": os.getpid(), 'placement_preflight': placement_report,
                   'cooperative_text_stop': True})
        while True:
            request = conn.recv()
            if request["action"] == "text":
                import openvino as ov
                import numpy as np
                tokenizer = pipeline.get_tokenizer()
                pieces, counts = [], [0]
                last_emit = [0.0]
                stop_requested = [False]

                def on_text(piece):
                    pieces.append(piece)
                    return genai.StreamingStatus.RUNNING

                # Preserve think/tool delimiters until the API parser has
                # separated channels; default decoding may remove them.
                decoder = genai.TextStreamer(tokenizer, on_text, {"skip_special_tokens": False})

                class Streamer(genai.StreamerBase):
                    def __init__(self):
                        super().__init__()
                        self.finished = False
                    def write(self, tokens):
                        if _text_stop_requested(conn, stop_requested):
                            return genai.StreamingStatus.CANCEL
                        ids = tokens if isinstance(tokens, list) else [tokens]
                        counts[0] += len(ids)
                        if timer:
                            timer.observe(counts[0])
                        decoder.write(tokens)
                        if time.monotonic() - last_emit[0] >= 0.02:
                            conn.send({"kind": "text", "text": "".join(pieces), "generated_tokens": counts[0], "accuracy": "exact",
                                       **({"timings": timer.snapshot(counts[0])} if timer else {})})
                            last_emit[0] = time.monotonic()
                        return genai.StreamingStatus.RUNNING
                    def end(self):
                        if not self.finished:
                            decoder.end()
                            self.finished = True

                config = pipeline.get_generation_config()
                configure_output(genai, config, request.get("response_format"))
                config.max_new_tokens = request["max_tokens"]
                config.do_sample = request.get("temperature", 0.0) > 0
                if config.do_sample:
                    config.temperature = request["temperature"]
                config.apply_chat_template = False
                config.num_return_sequences = 1
                config.num_beams = 1
                configure_generation(config, model, {**request.get("generation", {}), "temperature": request.get("temperature", 0)})
                apply_sampling(config, request.get('generation', {}))
                conn.send({'kind': 'sampling', 'settings': sampling_snapshot(config)})
                prepared = request["prepared"]
                inputs = (ov.Tensor(np.array([prepared["token_ids"]], dtype=np.int64))
                          if model["pipeline"] == "llm" else prepared["text"])
                timer = GenerationTimer(prepared["context"]["input_tokens"]) if request.get("timings_per_token") else None
                streamer = Streamer()
                image_options = {}
                if prepared.get("images"):
                    from PIL import Image
                    tensors = []
                    for info in prepared["images"]:
                        with Image.open(info["path"]) as im:
                            tensors.append(ov.Tensor(np.array(im.convert("RGB"), dtype=np.uint8)[None]))
                    image_options["images"] = tensors
                result = pipeline.generate(inputs, generation_config=config, streamer=streamer, **image_options)
                streamer.end()
                if _text_stop_requested(conn, stop_requested):
                    # CANCEL rolls back this generation in GenAI. Acknowledge
                    # only after native generate() and streamer cleanup return.
                    conn.send({'kind': 'cancelled', 'worker_reusable': True})
                    continue
                if timer:
                    timer.finish()
                if hasattr(result, "tokens"):
                    text = tokenizer.decode(result.tokens[0], skip_special_tokens=False)
                    generated = len(result.tokens[0])
                else:
                    text = "".join(pieces) if counts[0] else (result.texts[0] if hasattr(result, "texts") else str(result))
                    generated = counts[0]
                runtime_input = None
                if hasattr(result, "perf_metrics"):
                    runtime_input = result.perf_metrics.get_num_input_tokens()
                    generated = result.perf_metrics.get_num_generated_tokens()
                conn.send({"kind": "result", "text": text, "generated_tokens": generated,
                           "input_tokens": runtime_input, "accuracy": "exact",
                           **({"timings": timer.snapshot(generated, runtime_input, getattr(result, "perf_metrics", None))} if timer else {})})
            elif request["action"] in {"image", "image_edit"}:
                target = Path(request["output_path"])
                inference = _native_image(conn, pipeline, model, request)
                conn.send({"kind": "result", "path": str(target), 'inference': inference})
            elif request['action'] == 'cancel_text':
                # The result can win the race with Stop. Drain its subsequent
                # control message before acknowledging an idle, reusable worker.
                conn.send({'kind': 'cancelled', 'worker_reusable': True})
            else:
                raise ValueError("Unsupported worker action")
    except EOFError:
        pass
    except Exception as exc:
        try:
            failed_report = getattr(exc, 'placement_preflight', None) or placement_report
            conn.send({"kind": "error", "error": type(exc).__name__ + ": " + str(exc),
                       **kernel_build_failure(exc),
                       **attention_cache_failure(exc, properties),
                       **({'placement_preflight': failed_report} if failed_report else {})})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()
        if fault_stream is not None:
            import faulthandler
            faulthandler.disable()
            fault_stream.close()

def _spawn_child(conn, mode, model, job_name):
    if os.name != 'nt':
        os.setsid()
    _child(conn, mode, model, job_name)


class ProcessWorker:
    def __init__(self, mode, model, timeout, cancelled):
        self.conn, self.process, self._job = None, None, None
        self._placement_view = None
        self._closed = False
        self.load_memory = {}
        self.worker_diagnostics = {}
        self._fault_log_path = None
        memory_guard = LoadMemoryGuard()
        try:
            memory_guard.check(force=True)
            if mode == 'openvino':
                from .placement_view import prepare_manual_view

                def preparation_checkpoint():
                    if cancelled.is_set():
                        raise Cancelled('Job cancelled')
                    check_deadline(cancelled)
                    memory_guard.check()

                model, self._placement_view = prepare_manual_view(model, preparation_checkpoint)
            self._fault_log_path = create_fault_log()
            model = {**model, '_worker_fault_log': self._fault_log_path}
            if mode == "openvino" and model.get("runtime_profile", "default") != "default":
                from .conversion import ensure_profile, ROOT
                python = ensure_profile(model["runtime_profile"], cancelled, lambda t: print(t, end="", flush=True),
                                        ROOT / ".runtime" / "profile-setup.log")
                self._external(python, mode, model, cancelled, memory_guard=memory_guard)
            else:
                self._prepare_load_memory(memory_guard)
                ctx = mp.get_context("spawn")
                self.conn, child = ctx.Pipe()
                self.process = ctx.Process(target=_spawn_child, args=(child, mode, model, self._job.name if self._job else None), daemon=True)
                try:
                    self.process.start()
                finally:
                    child.close()
            event = next(self._receive(timeout, cancelled, memory_guard=memory_guard, stage='loading'))
            if event["kind"] != "ready":
                raise RuntimeError("Worker did not become ready")
            self._report_load_memory(memory_guard, 'ready')
        except BaseException as operation_error:
            if self._placement_view is None:
                self._placement_view = getattr(operation_error, 'placement_view', None)
            report = self._report_load_memory(memory_guard, 'error', publish=False)
            self._attach_diagnostics(operation_error)
            try:
                self.close()
            except BaseException as cleanup_error:
                error = WorkerCleanupError(self, operation_error, cleanup_error)
                error.load_memory = report
                error.worker_diagnostics = operation_error.worker_diagnostics
                raise error from cleanup_error
            operation_error.load_memory = report
            raise

    def _report_load_memory(self, guard, stage, publish=True):
        job = getattr(self, '_job', None)
        peak_commit = None
        if job is not None and hasattr(job, 'peak_commit_bytes'):
            try:
                peak_commit = job.peak_commit_bytes()
            except OSError:
                pass  # Unavailable telemetry must not replace the load result.
        report = {**guard.diagnostics(), 'stage': stage,
                  'job_commit_limit_bytes': getattr(job, 'memory_limit_bytes', None),
                  'job_peak_commit_bytes': peak_commit,
                  'job_commit_limit_scope': 'worker lifetime; committed memory, not VRAM'}
        self.load_memory = report
        if publish:
            publish_load_memory(report)
        return dict(report)

    def _attach_diagnostics(self, error):
        error.worker_diagnostics = copy.deepcopy(getattr(self, 'worker_diagnostics', {}))
        error.worker_diagnostics['fault_log'] = read_fault_log(getattr(self, '_fault_log_path', None))

    def _prepare_load_memory(self, guard):
        # Refresh after any isolated-runtime setup and before the native process
        # starts. The worker joins this capped Windows job before importing OV.
        guard.check(force=True)
        if os.name == 'nt':
            from .windows_job import WindowsJob
            self._job = WindowsJob(memory_limit_bytes=guard.allocation_limit_bytes)
        self._report_load_memory(guard, 'initial')

    def _exit_error(self, cancelled, stage, operation_deadline=None):
        # EOF can arrive just before the OS publishes the exit status. Reap for
        # a bounded interval before close() terminates anything or closes handles.
        # A still-live process with closed IPC must not be labelled a crash, nor
        # may its later cleanup/termination status replace the original evidence.
        if cancelled.is_set():
            raise Cancelled('Job cancelled')
        process = self.process
        deadline = time.monotonic() + WORKER_EXIT_GRACE_SECONDS
        if operation_deadline is not None:
            deadline = min(deadline, operation_deadline)
        while process.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            process.join(time_budget(min(.05, remaining), cancelled))
            if cancelled.is_set():
                raise Cancelled('Job cancelled')
        if cancelled.is_set():
            raise Cancelled('Job cancelled')
        exited = not process.is_alive()
        error = WorkerProcessError(getattr(process, 'exitcode', None) if exited else None,
                                   pid=process.pid, stage=stage, exited=exited)
        self._attach_diagnostics(error)
        return error

    def _receive(self, timeout, cancelled, memory_guard=None, stage='generation', cooperative_stop=False):
        deadline = time.monotonic() + time_budget(timeout, cancelled)
        stop_deadline = None
        while True:
            if cancelled.is_set():
                if not cooperative_stop:
                    raise Cancelled("Job cancelled")
                if stop_deadline is None:
                    try:
                        self.conn.send({'action': 'cancel_text'})
                    except (EOFError, OSError) as exc:
                        raise Cancelled('Generation stop required worker shutdown') from exc
                    stop_deadline = time.monotonic() + TEXT_STOP_GRACE_SECONDS
            if stop_deadline is not None and time.monotonic() >= stop_deadline:
                raise Cancelled('Generation stop required worker shutdown')
            if stop_deadline is None and time.monotonic() > deadline:
                raise TimeoutError("Model operation exceeded its configured deadline")
            if memory_guard is not None:
                memory_guard.check()
            try:
                available = self.conn.poll(.05 if stop_deadline is not None else time_budget(0.05, cancelled))
                item = self.conn.recv() if available else None
            except (EOFError, ConnectionError) as exc:
                raise self._exit_error(cancelled, stage, deadline) from exc
            if available:
                if item.get('kind') == 'worker_diagnostics':
                    if not hasattr(self, 'worker_diagnostics'):
                        self.worker_diagnostics = {}
                    for key in ('runtime', 'fault_handler_enabled', 'validation'):
                        if key in item:
                            self.worker_diagnostics[key] = copy.deepcopy(item[key])
                    continue
                if stop_deadline is not None:
                    if item.get('kind') == 'cancelled' and item.get('worker_reusable') is True:
                        raise Cancelled(worker_reusable=True)
                    if item.get('kind') == 'error':
                        raise Cancelled('Generation stop required worker shutdown')
                    # Drain queued sampling/text/result messages, including a
                    # completed result followed by the idle-stop acknowledgement.
                    continue
                if item["kind"] == "error":
                    error = NativeWorkerError(item["error"])
                    error.worker_stage = stage
                    if item.get('error_code') == 'gpu_kernel_build_failed':
                        error.error_code = item['error_code']
                    if item.get('error_code') == 'attention_cache_budget_exceeded':
                        error.error_code = item['error_code']
                        error.cache_settings = item.get('cache_settings', {})
                    if item.get('placement_preflight'):
                        error.placement_preflight = item['placement_preflight']
                    raise error
                if item['kind'] == 'load_stage' and stage == 'loading' and item.get('stage') == 'validation':
                    stage = 'validation'
                    if memory_guard is not None:
                        self._report_load_memory(memory_guard, 'validating')
                    continue
                if item['kind'] == 'ready':
                    if memory_guard is not None:
                        memory_guard.check(force=True)
                    self.worker_pid = item.get('pid', self.process.pid)
                    self.placement_preflight = item.get('placement_preflight', {})
                    self.cooperative_text_stop = item.get('cooperative_text_stop') is True
                yield item
                if item["kind"] in {"result", "ready"}:
                    return
            elif not self.process.is_alive():
                raise self._exit_error(cancelled, stage, deadline)

    def _external(self, python, mode, model, cancelled, memory_guard=None):
        # Authenticated loopback IPC keeps nightly libraries separate from the LLM.
        from multiprocessing.connection import Listener
        from .conversion import quiet_environment, ROOT
        secret = os.urandom(32)
        with Listener(("127.0.0.1", 0), authkey=secret) as listener:
            listener._listener._socket.settimeout(time_budget(.2, cancelled))
            env = quiet_environment()
            env["LAZYRIG_WORKER_SECRET"] = secret.hex()
            env["HF_HUB_OFFLINE"] = "1"
            if memory_guard is not None:
                self._prepare_load_memory(memory_guard)
            elif os.name == 'nt' and not getattr(self, '_job', None):
                from .windows_job import WindowsJob
                self._job = WindowsJob()
            process = subprocess.Popen([str(python), "-m", "lazyrig.worker", str(listener.address[1])],
                                       cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                       start_new_session=os.name != 'nt')
            self.process = ExternalProcess(process)
            deadline = time.monotonic() + time_budget(30, cancelled)
            try:
                while True:
                    if cancelled.is_set():
                        raise Cancelled()
                    if memory_guard is not None:
                        memory_guard.check()
                    if not self.process.is_alive():
                        raise self._exit_error(cancelled, 'connection')
                    if time.monotonic() > deadline:
                        raise RuntimeError("Isolated model runtime could not connect")
                    try:
                        self.conn = listener.accept()
                        break
                    except socket.timeout:
                        continue
                self.conn.send({"mode": mode, "model": model, 'job_name': self._job.name if getattr(self, '_job', None) else None})
            except BaseException:
                self.close()
                raise

    def run(self, request, timeout, cancelled):
        try:
            if cancelled.is_set():
                raise Cancelled("Job cancelled", worker_reusable=self.process.is_alive())
            try:
                self.conn.send(request)
            except (EOFError, ConnectionError) as exc:
                raise self._exit_error(cancelled, 'generation') from exc
            yield from self._receive(time_budget(timeout, cancelled), cancelled,
                                     cooperative_stop=request.get('action') == 'text' and
                                     getattr(self, 'cooperative_text_stop', False))
        except Cancelled as exc:
            if not exc.worker_reusable:
                self.close()
            raise
        except BaseException as error:
            self._attach_diagnostics(error)
            self.close()
            raise

    def close(self):
        if getattr(self, '_closed', False):
            self._close_placement_view()
            self._close_fault_log()
            return
        process = getattr(self, 'process', None)
        job = getattr(self, '_job', None)
        try:
            # Windows membership belongs to the native Python process, not just
            # the venv launcher. Wait for all job members before releasing a lane.
            if job:
                job.terminate()
            if process and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(3)
                # Clear descendants in our POSIX session even if the leader has
                # already exited. All external launches own a new session.
                if os.name != 'nt':
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.is_alive():
                    process.kill()
                process.join(3)
                if process.is_alive():
                    raise RuntimeError('Worker could not be terminated; device must not be reused')
            if job:
                job.close()
            if process:
                pid = process.pid
                process.close()  # Release OS handles after termination was verified.
                self.process = StoppedProcess(pid)
            self._closed = True
            self._close_placement_view()
            self._close_fault_log()
        finally:
            if getattr(self, 'conn', None):
                self.conn.close()

    def _close_placement_view(self):
        view = getattr(self, '_placement_view', None)
        if view is not None:
            # In particular, do not unlink files while Windows still holds the
            # native process/job handles. Preserve ownership if cleanup fails so
            # a subsequent close() can retry without repeating process shutdown.
            view.close()
            self._placement_view = None

    def _close_fault_log(self):
        path = getattr(self, '_fault_log_path', None)
        if path:
            try:
                os.unlink(path)
            except OSError:
                return  # Best effort; never turn successful GPU cleanup into a failure.
            self._fault_log_path = None


class ExternalProcess:
    def __init__(self, process):
        self.process = process

    def is_alive(self):
        return self.process.poll() is None

    @property
    def pid(self):
        return self.process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def join(self, timeout=None):
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        # Popen exposes the Windows process handle; wait() alone leaves it open.
        if os.name == 'nt':
            self.process._handle.Close()


class StoppedProcess:
    def __init__(self, pid):
        self.pid = pid

    def is_alive(self):
        return False


if __name__ == "__main__":
    from multiprocessing.connection import Client
    connection = Client(("127.0.0.1", int(sys.argv[1])), authkey=bytes.fromhex(os.environ.pop("LAZYRIG_WORKER_SECRET")))
    configuration = connection.recv()
    _child(connection, configuration["mode"], configuration["model"], configuration.get('job_name'))
