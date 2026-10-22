"""Compact runtime feedback; never log prompts, generated text or credentials."""
import logging
import math
import time
from .worker_status import exit_status

LOGGER = logging.getLogger('lazyrig.runtime')
# Never print arbitrary exception messages: these may contain generated tool
# arguments, source text or credentials. Full errors remain in the job details.
FAILURE_REASONS = {
    'invalid_tool_call': 'invalid_tool_call (model tool-call validation failed)',
    'incomplete_tool_call': 'incomplete_tool_call (model tool call was incomplete)',
    'undeclared_tool': 'undeclared_tool (model selected an unavailable tool)',
    'incomplete_reasoning': 'incomplete_reasoning (model stopped inside reasoning)',
    'invalid_response_format': 'invalid_response_format (model output was not valid JSON)',
    'request_timeout': 'request_timeout (request deadline exceeded)',
    'gpu_resource_error': 'gpu_resource_error (GPU recovery exhausted)',
    'gpu_kernel_build_failed': 'gpu_kernel_build_failed (OpenCL kernel compilation failed; run diagnose-gpu.bat)',
    'storage_failure': 'storage_failure (result persistence failed)',
    'host_memory_pressure': 'host_memory_pressure (model loading stopped to preserve system RAM)',
    'manual_split_rejected': 'manual_split_rejected (requested model split could not be loaded)',
    'split_inference_failed': 'split_inference_failed (split failed its prefill/decode check)',
    'attention_cache_budget_exceeded': 'attention_cache_budget_exceeded (check attention cache and sequence limit in Model Settings)',
    'worker_process_exited': 'worker_process_exited (model worker ended without a native error)',
    'worker_connection_closed': 'worker_connection_closed (model worker communication ended)',
    'device_budget_exceeded': 'device_budget_exceeded (reservation exceeds device budget)',
    'model_weight_budget_exceeded': 'model_weight_budget_exceeded (stored weights exceed GPU budget)',
    'mtp_model_weight_budget_exceeded': 'mtp_model_weight_budget_exceeded (main model exceeds GPU budget; MTP disables model splitting)',
}


def configure_logging():
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def number(value, suffix=''):
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return f'{value:,.1f}{suffix}'
    return 'n/a'


def context_label(context):
    used, limit = context.get('used_tokens', 0), context.get('context_limit', 0)
    accuracy = context.get('accuracy', 'estimated')
    note = '; visual tokens pending' if context.get('visual_tokens_included') is False else ''
    percent = 100 * used / limit if limit else 0
    return f'context {used:,}/{limit:,} ({percent:.1f}%; {accuracy}{note})'


class Reporter:
    def __init__(self, config, job, clock=time.monotonic):
        self.enabled = config.get('terminal_metrics', True)
        self.job = job
        self.clock, self.started = clock, clock()
        self.prefix = f'{job["id"][:8]} | '
        self.model = self.image_model = job['model']
        self.stage = 'initial'
        self.last_text = self.last_progress = float('-inf')
        self.image_started = None
        self.loads = {}

    def write(self, message):
        if self.enabled:
            LOGGER.info('%s%s', self.prefix, message)

    def event(self, kind, **data):
        model = data.get('model', self.model)
        if kind == 'image_started':
            self.start_image(model)
        elif kind == 'image_retry':
            self.write(f'{model} | {data["recovery"]["reason"]} | restarting image worker; automatic retry 1/1')
        elif kind == 'image_recovered':
            self.write(f'{model} | image recovered after automatic retry 1/1')
        elif kind == 'response_rejected':
            reason = FAILURE_REASONS.get(data.get('code'), 'response validation failed')
            self.write(f'{model} | response rejected: {reason} | model remains loaded')
        elif kind == 'load_memory':
            if data.get('stage') == 'validating':
                self.write(f'{model} | checking split inference (prefill and decode)')
                return
            stage = {'initial': 'start', 'starting': 'start', 'ready': 'ready', 'failed': 'failed'}.get(data.get('stage'), 'sample')
            if data.get('available_gib') is None:
                self.write(f'{model} | load memory guard v2 ({stage}) | system RAM measurement unavailable')
            else:
                message = (f'{model} | load memory guard v2 ({stage}) | RAM available '
                           f'{number(data.get("available_gib"), " GiB")} | headroom '
                           f'{number(data.get("headroom_gib"), " GiB")}')
                limit = data.get('job_commit_limit_bytes')
                if isinstance(limit, int) and limit > 0:
                    message += f' | Windows worker commit limit {number(limit / 1024**3, " GiB")}'
                peak = data.get('job_peak_commit_bytes')
                if type(peak) is int and peak >= 0:
                    message += f' | worker peak commit {number(peak / 1024**3, " GiB")}'
                if data.get('minimum_available_gib') is not None:
                    message += f' | lowest sampled RAM {number(data["minimum_available_gib"], " GiB")}'
                self.write(message)
        elif kind == 'placement_plan':
            targets = ' -> '.join(a['target'] for a in data.get('attempts', []))
            self.write(f'{model} | placement plan ({data.get("mode", "single")}) | {targets}')
            split = data.get('split', {})
            if split.get('status') == 'disabled_mtp':
                self.write(f'{model} | {split["reason"]}')
        elif kind == 'placement_attempt':
            if data.get('status') in {'skipped', 'failed', 'cancelled'}:
                # Keep terminal output one line, while the saved job retains the
                # original bounded native message including its line breaks.
                reason = ' '.join(str(data.get('reason', '')).split())[:1000]
                self.write(f'{model} | {data.get("target")} | {data.get("status")} | {data.get("reason_code")} | {reason}')
                diagnostics = data.get('worker_diagnostics', {})
                probe = diagnostics.get('validation', {})
                tokens = probe.get('reported_tokens')
                if type(tokens) is int and 0 <= tokens <= 2:
                    self.write(f'{model} | split validation reported {tokens}/2 tokens; '
                               'runtime versions and fault capture are in Copy load details')
        elif kind == 'placement_fallback':
            reason = {'weight_budget': 'stored weights exceed GPU budget',
                      'reservation_budget': 'reservation exceeds device budget',
                      'host_memory_pressure': 'system RAM headroom reached',
                      'split_inference_failed': 'split inference check failed',
                      'gpu_kernel_build_failed': 'GPU kernel compilation failed',
                      'native_load_rejected': 'native load rejected'}.get(data.get('reason_code'))
            self.write(f'{model} | automatic device fallback' + (f' | {reason}' if reason else '') +
                       f' | trying {data.get("to_target", data.get("device", "next placement"))}')
        elif kind in {'placement_selected', 'placement_rejected'}:
            placement = data.get('placement', {})
            rejected = kind == 'placement_rejected'
            if rejected:
                self.loads.pop(model, None)
                self.write(f'{model} | manual split rejected | {placement.get("target", data.get("device", "?"))}')
            elif len(placement.get('devices', [])) > 1 or 'fallback' in placement.get('label', '').lower() or placement.get('shared_memory_attempt'):
                self.write(f'{model} | {placement.get("label", "Device allocation")} | {placement.get("target", data.get("device", "?"))}')
            preflight = data.get('preflight') or {}
            if preflight.get('scope') == 'exported_language_graph' and preflight.get('devices'):
                requested = preflight.get('requested_weight_shares', {})
                shares = []
                for device in placement.get('devices', []):
                    value = preflight['devices'].get(device, {})
                    shares.append(f'{device} {number(value.get("weight_share_percent"), "%")}' +
                                  (f' (requested {number(requested[device], "%")})' if device in requested else ''))
                if shares:
                    heading = 'closest queried language weights' if rejected else 'preflight language weights'
                    if preflight.get('partition', {}).get('method') == 'explicit_affinity':
                        heading = 'assigned language weights'
                    self.write(f'{model} | {heading} | ' + ' | '.join(shares) +
                               ' | exported graph; not measured VRAM')
                state_order = preflight.get('state_order', {})
                assignments = state_order.get('assign_count')
                if (state_order.get('method') == 'assign_dependency_depth' and
                        state_order.get('reordered') is True and type(assignments) is int and
                        2 <= assignments <= 1_000_000):
                    self.write(f'{model} | manual split graph | {assignments:,} state updates ordered by dependency')
                calibration = preflight.get('calibration', {})
                count, maximum = calibration.get('query_count'), calibration.get('max_queries')
                if type(count) is int and type(maximum) is int and 1 <= count <= maximum <= 9:
                    reason = calibration.get('stop_reason')
                    known_reasons = {'within tolerance', 'query budget reached',
                                     'calibration time budget reached', 'tested graph-boundary plateau'}
                    self.write(f'{model} | manual split preflight checks {count}/{maximum}' +
                               (f' | {reason}' if reason in known_reasons else ''))
                input_device = preflight.get('language_input_device')
                if not rejected and input_device in placement.get('devices', []):
                    self.write(f'{model} | language input device {input_device} (preflight)')
        elif kind == 'loading':
            self.loads[model] = self.clock()
            message = f'{model} | loading on {data.get("device", "?")}'
            if data.get('mtp_device'):
                message += f' | MTP on {data["mtp_device"]}'
            self.write(message)
        elif kind == 'loaded':
            elapsed = self.clock() - self.loads.pop(model, self.clock())
            self.write(f'{model} | loaded in {elapsed:.2f}s')
        elif kind in {'unloading', 'unloaded', 'unload_failed'}:
            self.write(f'{model} | {kind.replace("_", " ")}')
        elif kind == 'generation_stopped' and data.get('worker_retained'):
            self.write(f'{model} | generation stopped | model kept loaded')
        elif kind in {'restoring', 'resuming'}:
            self.write(f'{model} | ' + ('reusing loaded worker' if data.get('worker_reused') else 'restoring worker'))
        elif kind == 'progress':
            now = self.clock()
            if now - self.last_progress >= 2 or data['step'] == data['steps']:
                elapsed = now - (self.image_started if self.image_started is not None else self.started)
                self.write(f'{self.image_model} | image step {data["step"]}/{data["steps"]} | {elapsed:.1f}s')
                self.last_progress = now

    def start_text(self, model, stage, context):
        self.model, self.stage = model, stage
        self.last_text = self.clock()
        self.write(f'{model} | {stage} | input {context["input_tokens"]:,} | output budget {context["output_budget"]:,} | {context_label(context)}')

    def text(self, context, timings=None, final=False):
        now = self.clock()
        if not final and now - self.last_text < 2:
            return
        self.last_text = now
        timings = timings or {}
        native = timings.get('x_openvino_metrics', {})
        rate = native.get('tokens_per_second', timings.get('predicted_per_second'))
        basis = 'native' if 'tokens_per_second' in native else 'worker clock'
        ttft = native.get('ttft_ms', timings.get('prompt_ms'))
        prompt_rate = timings.get('prompt_per_second')
        self.write(f'{self.model} | {self.stage} {"generation complete" if final else "generating"} | '
                   f'prompt {number(prompt_rate, " t/s")} (TTFT estimate) | '
                   f'output {context["generated_tokens"]:,} | decode {number(rate, " t/s")} ({basis}) | '
                   f'TTFT {number(ttft, " ms")} | {context_label(context)}')

    def start_image(self, model):
        self.image_model = model
        self.image_started = self.clock()
        self.last_progress = float('-inf')
        self.write(f'{model} | generating image')

    def finish(self, status):
        reason = (' | reason: ' + FAILURE_REASONS.get(self.job.get('error_code'),
                  'request failed; see job details') if status == 'failed' else '')
        if status == 'failed' and self.job.get('error_code') in {'worker_process_exited', 'worker_connection_closed'}:
            reason += ' | ' + exit_status(self.job.get('worker_exit'))
        self.write(f'job {status}{reason} | total {self.clock() - self.started:.2f}s')
