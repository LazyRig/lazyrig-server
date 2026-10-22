"""Serializable model controls and validation for native OpenVINO properties.

Omitted properties retain runtime defaults except the serial-worker sequence
limit, which defaults to one. Actual device/model support is
checked by OpenVINO when compiling; this catalog is not a hardware capability list.
"""
import math
import copy
from .mtp import enabled as mtp_enabled, ensure_export, device as mtp_device, draft_properties as mtp_draft_properties, cpu_threads as mtp_cpu_threads


PRECISIONS = [("u8", "INT8 · u8"), ("u4", "INT4 · u4"), ("f16", "FP16"),
              ("bf16", "BF16"), ("f32", "FP32"), ("undefined", "Automatic precision")]


def _field(scope, key, label, kind="integer", **extra):
    return {"scope": scope, "key": key, "label": label, "kind": kind, **extra}


CONTROLS = [
    _field("plugin_config", "KV_CACHE_PRECISION", "KV-cache precision", "select", section="kv", options=PRECISIONS),
    _field("scheduler_config", "cache_size", "Initial attention cache (GiB)", section="kv", minimum=0,
           help="Whole GiB, including KV and linear-attention state. Blank or 0 allocates on demand. This is not a hard RAM or VRAM limit."),
    _field("scheduler_config", "enable_prefix_caching", "Reuse cached prompt prefixes", "boolean", section="kv"),
    _field("scheduler_config", "max_num_batched_tokens", "Tokens per scheduler step", section="kv", minimum=1),
    _field("scheduler_config", "max_num_seqs", "Scheduler sequence limit", section="kv", minimum=1,
           default_label="1 · one request",
           help="Blank uses 1: each LazyRig model worker runs one request at a time. Higher values can reserve more attention state; they do not enable concurrent requests."),
    _field("scheduler_config", "cache_interval_multiplier", "Hybrid cache checkpoint interval", section="kv", minimum=0,
           help="Block-size multiplier for hybrid / linear-attention models. Blank keeps adaptive allocation; larger values use less checkpoint memory."),
    _field("plugin_config", "KEY_CACHE_PRECISION", "Key precision override", "select", section="kv_advanced", options=PRECISIONS),
    _field("plugin_config", "VALUE_CACHE_PRECISION", "Value precision override", "select", section="kv_advanced", options=PRECISIONS),
    _field("plugin_config", "KEY_CACHE_GROUP_SIZE", "Key quantization group size", section="kv_advanced", minimum=0),
    _field("plugin_config", "VALUE_CACHE_GROUP_SIZE", "Value quantization group size", section="kv_advanced", minimum=0),
    _field("plugin_config", "KEY_CACHE_QUANT_MODE", "Key quantization mode", "select", section="kv_advanced",
           options=[("BY_CHANNEL", "By channel"), ("BY_TOKEN", "By token")]),
    _field("scheduler_config", "dynamic_split_fuse", "Dynamic split / fuse", "boolean", section="kv_advanced"),
    _field("plugin_config", "PERFORMANCE_HINT", "Performance preference", "select", section="runtime",
           options=[("LATENCY", "Latency"), ("THROUGHPUT", "Throughput"), ("CUMULATIVE_THROUGHPUT", "Cumulative throughput")]),
    _field("plugin_config", "INFERENCE_PRECISION_HINT", "Compute precision hint", "select", section="runtime",
           options=[("f16", "FP16"), ("bf16", "BF16"), ("f32", "FP32"), ("undefined", "Automatic precision")]),
    _field("plugin_config", "INFERENCE_NUM_THREADS", "CPU inference threads", section="runtime", minimum=1),
    _field("plugin_config", "NUM_STREAMS", "Inference streams", "streams", section="runtime",
           help="Positive integer, AUTO, or NUMA (CPU). Blank uses the runtime default."),
    _field("plugin_config", "CACHE_DIR", "Compiled-model cache directory", "text", section="runtime",
           help="Server folder for compiled models. This is separate from the conversation's KV cache."),
    _field("plugin_config", "EXECUTION_MODE_HINT", "Execution preference", "select", section="runtime",
           options=[("PERFORMANCE", "Performance"), ("ACCURACY", "Accuracy")],
           help="Controls runtime optimization priorities. Accuracy may reduce performance; it does not restore precision already lost in quantized weights."),
    _field("plugin_config", "COMPILATION_NUM_THREADS", "GPU compilation threads", section="runtime", minimum=1, device_kind="GPU",
           help="Limits GPU model compilation threads. Applies only to selected GPUs; it does not change decoding threads. Blank uses the runtime default."),
    _field("plugin_config", "SCHEDULING_CORE_TYPE", "CPU core preference", "select", section="cpu_runtime", device_kind="CPU",
           options=[("ANY_CORE", "Any core"), ("PCORE_ONLY", "Performance cores"), ("ECORE_ONLY", "Efficiency cores")],
           help="Applies when the main model uses CPU, including CPU offload. Core types depend on the processor and OS; the runtime chooses a supported configuration."),
    _field("plugin_config", "ENABLE_HYPER_THREADING", "CPU hyper-threading", "boolean", section="cpu_runtime", device_kind="CPU",
           help="Use logical sibling threads during CPU inference when supported. This does not change the BIOS setting. Blank keeps the runtime default."),
    _field("plugin_config", "ENABLE_CPU_PINNING", "Pin CPU inference threads", "boolean", section="cpu_runtime", device_kind="CPU",
           help="Request CPU thread affinity where supported by the runtime and OS. Applies only to main-model CPU inference, including offload."),
]


def _integer(value, minimum, key):
    if type(value) is not int or not minimum <= value <= 2147483647:
        raise ValueError(f"{key} must be a whole number between {minimum} and 2147483647")


def validate_runtime(model):
    plugin, scheduler = model.get("plugin_config", {}), model.get("scheduler_config", {})
    known_scheduler = {c["key"] for c in CONTROLS if c["scope"] == "scheduler_config"}
    known_scheduler.update({"num_kv_blocks", "num_linear_attention_blocks"})
    for key, value in scheduler.items():
        if key not in known_scheduler:
            raise ValueError("Unsupported scheduler setting: " + key)
        if key in {"num_kv_blocks", "num_linear_attention_blocks"}:
            _integer(value, 0, key)
    for control in CONTROLS:
        values, key = (plugin if control["scope"] == "plugin_config" else scheduler), control["key"]
        if key not in values:
            continue
        value, kind = values[key], control["kind"]
        if kind == "integer":
            _integer(value, control.get("minimum", 0), key)
        elif kind == "boolean" and type(value) is not bool:
            raise ValueError(key + " must be true or false")
        elif kind == "select" and value not in [v for v, _ in control["options"]]:
            raise ValueError(key + " must be one of: " + ", ".join(v for v, _ in control["options"]))
        elif kind == "streams":
            if not isinstance(value, str) or value not in {"AUTO", "NUMA"}:
                _integer(value, 1, key)
        elif kind == "text" and not isinstance(value, str):
            raise ValueError(key + " must be a string")
    if scheduler.get("cache_size", 0) and scheduler.get("num_kv_blocks", 0):
        raise ValueError("Set either cache_size or num_kv_blocks, not both")
    if scheduler.get("enable_prefix_caching") and scheduler.get("cache_interval_multiplier") == 0:
        raise ValueError("cache_interval_multiplier must be positive when prefix caching is enabled")
    if model["pipeline"] == "image" and scheduler:
        raise ValueError("Scheduler / KV-cache settings apply to text models only")
    for key, value in plugin.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("OpenVINO settings must be finite: " + key)


def effective_cpu_thread_default(model, default):
    """Return a device CPU default only where no model-wide override applies.

    Individual VLM role overrides do not suppress defaults for other roles.
    The scheduler supplies an actual resolved plan, never the entire Auto pool.
    """
    if type(default) is not int or default < 1:
        return None
    plan = model.get('_placement', {})
    target = plan.get('target', model.get('device', ''))
    selected = plan.get('devices') or (target.split(':', 1)[1].split(',') if target.startswith('HETERO:') else [target])
    cpu_draft = mtp_enabled(model) and mtp_device(model).split('.')[0] == 'CPU'
    if 'CPU' not in selected and not cpu_draft:
        return None
    if cpu_draft and 'CPU' not in selected and mtp_cpu_threads(model) is not None:
        return None
    for properties in (model.get('plugin_config', {}), plan.get('properties', {})):
        if 'INFERENCE_NUM_THREADS' in properties:
            return None
        per_device = properties.get('DEVICE_PROPERTIES', {})
        cpu = per_device.get('CPU', {})
        target_properties = per_device.get(target, {})
        target_cpu = target_properties.get('DEVICE_PROPERTIES', {}).get('CPU', {})
        if any('INFERENCE_NUM_THREADS' in values for values in (cpu, target_properties, target_cpu)):
            return None
    return default


def _vlm_embedding_device(model):
    """Use the verified language input device for remote embedding handoff."""
    plan = model.get('_placement', {})
    selected = plan.get('devices', [])
    if (model.get('pipeline') not in {'vlm', 'vlm-text'} or
            model.get('placement', {}).get('mode') != 'manual' or
            sum(device.split('.')[0] == 'GPU' for device in selected) < 2):
        return None
    report = model.get('_placement_preflight', {})
    if report.get('status') != 'preflight_verified':
        return None
    device = report.get('language_input_device')
    if not isinstance(device, str) or not device:
        raise ValueError('Cannot verify the language inputs_embeds device for this multi-GPU VLM split')
    if device not in selected:
        raise ValueError('The verified language input device is outside the selected allocation')
    return device


def pipeline_target(model):
    """Allow per-component priorities only for a verified multi-GPU VLM split.

    A colon-qualified HETERO target fixes priorities for every GenAI component.
    Bare HETERO lets its language model keep the split while text embeddings
    run on the GPU that consumes their remote tensors.
    """
    if _vlm_embedding_device(model) is not None:
        return 'HETERO'
    return model.get('_placement', {}).get('target', model['device'])


def _merge_device_properties(base, overlay):
    result = copy.deepcopy(base)
    for device, values in overlay.items():
        result.setdefault(device, {}).update(copy.deepcopy(values))
    return result


def _overlay_inherited_device_values(per_device, values):
    """Keep higher-priority shared values from being re-overlaid by native DP."""
    changed = False
    for device_values in per_device.values():
        for key in device_values.keys() & values.keys():
            if key != 'DEVICE_PROPERTIES':
                device_values[key] = copy.deepcopy(values[key])
                changed = True
    return changed


def _route_vlm_device_properties(properties, model, managed_devices):
    """Carry physical-device settings through GenAI's VLM role resolver.

    GenAI 2026.4 get_model_properties removes DEVICE_PROPERTIES and flattens only
    its exact device key. For HETERO that key is the full virtual target, so a
    physical GPU's QUERY_MODEL_RATIO would otherwise disappear before compile.
    The exact-target entry carries a nested native DEVICE_PROPERTIES map into
    every component. Explicit per-role values retain precedence, while managed
    allocation ratios remain authoritative within a managed placement.
    """
    source_target = model.get('_placement', {}).get('target', '')
    if model['pipeline'] not in {'vlm', 'vlm-text'} or not source_target.startswith('HETERO:'):
        return
    target = pipeline_target(model)
    per_device = properties.get('DEVICE_PROPERTIES', {})
    if not per_device:
        return
    target_properties = copy.deepcopy(per_device.get(target, {}))
    if source_target != target:
        target_properties.update(copy.deepcopy(per_device.get(source_target, {})))
    physical = copy.deepcopy({device: values for device, values in per_device.items()
                              if device not in {target, source_target}})
    _overlay_inherited_device_values(physical, target_properties)
    physical = _merge_device_properties(physical, target_properties.get('DEVICE_PROPERTIES', {}))
    physical = _merge_device_properties(physical, managed_devices)
    target_properties['DEVICE_PROPERTIES'] = physical
    per_device[target] = target_properties
    for role_properties in properties.get('MODEL_PROPERTIES', {}).values():
        role_devices = copy.deepcopy(physical)
        # GenAI gives role-level values precedence over device-wide settings.
        # Native HETERO would otherwise re-overlay inherited CPU thread defaults
        # (or GPU precision settings) over those explicit role-level values.
        overrides_inherited = _overlay_inherited_device_values(role_devices, role_properties)
        if 'DEVICE_PROPERTIES' in role_properties or overrides_inherited:
            role_devices = _merge_device_properties(role_devices, role_properties.get('DEVICE_PROPERTIES', {}))
            role_properties['DEVICE_PROPERTIES'] = _merge_device_properties(role_devices, managed_devices)


def _pin_vlm_embedding_device(properties, model):
    device = _vlm_embedding_device(model)
    if device is None:
        return
    priorities = ','.join(model['_placement']['devices'])

    def set_priorities(values, expected, scope):
        current = values.get('MULTI_DEVICE_PRIORITIES', expected)
        if current != expected:
            raise ValueError('Conflicting MULTI_DEVICE_PRIORITIES for ' + scope +
                             '; managed VLM allocation requires ' + expected)
        values['MULTI_DEVICE_PRIORITIES'] = expected

    set_priorities(properties, priorities, 'the language model')
    bridge = properties.get('DEVICE_PROPERTIES', {}).get('HETERO', {})
    if 'MULTI_DEVICE_PRIORITIES' in bridge:
        set_priorities(bridge, priorities, 'the HETERO target')
    physical = bridge.get('DEVICE_PROPERTIES', {})
    roles = properties.setdefault('MODEL_PROPERTIES', {})
    embedding_roles = {'text_embeddings', 'text_embeddings_per_layer'}
    for role, values in roles.items():
        if role not in embedding_roles and 'MULTI_DEVICE_PRIORITIES' in values:
            set_priorities(values, priorities, role)
    for role in embedding_roles:
        values = roles.setdefault(role, {})
        set_priorities(values, device, role)
        # GenAI's incremental decoding directly forwards the embedding output
        # RemoteTensor. It must share the language input's GPU context. Do not
        # partition this auxiliary graph or retain properties for other GPUs.
        device_values = copy.deepcopy(values.get('DEVICE_PROPERTIES', physical))
        allowed = {device, device.split('.')[0]}
        device_values = {name: value for name, value in device_values.items() if name in allowed}
        for value in device_values.values():
            value.pop('QUERY_MODEL_RATIO', None)
        values['DEVICE_PROPERTIES'] = device_values


def pipeline_properties(genai, model):
    """Build typed scheduler options without modifying the stored dictionaries."""
    properties = copy.deepcopy(model.get("plugin_config", {}))
    overrides = copy.deepcopy(model.get('_placement', {}).get('properties', {}))
    device_overrides = overrides.pop('DEVICE_PROPERTIES', {})
    properties.update(overrides)
    if device_overrides:
        per_device = properties.setdefault('DEVICE_PROPERTIES', {})
        for device, values in device_overrides.items():
            per_device.setdefault(device, {}).update(values)
    cpu_threads = effective_cpu_thread_default(model, model.get('_cpu_inference_threads'))
    if cpu_threads is not None:
        target = model.get('_placement', {}).get('target', model['device'])
        if target == 'CPU':
            properties.setdefault('INFERENCE_NUM_THREADS', cpu_threads)
        else:
            properties.setdefault('DEVICE_PROPERTIES', {}).setdefault('CPU', {}).setdefault('INFERENCE_NUM_THREADS', cpu_threads)
    # Device-specific controls must not be sent to the other plugin in a mixed
    # plan. Keep saved choices inactive when that device kind is not executing.
    scoped = [control for control in CONTROLS if control.get('device_kind') and control['key'] in properties]
    if scoped:
        plan = model.get('_placement', {})
        target = plan.get('target', model.get('device', ''))
        selected = plan.get('devices') or (target.split(':', 1)[1].split(',') if target.startswith('HETERO:') else [target])
        for control in scoped:
            key = control['key']
            value = properties.pop(key)
            matching = [d for d in selected if d.split('.')[0] == control['device_kind']]
            if matching == [target]:
                properties[key] = value
            else:
                for device in matching:
                    properties.setdefault('DEVICE_PROPERTIES', {}).setdefault(device, {}).setdefault(key, value)
    _route_vlm_device_properties(properties, model, device_overrides)
    _pin_vlm_embedding_device(properties, model)
    stateful = properties.get('ATTENTION_BACKEND') == 'SDPA'
    if (model.get("scheduler_config") or mtp_enabled(model)) and model["pipeline"] != "image" and not stateful:
        scheduler = genai.SchedulerConfig()
        # LazyRig leases each model worker exclusively and generates one beam /
        # one returned sequence. GenAI's batching default (256 sequences) can
        # reserve hundreds of full recurrent states on hybrid models when a
        # fixed cache allocation is set, even for a single chat request.
        # Preserve explicit operator overrides; never increase cache_size to
        # compensate for this unnecessary reservation.
        scheduler.max_num_seqs = 1
        for key, value in model.get("scheduler_config", {}).items():
            if not hasattr(scheduler, key):
                raise ValueError("Installed GenAI does not expose scheduler option " + key + "; rerun setup-windows.bat")
            setattr(scheduler, key, value)
        properties["scheduler_config"] = scheduler
    if mtp_enabled(model):
        if stateful:
            raise ValueError('Disable MTP before using a stateful multi-device split')
        ensure_export(model)
        if not hasattr(genai, "draft_model"):
            raise ValueError("Installed OpenVINO GenAI lacks draft_model; rerun setup-windows.bat")
        properties["scheduler_config"].enable_prefix_caching = False
        properties["draft_model"] = genai.draft_model(model["path"], mtp_device(model), **mtp_draft_properties(model))
    return properties


def attention_cache_failure(error, properties):
    """Explain this specific native configuration failure without retrying it."""
    if 'Requested linear attention cache allocation exceeds the configured cache size' not in str(error):
        return {}
    scheduler = properties.get('scheduler_config')
    settings = {key: getattr(scheduler, key) for key in
                ('cache_size', 'max_num_seqs', 'num_linear_attention_blocks', 'enable_prefix_caching')
                if scheduler is not None and hasattr(scheduler, key)}
    values = ', '.join(f'{key}={value}' for key, value in settings.items())
    explanation = ('Attention-cache settings rejected' + (f' ({values})' if values else '') +
                   '. Set Scheduler sequence limit to 1 or leave it blank for one request per worker. '
                   'Set Initial attention cache to 0 for allocation on demand, or choose an allocation '
                   'that fits the model state and available memory.')
    if settings.get('num_linear_attention_blocks', 0):
        explanation += ' Clear the explicit num_linear_attention_blocks override to restore automatic sizing.'
    return {'error_code': 'attention_cache_budget_exceeded', 'cache_settings': settings,
            'error': explanation + '\nNative error: ' + type(error).__name__ + ': ' + str(error)}
