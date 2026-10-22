"""Bundled MTP-head setup using OpenVINO GenAI's draft_model API."""
from pathlib import Path
import re


def available(model):
    path = Path(model.get("path") or "__no_model__")
    return all((path / ("openvino_mtp_model." + suffix)).is_file() for suffix in ("xml", "bin"))


def enabled(model):
    return model.get("mtp", {}).get("enabled", False)


def device(model):
    """An omitted draft device follows the resolved main-model target."""
    return model.get('mtp', {}).get('device') or model.get('_placement', {}).get('target', model['device'])


def devices(model):
    return {device(model)} if enabled(model) else set()


def cpu_threads(model):
    """Explicit model CPU overrides also apply to a CPU MTP head."""
    properties = model.get('plugin_config', {})
    per_device = properties.get('DEVICE_PROPERTIES', {})
    return per_device.get(device(model), {}).get('INFERENCE_NUM_THREADS',
        per_device.get('CPU', {}).get('INFERENCE_NUM_THREADS', properties.get('INFERENCE_NUM_THREADS')))


def draft_properties(model):
    target = model.get('_placement', {}).get('target', model['device'])
    selected = device(model)
    # An empty native draft property map inherits the main model's properties.
    # On a different device that can send GPU-only settings to CPU (or another
    # GPU). LATENCY is GenAI's normal preference and makes the map independent.
    properties = {'PERFORMANCE_HINT': 'LATENCY'} if selected != target else {}
    if selected.split('.')[0] == 'CPU':
        threads = cpu_threads(model)
        if threads is None:
            threads = model.get('_cpu_inference_threads')
        if threads is not None and selected != target:
            properties['INFERENCE_NUM_THREADS'] = threads
    return properties


def ensure_export(model):
    if enabled(model) and not available(model):
        raise ValueError("MTP requires openvino_mtp_model.xml and openvino_mtp_model.bin in this export. A model's original MTP layer alone is insufficient; download/export its OpenVINO MTP head.")


def validate_settings(model, config=None):
    options = model.get("mtp", {})
    if not isinstance(options, dict) or set(options) - {"enabled", "num_assistant_tokens", "device"}:
        raise ValueError("mtp accepts enabled, num_assistant_tokens and device")
    if type(options.get("enabled", False)) is not bool:
        raise ValueError("mtp.enabled must be true or false")
    count = options.get("num_assistant_tokens", 2)
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("MTP num_assistant_tokens must be an integer from 1 to 64")
    selected = options.get('device', '')
    if not isinstance(selected, str) or (selected and not re.fullmatch(r'(?:CPU|GPU)(?:\.\d+)*', selected)):
        raise ValueError('MTP device must be a CPU or GPU device, or blank for Same as model')
    if selected and config is not None:
        from .device_info import canonical_device
        selected = canonical_device(selected, config['devices'])
        if selected not in {d['id'] for d in config['devices']}:
            raise ValueError('MTP device is not configured: ' + selected)
        options['device'] = selected
    if enabled(model):
        if model["pipeline"] == "image":
            raise ValueError("MTP is for LLM/VLM text generation")
        if model.get("scheduler_config", {}).get("enable_prefix_caching", False):
            raise ValueError("MTP requires prefix caching disabled in this OpenVINO integration")
        if model.get("generation", {}).get("temperature", 0) != 0:
            raise ValueError("MTP requires greedy decoding: set the model temperature to 0")
        if model.get("plugin_config", {}).get("prompt_lookup"):
            raise ValueError("MTP and prompt_lookup are mutually exclusive")


def request_options(model, body):
    active = enabled(model)
    temperature = body.get("temperature", model.get("generation", {}).get("temperature", 0 if active else .7))
    if active and float(temperature) != 0:
        raise ValueError("MTP requires temperature=0 (greedy decoding); change the frontend temperature or disable MTP for this model")
    count = body.get("num_assistant_tokens", model.get("mtp", {}).get("num_assistant_tokens", 2))
    if "num_assistant_tokens" in body and not active:
        raise ValueError("Enable MTP in Model Settings before using num_assistant_tokens")
    if active and (type(count) is not int or not 1 <= count <= 64):
        raise ValueError("num_assistant_tokens must be an integer from 1 to 64")
    return {"enabled": active, **({"num_assistant_tokens": count} if active else {})}


def configure_generation(config, model, request):
    settings = request_options(model, request)
    if settings["enabled"]:
        config.num_assistant_tokens = settings["num_assistant_tokens"]
        config.assistant_confidence_threshold = 0.0
        config.do_sample = False


def model_capabilities(model):
    from .vision import is_vlm
    from .image_edit import capabilities as edit_capabilities
    from .image_sizes import constraints as image_size_constraints
    from .diffusion_settings import describe as diffusion_scheduler_info
    text = model["task"] == "text-generation"
    edit = edit_capabilities(model)
    return {"input_modalities": ["text", "image"] if text and is_vlm(model) or edit['available'] else ["text"],
            "output_modalities": ["text"] if text else ["image"],
            **({'image_editing': edit, 'image_size_constraints': image_size_constraints(model),
                'diffusion_scheduler_info': diffusion_scheduler_info(model)} if not text else {}),
            "capabilities": {"vision": text and is_vlm(model), "image_input": text and is_vlm(model) or edit['available'],
                             'image_editing': edit['available'],
                             "mtp_available": available(model) if text else False,
                             "mtp_enabled": enabled(model) if text else False}}
