import copy
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit
from .runtime_settings import validate_runtime
from .mtp import validate_settings as validate_mtp
from .image_sizes import validate_settings as validate_image_settings
from .image_options import validate_options as validate_image_options
from .diffusion_settings import validate_settings as validate_diffusion_settings
from .sampling import FIELDS as SAMPLING_FIELDS, validate as validate_sampling
from .placement import validate_placement, migrate_manual_allocations

SERVER_DEFAULT = {
    "mode": "openvino", "data_dir": "server-data", "host": "127.0.0.1", "port": 8765,
    "api_key": "", "inference_api_key": "", "allowed_origins": [], "tls_cert": "", "tls_key": "", "models_dir": str(Path.cwd() / "models"),
    "browser_access": "origins",
    "public_base_url": "", "max_connections": 64, "http_timeout_seconds": 30,
    "max_jobs": 8, "max_preparing": 2,
    "event_retention_days": 0, "min_free_disk_mib": 256,
    "load_timeout_seconds": 900, "request_timeout_seconds": 3600,
    "devices": [], "models": [], "download_max_gib": 100
}

def normalize_origin(value):
    if not isinstance(value, str) or any(c.isspace() for c in value.strip()):
        raise ValueError("Enter a frontend HTTP(S) URL, such as http://localhost:5173")
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or "*" in parsed.netloc):
        raise ValueError("Allowed origins need a frontend HTTP(S) URL; wildcards and file/null origins are not supported")
    hostname = parsed.hostname.lower()
    hostname = "[" + hostname + "]" if ":" in hostname else hostname.encode("idna").decode()
    port = parsed.port
    authority = hostname + (":" + str(port) if port and port != {"http": 80, "https": 443}[parsed.scheme] else "")
    return parsed.scheme + "://" + authority


def validate(config):
    # Older settings must not turn plain client chats into server-managed jobs.
    # The chat image tool is now an explicit, per-request option only.
    for retired in ("default_auto_image", "default_image_model"):
        config.pop(retired, None)
    config.setdefault("mode", "openvino")
    config.setdefault('terminal_metrics', True)
    if not isinstance(config['terminal_metrics'], bool):
        raise ValueError('terminal_metrics must be a boolean')
    if config.get("mode", "openvino") != "openvino":
        raise ValueError("Only the OpenVINO backend is supported")
    config.setdefault("devices", [])
    config.setdefault("models", [])
    for key, default, maximum in (("max_connections", 64, 1024), ("max_jobs", 8, 128), ("max_preparing", 2, 32)):
        value = config.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}")
    for key, default, maximum in (("event_retention_days", 0, 3650), ("min_free_disk_mib", 256, 1048576)):
        value = config.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= maximum or not math.isfinite(value):
            raise ValueError(f"{key} must be between 0 and {maximum}")
    timeout = config.setdefault("http_timeout_seconds", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not .1 <= timeout <= 300:
        raise ValueError("http_timeout_seconds must be between 0.1 and 300 seconds")
    public_url = config.setdefault("public_base_url", "")
    if not isinstance(public_url, str):
        raise ValueError("public_base_url must be an HTTP(S) origin or an empty string")
    if public_url:
        parsed = urlsplit(public_url)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("public_base_url must be an HTTP(S) origin without a path, query or fragment")
        config["public_base_url"] = normalize_origin(public_url)
    if not 1 <= int(config.get("port", 8765)) <= 65535:
        raise ValueError("Port must be 1–65535")
    host = config.get("host", "127.0.0.1")
    if host not in {"127.0.0.1", "0.0.0.0", "localhost"}:
        raise ValueError("Listen host must be 127.0.0.1, localhost or 0.0.0.0")
    if not isinstance(config.get("api_key", ""), str):
        raise ValueError("API key must be a string")
    if host == "0.0.0.0" and len(config.get("api_key", "")) < 16:
        raise ValueError("Set an API key of at least 16 characters before enabling network access")
    inference_key = config.setdefault("inference_api_key", "")
    if not isinstance(inference_key, str):
        raise ValueError("Inference API key must be a string")
    if inference_key and (len(inference_key) < 16 or len(config.get("api_key", "")) < 16 or inference_key == config.get("api_key")):
        raise ValueError("Inference API key needs at least 16 characters and a different owner API key of at least 16 characters")
    if config.get("browser_access", "origins") not in {"origins", "api-key"}:
        raise ValueError("Browser access must be origins or api-key")
    if config.get("browser_access") == "api-key" and len(config.get("api_key", "")) < 16:
        raise ValueError("Any frontend with API key requires an API key of at least 16 characters")
    origins = config.get("allowed_origins", [])
    if not isinstance(origins, list):
        raise ValueError("Allowed origins must be an array of HTTP(S) origins")
    config["allowed_origins"] = list(dict.fromkeys(normalize_origin(x) for x in origins))
    for key in ("tls_cert", "tls_key"):
        if not isinstance(config.get(key, ""), str):
            raise ValueError(key + " must be a PEM file path")
    if bool(config.get("tls_cert")) != bool(config.get("tls_key")):
        raise ValueError("HTTPS requires both a TLS certificate and private key path")
    if not 0 < float(config.get("download_max_gib", 100)) <= 10000:
        raise ValueError("Download limit must be greater than zero and at most 10000 GiB")
    for key in ("load_timeout_seconds", "request_timeout_seconds"):
        if not 1 <= float(config.get(key, 3600)) <= 86400:
            raise ValueError("Timeout must be between 1 second and 24 hours")
    devices = {d["id"] for d in config["devices"]}
    if len(devices) != len(config["devices"]):
        raise ValueError("Duplicate device IDs")
    if len({m["id"] for m in config["models"]}) != len(config["models"]):
        raise ValueError("Duplicate model IDs")
    for d in config["devices"]:
        if d.get("policy") not in {"exclusive", "budget"} or not 0 < float(d["budget_gib"]) < 10000:
            raise ValueError("Invalid device policy or memory budget")
        if 'inference_threads' in d:
            threads = d['inference_threads']
            if d['id'].split('.')[0] != 'CPU':
                raise ValueError('Device inference threads apply to CPU devices only')
            elif threads is None:
                d.pop('inference_threads')  # Auto leaves OpenVINO's default unchanged.
            elif type(threads) is not int or not 1 <= threads <= 2147483647:
                raise ValueError('CPU inference threads must be a positive whole number, or null for Auto')
    for m in config["models"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", m["id"]):
            raise ValueError("Model alias must use letters, digits, dots, underscores or hyphens")
        if m["device"] not in devices:
            raise ValueError("Unknown device for " + m["id"])
        if m["task"] not in {"text-generation", "text-to-image"}:
            raise ValueError("Supported tasks are text-generation and text-to-image")
        if m["pipeline"] == "vlm-text":
            m["pipeline"] = "vlm"
        if m["pipeline"] not in {"llm", "vlm", "image"}:
            raise ValueError("Unknown pipeline")
        if m.get("runtime_profile", "default") not in {"default", "qwen-image-21"}:
            raise ValueError("Unknown model runtime profile")
        if m.get("runtime_profile") == "qwen-image-21" and m["pipeline"] != "image":
            raise ValueError("The Qwen-Image-2.1 runtime is for image generation")
        if (m["task"] == "text-to-image") != (m["pipeline"] == "image"):
            raise ValueError("Task and pipeline disagree")
        if not 0 < float(m["memory_gib"]) < 10000:
            raise ValueError("Invalid memory reservation")
        if m["task"] == "text-generation" and not 128 <= int(m["context_limit"]) <= 1048576:
            raise ValueError("Invalid context limit")
        if not isinstance(m.get("plugin_config", {}), dict) or not isinstance(m.get("generation", {}), dict):
            raise ValueError("Plugin and generation settings must be JSON objects")
        m.setdefault("enabled", True)
        m.setdefault("generation", {})
        m.setdefault("template_context", {})
        m.setdefault("scheduler_config", {})
        m.setdefault("idle_timeout_seconds", 0)
        if m['pipeline'] == 'image':
            m.setdefault('keep_image_model_loaded', 'auto')
            m.setdefault('restore_text_after_image', False)
        if 'default_image_tool' in m and (m['pipeline'] != 'image' or not isinstance(m['default_image_tool'], bool)):
            raise ValueError('default_image_tool must be a boolean for an image model')
        if 'keep_image_model_loaded' in m:
            value = m['keep_image_model_loaded']
            if m['pipeline'] != 'image' or not (isinstance(value, bool) or value == 'auto'):
                raise ValueError('keep_image_model_loaded must be auto or a boolean for an image model')
        if 'restore_text_after_image' in m and (m['pipeline'] != 'image' or not isinstance(m['restore_text_after_image'], bool)):
            raise ValueError('restore_text_after_image must be a boolean for an image model')
        if not isinstance(m["template_context"], dict) or not isinstance(m["scheduler_config"], dict):
            raise ValueError("Template and scheduler settings must be JSON objects")
        for key, value in m["generation"].items():
            if key not in {"max_tokens", "rng_seed", "stop_strings", "width", "height", "steps", "seed", "guidance_scale", "negative_prompt", "strength", "max_sequence_length", *SAMPLING_FIELDS}:
                raise ValueError("Unsupported generation setting: " + key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Generation numbers must be finite")
        generation = m["generation"]
        validate_sampling(generation)
        if m['pipeline'] == 'image' and {'min_p', 'presence_penalty', 'frequency_penalty'} & generation.keys():
            raise ValueError('min_p, presence_penalty and frequency_penalty are text sampling settings')
        if 'strength' in generation:
            from .image_edit import capabilities as edit_capabilities
            value = generation['strength']
            if not edit_capabilities(m)['supports_strength']:
                raise ValueError('Denoising strength is only supported by compatible classic image-to-image exports; omit it for Qwen-Image-2.1')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError('strength must be greater than 0 and at most 1')
        ranges = {"max_tokens": (1, m.get("context_limit", 1048576)),
                  "steps": (1, 100), "seed": (-1, 2147483647),
                  "rng_seed": (0, 2147483647), "guidance_scale": (0, 100)}
        for key, (low, high) in ranges.items():
            if key in generation and not low <= float(generation[key]) <= high:
                raise ValueError(f"{key} must be {low}–{high}")
        if "negative_prompt" in generation and not isinstance(generation["negative_prompt"], str):
            raise ValueError("negative_prompt must be a string")
        if "stop_strings" in generation and (not isinstance(generation["stop_strings"], list) or any(not isinstance(s, str) or not s for s in generation["stop_strings"])):
            raise ValueError("stop_strings must be an array of nonempty strings")
        if m["task"] == "text-to-image":
            validate_image_settings(m)
            validate_image_options(m, generation)
        elif 'max_sequence_length' in generation:
            raise ValueError('max_sequence_length is an image generation setting')
        if not 0 <= float(m["idle_timeout_seconds"]) <= 86400:
            raise ValueError("Idle unload timeout must be 0–86400 seconds")
        validate_runtime(m)
        validate_diffusion_settings(m)
        validate_mtp(m, config)
        validate_placement(m, config)
    if sum(m.get('default_image_tool') is True for m in config['models']) > 1:
        raise ValueError('Choose only one default image tool')
    return config

def load(path=None, data_dir=None):
    config = json.loads(Path(path).read_text(encoding="utf-8")) if path else copy.deepcopy(SERVER_DEFAULT)
    if path:
        base = Path(path).resolve().parent
        for m in config["models"]:
            if m.get("path"):
                m["path"] = str((base / m["path"]).resolve())
    config["data_dir"] = str(Path(data_dir or config.get("data_dir", "data")).resolve())
    return validate(migrate_manual_allocations(config))


def restore_settings(config):
    config = copy.deepcopy(config)
    path = Path(config["data_dir"]) / "settings.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("mode") == config["mode"]:
            config.update({k: v for k, v in saved.items() if k != "data_dir"})
            # Earlier model-settings forms persisted the old factory budget.
            # Upgrade it once; later explicit choices, including 1024, survive.
            if 'output_token_defaults_version' not in saved:
                from .context import DEFAULT_OUTPUT_TOKENS
                for model in config.get('models', []):
                    generation = model.get('generation', {})
                    if (model.get('task') == 'text-generation' and generation.get('max_tokens') == 1024
                            and int(model.get('context_limit', 0)) > DEFAULT_OUTPUT_TOKENS):
                        generation['max_tokens'] = DEFAULT_OUTPUT_TOKENS
    config.setdefault('output_token_defaults_version', 1)
    return validate(migrate_manual_allocations(config))
