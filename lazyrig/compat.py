"""OpenAI-shaped chat and image endpoints backed by the same scheduler."""
import base64
import json
import re
import ssl
import time
from .context import DEFAULT_OUTPUT_TOKENS, IMAGE_TOOL
from .service import TERMINAL
from .admission import keyed
from .deadlines import DeadlineEvent, current_deadline
from .tool_calls import parse_tool_calls, ToolCallError
from .request_options import template_model
from .reasoning import split_reasoning, assistant_message, tool_stream_parts
from .response_format import is_json
from .vision import check_content, text_content
from .mtp import request_options as mtp_request
from .sampling import FIELDS as SAMPLING_FIELDS, validate as validate_sampling
from .image_edit import EDIT_FIELDS
from .idempotency import image_identity, request_identity
from .chat_images import capture_sources, image_tools
from .prompt_enhancement import enabled as enhancement_enabled, FIELDS as ENHANCEMENT_FIELDS

def public_url(handler):
    # Native handler uses a configured canonical origin; lightweight test/client
    # handlers retain the pre-beta interface.
    return handler.base_url if hasattr(handler, 'base_url') else handler.scheme + '://' + handler.headers['Host']


def auto_image_enabled(service, body):
    """Server-managed image tools require explicit opt-in for this request."""
    return body.get("auto_image", False)

def check_messages(messages, model=None):
    if not isinstance(messages, list) or not messages or len(messages) > 10000:
        raise ValueError("messages must be a nonempty array")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
    check_content(messages, model)
    if messages[-1].get("content") is None:
        raise ValueError("The final message needs text content")

def validate_chat(service, body, preflight=True, origin=''):
    allowed = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature",
               *SAMPLING_FIELDS, "seed", "stop", "tools", "tool_choice", "n", "user",
               "auto_image", "image_model", "image_sources", "enhance_prompt", "width", "height", "steps", "image_seed",
               "chat_template_kwargs", "reasoning_effort", "timings_per_token", "response_format", "num_assistant_tokens"}
    extra = set(body) - allowed
    if extra:
        raise ValueError("Unsupported request fields: " + ", ".join(sorted(extra)))
    if body.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported")
    if not isinstance(body.get("stream", False), bool) or not isinstance(body.get("auto_image", False), bool):
        raise ValueError("stream and auto_image must be booleans")
    if "stream_options" in body and (not isinstance(body["stream_options"], dict) or set(body["stream_options"]) - {"include_usage"}):
        raise ValueError("stream_options accepts only include_usage")
    if body.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("tool_choice supports auto or none")
    messages = body.get("messages")
    model = service.model(body["model"], "text-generation")
    check_messages(messages, model)
    prompt_model = template_model(model, body)
    applied_mtp = mtp_request(model, body)
    tools = body.get("tools") if body.get("tool_choice") != "none" else None
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(t, dict) or t.get("type") != "function" or not isinstance(t.get("function"), dict) or not isinstance(t["function"].get("name"), str) for t in tools):
            raise ValueError("tools must contain function definitions")
    auto = auto_image_enabled(service, body)
    if 'image_sources' in body and not isinstance(body['image_sources'], list):
        raise ValueError('image_sources must be an array')
    if (enhancement_enabled(body) or body.get('image_sources')) and not auto:
        raise ValueError('Prompt enhancement and image_sources require explicit auto_image=true')
    if auto and body.get("tool_choice") == "none":
        raise ValueError("auto_image requires tools; remove tool_choice=none or disable auto_image")
    if auto and tools:
        raise ValueError("auto_image uses the server's image tool; omit client tools")
    if is_json(prompt_model["response_format"]) and (auto or tools):
        raise ValueError("JSON response_format cannot be combined with active tools or auto_image=true; use a separate structured-output request")
    image_model = body.get("image_model")
    if auto:
        if not image_model:
            image_model = next((m["id"] for m in service.config["models"] if m["task"] == "text-to-image" and m.get("enabled", True)), None)
        if not image_model:
            raise ValueError("Register an image model before using auto_image")
        service.model(image_model, "text-to-image")
    generation = {**model.get("generation", {}), **{k: v for k, v in body.items() if k in SAMPLING_FIELDS}}
    validate_sampling(generation)
    if applied_mtp["enabled"]:
        generation["temperature"] = 0
        generation["num_assistant_tokens"] = applied_mtp["num_assistant_tokens"]
    budget = int(body.get("max_completion_tokens", body.get("max_tokens", generation.get("max_tokens", DEFAULT_OUTPUT_TOKENS))))
    if not 1 <= budget <= model["context_limit"]:
        raise ValueError("Output token budget exceeds the configured context limit")
    if "seed" in body:
        generation["rng_seed"] = int(body["seed"])
    if "stop" in body:
        stops = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
            raise ValueError("stop must be a string or nonempty strings")
        generation["stop_strings"] = stops
    prepared = None
    if preflight:
        captured, sources, prepared_tools = chat_inputs(service, body, model, auto, image_model, tools, origin)
        prepared = service.tokens.prepare(prompt_model, captured, budget, auto, tools=prepared_tools, add_system=False)
    if prepared and prepared["context"]["fits"] is False:
        raise ValueError("Context overflow: prepared input plus output budget exceeds configured context limit")
    return model, tools, auto, image_model, generation, budget, prepared

def chat_inputs(service, body, model, auto, image_model, tools=None, origin=''):
    messages = service.images.capture(body['messages'], model)
    sources = capture_sources(service.images, body.get('image_sources', []), messages, origin) if auto else []
    return messages, sources, image_tools(service.model(image_model), sources, enhancement_enabled(body)) if auto else tools


def start_chat(service, body, preflight=True, background=True, origin='', idempotency_key=None):
    signal = DeadlineEvent(timeout=service.config.get('request_timeout_seconds', 1800))
    if idempotency_key is None:
        with service.admission(signal=signal):
            return _start_chat(service, body, preflight, background, origin)
    identity = request_identity(idempotency_key, body, 'chat')
    with keyed(service, identity, signal):
        prior = service.store.inference_request(identity)
        if prior is not None:
            return service.job(prior['id'])
        with service.admission(signal=signal):
            request_id = idempotency_key if re.fullmatch(r'[a-f0-9]{32}', idempotency_key) else None
            return _start_chat(service, body, preflight, background, origin, identity, request_id)


def _start_chat(service, body, preflight, background, origin, retry_identity=None, request_id=None):
    model, tools, auto, image_model, generation, budget, prepared = validate_chat(service, body, False)
    messages, sources, tools = chat_inputs(service, body, model, auto, image_model, tools, origin)
    if preflight:
        prepared = service.tokens.prepare(template_model(model, body), messages, budget, auto, tools=tools, add_system=False)
        if prepared["context"]["fits"] is False:
            raise ValueError("Context overflow: prepared input plus output budget exceeds configured context limit")
    if current_deadline() is not None:
        current_deadline().check()
    with service.lock:
        service._check_admission()
    sid = service.store.session("API · " + model["id"])
    for m in messages[:-1]:
        if current_deadline() is not None:
            current_deadline().check()
        service.store.append(sid, m)
    image_defaults = service.model(image_model).get("generation", {}) if auto else {}
    return service.submit({**generation, "session_id": sid, "model": model["id"], "prompt": text_content(messages[-1]["content"]) or "[Image input]",
        **{k: body[k] for k in ("chat_template_kwargs", "reasoning_effort", "timings_per_token", "response_format") if k in body},
        "input_message": messages[-1], "external": True, "tools": tools, "tools_enabled": auto, "image_model": image_model,
        "image_sources": sources, "enhance_prompt": enhancement_enabled(body),
        "image_size_explicit": 'width' in body or 'height' in body,
        "max_tokens": budget, "width": body.get("width", image_defaults.get("width", 512)), "height": body.get("height", image_defaults.get("height", 512)),
        "steps": body.get("steps", image_defaults.get("steps", 20)), "seed": body.get("image_seed", image_defaults.get("seed", -1)), "prepared": prepared}, captured=True, background=background,
        retry_identity=retry_identity, request_id=request_id)

def completion(job, body):
    text = job["text"]
    message = assistant_message(text, job.get("thinking_open", False))
    parts = split_reasoning(text, job.get("thinking_open", False))
    reason = "length" if job["context"] and job["context"]["generated_tokens"] >= job["request"]["max_tokens"] else "stop"
    if body.get("tools") and body.get("tool_choice") != "none":
        try:
            parsed = parse_tool_calls(text, body["tools"], job.get("thinking_open", False))
        except ToolCallError as exc:
            if not (exc.code == "incomplete_reasoning" or (exc.incomplete and reason == "length")):
                raise
            parsed = {"calls": []}  # Incomplete client calls remain text with finish_reason=length.
        if parsed["calls"]:
            calls = []
            for call in parsed["calls"]:
                args = call.get("arguments", {})
                calls.append({"id": "call_" + job["id"] + "_" + str(len(calls)), "type": "function", "function": {
                    "name": call["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}})
            message.update(content=parsed["content"] or None, tool_calls=calls)
            reason = "tool_calls"
    ci = job["context"] or {}
    return {"id": "chatcmpl-" + job["id"], "object": "chat.completion", "created": int(job["created"]),
            "model": job["model"], "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": job.get("usage", {"prompt_tokens": ci.get("input_tokens", 0), "completion_tokens": ci.get("generated_tokens", 0),
                      "total_tokens": ci.get("used_tokens", 0)}),
            "x_context": ci, "x_job_id": job["id"], "x_artifacts": job["artifacts"], "x_mode": job["mode"],
            "x_durable": job.get('durable', True), "x_storage_warning": job.get('storage_error'),
            "x_tool_attempts": len(job.get("tool_attempts", [])),
            "x_template_context": job.get("current_template_context", {}),
            "x_template_notes": job.get("template_notes", []),
            "x_reasoning_incomplete": parts["reasoning_incomplete"],
            "x_response_format": job.get("response_format", {"type": "text"}),
            "x_mtp": job.get("mtp", {"enabled": False}),
            "x_prompt_enhancement": job.get('prompt_enhancement'),
            "x_image_recovery": job.get('image_recovery'),
            "x_sampling": job.get("sampling"),
            **({"x_json_valid": job.get("json_valid")} if is_json(job.get("response_format")) else {}),
            **({"timings": job.get("timings"), "x_timing_passes": job.get("timing_passes", [])} if job.get("timings_per_token") else {})}

def wait_job(service, jid, image_ready=False, handler=None):
    while True:
        if handler is not None:
            handler.check_disconnect()
        job = service.job(jid)
        if job["status"] in TERMINAL or (image_ready and job.get('result_ready') and job['artifacts']):
            return job
        time.sleep(.03)

def job_receipt(job, origin=''):
    result = {"object": "inference.job", "id": job["id"], "x_job_id": job["id"], "model": job["model"],
            "status": job["status"], "phase": job.get("phase"), "progress": job.get("progress"),
            "x_context": job.get("context"), "x_artifacts": job.get("artifacts", []),
            "x_image_recovery": job.get('image_recovery'),
            "x_result_ready": bool(job.get('result_ready') or (job.get('kind') == 'image' and job.get('artifacts'))),
            "x_warning": job.get('error'),
            "x_durable": job.get('durable', True), "x_storage_warning": job.get('storage_error'),
            **({"timings": job.get("timings")} if job.get("timings_per_token") else {}),
            "status_url": "/api/jobs/" + job["id"], "result_url": "/api/jobs/" + job["id"] + "/result",
            "events_url": "/api/jobs/" + job["id"] + "/events",
            "cancel_url": "/api/jobs/" + job["id"] + "/cancel",
            "x_cancel_requested": bool(job.get('cancel_requested')),
            "disconnect_policy": "continue" if job.get('background', True) else "cancel",
            "x_handoff_complete": job["status"] == "completed"}
    if job.get('kind') == 'image':
        result['artifact_url'] = origin + '/artifacts/' + job['id'] + '.png'
        result['artifacts_url'] = origin + '/api/artifacts'
    return result


def job_error(job):
    cancelled = job['status'] == 'cancelled'
    return {"error": {"message": job["error"] or job["status"], "type": "request_cancelled" if cancelled else "server_error",
                      **({'code': job['error_code']} if job.get('error_code') else {})}, "x_job_id": job["id"],
            "x_status": job["status"], "x_artifacts": job["artifacts"], "x_context": job["context"],
            "x_image_recovery": job.get('image_recovery'),
            "x_durable": job.get('durable', True), "x_storage_warning": job.get('storage_error'),
            **({"x_retryable": False} if cancelled or job.get('error_code') else {}),
            "x_tool_diagnostics": job.get("tool_attempts", [])}


def job_error_status(job):
    # A cancelled result is permanently unavailable. 5xx (and some SDKs' 409
    # handling) invites automatic retries of a request the user just stopped.
    return 410 if job['status'] == 'cancelled' else 504 if job.get('error_code') == 'request_timeout' else 502 if job.get('error_code') else 500


def serve_job_result(handler, service, jid):
    job = service.job(jid)
    handler.job_id = jid
    chat = job.get("kind") == "chat" or "max_tokens" in job.get("request", {})
    image = not chat and (job.get("kind") == "image" or {"width", "steps"} <= set(job.get("request", {})))
    if not chat and not image:
        raise ValueError("This job has no inference result; use its status endpoint")
    if image and job["artifacts"] and (job.get('result_ready') or job['status'] in TERMINAL):
        handler.reply(image_completion(handler, service, job, job.get("response_format", "url")))
    elif job["status"] not in TERMINAL:
        handler.reply(job_receipt(job, public_url(handler)), 202)
    elif job["status"] != "completed":
        handler.reply(job_error(job), job_error_status(job))
    elif chat:
        handler.reply(completion(job, {"tools": job.get("tools")}))
    else:
        handler.reply(job_error(job), 500)


def serve_chat(handler, service, body, asynchronous=False):
    job = start_chat(service, body, preflight=not (asynchronous or body.get("stream", False)), background=asynchronous,
                     origin=public_url(handler), idempotency_key=handler.headers.get('Idempotency-Key'))
    jid = job["id"]
    handler.job_id = jid
    if job['status'] == 'cancelled':
        handler.reply(job_error(job), job_error_status(job))
        return
    if asynchronous:
        handler.reply(job_receipt(job), 202)
        return
    handler.cancel_on_disconnect = not job.get('background', True)
    if not body.get("stream", False):
        job = wait_job(service, jid, handler=handler)
        if job["status"] != "completed":
            handler.reply(job_error(job), job_error_status(job))
        else:
            handler.reply(completion(job, body))
        return
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache, no-transform")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.send_header("X-Job-Id", jid)
    handler.cors_headers()
    handler.end_headers()
    handler.close_connection = True
    def send(value):
        handler.wfile.write(("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode())
        handler.wfile.flush()
    def chunk(delta, finish=None, **extra):
        return {"id": "chatcmpl-"+jid, "object": "chat.completion.chunk", "created": int(job["created"]),
                "model": job["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **({"timings": job.get("timings")} if job.get("timings_per_token") else {}), **extra}
    try:
        send(chunk({"role": "assistant", "content": ""}))
        sent, after = {"content": "", "reasoning_content": ""}, 0
        server_tools = bool(job.get("image_model"))
        client_tools = bool(job.get("tools"))
        def send_parts(parts, **extra):
            delta = {}
            for field in sent:
                text = parts.get(field) or ""
                if not text.startswith(sent[field]):
                    raise RuntimeError("Runtime revised streamed " + field + "; request non-streaming output for this pipeline")
                if text != sent[field]:
                    delta[field] = text[len(sent[field]):]
                    sent[field] = text
            if delta:
                send(chunk(delta, **extra))
        heartbeat = time.monotonic()
        last_state = None
        while True:
            handler.check_disconnect()
            for event in service.store.events(jid, after):
                after = event["seq"]
                if event["kind"] == "text" and (not server_tools or event.get("stage") == "continuation"):
                    parts = event if "content" in event else split_reasoning(event["text"], event.get("thinking_open", job.get("thinking_open", False)), event.get("final", False))
                    if client_tools:
                        parts = tool_stream_parts(parts)
                    send_parts(parts, x_context=event["context"],
                               **({"timings": event.get("timings")} if job.get("timings_per_token") else {}))
            job = service.job(jid)
            if job["status"] in TERMINAL:
                break
            current_state = (job.get("phase"), len(job.get("artifacts", [])))
            if current_state != last_state or time.monotonic() - heartbeat > 5:
                send(chunk({}, x_job_id=jid, x_status={"status": job["status"], "phase": job.get("phase")},
                           x_artifacts=job["artifacts"], x_context=job["context"], x_image_recovery=job.get('image_recovery')))
                heartbeat, last_state = time.monotonic(), current_state
            time.sleep(.03)
        if job["status"] != "completed":
            send(job_error(job))
        else:
            result = completion(job, body)
            choice = result["choices"][0]
            message = choice["message"]
            send_parts(message)
            if "tool_calls" in message:
                send(chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(message["tool_calls"])]}))
            send(chunk({}, choice["finish_reason"], x_context=result["x_context"], x_artifacts=result["x_artifacts"], x_job_id=jid,
                       x_durable=result['x_durable'], x_storage_warning=result['x_storage_warning'],
                       x_template_context=result["x_template_context"], x_template_notes=result["x_template_notes"],
                       x_prompt_enhancement=result['x_prompt_enhancement'],
                       x_image_recovery=result['x_image_recovery'],
                       x_reasoning_incomplete=result["x_reasoning_incomplete"], x_response_format=result["x_response_format"], x_mtp=result["x_mtp"], x_sampling=result['x_sampling'],
                       **({"x_json_valid": result["x_json_valid"]} if "x_json_valid" in result else {}),
                       **({"x_timing_passes": result["x_timing_passes"]} if job.get("timings_per_token") else {})))
            if body.get("stream_options", {}).get("include_usage"):
                usage = chunk({}); usage.update(choices=[], usage=result["usage"], x_context=result["x_context"]); send(usage)
        send("[DONE]")
    except (ConnectionError, ssl.SSLEOFError):
        # The handler cancels synchronous work on disconnect. Background jobs
        # and their event subscribers have an independent lifetime.
        raise
    except Exception as exc:
        send({"error": {"message": str(exc), "type": "server_error"}, "x_job_id": jid}); send("[DONE]")

def start_image(service, body, editing=False, origin='', idempotency_key=None, background=True):
    signal = DeadlineEvent(timeout=service.config.get('request_timeout_seconds', 1800))
    if idempotency_key is None:
        with service.admission(signal=signal):
            return _start_image(service, body, editing, origin, background=background)
    identity = image_identity(idempotency_key, body, editing)
    with keyed(service, identity, signal):
        prior = service.store.image_request(identity)
        if prior is not None:
            return service.job(prior['id'])
        with service.admission(signal=signal):
            request_id = idempotency_key if re.fullmatch(r'[a-f0-9]{32}', idempotency_key) else None
            return _start_image(service, body, editing, origin, retry_identity=identity, request_id=request_id, background=background)


def _start_image(service, body, editing=False, origin='', retry_identity=None, request_id=None, background=True):
    allowed = {"model", "prompt", "n", "size", "response_format", "user", "steps", "num_inference_steps", "seed", "rng_seed", "guidance_scale", "negative_prompt", "max_sequence_length", *ENHANCEMENT_FIELDS}
    if editing:
        allowed |= EDIT_FIELDS
    if set(body) - allowed:
        raise ValueError("Unsupported image fields: " + ", ".join(sorted(set(body)-allowed)))
    if body.get("n", 1) != 1:
        raise ValueError("Only n=1 is currently supported")
    defaults = service.model(body["model"], "text-to-image").get("generation", {})
    selected_size = body.get('size', 'auto')
    if selected_size == 'auto':
        selected_size = f"{defaults.get('width', 512)}x{defaults.get('height', 512)}"
    size = re.fullmatch(r"([0-9]+)x([0-9]+)", selected_size) if isinstance(selected_size, str) else None
    if not size:
        raise ValueError("size must be WIDTHxHEIGHT")
    format_ = body.get("response_format", "b64_json" if editing else "url")
    if format_ not in {"url", "b64_json"}:
        raise ValueError("response_format must be url or b64_json")
    params = {**body, "width": int(size[1]), "height": int(size[2]), "response_format": format_,
              'image_size_explicit': body.get('size', 'auto') != 'auto'}
    if "num_inference_steps" in body:
        params["steps"] = body["num_inference_steps"]
    if "rng_seed" in body:
        params["seed"] = body["rng_seed"]
    return service.submit_image(params, editing=editing, origin=origin, retry_identity=retry_identity, request_id=request_id, background=background)


def serve_image(handler, service, body, asynchronous=False, editing=False):
    job = start_image(service, body, editing=editing, origin=public_url(handler),
                      idempotency_key=handler.headers.get('Idempotency-Key'), background=asynchronous)
    handler.job_id = job["id"]
    if job['status'] == 'cancelled' and not job['artifacts']:
        handler.reply(job_error(job), job_error_status(job))
        return
    if asynchronous:
        handler.reply(job_receipt(job, public_url(handler)), 202)
        return
    # Replaying a background request key must not change its lifetime policy.
    handler.cancel_on_disconnect = not job.get('background', True)
    job = wait_job(service, job["id"], image_ready=True, handler=handler)
    if not job["artifacts"]:
        handler.reply(job_error(job), job_error_status(job))
        return
    handler.reply(image_completion(handler, service, job, job.get("response_format", "url")))


def image_completion(handler, service, job, format_):
    a = job["artifacts"][0]
    data = {"revised_prompt": a["prompt"],
            "url": public_url(handler) + a["url"]}
    if format_ == "b64_json":
        data["b64_json"] = base64.b64encode((service.store.directory / "artifacts" / a["name"]).read_bytes()).decode()
    return {"created": int(job["created"]), "data": [data], "x_job_id": job["id"], "x_artifacts": job["artifacts"],
                   "x_mode": job["mode"], "x_warning": job["error"],
                   "x_durable": job.get('durable', True), "x_storage_warning": job.get('storage_error'),
                   "x_result_ready": True, "x_handoff_complete": job["status"] == "completed", "x_status": job['status'], "x_phase": job.get('phase'),
                   "artifact_url": data['url'], "artifacts_url": public_url(handler) + '/api/artifacts',
                   "status_url": "/api/jobs/" + job['id'], "result_url": "/api/jobs/" + job['id'] + "/result",
                   "x_operation": job.get('operation', 'generate'), "x_parameter_notes": job.get('parameter_notes', []),
                   "x_prompt_enhancement": job.get('prompt_enhancement'),
                   "x_seed": job['request'].get('seed'),
                   "x_inference": job.get('image_inference'),
                   "x_image_recovery": job.get('image_recovery'),
                   "x_media_type": "image/svg+xml" if a["simulated"] else "image/png"}
