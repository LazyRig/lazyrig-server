import hashlib
import json
import threading
from .templates import render_chat_details
from .tool_calls import parse_tool_calls, ToolCallError
from .response_format import format_messages, is_json
from .vision import text_content
from .deadlines import check_deadline, current_deadline, time_budget, cancellation_requested

DEFAULT_OUTPUT_TOKENS = 8192

SYSTEM = "You are a helpful local assistant. Use generate_image for new images and edit_image, when available, for changes to existing images. Do not claim to see an image unless image input was supplied."
IMAGE_TOOL = {"type": "function", "function": {"name": "generate_image",
    "description": "Generate one image from a detailed text prompt. The server controls its model and device.",
    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"], "additionalProperties": False}}}

def model_messages(messages):
    return [{k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}} for m in messages]

def context_info(count, limit, budget, generated=0, accuracy="exact", **extra):
    return {"input_tokens": count, "generated_tokens": generated,
            "used_tokens": count + generated, "context_limit": limit,
            "output_budget": budget, "remaining_tokens": max(0, limit - count - generated),
            "remaining_after_budget": max(0, limit - count - budget),
            "fits": count + budget <= limit, "accuracy": accuracy,
            "cache_occupancy": None, "scope": "current inference", **extra}

class TokenizerService:
    def __init__(self, mode, images=None):
        self.mode = mode
        self.images = images
        self.tokenizers = {}
        self.lock = threading.RLock()

    def prepare(self, model, history, budget, tools_enabled=True, tools=None, add_system=True):
        signal = current_deadline()

        def check_request():
            check_deadline(signal)
            if signal is not None and cancellation_requested(signal):
                from .worker import Cancelled
                raise Cancelled('Request cancelled during tokenization preparation')

        check_request()
        messages = ([{"role": "system", "content": SYSTEM}] if add_system else []) + model_messages(history)
        if self.images:
            messages, images = self.images.flatten(messages, model)
        else:
            images = []
            for message in messages:
                if isinstance(message.get("content"), list):
                    if any(p.get("type") != "text" for p in message["content"]):
                        raise ValueError("Image storage is not configured")
                    message["content"] = text_content(message["content"])
        messages = format_messages(messages, model.get("response_format"))
        tools = ([IMAGE_TOOL] if tools is None else tools) if tools_enabled else tools
        details = {"template_context": model.get("template_context", {}), "template_notes": []}
        acquired = False
        try:
            # Native tokenizers share a cache, but a request waiting for that
            # cache must still honor its own cancellation/deadline. An encode
            # already executing remains cooperative at the call boundaries.
            while not acquired:
                check_request()
                acquired = self.lock.acquire(timeout=time_budget(.05, signal))
            check_request()
            text, ids, count, accuracy, details = self.encode(model, messages, tools)
            check_request()
        finally:
            if acquired:
                self.lock.release()
        if is_json(model.get("response_format")):
            details["template_notes"].append("JSON response_format disables thinking for this request; native constraints apply from the first generated token.")
        prompt_bytes = text.encode() + (b"\0" + json.dumps([i["sha256"] for i in images]).encode() if images else b"")
        info = context_info(count, int(model["context_limit"]), budget, accuracy=accuracy,
                            prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
                            template_context=details["template_context"], template_notes=details["template_notes"])
        if images:
            info.update(accuracy="estimated-multimodal", image_count=len(images),
                        visual_tokens_included=False, fits=None if info["fits"] else False,
                        note="Preflight counts text and image placeholders. Full input usage is reconciled with native vision preprocessing after generation.")
        return {"text": text, "token_ids": ids, "messages": messages, "context": info,
                "images": images,
                "thinking_open": text.rstrip().endswith("<think>"),
                "template_context": details["template_context"], "template_notes": details["template_notes"]}

    def encode(self, model, messages, tools):
        import openvino_genai as genai
        path = model["path"]
        if path not in self.tokenizers:
            self.tokenizers[path] = genai.Tokenizer(path)
        tokenizer = self.tokenizers[path]
        details = render_chat_details(path, tokenizer, messages, tools, model.get("template_context", {}))
        text = details["text"]
        if is_json(model.get("response_format")) and text.rstrip().endswith("<think>"):
            text += "</think>\n\n"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids = encoded.input_ids.data[0].tolist()
        count = len(ids)
        # LLM receives these exact IDs. VLM receives the rendered string;
        # report preflight conservatively until reconciled with runtime metrics.
        accuracy = "exact" if model["pipeline"] == "llm" else "estimated-template"
        return text, ids, count, accuracy, details

def parse_image_tool(text, thinking_open=False):
    result = parse_tool_calls(text, [IMAGE_TOOL], thinking_open)
    if not result["calls"]:
        return None
    if len(result["calls"]) != 1:
        raise ToolCallError("Only one image call is allowed per request")
    args = result["calls"][0]["arguments"]
    if not isinstance(args, dict) or set(args) != {"prompt"} or not isinstance(args["prompt"], str):
        raise ToolCallError("Image tool must contain only a string prompt")
    if not 1 <= len(args["prompt"].strip()) <= 8000:
        raise ToolCallError("Image prompt must have 1–8000 characters")
    return args
